"""Traffic HTGNN: SGMP -> temporal/heterogeneous GNN -> +PE.

The dataset stores static positional features in ``ndata['pe']`` and dynamic
segment observations in ``ndata['dynamic_tN']``.  This module deliberately
keeps those branches separate so positional information is only added to the
final segment representation, as in the requested architecture.
"""

from collections import defaultdict

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from sgmp import SGMPImputer


def _base_relation(edge_type: str) -> str:
    relation, separator, timestamp = edge_type.rpartition("_")
    if not separator or not timestamp.startswith("t") or not timestamp[1:].isdigit():
        raise ValueError(f"Expected a time-suffixed edge type, got {edge_type!r}")
    return relation


class RelationAggregation(nn.Module):
    """Aggregate one message per incoming relation with learned attention."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, messages: list[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(messages, dim=1)
        weights = torch.softmax(self.score(stacked).mean(dim=0), dim=0)
        return (stacked * weights.unsqueeze(0)).sum(dim=1)


class HeterogeneousGraphLayer(nn.Module):
    """Relation-specific mean message passing and residual relation fusion."""

    def __init__(self, graph: dgl.DGLGraph, hidden_dim: int, dropout: float):
        super().__init__()
        keys = []
        for src_type, edge_type, dst_type in graph.canonical_etypes:
            key = f"{src_type}__{_base_relation(edge_type)}__{dst_type}"
            if key not in keys:
                keys.append(key)
        self.transforms = nn.ModuleDict(
            {key: nn.Linear(hidden_dim, hidden_dim, bias=False) for key in keys}
        )
        self.aggregators = nn.ModuleDict(
            {node_type: RelationAggregation(hidden_dim) for node_type in graph.ntypes}
        )
        self.norms = nn.ModuleDict(
            {node_type: nn.LayerNorm(hidden_dim) for node_type in graph.ntypes}
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph: dgl.DGLGraph, features: dict[str, torch.Tensor]):
        incoming = defaultdict(list)
        seen = set()
        for src_type, edge_type, dst_type in graph.canonical_etypes:
            relation = _base_relation(edge_type)
            key = f"{src_type}__{relation}__{dst_type}"
            if key in seen:  # topology is identical for every timestamp
                continue
            seen.add(key)
            relation_graph = graph[src_type, edge_type, dst_type]
            source, destination = relation_graph.edges()
            transformed = self.transforms[key](features[src_type])
            message = torch.zeros(
                relation_graph.num_dst_nodes(), transformed.shape[-1],
                device=transformed.device, dtype=transformed.dtype,
            )
            message.index_add_(0, destination, transformed[source])
            degree = torch.zeros(
                relation_graph.num_dst_nodes(), device=transformed.device,
                dtype=transformed.dtype,
            )
            degree.index_add_(0, destination, torch.ones_like(destination, dtype=degree.dtype))
            incoming[dst_type].append(message / degree.clamp_min(1).unsqueeze(-1))

        output = {}
        for node_type, residual in features.items():
            if not incoming[node_type]:
                output[node_type] = residual
                continue
            fused = self.aggregators[node_type](incoming[node_type])
            output[node_type] = self.norms[node_type](
                residual + self.dropout(F.relu(fused))
            )
        return output


class HTGNN(nn.Module):
    """Per-segment encoder following ``G^t -> SGMP -> HTGNN -> +PE``."""

    def __init__(
        self,
        graph: dgl.DGLGraph,
        n_hid: int,
        n_layers: int,
        time_window: int,
        dropout: float = 0.2,
        inp_list: dict[str, int] | None = None,
        dynamic_input_dim: int | None = None,
        velocity_feature_index: int | None = None,
        velocity_mask_feature_index: int | None = None,
        sgmp_order: int = 2,
        **_ignored,
    ):
        super().__init__()
        if inp_list is None or dynamic_input_dim is None:
            raise ValueError("HTGNN requires inp_list and dynamic_input_dim")
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if (velocity_feature_index is None) != (velocity_mask_feature_index is None):
            raise ValueError("Both velocity and observed-mask indices are required")

        self.timeframe = [f"t{i}" for i in range(time_window)]
        self.velocity_feature_index = velocity_feature_index
        self.velocity_mask_feature_index = velocity_mask_feature_index
        self.sgmp = SGMPImputer(sgmp_order) if velocity_feature_index is not None else None

        self.dynamic_projection = nn.Linear(dynamic_input_dim, n_hid)
        self.temporal_encoder = nn.GRU(n_hid, n_hid, batch_first=True)
        self.static_projection = nn.ModuleDict(
            {node_type: nn.Linear(inp_list[node_type], n_hid) for node_type in graph.ntypes}
        )
        self.layers = nn.ModuleList(
            [HeterogeneousGraphLayer(graph, n_hid, dropout) for _ in range(n_layers)]
        )
        self.output_norm = nn.LayerNorm(n_hid)
        self.dropout = nn.Dropout(dropout)

    def _segment_edges(self, graph: dgl.DGLGraph):
        for src_type, edge_type, dst_type in graph.canonical_etypes:
            if src_type == dst_type == "segment" and _base_relation(edge_type) in {
                "connects_to", "connects_with"
            }:
                return graph[src_type, edge_type, dst_type].edges()
        raise ValueError("SGMP requires a segment connects_to/connects_with relation")

    def forward(self, graph: dgl.DGLGraph, predict_type: str = "segment"):
        if predict_type != "segment":
            raise ValueError("This traffic HTGNN predicts segment nodes only")

        sequence = torch.stack(
            [graph.nodes["segment"].data[f"dynamic_{time}"] for time in self.timeframe],
            dim=1,
        )
        if self.sgmp is not None:
            source, destination = self._segment_edges(graph)
            completed = self.sgmp(
                sequence[:, :, self.velocity_feature_index],
                sequence[:, :, self.velocity_mask_feature_index],
                source,
                destination,
            )
            sequence = sequence.clone()
            sequence[:, :, self.velocity_feature_index] = completed

        encoded_sequence = self.dropout(F.relu(self.dynamic_projection(sequence)))
        _, hidden = self.temporal_encoder(encoded_sequence)
        features = {
            node_type: self.static_projection[node_type](graph.nodes[node_type].data["pe"])
            for node_type in graph.ntypes
        }
        # Dynamic traffic is the segment state passed into HTGNN.  Segment PE is
        # retained separately and added only after heterogeneous message passing.
        segment_pe = features["segment"]
        features["segment"] = hidden[-1]
        for layer in self.layers:
            features = layer(graph, features)
        return self.output_norm(features["segment"] + segment_pe)


class NodePredictor(nn.Module):
    """Map each segment embedding to y^(t+1), ..., y^(t+Q)."""

    def __init__(self, n_inp: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_inp, n_inp), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(n_inp, output_dim),
        )

    def forward(self, node_features: torch.Tensor):
        return self.network(node_features)
