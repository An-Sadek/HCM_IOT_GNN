"""HTGNN adapted to node-level traffic forecasting on the HCM heterograph.

The official EPFL-IMOS implementation targets graph-level regression with
low/high-frequency sensor types.  This adaptation keeps its core ordering:
type-specific temporal encoders -> heterogeneous graph layers -> prediction
head, while producing one embedding per road segment.
"""

from collections import defaultdict

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GraphConv
from sgmp import SGMPImputer


def _base_relation(edge_type: str) -> str:
    """Remove the final ``_tN`` suffix from a temporal DGL edge type."""
    relation, separator, time_name = edge_type.rpartition("_")
    if not separator or not time_name.startswith("t") or not time_name[1:].isdigit():
        raise ValueError(f"Expected a time-suffixed edge type, got {edge_type!r}")
    return relation


class TypeTemporalEncoder(nn.Module):
    """Encode each node's input window with a type-specific GRU."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        sequence = self.dropout(F.relu(self.input_projection(sequence)))
        _, hidden = self.gru(sequence)
        return self.norm(hidden[-1])


class HeterogeneousGraphLayer(nn.Module):
    """Relation-specific message passing followed by learned relation fusion."""

    def __init__(self, graph: dgl.DGLGraph, hidden_dim: int, dropout: float):
        super().__init__()
        relation_keys = []
        relation_destinations = {}
        for source_type, edge_type, destination_type in graph.canonical_etypes:
            relation = _base_relation(edge_type)
            key = f"{source_type}__{relation}__{destination_type}"
            if key not in relation_destinations:
                relation_keys.append(key)
                relation_destinations[key] = destination_type

        self.relation_keys = relation_keys
        self.relation_destinations = relation_destinations
        self.transforms = nn.ModuleDict(
            {key: nn.Linear(hidden_dim, hidden_dim, bias=False) for key in relation_keys}
        )
        self.relation_scores = nn.ParameterDict(
            {key: nn.Parameter(torch.zeros(())) for key in relation_keys}
        )
        self.norms = nn.ModuleDict(
            {node_type: nn.LayerNorm(hidden_dim) for node_type in graph.ntypes}
        )
        self.graph_conv = GraphConv(norm="right")
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph: dgl.DGLGraph, features: dict[str, torch.Tensor]):
        messages = defaultdict(list)
        scores = defaultdict(list)
        seen = set()

        # Topology is identical at every timestamp, so use one copy per base relation.
        for source_type, edge_type, destination_type in graph.canonical_etypes:
            relation = _base_relation(edge_type)
            key = f"{source_type}__{relation}__{destination_type}"
            if key in seen:
                continue
            seen.add(key)
            relation_graph = graph[source_type, edge_type, destination_type]
            message = self.graph_conv(
                relation_graph,
                (features[source_type], features[destination_type]),
            )
            messages[destination_type].append(self.transforms[key](message))
            scores[destination_type].append(self.relation_scores[key])

        output = {}
        for node_type, residual in features.items():
            if node_type not in messages:
                output[node_type] = residual
                continue
            weights = F.softmax(torch.stack(scores[node_type]), dim=0)
            fused = sum(weight * message for weight, message in zip(weights, messages[node_type]))
            output[node_type] = self.norms[node_type](
                residual + self.dropout(F.relu(fused))
            )
        return output


class HTGNN(nn.Module):
    """Heterogeneous Temporal GNN for per-segment traffic forecasting."""

    def __init__(
        self,
        graph: dgl.DGLGraph,
        n_hid: int,
        n_layers: int,
        time_window: int,
        dropout: float = 0.2,
        inp_list: dict[str, int] | None = None,
        velocity_feature_index: int | None = None,
        velocity_mask_feature_index: int | None = None,
        sgmp_order: int = 2,
        **_ignored,
    ):
        super().__init__()
        if inp_list is None:
            raise ValueError("HTGNN requires inp_list with one input dimension per node type")
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")

        self.timeframe = [f"t{i}" for i in range(time_window)]
        self.velocity_feature_index = velocity_feature_index
        self.velocity_mask_feature_index = velocity_mask_feature_index
        if (velocity_feature_index is None) != (velocity_mask_feature_index is None):
            raise ValueError("Both SGMP velocity and mask feature indices are required")
        self.sgmp = (
            SGMPImputer(order=sgmp_order)
            if velocity_feature_index is not None
            else None
        )
        self.temporal_encoders = nn.ModuleDict(
            {
                node_type: TypeTemporalEncoder(inp_list[node_type], n_hid, dropout)
                for node_type in graph.ntypes
            }
        )
        self.gnn_layers = nn.ModuleList(
            [HeterogeneousGraphLayer(graph, n_hid, dropout) for _ in range(n_layers)]
        )

    def forward(self, graph: dgl.DGLGraph, predict_type: str = "segment"):
        features = {}
        for node_type in graph.ntypes:
            sequence = torch.stack(
                [graph.nodes[node_type].data[time] for time in self.timeframe], dim=1
            )
            if node_type == "segment" and self.sgmp is not None:
                relation_graph = None
                for source_type, edge_type, destination_type in graph.canonical_etypes:
                    if (
                        source_type == "segment"
                        and destination_type == "segment"
                        and _base_relation(edge_type) == "connects_to"
                    ):
                        relation_graph = graph[source_type, edge_type, destination_type]
                        break
                if relation_graph is None:
                    raise ValueError("SGMP requires a segment-connects_to-segment relation")
                source, destination = relation_graph.edges()
                completed_velocity = self.sgmp(
                    sequence[:, :, self.velocity_feature_index],
                    sequence[:, :, self.velocity_mask_feature_index],
                    source,
                    destination,
                )
                sequence = sequence.clone()
                sequence[:, :, self.velocity_feature_index] = completed_velocity
            features[node_type] = self.temporal_encoders[node_type](sequence)

        for layer in self.gnn_layers:
            features = layer(graph, features)
        return features[predict_type]


class NodePredictor(nn.Module):
    def __init__(self, n_inp: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_inp, n_inp),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_inp, output_dim),
        )

    def forward(self, node_features: torch.Tensor):
        return self.network(node_features)
