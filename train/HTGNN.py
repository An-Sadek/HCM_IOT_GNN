"""HTGNN adapted from the authors' reference implementation.

The model keeps the original HTGNN computation order:

1. relation-specific GAT aggregation at every timestamp;
2. attention-based fusion of incoming relations;
3. self-attention across timestamps;
4. a learned residual gate; and
5. summation of the timestamp representations.

Only the input adapters differ from the reference repository: traffic node types
have different feature widths, so ``inp_list`` supplies one width per node type.
"""

import math

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv


class RelationAgg(nn.Module):
    """Attention-based aggregation over incoming relation types."""

    def __init__(self, n_inp: int, n_hid: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(n_inp, n_hid), nn.Tanh(), nn.Linear(n_hid, 1, bias=False)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.project(features).mean(0), dim=0)
        weights = weights.expand((features.shape[0],) + weights.shape)
        return (weights * features).sum(1)


class TemporalAgg(nn.Module):
    """Original HTGNN dot-product attention across timestamps."""

    def __init__(
        self, n_inp: int, n_hid: int, time_window: int, device: torch.device
    ):
        super().__init__()
        self.proj = nn.Linear(n_inp, n_hid)
        self.q_w = nn.Linear(n_hid, n_hid, bias=False)
        self.k_w = nn.Linear(n_hid, n_hid, bias=False)
        self.v_w = nn.Linear(n_hid, n_hid, bias=False)
        self.fc = nn.Linear(n_hid, n_hid)
        self.register_buffer(
            "pe",
            torch.tensor(
                self.generate_positional_encoding(n_hid, time_window),
                dtype=torch.float32,
                device=device,
            ),
        )

    @staticmethod
    def generate_positional_encoding(d_model: int, max_len: int) -> np.ndarray:
        pe = np.zeros((max_len, d_model))
        for timestamp in range(max_len):
            for channel in range(0, d_model, 2):
                divisor = math.exp(channel * -math.log(100000.0) / d_model)
                pe[timestamp, channel] = math.sin((timestamp + 1) * divisor)
                if channel + 1 < d_model:
                    pe[timestamp, channel + 1] = math.cos(
                        (timestamp + 1) * divisor
                    )
        return pe

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.proj(features.permute(1, 0, 2)) + self.pe
        query = self.q_w(hidden)
        key = self.k_w(hidden)
        value = self.v_w(hidden)
        scores = torch.softmax(torch.matmul(query, key.permute(0, 2, 1)), dim=-1)
        return F.relu(self.fc(torch.matmul(scores, value)))


class HTGNNLayer(nn.Module):
    """One spatial, relation, temporal, and residual HTGNN block."""

    def __init__(
        self,
        graph: dgl.DGLGraph,
        n_inp: int,
        n_hid: int,
        n_heads: int,
        timeframe: list[str],
        norm: bool,
        device: torch.device,
        dropout: float,
    ):
        super().__init__()
        if n_heads != 1:
            raise ValueError(
                "The reference HTGNN implementation requires n_heads=1 because "
                "it squeezes the GAT head dimension before relation aggregation."
            )
        self.timeframe = timeframe
        self.norm = norm
        self.intra_rel_agg = nn.ModuleDict(
            {
                etype: GATConv(
                    n_inp,
                    n_hid,
                    n_heads,
                    feat_drop=dropout,
                    allow_zero_in_degree=True,
                )
                for _, etype, _ in graph.canonical_etypes
            }
        )
        self.inter_rel_agg = nn.ModuleDict(
            {timestamp: RelationAgg(n_hid, n_hid) for timestamp in timeframe}
        )
        self.cross_time_agg = nn.ModuleDict(
            {
                ntype: TemporalAgg(n_hid, n_hid, len(timeframe), device)
                for ntype in graph.ntypes
            }
        )
        self.res_fc = nn.ModuleDict(
            {ntype: nn.Linear(n_inp, n_hid) for ntype in graph.ntypes}
        )
        self.res_weight = nn.ParameterDict(
            {ntype: nn.Parameter(torch.randn(1)) for ntype in graph.ntypes}
        )
        self.norm_layer = (
            nn.ModuleDict({ntype: nn.LayerNorm(n_hid) for ntype in graph.ntypes})
            if norm
            else None
        )
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for layer in self.res_fc.values():
            nn.init.xavier_normal_(layer.weight, gain=gain)

    def forward(
        self, graph: dgl.DGLGraph, node_features: dict[str, dict[str, torch.Tensor]]
    ) -> dict[str, dict[str, torch.Tensor]]:
        intra_features = {timestamp: {} for timestamp in self.timeframe}
        for src_type, edge_type, dst_type in graph.canonical_etypes:
            timestamp = edge_type.rsplit("_", 1)[-1]
            relation_graph = graph[src_type, edge_type, dst_type]
            destination = self.intra_rel_agg[edge_type](
                relation_graph,
                (
                    node_features[src_type][timestamp],
                    node_features[dst_type][timestamp],
                ),
            )
            intra_features[timestamp][(src_type, edge_type, dst_type)] = (
                destination.squeeze(1)
            )

        inter_features = {ntype: {} for ntype in graph.ntypes}
        for timestamp in self.timeframe:
            for ntype in graph.ntypes:
                relations = [
                    feature
                    for (_, _, dst_type), feature in intra_features[timestamp].items()
                    if dst_type == ntype
                ]
                # The traffic graph has source-only node/way types. The authors'
                # datasets do not; carrying their state forward is the neutral
                # extension needed for stacking multiple reference HTGNN layers.
                inter_features[ntype][timestamp] = (
                    self.inter_rel_agg[timestamp](torch.stack(relations, dim=1))
                    if relations
                    else node_features[ntype][timestamp]
                )

        output_features = {}
        for ntype in graph.ntypes:
            embeddings = torch.stack(
                [inter_features[ntype][timestamp] for timestamp in self.timeframe],
                dim=0,
            )
            temporal = self.cross_time_agg[ntype](embeddings).permute(1, 0, 2)
            alpha = torch.sigmoid(self.res_weight[ntype])
            output_features[ntype] = {}
            for index, timestamp in enumerate(self.timeframe):
                hidden = temporal[index] * alpha + self.res_fc[ntype](
                    node_features[ntype][timestamp]
                ) * (1 - alpha)
                output_features[ntype][timestamp] = (
                    self.norm_layer[ntype](hidden) if self.norm else hidden
                )
        return output_features


class HTGNN(nn.Module):
    """Reference HTGNN encoder with per-node-type traffic input adapters."""

    def __init__(
        self,
        graph: dgl.DGLGraph,
        n_inp: int,
        n_hid: int,
        n_layers: int,
        n_heads: int,
        time_window: int,
        norm: bool,
        device: torch.device,
        dropout: float = 0.2,
        inp_list: dict[str, int] | None = None,
        **_ignored,
    ):
        super().__init__()
        if inp_list is None:
            inp_list = {ntype: n_inp for ntype in graph.ntypes}
        self.n_layers = n_layers
        self.timeframe = [f"t{index}" for index in range(time_window)]
        self.adaption_layer = nn.ModuleDict(
            {ntype: nn.Linear(inp_list[ntype], n_hid) for ntype in graph.ntypes}
        )
        self.gnn_layers = nn.ModuleList(
            [
                HTGNNLayer(
                    graph,
                    n_hid,
                    n_hid,
                    n_heads,
                    self.timeframe,
                    norm,
                    device,
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, graph: dgl.DGLGraph, predict_type: str = "segment"):
        features = {
            ntype: {
                timestamp: self.adaption_layer[ntype](
                    graph.nodes[ntype].data[timestamp]
                )
                for timestamp in self.timeframe
            }
            for ntype in graph.ntypes
        }
        for layer in self.gnn_layers:
            features = layer(graph, features)
        return sum(features[predict_type][timestamp] for timestamp in self.timeframe)


class NodePredictor(nn.Module):
    """Two-layer node predictor from the reference implementation."""

    def __init__(self, n_inp: int, output_dim: int, **_ignored):
        super().__init__()
        self.fc1 = nn.Linear(n_inp, n_inp)
        self.fc2 = nn.Linear(n_inp, output_dim)

    def forward(self, node_features: torch.Tensor):
        return F.relu(self.fc2(F.relu(self.fc1(node_features))))
