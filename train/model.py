import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from torch.nn import GRU
try:
    from torch_geometric.nn.norm import LayerNorm
except ImportError:
    LayerNorm = nn.LayerNorm
from collections import defaultdict

import torch as th
from torch import nn
from dgl import function as fn
from dgl.utils import expand_as_pair

class GraphConv(nn.Module):

    def __init__(
        self,
        norm="both",
        activation=None,
    ):
        super(GraphConv, self).__init__()

        self._norm = norm
        self._activation = activation

    def forward(self, graph, feat,):

        with graph.local_scope():

            aggregate_fn = fn.copy_u("h", "m")
            feat_src, feat_dst = expand_as_pair(feat, graph)

            if self._norm in ["left", "both"]:
                degs = graph.out_degrees().to(feat_src).clamp(min=1)+1
                if self._norm == "both":
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = th.reshape(norm, shp)
                feat_src = feat_src * norm

            graph.srcdata["h"] = feat_src
            graph.update_all(aggregate_fn, fn.sum(msg="m", out="h_neigh"))
            rst = feat_dst+graph.dstdata["h_neigh"]
            # rst = graph.dstdata["h_neigh"]

            if self._norm in ["right", "both"]:
                degs = graph.in_degrees().to(feat_dst).clamp(min=1) +1
                if self._norm == "both":
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm


            if self._activation is not None:
                rst = self._activation(rst)

            return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = "in={_in_feats}, out={_out_feats}"
        summary += ", normalization={_norm}"
        if "_activation" in self.__dict__:
            summary += ", activation={_activation}"
        return summary.format(**self.__dict__)


class Dyanmicatt(nn.Module):
    def __init__(self,n_inp, n_hid, layer=1):
        super(Dyanmicatt, self).__init__()
        self.gru = GRU(n_inp, 1, layer, batch_first=True)   # batch_first=True

    def forward(self, x, mask):
        out, _ = self.gru(x, mask)
        mask = torch.split(out.mean(0), 1, dim=0)
        dict_mask = {f"t{i}":mask[i].squeeze() for i in range(len(mask))} #(1)[0]

        return dict_mask


class linearproj(nn.Module):
    def __init__(self, n_hid, timeframe):
        super(linearproj, self).__init__()
        self.project = nn.Linear(len(timeframe), 1)

    def forward(self, h):
        out = self.project(h).mean(-1)
        return out


class HTGNNLayer(nn.Module):
    def __init__(self, graph: dgl.DGLGraph, n_inp: int, n_hid: int, n_heads: int, timeframe: list, norm: bool, device: torch.device, dropout: float,LLM_feature):
        """

        :param graph    : dgl.DGLGraph, a heterogeneous graph
        :param n_inp    : int         , input dimension
        :param n_hid    : int         , hidden dimension
        :param n_heads  : int         , number of attention heads
        :param timeframe: list        , list of time slice
        :param norm     : bool        , use LayerNorm or not
        :param device   : torch.device, gpu
        :param dropout  : float       , dropout rate
        """
        super(HTGNNLayer, self).__init__()
        
        self.n_inp     = n_inp
        self.n_hid     = n_hid
        self.n_heads   = n_heads
        self.timeframe = timeframe
        self.norm      = norm
        self.dropout   = dropout
        self.device    = device
        self.GRUlayer = 1
        # intra reltion aggregation modules
        self.LLM_features = LLM_feature
        self.intra_rel_agg = GraphConv(norm='right',activation= nn.ELU())

        # dynamic-attention
        self.predict = nn.ModuleDict({
            etype[0:-(len(etype.split('_')[-1]) + 1)]: Dyanmicatt(n_inp, n_hid, self.GRUlayer)
            for srctype, etype, dsttype in graph.canonical_etypes
        })

        if norm:
            self.norm_layer = nn.ModuleDict({ntype: LayerNorm(n_hid,) for ntype in graph.ntypes})


    def forward(self, graph: dgl.DGLGraph, node_features: dict, init_parameter):
        """

        :param graph: dgl.DGLGraph
        :param node_features: dict, {'ntype': {'ttype': features}}
        :return: output_features: dict, {'ntype': {'ttype': features}}
        """
        init_attention= init_parameter

        # same type neighbors aggregation
        # intra_features, dict, {'ttype': {(stype, etype, dtype): features}}
        intra_features = dict({ttype:{} for ttype in self.timeframe})
        h_mask = {etype[0:-(len(etype.split('_')[-1])+1)]: [] for stype, etype, dtype in graph.canonical_etypes}

        for stype, etype, dtype in graph.canonical_etypes:
            rel_graph = graph[stype, etype, dtype]
            ttype = etype.split('_')[-1]
            reltype = etype[0:-(len(ttype)+1)]
            dst_feat = self.intra_rel_agg(rel_graph, (node_features[stype][ttype], node_features[dtype][ttype]), ) #
            intra_features[ttype][(stype, etype, dtype)] = dst_feat
            h_mask[reltype].append(dst_feat)

        # attention_coefficient predcit
        h_mask = {key: self.predict[key](torch.stack(h_mask[key], dim=1), #[key]
                 init_attention[key].expand(self.GRUlayer, h_mask[key][0].size(0), 1).to(self.device), ) for
                  key in
                  h_mask.keys()
                  }

        # dynamic-attention-based fusion
        # inter_features, dict, {'ntype': {ttype: features}}
        inter_features = dict({ntype:{} for ntype in graph.ntypes})
        for ttype in intra_features.keys():
            for ntype in graph.ntypes:
                types_features = []
                types_weight = []
                for stype, etype, dtype in intra_features[ttype]:
                    if ntype == dtype:
                        reltype = etype[0:-(len(ttype) + 1)]
                        weight = h_mask[reltype][ttype]
                        types_weight.append(weight)
                        types_features.append(intra_features[ttype][(stype, etype, dtype)])
                out_feat = []
                #weight softmax
                types_weight = F.softmax(torch.stack(types_weight, dim=0), dim=0)
                for i in range(len(types_features)):
                    out_feat.append(types_features[i]*types_weight[i])
                out_feat = sum(out_feat)
                inter_features[ntype][ttype] = self.norm_layer[ntype](out_feat) if self.norm else (out_feat+node_features[ntype][ttype])


        return inter_features


class LLM4init(nn.Module):

    def __init__(self, graph: dgl.DGLGraph, n_inp: int, n_hid: int, LLM_feature, device):
        super(LLM4init, self).__init__()
        self.device = device
        self.nid = n_hid
        self.consistent = nn.Linear(n_inp, n_hid)
        # self.k_w = nn.Linear(n_hid, n_hid, bias=False)
        # self.q_w = nn.Linear(n_hid, n_hid, bias=False)
        self.LLM_feature = LLM_feature
        self.reltype = []
        self.Relation_feat_init(graph)

    def Relation_feat_init(self, graph):
        for stype, etype, dtype in graph.canonical_etypes:
            ttype = etype.split('_')[-1]
            reltype = etype[0:-(len(ttype) + 1)]
            if (stype, reltype, dtype) not in self.reltype:
                self.reltype.append((stype, reltype, dtype))

    def forward(self):
        feature_dict = {}
        for key, feature in self.LLM_feature.items():
            feature = feature.to(self.device)
            feature_dict[key] = feature
            # feature_dict[key] = self.consistent(feature)

        grouped_edges = defaultdict(list)
        for stype, etype, dtype in  self.reltype:
            # stype_feature = self.k_w(feature_dict[stype])
            # dtype_feature = self.q_w(feature_dict[dtype])
            stype_feature = feature_dict[stype]
            dtype_feature = feature_dict[dtype]
            inner_product = torch.dot(stype_feature.squeeze(), dtype_feature.squeeze())
            grouped_edges[dtype].append((inner_product, stype, etype, dtype))

        normalized_inner_products = {}
        for dtype, edges in grouped_edges.items():
            inner_products = torch.tensor([edge[0] for edge in edges])
            softmax_weights = F.softmax(torch.log(inner_products), dim=0)
            for i, (inner_product, stype, etype, dtype) in enumerate(edges):
                normalized_inner_products[etype] = softmax_weights[i]
                # normalized_inner_products[etype] = 0.5

        return normalized_inner_products,  feature_dict


class SEHTGNN(nn.Module):
    def __init__(self, graph: dgl.DGLGraph, n_inp: int, n_hid: int, n_layers: int, n_heads: int, time_window: int, norm: bool, device: torch.device, dropout: float = 0.2, LLM_feature = None, inp_list = None):
        """

        :param graph      : dgl.DGLGraph, a dgl heterogeneous graph
        :param n_inp      : int         , input dimension
        :param n_hid      : int         , hidden dimension
        :param n_layers   : int         , number of stacked layers
        :param n_heads    : int         , number of attention heads

        :param time_window: int         , number of timestamps
        :param norm       : bool        , use Norm or not
        :param device     : torch.device, gpu
        :param dropout    : float       , dropout rate
        """
        super(SEHTGNN, self).__init__()

        self.n_inp     = n_inp
        self.n_hid     = n_hid
        self.n_layers  = n_layers
        self.n_heads   = n_heads
        self.timeframe = [f't{_}' for _ in range(time_window)]
        self.drop = nn.Dropout(dropout)
        self.norm = norm

        # feature projector
        if inp_list is None:
            self.adaption_layer = nn.ModuleDict({ntype: nn.Linear(n_inp, n_hid) for ntype in graph.ntypes})
        else:
            self.adaption_layer = nn.ModuleDict({ntype: nn.Linear(inp_list[ntype], n_hid) for ntype in graph.ntypes})

        # LLM-enhanced prompt
        self.LLM_init = LLM4init(graph, 4096, n_hid, LLM_feature,device)

        # Dynamic-attention-based graph learning
        self.gnn_layers = nn.ModuleDict({str(i): HTGNNLayer(graph, n_hid, n_hid, n_heads, self.timeframe, norm, device, dropout, LLM_feature) for i in range(self.n_layers)})

        # Linear projection
        self.LinearProj = linearproj(n_hid, self.timeframe)


    def forward(self, graph: dgl.DGLGraph, predict_type: str):
        """
        :param graph       : dgl.DGLGraph, a dgl heterogeneous graph
        :param predict_type: str         , predicted node type
        """
        # LLM-enhanced initialization
        init_attention, init_temporal = self.LLM_init()

        # Heterogeneous feature projection & drop
        # inp_feat: dict, {'ntype': {'ttype': features}}
        inp_feat = {}
        for ntype in graph.ntypes:
            inp_feat[ntype] = {}
            for ttype in self.timeframe:
                inp_feat[ntype][ttype] = self.adaption_layer[ntype](self.drop(graph.nodes[ntype].data[ttype]),)
        spatial_feat = inp_feat

        # Dynamic attention-based spatial learning
        for step in range(self.n_layers):
            spatial_feat = self.gnn_layers[str(step)](graph, spatial_feat, init_attention)

        # Linear project
        out_feat = self.LinearProj(torch.stack([spatial_feat[predict_type][ttype] for ttype in self.timeframe],dim=2) )

        return out_feat



class NodePredictor(nn.Module):
    def __init__(self, n_inp: int, n_classes: int):
        """
    
        :param n_inp      : int, input dimension
        :param n_classes  : int, number of classes
        """
        super().__init__()

        self.fc1 = nn.Linear(n_inp, n_inp)
        self.fc2 =  nn.Linear(n_inp, n_classes)

    def forward(self, node_feat: torch.tensor):
        """
        
        :param node_feat: torch.tensor
        """

        node_feat = F.relu(self.fc1(node_feat))
        # pred = F.relu(self.fc2(node_feat))
        pred = self.fc2(node_feat)

        return pred
