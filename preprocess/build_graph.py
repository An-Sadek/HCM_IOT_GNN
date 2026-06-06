import os
from pathlib import Path
import json

import numpy as np
import pandas as pd

import torch

from torch_geometric.data import HeteroData

PREPROCESS_ROOT = Path("data/preprocess")
RAW_ROOT = Path("../data/raw")


def build_static_graph():
    # Load csv
    nodes_df = pd.read_csv(PREPROCESS_ROOT / "nodes.csv")
    segments_df = pd.read_csv(PREPROCESS_ROOT / "segments.csv")
    ways_df = pd.read_csv(PREPROCESS_ROOT / "ways.csv")
    nodes_segments_edges_df = pd.read_csv(
        PREPROCESS_ROOT / "nodes_segments_edges_df.csv"
    )
    way2way_df = pd.read_csv(PREPROCESS_ROOT / "way2way.csv")
    way_restrictions_df = pd.read_csv(PREPROCESS_ROOT / "relation_members.csv")

    static_node_features = np.load(PREPROCESS_ROOT / "static_nodes.npy")
    static_segment_features = np.load(PREPROCESS_ROOT / "static_segments.npy")
    static_way_features = np.load(PREPROCESS_ROOT / "static_ways.npy")

    data = HeteroData()

    # Node
    data["node"] = static_node_features
    data["segment"] = static_segment_features
    data["way"] = static_way_features

    # Edge
    # way <-> segment
    contain_segments = (
        segments_df[["street_id", "id"]]
        .drop_duplicates()
        .to_numpy()
        .astype(int)
    )
    data["way", "contains_segment", "segment"] = contain_segments
    data["segment", "contained_in", "way"] = contain_segments[:, [1, 0]]

    # segment <-> node
    ## segment <-> start_node
    starts_with = (
        segments_df[["id", "s_node_id"]]        
        .drop_duplicates()
        .to_numpy()
        .astype(int)
    )
    data["way", "starts_swith", "s_node"] = starts_with
    data["s_node", "start_of", "way"] = starts_with[:, [1, 0]]

    ## segment <-> end_node
    ends_with = (
        segments_df[["id", "e_node_id"]]        
        .drop_duplicates()
        .to_numpy()
        .astype(int)
    )
    data["way", "ends_with", "s_node"] = ends_with
    data["s_node", "end_of", "way"] = ends_with[:, [1, 0]]

    # way <-> way
    way_start_nodes_df = way2way_df[["id", "start_node"]].rename(columns={"id": "start_id"})
    way_end_nodes_df = way2way_df[["id", "end_node"]].rename(columns={"id": "end_id"})
    merge_way2way_df = way_start_nodes_df.merge(
        way_end_nodes_df,
        how="inner",
        left_on = "start_id",
        right_on = "end_id"
    )
    way2way_connections_df = merge_way2way_df[["start_id", "end_id"]]
    
    # Loại bỏ đường cấm rẽ
    way_restrictions_df = way_restrictions_df[way_restrictions_df["type"] == "node"] # Bỏ node ra (via)

    ## Tạo các restriction
    way_restrictions = []
    way_restrictions_group = way_restrictions_df.groupby("id")
    for _, group in way_restrictions_group:
        start_way_id = group[group["role" == "from"]]["ref"].values[0]
        start_way_id = int(start_way_id)
        end_way_id = group[group["role" == "to"]]["ref"].values[0]
        end_way_id = int(end_way_id)
        way_restrictions = way_restrictions.append([start_way_id, end_way_id])

    ## Lọc các restriction
    way2way_connections_df = way2way_connections_df[~
        way2way_connections_df[["start_node", "end_node"]].isin(way_restrictions) 
    ]

    
if __name__ == "__main__":
    build_static_graph()