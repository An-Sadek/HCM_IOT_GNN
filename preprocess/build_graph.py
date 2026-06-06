from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import HeteroData


PREPROCESS_ROOT = Path("data/preprocess")
METADATA_ROOT = Path("metadata")


def load_id2index(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f)

    return {
        int(raw_id): int(index)
        for raw_id, index in metadata["conversion"]["id2index"].items()
    }

def build_node_pairs():
    pass

def build_restricted_way_pairs(relation_members_df, way_id2index):
    relation_col = (
        "relation_id"
        if "relation_id" in relation_members_df.columns
        else "id"
    )
    way_members_df = relation_members_df[
        relation_members_df["type"] == "way"
    ].copy()

    restricted_pairs = set()
    for _, group in way_members_df.groupby(relation_col):
        from_refs = group.loc[group["role"] == "from", "ref"]
        to_refs = group.loc[group["role"] == "to", "ref"]
        if from_refs.empty or to_refs.empty:
            continue

        from_way = way_id2index.get(int(from_refs.iloc[0]))
        to_way = way_id2index.get(int(to_refs.iloc[0]))
        if from_way is None or to_way is None:
            continue

        restricted_pairs.add((from_way, to_way))

    return restricted_pairs


def build_way_connections(way2way_df, restricted_pairs):
    way_start_nodes_df = way2way_df[["id", "start_node"]].rename(
        columns={"id": "to_way", "start_node": "junction_node"}
    )
    way_end_nodes_df = way2way_df[["id", "end_node"]].rename(
        columns={"id": "from_way", "end_node": "junction_node"}
    )

    connections_df = way_end_nodes_df.merge(
        way_start_nodes_df,
        how="inner",
        on="junction_node",
    )
    connections_df = connections_df[
        connections_df["from_way"] != connections_df["to_way"]
    ].copy()

    is_restricted = connections_df.apply(
        lambda row: (
            int(row["from_way"]),
            int(row["to_way"]),
        ) in restricted_pairs,
        axis=1,
    )

    connections = (
        connections_df.loc[~is_restricted, ["from_way", "to_way"]]
        .drop_duplicates()
        .to_numpy()
        .astype(int)
    )
    return connections, len(connections_df), int(is_restricted.sum())


def to_edge_index(edges):
    return torch.tensor(edges.T, dtype=torch.long)


def build_static_graph():
    segments_df = pd.read_csv(PREPROCESS_ROOT / "segments.csv")
    way2way_df = pd.read_csv(PREPROCESS_ROOT / "way2way.csv")
    relation_members_df = pd.read_csv(PREPROCESS_ROOT / "relation_members.csv")

    static_node_features = np.load(PREPROCESS_ROOT / "static_nodes.npy")
    static_segment_features = np.load(PREPROCESS_ROOT / "static_segments.npy")
    static_way_features = np.load(PREPROCESS_ROOT / "static_ways.npy")

    data = HeteroData()

    data["node"].x = torch.from_numpy(static_node_features)
    data["segment"].x = torch.from_numpy(static_segment_features)
    data["way"].x = torch.from_numpy(static_way_features)

    contain_segments = (
        segments_df[["street_id", "id"]]
        .drop_duplicates()
        .to_numpy()
        .astype(int)
    )
    data["way", "contains_segment", "segment"].edge_index = to_edge_index(
        contain_segments
    )
    data["segment", "contained_in", "way"].edge_index = to_edge_index(
        contain_segments[:, [1, 0]]
    )

    starts_with = (
        segments_df[["id", "s_node_id"]]
        .drop_duplicates()
        .to_numpy()
        .astype(int)
    )
    data["segment", "starts_with", "node"].edge_index = to_edge_index(
        starts_with
    )
    data["node", "start_of", "segment"].edge_index = to_edge_index(
        starts_with[:, [1, 0]]
    )

    ends_with = (
        segments_df[["id", "e_node_id"]]
        .drop_duplicates()
        .to_numpy()
        .astype(int)
    )
    data["segment", "ends_with", "node"].edge_index = to_edge_index(ends_with)
    data["node", "end_of", "segment"].edge_index = to_edge_index(
        ends_with[:, [1, 0]]
    )

    way_id2index = load_id2index(METADATA_ROOT / "ways.csv")
    restricted_pairs = build_restricted_way_pairs(
        relation_members_df,
        way_id2index,
    )
    way_connections, total_connections, removed_connections = (
        build_way_connections(way2way_df, restricted_pairs)
    )
    data["way", "connects_to", "way"].edge_index = to_edge_index(
        way_connections
    )

    print("Way connections before filtering:", total_connections)
    print("Restricted way turns removed:", removed_connections)
    print("Way connections after filtering:", way_connections.shape[0])

    return data


if __name__ == "__main__":
    data = build_static_graph()
    print(data)
