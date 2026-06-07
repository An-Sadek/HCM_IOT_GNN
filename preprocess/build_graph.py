from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


PREPROCESS_ROOT = Path("data/preprocess")

segments_df = pd.read_csv(PREPROCESS_ROOT / "segments.csv")
way2way_df = pd.read_csv(PREPROCESS_ROOT / "way2way.csv")
relation_df = pd.read_csv(PREPROCESS_ROOT / "relation_members.csv")

static_node_features = np.load(PREPROCESS_ROOT / "static_nodes.npy")
static_segment_features = np.load(PREPROCESS_ROOT / "static_segments.npy")
static_way_features = np.load(PREPROCESS_ROOT / "static_ways.npy")


def build_segment_pairs():
    filtered_segments_df = segments_df[["s_node_id", "e_node_id"]]


def build_restricted_way_pairs():
    """
    Tìm các tuyến đường bị cấm
    """
    start_way_list = []
    end_way_list = []

    relation_group =  relation_df.groupby("id")
    for _, group in relation_group:
        from_series = group.loc[group["role"] == "from", "ref"]
        to_series = group.loc[group["role"] == "to", "ref"]

        # Kiểm tra xem có bị rỗng không
        if from_series.empty or to_series.empty:
            continue

        from_way = from_series.values[0]
        to_way = to_series.values[0]

        start_way_list.append(int(from_way))
        end_way_list.append(int(to_way))

    restriction_df = pd.DataFrame({
        "from_way_id": start_way_list,
        "to_way_id": end_way_list
    })

    return restriction_df

def build_way_pairs():
    # Xây dựng df cặp
    way_pairs_df = way2way_df[["id", "end_node"]].merge(
        way2way_df[["id", "start_node"]],
        left_on="end_node",
        right_on="start_node",
        how="inner"
    ).drop_duplicates()
    way_pairs_df = way_pairs_df.rename(
        columns={"id_x": "from_way_id", "id_y": "to_way_id"}
    )
    way_pairs_df = way_pairs_df[way_pairs_df["from_way_id"] != way_pairs_df["to_way_id"]]
    print("Shape của các cặp đường gốc", way_pairs_df.shape)

    # Lấy nghịch đảo
    reverse_df = way_pairs_df.copy()
    reverse_df = reverse_df.rename(columns={
        "from_way_id": "to_way_id",
        "to_way_id": "from_way_id"
    })

    # Lọc các restrict
    restriction_df = build_restricted_way_pairs()
    merged_df = way_pairs_df.merge(
        restriction_df[["from_way_id", "to_way_id"]], 
        on=["from_way_id", "to_way_id"], 
        how="left", 
        indicator="is_restricted"
    )
    allowed_way_pairs_df = merged_df[merged_df["is_restricted"] == "left_only"].drop(columns=["is_restricted"])
    print("Shape sau khi lọc các đường cấm rẽ:", allowed_way_pairs_df.shape)

    # Gộp 2 data lại
    full_df = pd.concat([allowed_way_pairs_df, reverse_df], axis=0).drop_duplicates()
    print(full_df.isnull().sum())
    print("Shape sau khi kết hợp 2 data", full_df.shape)

    return full_df

def to_edge_index(edges):
    return torch.tensor(edges.T, dtype=torch.long)


def build_static_graph():

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

    return data


if __name__ == "__main__":
    data = build_static_graph()
    print(data)
    build_way_pairs()
