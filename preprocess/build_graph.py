from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


PREPROCESS_ROOT = Path("data/preprocess")
RAW_ROOT = Path("data/raw")

train_df = pd.read_csv(RAW_ROOT / "train.csv")
raw_ways_df = pd.read_csv(PREPROCESS_ROOT / "combine_ways_df.csv")

ways_df = pd.read_csv(PREPROCESS_ROOT / "ways.csv")
segments_df = pd.read_csv(PREPROCESS_ROOT / "segments.csv")
way2way_df = pd.read_csv(PREPROCESS_ROOT / "way2way.csv")
relation_df = pd.read_csv(PREPROCESS_ROOT / "relation_members.csv")
segment2segment_df = pd.read_csv(PREPROCESS_ROOT / "segment2segment.csv")
oneway_df = pd.read_csv(PREPROCESS_ROOT / "oneway_df.csv")
nodes_segments_edges_df = pd.read_csv(PREPROCESS_ROOT / "nodes_segments_edges_df.csv")

static_node_features = np.load(PREPROCESS_ROOT / "static_nodes.npy")
static_segment_features = np.load(PREPROCESS_ROOT / "static_segments.npy")
static_way_features = np.load(PREPROCESS_ROOT / "static_ways.npy")


def build_connects_with():
    from_df = segments_df[["id", "e_node_id", "street_id"]].rename(columns={
        "id": "from_segment_id",
        "e_node_id": "node_id",
        "street_id": "from_street_id"
    })

    to_df = segments_df[["id", "s_node_id", "street_id"]].rename(columns={
        "id": "to_segment_id",
        "s_node_id": "node_id",
        "street_id": "to_street_id"
    })

    connects_with_df = from_df.merge(to_df, on="node_id", how="inner").drop_duplicates()
    print("Shape thuận gốc:", connects_with_df.shape)

    connects_with_df["same_way"] = (
        connects_with_df["from_street_id"] == connects_with_df["to_street_id"]
    ).astype(int)

    # Gắn oneway cho street nguồn của chiều thuận
    connects_with_df = connects_with_df.merge(
        raw_ways_df[["id", "tags.oneway"]],
        how="left",
        left_on="from_street_id",
        right_on="id"
    ).drop(columns="id")

    connects_with_df["tags.oneway"] = connects_with_df["tags.oneway"].fillna("no")

    # Bỏ chiều thuận nếu cùng way và oneway = -1
    invalid_forward_mask = (
        (connects_with_df["tags.oneway"] == "-1") &
        (connects_with_df["same_way"] == 1)
    )

    forward_connects = connects_with_df[~invalid_forward_mask].copy()
    print("Shape thuận sau khi bỏ oneway=-1:", forward_connects.shape)

    # Tạo chiều nghịch
    reverse_connects = connects_with_df.rename(columns={
        "from_segment_id": "to_segment_id",
        "to_segment_id": "from_segment_id",
        "from_street_id": "to_street_id",
        "to_street_id": "from_street_id"
    }).copy()

    # Xoá tags.oneway cũ vì nó thuộc from_street_id cũ
    reverse_connects = reverse_connects.drop(columns=["tags.oneway"])

    # Gắn lại oneway theo from_street_id mới
    reverse_connects = reverse_connects.merge(
        raw_ways_df[["id", "tags.oneway"]],
        how="left",
        left_on="from_street_id",
        right_on="id"
    ).drop(columns="id")
    reverse_connects["tags.oneway"] = reverse_connects["tags.oneway"].fillna("no")

    # Bỏ chiều nghịch nếu cùng way và oneway = yes
    invalid_reverse_mask = (
        (reverse_connects["tags.oneway"] == "yes") &
        (reverse_connects["same_way"] == 1)
    )

    reverse_connects = reverse_connects[~invalid_reverse_mask].copy()
    print("Shape nghịch sau khi bỏ oneway=yes:", reverse_connects.shape)

    connects_with_df = pd.concat(
        [forward_connects, reverse_connects],
        ignore_index=True
    ).drop_duplicates()
    print("Shape trước khi bỏ restriction:", connects_with_df.shape)

    # Bỏ restriction
    restriction_wide = (
        relation_df
        .pivot_table(
            index="id",
            columns="role",
            values="ref",
            aggfunc="first"
        )
        .reset_index()
    )

    # Chỉ lấy restriction có đủ from, via, to
    restriction_wide = restriction_wide.dropna(subset=["from", "via", "to"])

    restriction_wide = restriction_wide.rename(columns={
        "from": "from_street_id",
        "via": "node_id",
        "to": "to_street_id"
    })

    # Ép kiểu cho chắc
    for col in ["from_street_id", "node_id", "to_street_id"]:
        restriction_wide[col] = restriction_wide[col].astype(connects_with_df[col].dtype)

    # Đánh dấu các cặp bị cấm
    connects_with_df = connects_with_df.merge(
        restriction_wide[["from_street_id", "node_id", "to_street_id"]],
        on=["from_street_id", "node_id", "to_street_id"],
        how="left",
        indicator="restriction_check"
    )

    is_restricted = connects_with_df["restriction_check"] == "both"

    print("Số cặp bị cấm rẽ:", is_restricted.sum())

    connects_with_df = (
        connects_with_df[~is_restricted]
        .drop(columns=["restriction_check"])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print("Shape cuối:", connects_with_df.shape)
    return connects_with_df

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

    connects_to_segment = build_segment_pairs()
    data["segment", "connects_to", "segment"].edge_index = to_edge_index(connects_to_segment)

    return data


if __name__ == "__main__":
    build_connects_with()
    #data = build_static_graph()
    #print(data)
    #torch.save(data, "data/preprocess/hetero_data.pt")
