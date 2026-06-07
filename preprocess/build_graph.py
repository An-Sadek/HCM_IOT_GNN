from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


PREPROCESS_ROOT = Path("data/preprocess")

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


def build_segment_pairs():
    # Xây dựng df cặp
    segment_pairs = segment2segment_df[["from_segment_id", "to_segment_id"]]
    print("Shape của cặp thuận:", segment_pairs.shape)

    # Tạo các segment 2 chiều
    twoway_segments = oneway_df[oneway_df["tags.oneway"] == "no"]["segment_id"]
    print("Số cặp way 2 chiều:", twoway_segments.shape[0])
    print("Số segment một chiều:", segment_pairs.shape[0] - twoway_segments.shape[0])

    all_pairs = segment_pairs.copy()
    reverse_pairs = segment_pairs.rename(
        columns={"from_segment_id": "to_segment_id", "to_segment_id": "from_segment_id"}
    )

    # Tạo reverse pair hợp lệ
    reverse_pairs_filtered = reverse_pairs[
        reverse_pairs["from_segment_id"].isin(twoway_segments)
        & reverse_pairs["to_segment_id"].isin(twoway_segments)
    ]
    print("Số cặp reverse segment hợp lệ", reverse_pairs_filtered.shape[0])

    # Tổng hợp
    full_segment_df = (
        pd.concat([all_pairs, reverse_pairs_filtered], ignore_index=True)
        .drop_duplicates()
    )
    print("Tổng cặp segment hợp lệ", full_segment_df.shape[0])

    # Bỏ các edge thông qua restriction
    segment_node_df = segments_df[["s_node_id", "e_node_id"]]
    restriction_df = build_restricted_way_pairs()



    return full_segment_df[["from_segment_id", "to_segment_id"]].to_numpy().astype(int)

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
    data["segment", "connects_to_segment", "segment"].edge_index = to_edge_index(connects_to_segment)

    return data


if __name__ == "__main__":
    data = build_static_graph()
    print(data)