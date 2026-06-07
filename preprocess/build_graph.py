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
    # Lấy cặp segment thuận
    segment_pairs = segment2segment_df[["from_segment_id", "to_segment_id"]]
    
    # Tìm cặp đảo ngược hợp lệ
    twoway_segs = set(oneway_df.loc[oneway_df["tags.oneway"] == "no", "segment_id"])
    mask_twoway = segment_pairs["from_segment_id"].isin(twoway_segs) & segment_pairs["to_segment_id"].isin(twoway_segs)
    
    reverse_pairs = segment_pairs[mask_twoway].rename(
        columns={"from_segment_id": "to_segment_id", "to_segment_id": "from_segment_id"}
    )

    # Tổng hợp lại
    valid_df = pd.concat([segment_pairs, reverse_pairs], ignore_index=True).drop_duplicates()

    # Xác định danh sách các node cấm rẽ (via nodes)
    restriction_nodes = set(relation_df.loc[relation_df["role"] == "via", "ref"])

    # Map start_node và end_node trực tiếp vào valid_df
    e_node_map = segments_df.set_index("id")["e_node_id"]
    s_node_map = segments_df.set_index("id")["s_node_id"]

    from_e_nodes = valid_df["from_segment_id"].map(e_node_map)
    to_s_nodes = valid_df["to_segment_id"].map(s_node_map)

    # Lọc bỏ restriction
    is_restricted = (from_e_nodes == to_s_nodes) & (from_e_nodes.isin(restriction_nodes))
    final_valid_df = valid_df[~is_restricted] # <--- Dùng dấu ~ để giữ lại các dòng KHÔNG bị cấm

    # In log kiểm tra
    print(f"Tổng cặp ban đầu (gồm 1 chiều + 2 chiều): {len(valid_df)}")
    print(f"Số cặp bị cấm rẽ (restriction): {is_restricted.sum()}")
    print(f"Tổng cặp hợp lệ cuối cùng: {len(final_valid_df)}")

    return final_valid_df[["from_segment_id", "to_segment_id"]].to_numpy().astype(int)

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
    data = build_static_graph()
    print(data)
    torch.save(data, "data/preprocess/hetero_data.pt")