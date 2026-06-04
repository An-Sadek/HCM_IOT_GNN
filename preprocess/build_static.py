import argparse
from pathlib import Path

import pandas as pd
import torch


DEFAULT_INPUT = Path("data/preprocess/all.csv")
DEFAULT_OUTPUT = Path("data/preprocess/static_graph.pt")
DEFAULT_NODE_CSV = Path("data/preprocess/static_nodes.csv")
DEFAULT_EDGE_CSV = Path("data/preprocess/static_edges.csv")

STATIC_COLUMNS = [
    "segment_id",
    "s_node_id",
    "e_node_id",
    "length",
    "max_velocity",
    "street_level",
    "street_name",
    "segment_type",
    "long_snode",
    "lat_snode",
    "long_enode",
    "lat_enode",
    "street_type",
]

NUMERIC_EDGE_COLUMNS = [
    "length",
    "max_velocity",
    "street_level",
    "long_snode",
    "lat_snode",
    "long_enode",
    "lat_enode",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build static road graph data.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--node-csv", type=Path, default=DEFAULT_NODE_CSV)
    parser.add_argument("--edge-csv", type=Path, default=DEFAULT_EDGE_CSV)
    return parser.parse_args()


def validate_static_columns(df: pd.DataFrame) -> None:
    grouped = df.groupby("segment_id", sort=False)
    violations = []

    for column in STATIC_COLUMNS:
        if grouped[column].nunique(dropna=False).max() > 1:
            violations.append(column)

    if violations:
        columns = ", ".join(violations)
        raise ValueError(f"Static columns vary within segment_id: {columns}")


def build_node_frame(segment_df: pd.DataFrame) -> pd.DataFrame:
    start_nodes = (
        segment_df[["s_node_id", "long_snode", "lat_snode"]]
        .rename(
            columns={
                "s_node_id": "node_id",
                "long_snode": "longitude",
                "lat_snode": "latitude",
            }
        )
    )
    end_nodes = (
        segment_df[["e_node_id", "long_enode", "lat_enode"]]
        .rename(
            columns={
                "e_node_id": "node_id",
                "long_enode": "longitude",
                "lat_enode": "latitude",
            }
        )
    )

    node_df = pd.concat([start_nodes, end_nodes], ignore_index=True)
    node_df = node_df.drop_duplicates(subset=["node_id"]).sort_values("node_id").reset_index(drop=True)
    node_df["node_index"] = range(len(node_df))
    return node_df


def encode_categories(values: pd.Series) -> tuple[torch.Tensor, dict[str, int]]:
    categories = sorted(values.fillna("NA").astype(str).unique().tolist())
    mapping = {category: index for index, category in enumerate(categories)}
    encoded = values.fillna("NA").astype(str).map(mapping).to_numpy()
    return torch.tensor(encoded, dtype=torch.long), mapping


def main() -> None:
    args = parse_args()

    all_df = pd.read_csv(args.input)
    validate_static_columns(all_df)

    segment_df = (
        all_df[STATIC_COLUMNS]
        .drop_duplicates(subset=["segment_id"])
        .sort_values("segment_id")
        .reset_index(drop=True)
    )

    node_df = build_node_frame(segment_df)
    node_id_to_index = dict(zip(node_df["node_id"], node_df["node_index"]))

    edge_df = segment_df.copy()
    edge_df["src"] = edge_df["s_node_id"].map(node_id_to_index)
    edge_df["dst"] = edge_df["e_node_id"].map(node_id_to_index)
    edge_df["edge_index"] = range(len(edge_df))

    edge_index = torch.tensor(edge_df[["src", "dst"]].to_numpy().T, dtype=torch.long)
    node_pos = torch.tensor(node_df[["longitude", "latitude"]].to_numpy(), dtype=torch.float32)
    edge_attr = torch.tensor(edge_df[NUMERIC_EDGE_COLUMNS].to_numpy(), dtype=torch.float32)
    segment_type, segment_type_map = encode_categories(edge_df["segment_type"])
    street_type, street_type_map = encode_categories(edge_df["street_type"])

    graph = {
        "edge_index": edge_index,
        "node_pos": node_pos,
        "edge_attr": edge_attr,
        "node_ids": torch.tensor(node_df["node_id"].to_numpy(), dtype=torch.long),
        "segment_ids": torch.tensor(edge_df["segment_id"].to_numpy(), dtype=torch.long),
        "segment_type": segment_type,
        "street_type": street_type,
        "node_feature_names": ["longitude", "latitude"],
        "edge_feature_names": NUMERIC_EDGE_COLUMNS,
        "segment_type_map": segment_type_map,
        "street_type_map": street_type_map,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, args.output)

    node_df.to_csv(args.node_csv, index=False)
    edge_df.to_csv(args.edge_csv, index=False)

    print(f"Saved static graph to {args.output}")
    print(f"Nodes: {node_pos.shape[0]}")
    print(f"Edges: {edge_index.shape[1]}")
    print(f"Edge features: {edge_attr.shape[1]}")


if __name__ == "__main__":
    main()
