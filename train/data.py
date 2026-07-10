from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import dgl


def _require_dgl():
    try:
        import dgl
    except Exception as exc:
        raise RuntimeError(
            "DGL is required to build SEHTGNN temporal heterographs. "
            "The current environment cannot import dgl; reinstall a working "
            "DGL build before running training."
        ) from exc
    return dgl


class SEHTGNNDataset(Dataset):
    """Sliding-window dataset that feeds the official SEHTGNN DGL interface.

    Each sample returns:
    - graph: a DGL heterograph with time-suffixed edge types, e.g. connects_to_t0
    - y: LOS targets for segment nodes, shape (num_segments, horizon)

    Node data is stored as graph.nodes[ntype].data["t{i}"], matching model.py.
    Segment snapshots contain static segment attributes concatenated with
    dynamic features at that timestamp.
    """

    def __init__(
        self,
        preprocess_root="data/preprocess",
        dynamic_path=None,
        window_size=12,
        horizon=6,
        target_channel=0,
        mmap_mode="r",
    ):
        self.preprocess_root = Path(preprocess_root)
        self.window_size = int(window_size)
        self.horizon = int(horizon)
        self.target_channel = int(target_channel)

        if dynamic_path is None:
            dynamic_path = self.preprocess_root / "dynamic_features.npy"
        self.dynamic = np.load(dynamic_path, mmap_mode=mmap_mode)

        if self.dynamic.ndim != 6:
            raise ValueError(
                "dynamic_features must have shape (time, num_segments, channels), "
                f"got {self.dynamic.shape}"
            )

        self.total_timesteps, self.num_segments, self.dynamic_dim = self.dynamic.shape
        self.num_samples = self.total_timesteps - self.window_size - self.horizon + 1
        if self.num_samples <= 0:
            raise ValueError(
                "Not enough timesteps for the requested window/horizon: "
                f"T={self.total_timesteps}, window={self.window_size}, horizon={self.horizon}"
            )

        self.static_features = self._load_static_features()
        self.inp_list = {
            "node": self.static_features["node"].shape[1],
            "way": self.static_features["way"].shape[1],
            "segment": self.static_features["segment"].shape[1] + self.dynamic_dim,
        }
        self.num_nodes_dict = {
            ntype: features.shape[0]
            for ntype, features in self.static_features.items()
        }

        self.base_edges = self._load_base_edges()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx < 0:
            idx += self.num_samples
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(idx)

        graph = self.build_graph(idx)
        target_start = idx + self.window_size
        target_end = target_start + self.horizon
        y = torch.from_numpy(
            np.asarray(
                self.dynamic[target_start:target_end, :, self.target_channel].T,
                dtype=np.float32,
            )
        )
        return graph, y

    def build_graph(self, start_idx):
        dgl = _require_dgl()
        data_dict = {}
        for t in range(self.window_size):
            suffix = f"t{t}"
            for (src_type, rel_type, dst_type), edges in self.base_edges.items():
                src, dst = edges
                data_dict[(src_type, f"{rel_type}_{suffix}", dst_type)] = (src, dst)

        graph = dgl.heterograph(data_dict, num_nodes_dict=self.num_nodes_dict)

        static_node = torch.from_numpy(self.static_features["node"])
        static_way = torch.from_numpy(self.static_features["way"])
        static_segment = self.static_features["segment"]

        for t in range(self.window_size):
            key = f"t{t}"
            dynamic_t = np.asarray(self.dynamic[start_idx + t], dtype=np.float32)
            segment_t = np.concatenate([static_segment, dynamic_t], axis=1)

            graph.nodes["node"].data[key] = static_node
            graph.nodes["way"].data[key] = static_way
            graph.nodes["segment"].data[key] = torch.from_numpy(segment_t)

        return graph

    def _load_static_features(self):
        return {
            "node": np.load(self.preprocess_root / "static_nodes.npy").astype(np.float32),
            "segment": np.load(self.preprocess_root / "static_segments.npy").astype(np.float32),
            "way": np.load(self.preprocess_root / "static_ways.npy").astype(np.float32),
        }

    def _load_base_edges(self):
        segments_df = pd.read_csv(self.preprocess_root / "segments.csv")
        raw_ways_df = pd.read_csv(self.preprocess_root / "combine_ways_df.csv")
        relation_df = pd.read_csv(self.preprocess_root / "relation_members.csv")

        contains = segments_df[["street_id", "id"]].drop_duplicates().to_numpy(dtype=np.int64)
        starts = segments_df[["id", "s_node_id"]].drop_duplicates().to_numpy(dtype=np.int64)
        ends = segments_df[["id", "e_node_id"]].drop_duplicates().to_numpy(dtype=np.int64)
        connects = self._build_connects_with(segments_df, raw_ways_df, relation_df)

        return {
            ("way", "contains_segment", "segment"): (contains[:, 0], contains[:, 1]),
            ("segment", "contained_in", "way"): (contains[:, 1], contains[:, 0]),
            ("segment", "starts_with", "node"): (starts[:, 0], starts[:, 1]),
            ("node", "start_of", "segment"): (starts[:, 1], starts[:, 0]),
            ("segment", "ends_with", "node"): (ends[:, 0], ends[:, 1]),
            ("node", "end_of", "segment"): (ends[:, 1], ends[:, 0]),
            ("segment", "connects_to", "segment"): (connects[:, 0], connects[:, 1]),
        }

    @staticmethod
    def _build_connects_with(segments_df, raw_ways_df, relation_df):
        from_df = segments_df[["id", "e_node_id", "street_id"]].rename(
            columns={
                "id": "from_segment_id",
                "e_node_id": "node_id",
                "street_id": "from_street_id",
            }
        )
        to_df = segments_df[["id", "s_node_id", "street_id"]].rename(
            columns={
                "id": "to_segment_id",
                "s_node_id": "node_id",
                "street_id": "to_street_id",
            }
        )

        connects_df = from_df.merge(to_df, on="node_id", how="inner").drop_duplicates()
        connects_df["same_way"] = (
            connects_df["from_street_id"] == connects_df["to_street_id"]
        ).astype(int)

        connects_df = connects_df.merge(
            raw_ways_df[["id", "tags.oneway"]],
            how="left",
            left_on="from_street_id",
            right_on="id",
        ).drop(columns="id")
        connects_df["tags.oneway"] = connects_df["tags.oneway"].fillna("no")

        forward = connects_df[
            ~((connects_df["tags.oneway"] == "-1") & (connects_df["same_way"] == 1))
        ].copy()

        reverse = connects_df.rename(
            columns={
                "from_segment_id": "to_segment_id",
                "to_segment_id": "from_segment_id",
                "from_street_id": "to_street_id",
                "to_street_id": "from_street_id",
            }
        ).drop(columns=["tags.oneway"])

        reverse = reverse.merge(
            raw_ways_df[["id", "tags.oneway"]],
            how="left",
            left_on="from_street_id",
            right_on="id",
        ).drop(columns="id")
        reverse["tags.oneway"] = reverse["tags.oneway"].fillna("no")
        reverse = reverse[
            ~((reverse["tags.oneway"] == "yes") & (reverse["same_way"] == 1))
        ].copy()

        connects_df = pd.concat([forward, reverse], ignore_index=True).drop_duplicates()

        restrictions = (
            relation_df.pivot_table(
                index="id",
                columns="role",
                values="ref",
                aggfunc="first",
            )
            .reset_index()
            .dropna(subset=["from", "via", "to"])
            .rename(
                columns={
                    "from": "from_street_id",
                    "via": "node_id",
                    "to": "to_street_id",
                }
            )
        )

        if not restrictions.empty:
            for col in ["from_street_id", "node_id", "to_street_id"]:
                restrictions[col] = restrictions[col].astype(connects_df[col].dtype)

            connects_df = connects_df.merge(
                restrictions[["from_street_id", "node_id", "to_street_id"]],
                on=["from_street_id", "node_id", "to_street_id"],
                how="left",
                indicator="restriction_check",
            )
            connects_df = connects_df[
                connects_df["restriction_check"] != "both"
            ].drop(columns=["restriction_check"])

        return (
            connects_df[["from_segment_id", "to_segment_id"]]
            .drop_duplicates()
            .to_numpy(dtype=np.int64)
        )


def collate_sehtgnn(samples):
    dgl = _require_dgl()
    graphs, targets = zip(*samples)
    return dgl.batch(graphs), torch.cat(targets, dim=0)


if __name__ == "__main__":
    test = SEHTGNNDataset()
    print(len(test))
    print(len(test[0][1]))