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
    - y: continuous targets for segment nodes, shape (num_segments, horizon)

    Node data is stored as graph.nodes[ntype].data["t{i}"], matching model.py.
    Segment snapshots contain static segment attributes concatenated with
    dynamic features at that timestamp.
    """

    def __init__(
        self,
        preprocess_root="data/preprocess",
        dynamic_path=None,
        target_path=None,
        window_size=12,
        horizon=6,
        mmap_mode="r",
    ):
        self.preprocess_root = Path(preprocess_root)
        self.window_size = int(window_size)
        self.horizon = int(horizon)
        if dynamic_path is None:
            dynamic_path = self.preprocess_root / "dynamic_features.npy"
        self.dynamic = np.load(dynamic_path, mmap_mode=mmap_mode)

        if target_path is None:
            target_path = self.preprocess_root / "dynamic_velocity.npy"
        self.targets = np.load(target_path, mmap_mode=mmap_mode)
        self.velocity_channel = None
        self.velocity_mean = None
        self.velocity_scale = None

        if self.dynamic.ndim != 3:
            raise ValueError(
                "dynamic_features must have shape (time, num_segments, channels), "
                f"got {self.dynamic.shape}"
            )
        if self.targets.ndim != 2:
            raise ValueError(
                "Velocity targets must have shape (time, num_segments), "
                f"got {self.targets.shape}"
            )
        if self.targets.shape != self.dynamic.shape[:2]:
            raise ValueError(
                "Velocity target dimensions must match dynamic features: "
                f"targets={self.targets.shape}, dynamic={self.dynamic.shape[:2]}"
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

    def fit_input_velocity_scaler(
        self,
        timestamp_start,
        timestamp_end,
        velocity_channel=1,
        chunk_size=256,
    ):
        """Fit per-segment input-velocity statistics on train timestamps only."""
        timestamp_start = int(timestamp_start)
        timestamp_end = int(timestamp_end)
        velocity_channel = int(velocity_channel)
        if not 0 <= timestamp_start < timestamp_end <= self.total_timesteps:
            raise ValueError(
                "Invalid velocity scaler timestamp range: "
                f"[{timestamp_start}, {timestamp_end}) for T={self.total_timesteps}"
            )
        if not 0 <= velocity_channel < self.dynamic_dim:
            raise ValueError(
                f"velocity_channel={velocity_channel} is outside dynamic feature "
                f"dimension {self.dynamic_dim}"
            )
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

        check_timestamps = np.linspace(
            timestamp_start,
            timestamp_end - 1,
            num=min(16, timestamp_end - timestamp_start),
            dtype=np.int64,
        )
        stored_velocity = np.asarray(
            self.dynamic[check_timestamps, :, velocity_channel], dtype=np.float32
        )
        raw_velocity = np.asarray(
            self.targets[check_timestamps, :], dtype=np.float32
        )
        if not np.allclose(stored_velocity, raw_velocity, rtol=1e-5, atol=1e-5):
            raise ValueError(
                "Configured velocity channel does not contain raw velocity: "
                f"channel={velocity_channel}. Regenerate dynamic_features.npy with "
                "the current preprocess/pipelines/dynamic.py or select the correct "
                "--velocity-channel."
            )

        velocity_sum = np.zeros(self.num_segments, dtype=np.float64)
        count = 0
        for start in range(timestamp_start, timestamp_end, chunk_size):
            end = min(start + chunk_size, timestamp_end)
            chunk = np.asarray(self.targets[start:end], dtype=np.float64)
            if not np.isfinite(chunk).all():
                raise ValueError(
                    f"Velocity contains a non-finite value in timestamps [{start}, {end})"
                )
            velocity_sum += chunk.sum(axis=0)
            count += chunk.shape[0]

        mean = velocity_sum / count
        squared_deviation_sum = np.zeros(self.num_segments, dtype=np.float64)
        for start in range(timestamp_start, timestamp_end, chunk_size):
            end = min(start + chunk_size, timestamp_end)
            chunk = np.asarray(self.targets[start:end], dtype=np.float64)
            squared_deviation_sum += np.square(chunk - mean).sum(axis=0)
        variance = squared_deviation_sum / count
        scale = np.sqrt(variance)
        # Match sklearn StandardScaler: constant features are left with scale 1.
        scale[scale < 1e-12] = 1.0

        self.velocity_channel = velocity_channel
        self.velocity_mean = mean.astype(np.float32)
        self.velocity_scale = scale.astype(np.float32)
        return self.velocity_mean, self.velocity_scale

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
        raw_target = np.asarray(
            self.targets[target_start:target_end, :].T
        ).copy()
        if not np.isfinite(raw_target).all():
            raise ValueError(f"Target contains a non-finite value at sample {idx}")
        y = torch.from_numpy(raw_target.astype(np.float32, copy=False))
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
            timestamp = start_idx + t
            dynamic_t = np.asarray(self.dynamic[timestamp], dtype=np.float32).copy()
            if self.velocity_channel is not None:
                raw_velocity = np.asarray(self.targets[timestamp], dtype=np.float32)
                dynamic_t[:, self.velocity_channel] = (
                    raw_velocity - self.velocity_mean
                ) / self.velocity_scale
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
            ("node", "start_of", "segment"): (starts[:, 1], starts[:, 0]),
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
