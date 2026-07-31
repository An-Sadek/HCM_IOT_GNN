from pathlib import Path

import pandas as pd
import numpy as np

from general import Preprocess


class DynamicPreprocess(Preprocess):
    def __init__(self, raw_root:str="data/raw"):
        raw_root = Path(raw_root)
        assert raw_root.exists(), "Đường dẫn không tồn tại"

        # Sắp xếp để đảm bảo nhất quán
        self.train_df = pd.read_csv(raw_root / "train.csv")
        self.train_df = self.train_df[["segment_id", "date", "weekday", "period", "LOS"]]
        self.train_df = self.train_df.sort_values("segment_id")

        self.status_df = pd.read_csv(raw_root / "segment_status.csv").sort_values("segment_id")
        self.status_df["updated_at"] = (
            pd.to_datetime(self.status_df["updated_at"], utc=True)
              .dt.tz_localize(None)
        )

        # Mốc thời gian
        min_timestamp = self.status_df["updated_at"].min().floor("30min")
        max_timestamp = self.status_df["updated_at"].max().floor("30min")
        self.full_time = pd.date_range(start=min_timestamp, end=max_timestamp, freq='30min')

        # Metadata
        self.metadata = dict()

    def velocity_preprocess(self):
        # Chuyển thành datetime
        self.status_df["updated_at"] = pd.to_datetime(self.status_df["updated_at"])

        # Chuyển thành time bucket
        # bucket 30 phút
        self.status_df["timestamp"] = (
            self.status_df["updated_at"]
            .dt.floor("30min")
        )
        
        # Trung bình
        status_30m = (
            self.status_df
            .groupby(
                ["segment_id", "timestamp"],
                as_index=False
            )
            .agg({
                "velocity": "mean"
            })
        )
        
        # pivot
        velocity_mat = status_30m.pivot(
            index="timestamp",
            columns="segment_id",
            values="velocity"
        ).reindex(self.full_time)

        # Thế bằng mode, nếu vượt ngưỡng thì [quantile(0.25), 120]
        velocity_arr = velocity_mat.reindex(self.full_time)
        velocity_arr = velocity_arr.clip(lower=20, upper=120)
        velocity_arr = velocity_arr.interpolate(
            method="linear",
            axis=0,
            limit_direction="both",
        )
        
        print("Kích thước của pivot table velocity trong status df:", velocity_arr.shape)
        
        return velocity_arr

    def los_preprocess(self):
        self.train_df["LOS"] = self.train_df["LOS"].apply(lambda x: ord(x) - ord('A'))
        self.train_df["date"] = pd.to_datetime(
            self.train_df["date"]
        )
        
        self.train_df[["hour", "minute"]] = (
            self.train_df["period"]
            .str.extract(r"period_(\d+)_(\d+)")
            .astype(int)
        )
        
        self.train_df["timestamp"] = (
            self.train_df["date"]
            + pd.to_timedelta(
                self.train_df["hour"],
                unit="h"
            )
            + pd.to_timedelta(
                self.train_df["minute"],
                unit="m"
            )
        )

        los_mat = self.train_df.pivot(
            index="timestamp",
            columns="segment_id",
            values="LOS"
        ).reindex(self.full_time)

        # Gộp lại, ffill rồi bfill
        los_arr = los_mat.reindex(self.full_time)
        los_arr = los_arr.ffill().bfill()

        print("Kích thước của pivot table LOS và mask trong train df:", los_arr.shape)

        return los_arr

    def dayweek_preprocess(self, columns=None):
        dayweek_arr = pd.DataFrame(
            {
                "dayweek": self.full_time.dayofweek.astype(np.int8)
            },
            index=self.full_time
        )
        dayweek_arr.index.name = "timestamp"

        if columns is not None:
            dayweek_arr = pd.DataFrame(
                np.repeat(dayweek_arr[["dayweek"]].to_numpy(), len(columns), axis=1),
                index=self.full_time,
                columns=columns
            )
            dayweek_arr.index.name = "timestamp"

        print("Kích thước của pivot table dayweek:", dayweek_arr.shape)

        return dayweek_arr

    def cyclic_time_preprocess(self, columns=None):
        minute_of_day = (
            self.full_time.hour * 60
            + self.full_time.minute
            + self.full_time.second / 60
        )
        time_angle = 2 * np.pi * minute_of_day / (24 * 60)

        day_of_week = self.full_time.dayofweek
        day_angle = 2 * np.pi * day_of_week / 7

        cyclic_arr = pd.DataFrame(
            {
                "time_in_day_sin": np.sin(time_angle).astype(np.float32),
                "time_in_day_cos": np.cos(time_angle).astype(np.float32),
                "day_in_week_sin": np.sin(day_angle).astype(np.float32),
                "day_in_week_cos": np.cos(day_angle).astype(np.float32),
            },
            index=self.full_time
        )
        cyclic_arr.index.name = "timestamp"

        if columns is not None:
            cyclic_arr = pd.DataFrame(
                np.repeat(cyclic_arr.to_numpy(), len(columns), axis=1),
                index=self.full_time,
                columns=pd.MultiIndex.from_product([cyclic_arr.columns, columns])
            )
            cyclic_arr.index.name = "timestamp"

        print("Kích thước của pivot table cyclic time:", cyclic_arr.shape)

        return cyclic_arr


    def save_dynamic_features(self):
        """
        """
        los_arr = self.los_preprocess()
        velocity_arr = self.velocity_preprocess().reindex(columns=los_arr.columns)
        if velocity_arr.isna().any().any():
            missing_segments = velocity_arr.columns[velocity_arr.isna().any()].tolist()
            raise ValueError(
                "Velocity is missing after alignment with LOS segments: "
                f"{missing_segments[:10]}"
            )

        cyclic_time_arr = self.cyclic_time_preprocess()
        cyclic_time_features = np.repeat(
            cyclic_time_arr.to_numpy()[:, None, :],
            len(los_arr.columns),
            axis=1
        )

        los_dayweek_features = np.stack(
            [
                los_arr.to_numpy().astype(np.float32),
            ],
            axis=-1
        )

        dynamic_features = np.concatenate(
            [
                velocity_arr.to_numpy(dtype=np.float32)[:, :, None],
                los_arr.to_numpy(dtype=np.float32)[:, :, None],
                cyclic_time_features,
            ],
            axis=2
        ).astype(np.float32)
        print("Kích thước của dynamic LOS + dayweek:", los_dayweek_features.shape)
        print("Kích thước của dynamic feature:", dynamic_features.shape)

        output_dir = Path("data/preprocess")
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "dynamic_features.npy", dynamic_features)
        # Keep targets raw. Train-only normalization is fitted after the
        # chronological split in train/run_train.py.
        np.save(
            output_dir / "dynamic_velocity.npy",
            velocity_arr.to_numpy(dtype=np.float32),
        )

        return dynamic_features

    def preprocess(self):
        print("\n=== Tiến hành tạo dynamic feature cho status và train ===")
        self.save_dynamic_features()

        print("=== Xử lý xong status và train ===\n")
