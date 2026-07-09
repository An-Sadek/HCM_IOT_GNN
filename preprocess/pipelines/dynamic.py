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

        # Gộp lại, ffill rồi bfill
        velocity_arr = velocity_mat.reindex(self.full_time)
        velocity_arr = velocity_mat.ffill().bfill()
        
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


    def save_dynamic_features(self):
        """
        """
        los_arr = self.los_preprocess()
        velocity_arr = self.velocity_preprocess()
        dayweek_arr = self.dayweek_preprocess(columns=los_arr.columns)

        los_dayweek_features = np.stack(
            [
                los_arr.to_numpy().astype(np.float32),
                dayweek_arr.to_numpy().astype(np.float32)
            ],
            axis=-1
        )

        np.save(
            "data/preprocess/dynamic_LOS.npy",
            los_arr.to_numpy().astype(np.float32)
        )
        np.save(
            "data/preprocess/dynamic_velocity.npy",
            velocity_arr.to_numpy().astype(np.float32)
        )
        np.save(
            "data/preprocess/dynamic_dayweek.npy",
            dayweek_arr.to_numpy().astype(np.float32)
        )
        np.save(
            "data/preprocess/dynamic_LOS_dayweek.npy",
            los_dayweek_features
        )

        dynamic_features = np.stack([los_arr, velocity_arr, dayweek_arr], axis=1)
        dynamic_features = np.transpose(dynamic_features, (0, 2, 1))
        print("Kích thước của dynamic LOS + dayweek:", los_dayweek_features.shape)
        print("Kích thước của dynamic feature:", dynamic_features.shape)

        np.save(
            "data/preprocess/dynamic_features.npy",
            dynamic_features
        )

        return dynamic_features

    def preprocess(self):
        print("\n=== Tiến hành tạo dynamic feature cho status và train ===")
        self.save_dynamic_features()

        print("=== Xử lý xong status và train ===\n")
