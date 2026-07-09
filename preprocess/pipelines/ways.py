import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from general import Preprocess

class WayPreprocess(Preprocess):
    def __init__(self, raw_root: str, osm_path:str):
        super().__init__(raw_root, osm_path)
        self.df = self.streets_df.rename(columns={"_id": "id"})

        # Mask 
        outlier_mask = (
            ~self.combine_ways_df["tags.oneway"].isin(["yes", "no", "-1"]) &
            self.combine_ways_df["tags.oneway"].notna()
        )
        self.combine_ways_df.loc[outlier_mask, "tags.oneway"] = "no"

        # Lưu 1 bản có -1
        self.combine_ways_df.to_csv("data/preprocess/combine_ways_df.csv", index=False)

        self.oh_tags = [
            "tags.surface",
            "tags.bridge",
            "tags.oneway",
            "tags.motorroad"
        ]
        
        self.num_tags = [
            "tags.lanes",
            "max_velocity",
            "tags.minspeed",
            "level"
        ]

    def max_velocity_rules(self, row: pd.Series):
        if int(row["max_velocity"]) == -1:
            # Đường cao tốc
            if row["type"] == "motorway":
                return 120
    
            oneway = row["tags.oneway"] in ["yes", "-1"]
            if oneway:
                return 50
            else:
                return 60
        return row["max_velocity"]

    def fill(self):
        valid_type = [x for x in self.combine_ways_df["type"] if not (x == "unclassified")]
        for col in self.num_tags:
            self.combine_ways_df[col] = pd.to_numeric(
                self.combine_ways_df[col],
                errors="coerce",
            )

        # Surface
        surface_mask = (
            self.combine_ways_df["type"].isin(valid_type) & 
            self.combine_ways_df["tags.surface"].isna()
        )
        self.combine_ways_df.loc[surface_mask, "tags.surface"] = "paved"
        self.combine_ways_df["tags.surface"] = self.combine_ways_df["tags.surface"].fillna("unpaved")
        self.metadata["fill"]["surface"] = "\"paved\" if tags.surface != \"unclassified\" else \"unpaved\""

        # Bridge, oneway
        self.combine_ways_df["tags.bridge"] = self.combine_ways_df["tags.bridge"].fillna("no")
        self.combine_ways_df["tags.oneway"] = self.combine_ways_df["tags.oneway"].fillna("no")
        self.metadata["fill"]["bridge"] = "no"
        self.metadata["fill"]["oneway"] = "no"
        
        # Lanes
        doubleway_mask = (
            self.combine_ways_df["type"].isin(valid_type) & 
            self.combine_ways_df["tags.lanes"].isna()
        )
        self.combine_ways_df.loc[doubleway_mask, "tags.lanes"] = 2
        self.combine_ways_df["tags.lanes"] = self.combine_ways_df["tags.lanes"].fillna(1)
        self.metadata["fill"]["lanes"] = 1

        # Max velocity
        self.combine_ways_df["max_velocity"] = self.combine_ways_df["max_velocity"].fillna(-1)
        self.combine_ways_df["max_velocity"] = self.combine_ways_df["max_velocity"].astype(float)
        self.combine_ways_df["max_velocity"] = self.combine_ways_df.apply(self.max_velocity_rules, axis=1)
        self.metadata["fill"]["max_velocity"] = "120 if tags.highway == \"motorway\" else (50 if tags.oneway == \"yes\" else 60)"
        
        # Min speed
        self.combine_ways_df["tags.minspeed"] = self.combine_ways_df["tags.minspeed"].fillna(0)

        print("Thế thành công các giá trị rỗng")

    def zscore_std(self):
        num_tags = [
            "tags.lanes",
            "max_velocity",
            "tags.minspeed",
        ]

        mm_scaler = MinMaxScaler()

        for col in num_tags:
            data = self.df[[col]]
            encoded_data = mm_scaler.fit_transform(data).reshape(-1, 1)
            self.df[col] = encoded_data
        
        print("Chuẩn hoá Z-Score thành công cho dữ liệu liên tục")
    
    def onehot_encoding(self):
        way_oh_encoder = OneHotEncoder(
            drop='first',        
            sparse_output=False,
        )
        
        # DF one-hot
        way_encoded_array = way_oh_encoder.fit_transform(self.combine_ways_df[self.oh_tags])
        way_oh_encoded_df = pd.DataFrame(
            way_encoded_array,
            columns=way_oh_encoder.get_feature_names_out(),
            index=self.combine_ways_df.index
        )
        way_oh_encoded_df.insert(0, "id", self.combine_ways_df["id"].to_numpy())

        # Merge one-hot
        self.df = self.df.merge(
            way_oh_encoded_df,
            how="inner",
            on="id"
        )

        # Viết metadata
        self.metadata["onehot"]["features"] = self.oh_tags
        self.metadata["onehot"]["onehot_feature_names"] = way_oh_encoder.get_feature_names_out().tolist()
        
        self.feature_names_out = way_oh_encoder.get_feature_names_out().tolist()
        print("OH thành công")

    def save_data_grid(self):
        self.df = self.df.sort_values("id")

        static_way_savepath = "data/preprocess/static_ways.npy"
        static_way_grid = self.df[self.feature_names_out + self.num_tags].to_numpy().astype(np.float32)
        
        self.metadata["conversion"] = self.conversion_dict["way"]
        np.save(
            static_way_savepath,
            static_way_grid
        )
        print("Đã lưu static way tại:", static_way_savepath)

    def save_way2way(self):
        filtered_osm_ways_df = self.combine_ways_df[["id", "nodes"]]
        
        # Thêm node
        filtered_osm_ways_df["start_node"] = filtered_osm_ways_df["nodes"].apply(lambda x: x[0])
        filtered_osm_ways_df["end_node"] = filtered_osm_ways_df["nodes"].apply(lambda x: x[-1])

        # Thêm tag oneway
        filtered_osm_ways_df = filtered_osm_ways_df.merge(
            self.combine_ways_df[["id", "tags.oneway"]],
            how="inner",
            on="id"
        )

        filtered_osm_ways_df.to_csv("data/preprocess/way2way.csv", index=False)
        print("Đã xử lý xong way2way")

    def save_way_segment(self):
        oneway_df = self.combine_ways_df[["id", "tags.oneway"]]
        filtered_segment_df = self.segments_df[["_id", "street_id"]].rename(
            columns={
                "_id": "segment_id", 
                "street_id": "id"
            }
        )
        oneway_df = oneway_df.merge(
            filtered_segment_df,
            how="inner",
            on="id"
        )
        oneway_df.to_csv("data/preprocess/oneway_df.csv", index=False)

    def preprocess(self): 
        print("\n=== Đang xử lý way ===")
        self.fill()
        self.onehot_encoding()
        self.df = self.df.drop(columns=self.num_tags, errors="ignore")

        self.df = self.df.merge(
            self.combine_ways_df[["id"] + self.num_tags],
            how="inner",
            on="id"
        )
        self.zscore_std()
        self.save_way_segment()
        self.save_way2way()
        self.save_data_grid()

        self.write_meta("metadata/ways.yaml")
        self.save("data/preprocess/ways.csv")
        print("=== Xử lý xong way ===\n")