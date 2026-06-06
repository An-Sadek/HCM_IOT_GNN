from pathlib import Path

import pandas as pd
import numpy as np

import yaml
import json

from sklearn.preprocessing import OneHotEncoder


class Preprocess:
    def __init__(self, 
                 raw_root: str="data/raw", 
                 osm_path: str="data/raw/osm_train_2019_01_03.json"
    ):
        self.raw_root = Path(raw_root)
        osm_path = Path(osm_path)
        assert self.raw_root.exists(), "Đường dẫn không tồn tại"
        assert osm_path.exists(), "File không tồn tại"

        # Load HCM Traffic Flow
        self.nodes_df = pd.read_csv(self.raw_root / "nodes.csv").sort_values("_id")
        self.segments_df = pd.read_csv(self.raw_root / "segments.csv").sort_values("_id")
        self.streets_df = pd.read_csv(self.raw_root / "streets.csv").sort_values("_id")
        self.status_df = pd.read_csv(self.raw_root/ "segment_status.csv").sort_values("_id")
        self.train_df = pd.read_csv(self.raw_root / "train.csv").sort_values("_id")

        train_nodes = sorted(
            set(self.train_df["s_node_id"]) |
            set(self.train_df["e_node_id"])
        )
        train_segments = sorted(self.train_df["segment_id"].unique().tolist())
        train_ways = sorted(self.train_df["street_id"].unique().tolist())

        # Xây dựng dict để chuyển id sang index
        ## Node
        node_index2id = dict()
        node_id2index = dict()
        for index, node_id in enumerate(train_nodes):
            node_index2id[index] = node_id
            node_id2index[node_id] = index
        node_dict = {
            "index2id": node_index2id,
            "id2index": node_id2index
        }

        ## Segment
        segment_index2id = dict()
        segment_id2index = dict()
        for index, segment_id in enumerate(train_segments):
            segment_index2id[index] = segment_id
            segment_id2index[segment_id] = index
        segment_dict = {
            "index2id": segment_index2id,
            "id2index": segment_id2index
        }

        # Way
        way_index2id = dict()
        way_id2index = dict()
        for index, way_id in enumerate(train_ways):
            way_index2id[index] = way_id
            way_id2index[way_id] = index
        way_dict = {
            "index2id": way_index2id,
            "id2index": way_id2index
        }
        self.conversion_dict = {
            "node": node_dict,
            "segment": segment_dict,
            "way": way_dict
        }

        # Filter raw ids first
        self.nodes_df = self.nodes_df[self.nodes_df["_id"].isin(train_nodes)].copy()
        self.segments_df = self.segments_df[self.segments_df["_id"].isin(train_segments)].copy()
        self.streets_df = self.streets_df[self.streets_df["_id"].isin(train_ways)].copy()

        # Then convert to index
        self.nodes_df["_id"] = self.nodes_df["_id"].map(node_id2index)

        self.segments_df["s_node_id"] = self.segments_df["s_node_id"].map(node_id2index)
        self.segments_df["e_node_id"] = self.segments_df["e_node_id"].map(node_id2index)
        self.segments_df["_id"] = self.segments_df["_id"].map(segment_id2index)
        self.segments_df["street_id"] = self.segments_df["street_id"].map(way_id2index)

        self.streets_df["_id"] = self.streets_df["_id"].map(way_id2index)

        self.train_df["s_node_id"] = self.train_df["s_node_id"].map(node_id2index)
        self.train_df["e_node_id"] = self.train_df["e_node_id"].map(node_id2index)
        self.train_df["segment_id"] = self.train_df["segment_id"].map(segment_id2index)
        self.train_df["street_id"] = self.train_df["street_id"].map(way_id2index)

        self.status_df["segment_id"] = (
            self.status_df["segment_id"]
            .map(self.conversion_dict["segment"])
        )

        # Load OSM
        with open(osm_path, "r", encoding="utf-8") as f:
            osm_data = json.load(f)
        self.osm_elements_df = pd.json_normalize(osm_data["elements"])

        # OSM Node
        self.osm_nodes_df = self.osm_elements_df[self.osm_elements_df["type"] == "node"]
        self.osm_nodes_df = self.osm_nodes_df[self.osm_nodes_df["id"].isin(train_nodes)]
        self.osm_nodes_df["id"] = self.osm_nodes_df["id"].map(node_id2index)
        self.osm_nodes_df = self.osm_nodes_df.drop(columns=["type"])
        self.combine_nodes_df = self.nodes_df.merge(
            self.osm_nodes_df,
            how="inner",
            left_on="_id",
            right_on="id"
        ).drop(columns="_id")

        # OSM Way
        self.osm_ways_df = self.osm_elements_df[self.osm_elements_df["type"] == "way"]
        self.osm_ways_df = self.osm_ways_df[self.osm_ways_df["id"].isin(train_ways)]
        self.osm_ways_df["id"] = self.osm_ways_df["id"].map(way_id2index)
        self.osm_ways_df = self.osm_ways_df.drop(columns=["type"])
        self.combine_ways_df = self.streets_df.merge(
            self.osm_ways_df,
            how="inner",
            left_on="_id",
            right_on="id"
        ).drop(columns="_id")

        self.df = pd.DataFrame()
        self.metadata = {
            "fill": {}, 
            "onehot": {},
            "conversion": {}
        }

        self.feature_names_out = None

    def save(self, output_file):
        assert Path(output_file).parent.exists(), "Thư mục cha không tồn tại"
        assert len(self.df.columns) > 0, "DF rỗng kiểm tra lại"
        self.df.to_csv(output_file, index=False)

    def write_meta(self, output_file):
        assert Path(output_file).parent.exists(), "Thư mục cha không tồn tại"
        with open(output_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.metadata, f, sort_keys=False)

    def fill(self):
        pass

    def onehot_encoding(self):
        pass


class NodePreprocess(Preprocess):
    def __init__(self, raw_root:str, osm_path:str):
        super().__init__(raw_root, osm_path)
        self.df = self.nodes_df

        self.oh_tags = [
            "tags.railway",
            "tags.junction",
            "tags.crossing",
            "tags.highway",
            "tags.bus"
        ]

    def fill(self):
        for tag in self.oh_tags:
            self.combine_nodes_df[tag] = self.combine_nodes_df[tag].fillna("no")
        self.metadata["fill"] = {key: "no" for key in self.oh_tags}

    def onehot_encoding(self):
        node_oh_encoder = OneHotEncoder(
            drop='first',        
            sparse_output=False,
        )

        # DF one-hot
        node_encoded_array = node_oh_encoder.fit_transform(self.combine_nodes_df[self.oh_tags])
        node_oh_encoded_df = pd.DataFrame(
            node_encoded_array,
            columns=node_oh_encoder.get_feature_names_out(),
            index=self.combine_nodes_df.index
        )
        node_oh_encoded_df.insert(0, "id", self.combine_nodes_df["id"].to_numpy())

        # Gộp lại
        self.df = self.df.merge(
            node_oh_encoded_df,
            how="inner",
            left_on="_id",
            right_on="id"
        ).drop(columns="_id")

        # Viết metadata
        self.metadata["onehot"]["features"] = self.oh_tags
        self.metadata["onehot"]["onehot_feature_names"] = node_oh_encoder.get_feature_names_out().tolist()

        # Node thực ra không cần
        self.feature_names_out = node_oh_encoder.get_feature_names_out().tolist()

    def save_data_grid(self):
        self.df = self.df.sort_values("id")
        static_node_savepath = "data/preprocess/static_nodes.npy"

        static_node_features = self.df.drop(columns=["id", "long", "lat"]).to_numpy().astype(np.float32)
        
        self.metadata["conversion"] = self.conversion_dict["node"]
        np.save(
            static_node_savepath,
            static_node_features
        )

        print("Đã lưu static_node_grid tại:", static_node_savepath)

    def preprocess(self):
        print("\n=== Tiến hành xử lý node ===")
        self.fill()
        self.onehot_encoding()
        self.save_data_grid()

        self.write_meta("metadata/nodes.csv")
        self.save("data/preprocess/nodes.csv")
        print("=== Xử lý xong node ===\n")

        
class WayPreprocess(Preprocess):
    def __init__(self, raw_root: str, osm_path:str):
        super().__init__(raw_root, osm_path)
        self.df = self.streets_df.rename(columns={"_id": "id"})
        
        self.oh_tags = [
            "tags.surface",
            "tags.bridge",
            "tags.oneway"
        ]
        
        self.num_tags = [
            "tags.lanes",
            "tags.layer",
            "max_velocity",
            "tags.minspeed",
            "level"
        ]

    def max_velocity_rules(self, row: pd.Series):
        if int(row["max_velocity"]) == -1:
            # Đường cao tốc
            if row["type"] == "motorway":
                return 120
    
            oneway = row["tags.oneway"] == "yes"
            if oneway:
                return 50
            else:
                return 60
        return row["max_velocity"]

    def fill(self):
        valid_type = [x for x in self.combine_ways_df["type"] if not (x == "unclassified")]

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

        # Layer
        self.combine_ways_df["tags.layer"] = self.combine_ways_df["tags.layer"].fillna(0)
        self.metadata["fill"]["lanes"] = 0

        # Max velocity
        self.combine_ways_df["max_velocity"] = self.combine_ways_df["max_velocity"].fillna(-1)
        self.combine_ways_df["max_velocity"] = self.combine_ways_df["max_velocity"].astype(float)
        self.combine_ways_df["max_velocity"] = self.combine_ways_df.apply(self.max_velocity_rules, axis=1)
        self.metadata["fill"]["max_velocity"] = "120 if tags.highway == \"motorway\" else (50 if tags.oneway == \"yes\" else 60)"
        print("Thế thành công các giá trị rỗng")

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
        filtered_osm_ways_df["start_node"] = filtered_osm_ways_df["nodes"].apply(lambda x: x[0])
        filtered_osm_ways_df["end_node"] = filtered_osm_ways_df["nodes"].apply(lambda x: x[-1])
        filtered_osm_ways_df.to_csv("data/preprocess/way2way.csv", index=False)
        print("Đã xử lý xong way2way")

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
        self.save_way2way()
        self.save_data_grid()

        self.write_meta("metadata/ways.csv")
        self.save("data/preprocess/ways.csv")
        print("=== Xử lý xong way ===\n")


class SegmentPreprocess(Preprocess):
    def __init__(self, raw_root:str, osm_path:str):
        super().__init__(raw_root, osm_path)

        # Vì streets.name != segments.street_name và 
        # streets.type != segments.street_type
        rename_dict = {
            "_id": "id",
            "street_name": "name",
            "street_type": "type"
        }
        self.df = self.segments_df.rename(columns=rename_dict)
        
        self.metadata["rename"] = rename_dict
        self.metadata["onehot"] = {}

    def onehot_encoding(self):
        segment_oh_encoder = OneHotEncoder(
            drop='first',        
            sparse_output=False,
        )
        
        segment_encoded_array = segment_oh_encoder.fit_transform(self.df[["type"]])
        segment_oh_encoded_df = pd.DataFrame(
            segment_encoded_array, 
            columns=segment_oh_encoder.get_feature_names_out(),
            index=self.df.index
        )
        segment_oh_encoded_df.insert(0, "id", self.df["id"])

        self.df = self.df.merge(
            segment_oh_encoded_df,
            how="inner",
            on="id"
        )

        self.metadata["onehot"]["features"] = ["type"]
        self.metadata["onehot"]["onehot_feature_names"] = segment_oh_encoder.get_feature_names_out().tolist()
        self.feature_names_out = segment_oh_encoder.get_feature_names_out().tolist()
        print("OH thành công")

    def normalize_length(self):
        """
        Chuẩn hóa m -> km
        """
        self.df["length"] = self.df["length"] / 1000
        self.metadata["normalize"] = "length / 1000"
        print("Chuẩn hóa từ km -> m cho length")

    def create_edges(self):
        node_segment_edges_df = self.segments_df[["_id", "s_node_id", "e_node_id"]]
        node_segment_edges_df = node_segment_edges_df.rename(columns={"_id": "id"})
        print("Kích thước của [segments] ---(has[startswith|endswiths])---> [nodes]", node_segment_edges_df.shape)
        node_segment_edges_df.to_csv("data/preprocess/nodes_segments_edges_df.csv", index=False)
        print("Tạo và lưu các cạnh từ edge và node gốc")


    def save_data_grid(self):
        self.df = self.df.sort_values("id")

        segment_grid_savepath = "data/preprocess/static_segments.npy"
        static_segment_grid = self.df[self.feature_names_out + ["length"]].to_numpy().astype(np.float32)
        
        np.save(
            segment_grid_savepath,
            static_segment_grid
        )
        self.metadata["conversion"] = self.conversion_dict["segment"]
        print("Đã lưu static segment tại:", segment_grid_savepath)


    def preprocess(self):
        print("\n=== Xử lý segments ===")
        self.onehot_encoding()
        self.normalize_length()
        self.create_edges()
        self.save_data_grid()
        
        self.write_meta("metadata/segments.csv")
        self.save("data/preprocess/segments.csv")
        print("=== Xử lý xong segments ===\n")


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
        min_timestamp = self.status_df["updated_at"].min()
        max_timestamp = self.status_df["updated_at"].max()
        self.full_time = pd.date_range(
            min_timestamp,
            max_timestamp,
            freq="30min"
        )

        # Metadata
        self.metadata = dict()

    def status_preprocess(self):
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

        # Gộp lại, fill với -1 thể hiện nó bị trống
        velocity_arr = velocity_mat.reindex(self.full_time)
        velocity_arr = velocity_mat.fillna(-1).to_numpy().astype(np.float32)
        mask_arr = (velocity_mat == 0).astype(np.float32)
        X = np.stack(
            [velocity_arr, mask_arr],
            axis=-1
        )
        
        print("Kích thước của pivot table velocity và mask trong status df:", X.shape)
        np.save("data/preprocess/dynamic_velocity.npy", X)
        print("Xử lý xong velocity của status")

    def train_preprocess(self):
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

        # Gộp lại, fill -1 thể hiện data bị trống
        los_arr = los_mat.reindex(self.full_time)
        mask_arr = los_mat.isna().to_numpy().astype(np.float32)
        los_arr = los_mat.fillna(-1).to_numpy().astype(np.float32)
        X = np.stack(
            [los_arr, mask_arr],
            axis=-1
        )
        print("Kích thước của pivot table LOS và mask trong train df:", X.shape)
        np.save("data/preprocess/dynamic_LOS.npy", X)
        print("Xử lý thành công LOS của train")

    def target_preprocess(self):
        los_mat = self.train_df.pivot(
            index="timestamp",
            columns="segment_id",
            values="LOS"
        ).reindex(self.full_time)
        los_mat = los_mat.ffill().bfill()
        print("Xử lý xong target của dynamic")

    def preprocess(self):
        print("\n=== Tiến hành tạo dynamic feature cho status và train ===")
        self.train_preprocess()
        self.status_preprocess()
        self.target_preprocess()

        print("=== Xử lý xong status và train ===\n")


class RelationPreprocess(Preprocess):
    def __init__(self, 
                 raw_root: str="data/raw", 
                 osm_path: str="data/raw/osm_train_2019_01_03.json"
    ):
        super().__init__(raw_root, osm_path)

    def save_relation_df(self):
        osm_relation_df = self.osm_elements_df[
            self.osm_elements_df["type"] == "relation"
        ]
        osm_relation_df = osm_relation_df.drop(columns=
            ["lat", "lon", "nodes"] + 
            [x for x in osm_relation_df.columns if x.startswith("tags.")]
        )
        df_exploded = osm_relation_df.explode('members').reset_index(drop=True)
        member_df = pd.json_normalize(df_exploded['members'])
        member_df.to_csv("data/preprocess/relation_members.csv", index=False)
        print("Lưu thành công đường cấm rẽ")

    def preprocess(self):
        print("\n=== Đang xử lý relation ===")
        self.save_relation_df()
        print("\n=== Xử lý xong relation ===")


if __name__ == "__main__":
    node_process = NodePreprocess(
        "data/raw", 
        "data/raw/osm_train_2019_01_03.json"
    )
    node_process.preprocess()
    del node_process

    way_process = WayPreprocess(
        "data/raw", 
        "data/raw/osm_train_2019_01_03.json"
    )
    way_process.preprocess()
    del way_process

    segment_process = SegmentPreprocess(
        "data/raw", 
        "data/raw/osm_train_2019_01_03.json"
    )
    segment_process.preprocess()
    del segment_process
    
    dynamic_process = DynamicPreprocess()
    dynamic_process.preprocess()
    del dynamic_process

    relation_process = RelationPreprocess()
    relation_process.preprocess()
    del relation_process