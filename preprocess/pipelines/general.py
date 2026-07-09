from pathlib import Path
import os
import tempfile

import pandas as pd

import yaml
import json


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

    def to_index(self, x, ref_dict: dict):
        try:
            return ref_dict[int(x)]
        except (KeyError, TypeError, ValueError):
            return x
        
    def convert_row_ref(self, row):
        obj_type = row["type"]
        obj_ref = row["ref"]
        
        if obj_type in ["node", "way"]:
            mapping_dict = self.conversion_dict[obj_type]["id2index"]
            return self.to_index(obj_ref, mapping_dict)
        
        return obj_ref

    def save(self, output_file):
        output_path = Path(output_file)
        assert output_path.parent.exists(), "Thư mục cha không tồn tại"
        assert len(self.df.columns) > 0, "DF rỗng kiểm tra lại"
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                dir=output_path.parent,
            )
            os.close(fd)
            self.df.to_csv(tmp_path, index=False)
            os.replace(tmp_path, output_path)
        finally:
            if tmp_path is not None and Path(tmp_path).exists():
                os.remove(tmp_path)

    def write_meta(self, output_file):
        assert Path(output_file).parent.exists(), "Thư mục cha không tồn tại"
        with open(output_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.metadata, f, sort_keys=False)

    def fill(self):
        pass

    def onehot_encoding(self):
        pass