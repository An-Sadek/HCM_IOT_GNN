import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from general import Preprocess

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

    def save_segment2segment(self):
        start_segment_df = self.segments_df[["_id", "s_node_id"]]
        end_segment_df = self.segments_df[["_id", "e_node_id"]]

        segment2segment_df = end_segment_df.merge(
            start_segment_df,
            how="inner",
            left_on="e_node_id",
            right_on="s_node_id"
        ).drop_duplicates()

        segment2segment_df = segment2segment_df.rename(columns={
            "_id_x": "from_segment_id", "_id_y": "to_segment_id"
        })
        segment2segment_df.to_csv("data/preprocess/segment2segment.csv", index=False)


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
        self.save_segment2segment()
        self.save_data_grid()
        
        self.write_meta("metadata/segments.yaml")
        self.save("data/preprocess/segments.csv")
        print("=== Xử lý xong segments ===\n")