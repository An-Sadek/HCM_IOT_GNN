import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from general import Preprocess


class NodePreprocess(Preprocess):
    def __init__(self, raw_root:str, osm_path:str):
        super().__init__(raw_root, osm_path)
        self.df = self.nodes_df

        self.oh_tags = [
            "tags.railway",
            "tags.crossing",
            "tags.highway",
        ]

    def add_junction_feature(self):
        segment_endpoints = pd.concat(
            [
                self.segments_df[["_id", "s_node_id"]].rename(
                    columns={"_id": "segment_id", "s_node_id": "node_id"}
                ),
                self.segments_df[["_id", "e_node_id"]].rename(
                    columns={"_id": "segment_id", "e_node_id": "node_id"}
                ),
            ],
            ignore_index=True,
        ).drop_duplicates()

        connection_counts = segment_endpoints.groupby("node_id")["segment_id"].nunique()
        self.df["junction"] = (
            self.df["_id"]
            .map(connection_counts)
            .fillna(0)
            .ge(3)
            .astype(np.int8)
        )

        print("Thêm điểm giao nhau, có tổng cộng có", self.df["junction"].sum(), "điểm giao nhau")
        self.metadata["junction"] = "1 if node connects to at least 3 distinct segments else 0"

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
        self.add_junction_feature()
        self.onehot_encoding()
        self.save_data_grid()

        self.write_meta("metadata/nodes.yaml")
        self.save("data/preprocess/nodes.csv")
        print("=== Xử lý xong node ===\n")