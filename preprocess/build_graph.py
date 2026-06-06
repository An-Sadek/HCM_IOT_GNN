import os
from pathlib import Path

import numpy as np
import pandas as pd

import torch

from torch_geometric.data import HeteroData

PREPROCESS_ROOT = Path("../data/preprocess")
RAW_ROOT = Path("../data/raw")



def build_static_graph():
    # Load csv
    nodes_df = pd.read_csv(PREPROCESS_ROOT / "nodes.csv")
    segments_df = pd.read_csv(PREPROCESS_ROOT / "segments.csv")
    ways_df = pd.read_csv(PREPROCESS_ROOT / "ways.csv")
    nodes_segments_edges_df = pd.read_csv(
        PREPROCESS_ROOT / "nodes_segments_edges_df.csv"
    )

    static_node_features = np.load(PREPROCESS_ROOT / "static_nodes.npy")
    static_segment_features = np.load(PREPROCESS_ROOT / "static_segments.npy")
    static_way_features = np.load(PREPROCESS_ROOT / "static_ways.npy")

    data = HeteroData()

    