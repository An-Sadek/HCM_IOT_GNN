import sys
sys.path.append(".")

from pathlib import Path
from torch_geometric.data import HeteroData
from utils import load_raw_data

PATH = Path("data/raw")

raw_dataset = load_raw_data()
data = HeteroData()

data["node"].x      = raw_dataset["nodes"].drop(columns=["_id"])
data["street"].x    = raw_dataset["streets"].drop(columns=["_id"])
data["segment"].x   = raw_dataset["segments"].drop(columns=["_id"])
data["status"].x    = raw_dataset["status"].drop(columns=["_id"])
data["train"].x     = raw_dataset["train"].drop(columns=["_id"])

