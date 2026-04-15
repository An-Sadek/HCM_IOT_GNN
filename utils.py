from pathlib import Path
import pandas as pd

PATH = Path("data/raw")

def load_raw_data(raw_path: Path=PATH) -> dict[str, pd.DataFrame]:
    nodes_df    = pd.read_csv(raw_path / "nodes.csv")
    streets_df  = pd.read_csv(raw_path / "streets.csv")
    segments_df = pd.read_csv(raw_path / "segments.csv")
    status_df   = pd.read_csv(raw_path / "segment_status.csv")
    train_df    = pd.read_csv(raw_path / "train.csv")

    return {
        "nodes": nodes_df,
        "streets": streets_df,
        "segments": segments_df,
        "status": status_df,
        "train": train_df
    }