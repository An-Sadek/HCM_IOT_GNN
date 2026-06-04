from pathlib import Path
import pandas as pd
from matplotlib_venn import venn3
import matplotlib.pyplot as plt

from collections.abc import Iterable

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

def create_venn_diagram(A: set[int], B: set[int], C: set[int], figsize=(6, 6), label=""):
    plt.figure(figsize=figsize)

    # Tập riêng
    _A = A - B - C
    _B = B - A - C
    _C = C - A - B

    # Giao 2 tập (loại bỏ giao 3)
    A_B = (A & B) - C
    A_C = (A & C) - B
    B_C = (B & C) - A

    # Giao 3 tập
    A_B_C = A & B & C

    venn = venn3(
        subsets=(
            len(_A), len(_B), len(_C),
            len(A_B), len(A_C), len(B_C),
            len(A_B_C)
        )
    )

    venn.get

    plt.title(label)
    plt.show()



