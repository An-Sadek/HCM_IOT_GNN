import pandas as pd
import os

def split_nodes_df(
    input_csv="data/raw/nodes.csv",
    output_dir="data/raw/nodes_df",
    chunk_size=10_000
):

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)

    total = len(df)

    print(f"Total rows: {total}")

    for i in range(0, total, chunk_size):

        chunk = df.iloc[i:i+chunk_size]

        file_idx = i // chunk_size

        save_path = os.path.join(
            output_dir,
            f"nodes_{file_idx:03d}.csv"
        )

        chunk.to_csv(save_path, index=False)

        print(
            f"Saved {save_path} "
            f"({len(chunk)} rows)"
        )

if __name__ == "__main__":
    split_nodes_df()