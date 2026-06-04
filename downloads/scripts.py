import requests
import pandas as pd
import json
import os
import time
from pathlib import Path
from glob import glob
from tqdm import tqdm

OVERPASS_URL = "https://lz4.overpass-api.de/api/interpreter"
# Backup: "https://overpass-api.de/api/interpreter"

def query_node_full_history(node_ids, save_path, max_retries=5):
    """Lấy full history + lat/lon cho từng node"""
    if not node_ids:
        return False

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    all_elements = []
    
    for node_id in tqdm(node_ids, desc=f"History {Path(save_path).name}", leave=False):
        query = f"""
        [out:json][timeout:120];
        timeline(node, {node_id});
        for (t["created"]) {{
          retro(_.val) {{
            node({node_id});
            out meta;
          }}
        }}
        """
        
        for retry in range(max_retries):
            try:
                resp = requests.post(
                    OVERPASS_URL,
                    data=query.strip(),
                    headers={
                        "User-Agent": "HCM_Traffic_Flow/1.0 (contact: nhankhdl2211012@student.ctuet.edu.vn)",
                        "Accept": "application/json"
                    },
                    timeout=180
                )
                
                if resp.status_code == 429:
                    sleep_time = 12 * (retry + 1)
                    print(f"  Rate limit → sleep {sleep_time}s")
                    time.sleep(sleep_time)
                    continue
                    
                resp.raise_for_status()
                data = resp.json()
                
                if "elements" in data and data["elements"]:
                    all_elements.extend(data["elements"])
                
                time.sleep(0.75)   # Điều chỉnh tùy server
                break
                
            except requests.exceptions.RequestException as e:
                print(f"Error node {node_id} (retry {retry+1}): {e}")
                time.sleep(6 * (retry + 1))
            except Exception as e:
                print(f"Unexpected error node {node_id}: {e}")
                time.sleep(5)
        
        else:
            print(f"❌ Thất bại sau {max_retries} lần: node {node_id}")
    
    if all_elements:
        result = {"elements": all_elements}
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        total_versions = len(all_elements)
        unique_nodes = len(set(el.get('id') for el in all_elements))
        print(f"✅ Saved {unique_nodes} nodes - {total_versions} versions (có lat/lon) → {Path(save_path).name}")
        return True
    else:
        print(f"⚠️ Không có dữ liệu: {Path(save_path).name}")
        return False


def process_all_batches_sync(
    nodes_dir="data/raw/nodes_meta",
    output_dir="data/raw/osm_full_history",
    batch_size=12          # Khuyến nghị nhỏ vì query nặng
):
    os.makedirs(output_dir, exist_ok=True)
    
    if not glob(os.path.join(nodes_dir, "*.csv")):
        split_train_nodes_df(chunk_size=batch_size)
    
    csv_files = sorted(glob(os.path.join(nodes_dir, "*.csv")))
    print(f"Found {len(csv_files)} batches\n")
    
    for idx, csv_path in enumerate(csv_files):
        save_path = os.path.join(output_dir, f"history_{idx:04d}.json")
        
        if os.path.exists(save_path):
            print(f"[{idx+1:03d}/{len(csv_files)}] Skip (đã tồn tại)")
            continue
        
        df = pd.read_csv(csv_path)
        node_ids = df["_id"].astype(int).tolist()
        
        query_node_full_history(node_ids, save_path)
        time.sleep(1.0)   # Nghỉ giữa các batch


def split_train_nodes_df(input_csv="data/raw/train.csv",
                        nodes_dir="data/raw/nodes_meta",
                        chunk_size=12):
    os.makedirs(nodes_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    
    node_ids = sorted(set(df["s_node_id"]) | set(df["e_node_id"]))
    print(f"Total unique nodes: {len(node_ids)}")
    
    for i in range(0, len(node_ids), chunk_size):
        batch = node_ids[i:i + chunk_size]
        batch_df = pd.DataFrame({"_id": batch})
        save_path = os.path.join(nodes_dir, f"nodes_batch_{i//chunk_size:04d}.csv")
        batch_df.to_csv(save_path, index=False)
    
    print(f"✅ Đã chia {len(glob(os.path.join(nodes_dir, '*.csv')))} batches")


if __name__ == "__main__":
    process_all_batches_sync(batch_size=12)