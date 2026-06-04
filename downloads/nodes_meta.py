import requests
import pandas as pd
import json
import os
import time
from pathlib import Path
from glob import glob
from tqdm import tqdm

OVERPASS_URL = "https://lz4.overpass-api.de/api/interpreter"
# OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# OVERPASS_URL = "https://overpass.private.coffee/api/interpreter"

def query_node_history_overpass(node_ids, save_path, max_retries=6):
    """Fetch full history for a list of nodes (one by one - safer)"""
    if not node_ids:
        return False
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    all_elements = []
    
    for node_id in tqdm(node_ids, desc=f"Fetching history {Path(save_path).name}", leave=False):
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
                
                if resp.status_code == 429:  # Rate limit
                    sleep_time = 15 * (retry + 1)
                    print(f" ⏳ Rate limit on node {node_id} - sleep {sleep_time}s")
                    time.sleep(sleep_time)
                    continue
                
                if resp.status_code in (502, 503, 504):
                    time.sleep(10 * (retry + 1))
                    continue
                
                resp.raise_for_status()
                data = resp.json()
                
                if "elements" in data and data["elements"]:
                    all_elements.extend(data["elements"])
                
                time.sleep(1.0)  # Be gentle with the public instance
                break
                
            except requests.exceptions.RequestException as e:
                print(f"❌ Error node {node_id} (retry {retry+1}): {e}")
                time.sleep(8 * (retry + 1))
            except Exception as e:
                print(f"Unexpected error node {node_id}: {e}")
                time.sleep(5)
        
        else:
            print(f"⚠️ Failed after {max_retries} retries: node {node_id}")
    
    # Save
    if all_elements:
        result = {"elements": all_elements}
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        num_versions = len(all_elements)
        num_unique = len(set(el.get("id") for el in all_elements))
        print(f"✅ Saved {num_unique} nodes — {num_versions} versions → {Path(save_path).name}")
        return True
    else:
        print(f"⚠️ No data for batch: {Path(save_path).name}")
        return False


def process_all_batches_sync(
    nodes_dir="data/raw/osm_node_ids",
    output_dir="data/raw/osm_node_history_all",
    batch_size=10
):
    os.makedirs(output_dir, exist_ok=True)
    
    # Tạo batch nếu chưa có
    if not glob(os.path.join(nodes_dir, "*.csv")):
        split_train_nodes_df(chunk_size=batch_size)
    
    csv_files = sorted(glob(os.path.join(nodes_dir, "*.csv")))
    print(f"📁 Tìm thấy {len(csv_files)} batches\n")
    
    for idx, csv_path in enumerate(csv_files):
        save_path = os.path.join(output_dir, f"history_{idx:04d}.json")
        
        if os.path.exists(save_path):
            print(f"[{idx+1:03d}/{len(csv_files)}] Skip (đã tồn tại): {os.path.basename(csv_path)}")
            continue
        
        df = pd.read_csv(csv_path)
        node_ids = df["_id"].astype(int).tolist()
        
        print(f"\n[{idx+1:03d}/{len(csv_files)}] Processing batch: {len(node_ids)} nodes")
        query_node_history_overpass(node_ids, save_path)
        
        time.sleep(1.2)  # Nghỉ giữa các batch


def split_train_nodes_df(input_csv="data/raw/train.csv",
                        nodes_dir="data/raw/osm_node_ids",
                        chunk_size=10):
    os.makedirs(nodes_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    
    # Lấy danh sách các node bị lệch
    """
    with open("data/raw/osm_train_2020-07-03.json", "r", encoding="utf-8") as f:
        osm_data = json.load(f)
    osm_elements = pd.DataFrame(osm_data["elements"])
    osm_nodes_df = osm_elements[osm_elements["type"] == "node"]

    osm_node_locs = set(osm_nodes_df[["id", "lon", "lat"]].apply(tuple, axis=1))
    train_node_locs = set(df[["s_node_id", "long_snode", "lat_snode"]].apply(tuple, axis=1)) | \
                    set(df[["e_node_id", "long_enode", "lat_enode"]].apply(tuple, axis=1))
    conflict_node_locs = train_node_locs - osm_node_locs
    conflict_node_ids = sorted([x[0] for x in conflict_node_locs])
    print("Số node bị lệch tọa độ:", len(conflict_node_ids))
    """
    node_ids = sorted(
        set(df["s_node_id"]) |
        set(df["e_node_id"])
    )
    print("Tổng số node:", len(node_ids))

    # Chia batch
    for i in range(0, len(node_ids), chunk_size):
        batch = node_ids[i:i + chunk_size]
        batch_df = pd.DataFrame({"_id": batch})
        save_path = os.path.join(nodes_dir, f"nodes_batch_{i//chunk_size:04d}.csv")
        batch_df.to_csv(save_path, index=False)
    
    print(f"✅ Đã chia thành {len(glob(os.path.join(nodes_dir, '*.csv')))} batches")


def merge_history_files(
    input_dir="data/raw/osm_node_history_all",
    output_file="data/raw/osm_full_history_all.json"
):
    """
    Gộp tất cả các file history_*.json thành một file duy nhất
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    json_files = sorted(glob(os.path.join(input_dir, "history_*.json")))
    
    if not json_files:
        print("❌ Không tìm thấy file history nào!")
        return False
    
    print(f"📁 Tìm thấy {len(json_files)} file history. Đang gộp...")
    
    all_elements = []
    total_versions = 0
    
    for file_path in tqdm(json_files, desc="Merging files"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            if "elements" in data and isinstance(data["elements"], list):
                all_elements.extend(data["elements"])
                total_versions += len(data["elements"])
        except Exception as e:
            print(f"⚠️ Lỗi khi đọc {Path(file_path).name}: {e}")
    
    # Lưu file gộp
    result = {
        "elements": all_elements,
        "meta": {
            "total_nodes": len(set(el.get("id") for el in all_elements)),
            "total_versions": len(all_elements),
            "source_files": len(json_files)
        }
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ GỘP HOÀN TẤT!")
    print(f"   • Số file gốc     : {len(json_files)}")
    print(f"   • Tổng versions   : {total_versions:,}")
    print(f"   • Số node unique  : {result['meta']['total_nodes']:,}")
    print(f"   • File đầu ra     : {output_file}")
    
    return True



if __name__ == "__main__":
    # ================== CẤU HÌNH ==================
    BATCH_SIZE = 10          # Khuyến nghị: 8 - 12 (query nặng)
    # =============================================
    
    # process_all_batches_sync(batch_size=BATCH_SIZE)
    merge_history_files()