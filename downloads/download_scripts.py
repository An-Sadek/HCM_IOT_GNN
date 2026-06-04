import requests
import pandas as pd
import json
import os
import time
from pathlib import Path
from glob import glob
from tqdm import tqdm

# OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_URL = "https://lz4.overpass-api.de/api/interpreter"

query_scripts = {
    "node": """
        [out:json][timeout:600][date:"2019-01-03T00:00:00Z"];
        (
          node(id:{});
        );
        (._;>;);
        out body;
        """,

    "way"
}

def split_train_nodes_df(
    input_csv="data/raw/train.csv",
    nodes_dir="data/raw/node_id_batches",
    chunk_size=100
):
    os.makedirs(nodes_dir, exist_ok=True)
    
    df = pd.read_csv(input_csv)
    
    # Lấy tất cả node_id unique
    node_ids = sorted(
        set(df["s_node_id"].tolist()) | 
        set(df["e_node_id"].tolist())
    )
    print(f"Total unique nodes: {len(node_ids)}")
    
    # Chia thành các batch
    for i in range(0, len(node_ids), chunk_size):
        batch = node_ids[i:i + chunk_size]
        batch_df = pd.DataFrame({"_id": batch})
        
        save_path = os.path.join(nodes_dir, f"nodes_batch_{i//chunk_size:04d}.csv")
        batch_df.to_csv(save_path, index=False)
        
        if i % (chunk_size * 10) == 0:  # in mỗi 10 batch
            print(f"Saved batch {i//chunk_size:04d} with {len(batch)} nodes")
    
    print(f"✅ Đã chia xong {len(glob(os.path.join(nodes_dir, '*.csv')))} batches")


def query_batch_nodes(node_ids, save_path):
    node_str = ",".join(map(str, node_ids))
    
    query = f"""
    [out:json][timeout:600][date:"2019-01-03T00:00:00Z"];
    (
      node(id:{node_str});
      way(bn)["highway"];
      relation(bw)["type"="restriction"];
      relation(bw)["route"];
    );
    (._;>;);
    out body;
    """

    headers = {
        "User-Agent": "HCM_Traffic_Flow/1.0",
        "Accept": "application/json"
    }

    for retry in range(5):
        try:
            response = requests.post(
                OVERPASS_URL,
                data=query,
                headers=headers,
                timeout=300
            )
            response.raise_for_status()
            data = response.json()

            if not data.get("elements"):
                print(f"⚠️ Batch rỗng: {save_path}")
            else:
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"✅ Saved {len(data['elements'])} elements → {save_path}")
            return True

        except Exception as e:
            print(f"Retry {retry+1}/5: {e}")
            time.sleep(15 * (retry + 1))  # backoff

    print(f"❌ Failed after 5 retries: {save_path}")
    return False


def process_all_batches(
    nodes_dir="data/raw/node_id_batches",
    output_dir="data/raw/osm_node_batches"
):
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = sorted(glob(os.path.join(nodes_dir, "*.csv")))
    print(f"Found {len(csv_files)} batches to process\n")
    
    for idx, csv_path in enumerate(csv_files):
        print(f"[{idx+1:03d}/{len(csv_files)}] Processing: {os.path.basename(csv_path)}")
        
        df = pd.read_csv(csv_path)
        node_ids = df["_id"].tolist()
        
        save_path = os.path.join(output_dir, f"osm_{idx:04d}.json")
        
        # Skip nếu đã tồn tại
        if os.path.exists(save_path):
            print("   → Skip (đã tồn tại)")
            continue
            
        success = query_batch_nodes(node_ids, save_path)
        
        if success:
            time.sleep(8)   # Nghỉ giữa các batch thành công
        else:
            time.sleep(30)  # Nghỉ lâu hơn nếu fail

def merge_all_osm_batches(
    input_dir="data/raw/osm_node_batches",
    output_file="data/raw/osm_train_2019_01_03.json"
):
    """
    Gộp tất cả các file osm_*.json thành một file duy nhất
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    json_files = sorted(glob(os.path.join(input_dir, "osm_*.json")))
    print(f"Found {len(json_files)} batch files to merge...\n")
    
    all_elements = []
    total_elements = 0
    metadata = None
    
    for file_path in tqdm(json_files, desc="Merging batches"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            elements = data.get("elements", [])
            all_elements.extend(elements)
            total_elements += len(elements)
            
            # Lấy metadata từ file đầu tiên
            if metadata is None and "version" in data:
                metadata = {
                    "version": data.get("version"),
                    "generator": data.get("generator"),
                    "osm3s": data.get("osm3s")
                }
                
        except Exception as e:
            print(f"❌ Lỗi khi đọc {file_path}: {e}")
            continue
    
    # Loại bỏ phần tử trùng lặp theo id (rất quan trọng)
    print("Removing duplicate elements by id...")
    seen = {}
    unique_elements = []
    
    for elem in tqdm(all_elements, desc="Deduplicating"):
        elem_id = (elem.get("type"), elem.get("id"))
        if elem_id not in seen:
            seen[elem_id] = True
            unique_elements.append(elem)
    
    # Tạo dữ liệu cuối cùng
    merged_data = {
        "version": metadata.get("version") if metadata else 0.6,
        "generator": metadata.get("generator") if metadata else "HCM_Traffic_Flow_Merge",
        "osm3s": metadata.get("osm3s") if metadata else None,
        "elements": unique_elements
    }
    
    print(f"\n✅ Merge hoàn tất!")
    print(f"   Tổng elements trước dedup : {total_elements:,}")
    print(f"   Sau khi loại trùng       : {len(unique_elements):,}")
    print(f"   Tiết kiệm                : {total_elements - len(unique_elements):,} elements")
    
    # Lưu file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Đã lưu file tổng hợp tại: {output_file}")
    print(f"   Kích thước: {os.path.getsize(output_file) / (1024*1024):.1f} MB")


if __name__ == "__main__":
    split_train_nodes_df()
    process_all_batches()
    merge_all_osm_batches()