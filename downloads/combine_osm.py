import json
import os
from glob import glob
from tqdm import tqdm


def merge_all_osm_batches(
    input_dir="data/raw/osm_edge_batches",
    output_file="data/raw/osm_train_edges_2020-07-03.json"
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

def merge_two_osm_files(
    file1,
    file2,
    output_file="data/raw/osm_merged.json"
):
    """
    Gộp 2 file OSM JSON thành 1 file duy nhất
    và loại bỏ phần tử trùng theo (type, id)
    """

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    all_elements = []
    metadata = None

    for file_path in [file1, file2]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            elements = data.get("elements", [])
            all_elements.extend(elements)

            print(f"✅ Loaded {len(elements):,} elements from: {file_path}")

            # Lấy metadata từ file đầu tiên
            if metadata is None:
                metadata = {
                    "version": data.get("version", 0.6),
                    "generator": data.get("generator", "OSM_Merge"),
                    "osm3s": data.get("osm3s")
                }

        except Exception as e:
            print(f"❌ Lỗi khi đọc {file_path}: {e}")
            return

    print("\nRemoving duplicates...")

    seen = set()
    unique_elements = []

    for elem in all_elements:
        elem_key = (elem.get("type"), elem.get("id"))

        if elem_key not in seen:
            seen.add(elem_key)
            unique_elements.append(elem)

    merged_data = {
        "version": metadata["version"],
        "generator": metadata["generator"],
        "osm3s": metadata["osm3s"],
        "elements": unique_elements
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print("\n✅ Merge hoàn tất!")
    print(f" Tổng elements ban đầu : {len(all_elements):,}")
    print(f" Sau dedup            : {len(unique_elements):,}")
    print(f" Đã loại trùng        : {len(all_elements) - len(unique_elements):,}")
    print(f"💾 Saved to: {output_file}")
    print(f"📦 Size: {os.path.getsize(output_file)/(1024*1024):.1f} MB")


if __name__ == "__main__":
    merge_all_osm_batches()
    merge_two_osm_files(
        "data/raw/osm_train_edges_2020-07-03.json",
        "data/raw/osm_train_nodes_2020-07-03.json",
        "data/raw/osm_train_2020-07-03.json"
    )