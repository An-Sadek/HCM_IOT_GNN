from general import Preprocess
import pandas as pd


class RelationPreprocess(Preprocess):
    def __init__(self, 
                 raw_root: str="data/raw", 
                 osm_path: str="data/raw/osm_train_2019_01_03.json"
    ):
        super().__init__(raw_root, osm_path)

    def abc(self):
        self.df = self.osm_elements_df[
            self.osm_elements_df["type"] == "relation"
        ].copy()
        self.df = self.df.dropna(how="all", axis=1)
        self.df.to_csv("data/preprocess/relation.csv")

    def save_relation_df(self):
        """
        Lưu các relation members lại kèm theo luật cấm rẽ (tags.restriction)
        """
        # 1. Định nghĩa các cột muốn xóa, NHƯNG giữ lại 'tags.restriction' nếu có
        columns_to_drop = ["lat", "lon", "nodes"]
        
        # Tìm tất cả các cột tags.* NGOẠI TRỪ tags.restriction
        tags_to_drop = [
            col for col in self.df.columns 
            if col.startswith("tags.") and col != "tags.restriction"
        ]
        columns_to_drop.extend(tags_to_drop)

        try:
            # Drop các cột không cần thiết, giữ lại id, members, và tags.restriction
            df_filtered = self.df.drop(columns=columns_to_drop, errors='ignore')
        except KeyError:
            df_filtered = self.df.copy()

        # 2. Explode để bung mớ hỗn độn members ra
        df_exploded = df_filtered.explode('members').reset_index(drop=True)
        
        # 3. Normalize thông tin chi tiết của member (type, ref, role)
        member_df = pd.json_normalize(df_exploded['members'])
        
        # 4. Chèn id gốc của relation vào đầu bảng
        member_df.insert(0, "id", df_exploded["id"].to_numpy())
        
        # 5. CHÈN THÊM tags.restriction VÀO ĐÂY
        if "tags.restriction" in df_exploded.columns:
            member_df.insert(1, "restriction", df_exploded["tags.restriction"].to_numpy())
        else:
            # Trường hợp dữ liệu không có bất kỳ relation restriction nào
            member_df.insert(1, "restriction", None)

        # 6. Convert sang index hệ thống của bạn
        member_df["ref"] = member_df.apply(self.convert_row_ref, axis=1)
        
        # Lưu file
        member_df.to_csv("data/preprocess/relation_members.csv", index=False)
        print("Lưu thành công đường cấm rẽ với thuộc tính restriction")

        # Convert sang index
        member_df["ref"] = member_df.apply(self.convert_row_ref, axis=1)
        
        member_df.to_csv("data/preprocess/relation_members.csv", index=False)
        print("Lưu thành công đường cấm rẽ")

    def preprocess(self):
        print("\n=== Đang xử lý relation ===")
        self.abc()
        self.save_relation_df()
        print("\n=== Xử lý xong relation ===")
