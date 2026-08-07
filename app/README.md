# Ứng dụng dự báo HTGNN

Từ thư mục gốc của repository, tạo file dự báo toàn bộ rồi chạy app:

```powershell
python test/forecast_all.py --device cuda --batch-size 16
streamlit run app/app.py
```

`result/htgnn/forecast_all.csv` chứa ma trận dự báo `|V| × |T|`: mỗi dòng là
một đoạn đường và mỗi cột là một timestamp. Các horizon chồng lấn được lấy trung
bình. Với các timestamp đầu chuỗi chưa đủ lịch sử, phần đầu vào thiếu được padding
bằng giá trị 0 với trạng thái không quan sát; vì vậy model vẫn trả về dự báo cho
toàn bộ `|T|` timestamp.

Trong app:

- **Dự báo một phần** chạy model cho đoạn đường và khoảng chỉ số hợp lệ đã chọn.
- **Dự báo toàn bộ** đọc dòng tương ứng từ `forecast_all.csv`.
