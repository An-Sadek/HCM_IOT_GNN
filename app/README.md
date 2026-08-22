# Ứng dụng dự báo HTGNN

Từ thư mục gốc của repository, chạy app:

```powershell
streamlit run app/app.py
```

App tự động hiển thị các thư mục model trong `result/` có checkpoint
`htgnn_best.pt` hoặc `sehtgnn_best.pt`.
Chế độ dự báo trực tiếp nhận đúng một chỉ số thời gian; app hiển thị khoảng ngày giờ
hợp lệ tương ứng.

Để tạo dự báo toàn bộ cho một model (ví dụ `htgnn1`):

```powershell
python test/forecast_all.py --checkpoint result/htgnn1/htgnn_best.pt --output result/htgnn1/forecast_all.csv --device cuda --batch-size 16
```

Với SEHTGNN, đổi checkpoint và output tương ứng, ví dụ
`result/sehtgnn1/sehtgnn_best.pt` và `result/sehtgnn1/forecast_all.csv`.

`result/<model>/forecast_all.csv` chứa ma trận dự báo `|V| × |T|`: mỗi dòng là
một đoạn đường và mỗi cột là một timestamp. Các horizon chồng lấn được lấy trung
bình. Với các timestamp đầu chuỗi chưa đủ lịch sử, phần đầu vào thiếu được padding
bằng giá trị 0 với trạng thái không quan sát; vì vậy model vẫn trả về dự báo cho
toàn bộ `|T|` timestamp.

Trong app:

- **Dự báo mốc đã chọn** chạy model cho đoạn đường và đúng một timestamp.
- **Dự báo toàn bộ** đọc dòng tương ứng từ `forecast_all.csv`.
