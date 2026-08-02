# Model18 Streamlit app

Chạy từ thư mục gốc của repository:

```powershell
pip install -r app/requirements.txt
streamlit run app/app.py
```

App nạp `result/model18/htgnn_best.pt`, dự báo mọi sliding window, lấy trung
bình các horizon chồng lấn, rồi cache kết quả riêng cho từng segment trong
`app/.cache/`. Lần suy luận đầu tiên có thể lâu vì mỗi lượt model phải xử lý
toàn bộ 10.027 segment của đồ thị.
