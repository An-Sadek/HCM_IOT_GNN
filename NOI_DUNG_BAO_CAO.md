# DỰ BÁO VẬN TỐC GIAO THÔNG TP.HCM BẰNG MẠNG NƠ-RON ĐỒ THỊ DỊ THỂ THEO THỜI GIAN

> Bản nội dung dùng cho báo cáo/slide. Các số liệu trong tài liệu được lấy trực tiếp từ dữ liệu, mã nguồn và thư mục `result` của dự án.

## Slide 1 — Trang bìa

**Dự báo vận tốc giao thông TP.HCM bằng mạng nơ-ron đồ thị dị thể theo thời gian**

- Sinh viên thực hiện: [Điền tên]
- Giảng viên hướng dẫn: [Điền tên]
- Đơn vị/lớp: [Điền thông tin]

**Lời nói gợi ý:**

Xin chào thầy/cô và các bạn. Đề tài của em tập trung vào dự báo vận tốc giao thông trên các đoạn đường tại TP.HCM. Khác với chuỗi thời gian thông thường, giao thông vừa phụ thuộc vào lịch sử của chính đoạn đường, vừa chịu ảnh hưởng từ các đoạn đường kết nối xung quanh. Vì vậy, em biểu diễn mạng đường dưới dạng đồ thị dị thể theo thời gian và sử dụng HTGNN để học đồng thời quan hệ không gian và thời gian.

## Slide 2 — Bối cảnh và vấn đề

- Ùn tắc làm tăng thời gian di chuyển, tiêu hao nhiên liệu và gây khó khăn cho quản lý đô thị.
- Vận tốc tại một đoạn đường biến đổi theo thời gian và bị ảnh hưởng bởi các đoạn đường lân cận.
- Mô hình chuỗi thời gian độc lập khó mô tả cấu trúc liên kết của mạng giao thông.
- Bài toán: sử dụng dữ liệu lịch sử để dự báo vận tốc của từng đoạn đường trong các mốc thời gian tiếp theo.

**Lời nói gợi ý:**

Vận tốc giao thông không phải là các chuỗi độc lập. Chẳng hạn, khi một đoạn đường bị chậm, ảnh hưởng có thể lan sang các đoạn nối tiếp. Đồng thời, lưu lượng còn có tính chu kỳ theo giờ trong ngày và ngày trong tuần. Do đó, bài toán cần một mô hình có thể học cả phụ thuộc không gian trên mạng đường và phụ thuộc thời gian.

## Slide 3 — Mục tiêu và đóng góp

**Mục tiêu**

- Xây dựng pipeline từ dữ liệu giao thông và OpenStreetMap đến đồ thị có thể huấn luyện.
- Dự báo vận tốc cho 10.027 đoạn đường theo nhiều bước thời gian.
- Đánh giá mô hình bằng RMSE, R² và MAPE trên tập kiểm thử tương lai.

**Đóng góp chính**

- Xây dựng đồ thị dị thể gồm `node`, `way` và `segment`.
- Kết hợp đặc trưng tĩnh của hạ tầng với đặc trưng giao thông động.
- Xét đường một chiều và quan hệ cấm rẽ khi tạo cạnh giữa các đoạn đường.
- Chia dữ liệu theo trình tự thời gian và chuẩn hóa chỉ trên tập train để hạn chế data leakage.

## Slide 4 — Nguồn và quy mô dữ liệu

| Thành phần | Quy mô |
|---|---:|
| Bản ghi vận tốc | 90.938 |
| Bản ghi LOS | 33.441 |
| Mốc thời gian sau đồng bộ | 14.049 |
| Tần suất | 30 phút/mốc |
| Khoảng thời gian | 03/07/2020 – 22/04/2021 |
| Nút giao/điểm OSM | 11.314 |
| Tuyến đường (`way`) | 1.967 |
| Đoạn đường (`segment`) | 10.027 |
| Quan hệ giao thông OSM | 63 quan hệ, 177 thành viên |

**Lời nói gợi ý:**

Dữ liệu động gồm vận tốc và mức phục vụ giao thông LOS từ A đến F. Dữ liệu bản đồ được bổ sung từ OpenStreetMap để mô tả nút giao, tuyến đường, loại đường, đường một chiều và các hạn chế rẽ. Sau khi đồng bộ, dữ liệu có 14.049 mốc, mỗi mốc cách nhau 30 phút và bao phủ 10.027 đoạn đường.

## Slide 5 — Khám phá và thách thức dữ liệu

- Dữ liệu đến từ nhiều nguồn nên khóa định danh và thời gian cần được đồng bộ.
- Quan sát vận tốc không phủ đều mọi đoạn đường và mọi thời điểm.
- LOS là biến phân loại A–F; vận tốc là biến liên tục.
- Thuộc tính OSM có nhiều giá trị thiếu và nhiều trường phân loại.
- Quy mô tensor lớn: 14.049 × 10.027 × 7 đặc trưng động.

**Lời nói gợi ý:**

Thách thức lớn nhất không chỉ là xây mô hình mà còn là đưa các nguồn dữ liệu về cùng một cấu trúc. Nếu điền dữ liệu thiếu hoặc chia tập không cẩn thận, mô hình có thể nhìn thấy thông tin tương lai và kết quả sẽ không phản ánh khả năng triển khai thực tế.

## Slide 6 — Tiền xử lý dữ liệu động

1. Chuyển thời gian về cùng định dạng và gom các quan sát vào bucket 30 phút.
2. Nếu một bucket có nhiều quan sát vận tốc, lấy giá trị trung bình.
3. Pivot dữ liệu thành ma trận `[thời gian, đoạn đường]`.
4. Điền về phía trước (`forward fill`) và giới hạn vận tốc trong khoảng 1–120.
5. Mã hóa LOS: A→0, B→1, …, F→5; giá trị chưa quan sát được biểu diễn bằng −1.
6. Tạo mask để phân biệt giá trị quan sát thật với giá trị thiếu/được điền.
7. Mã hóa chu kỳ thời gian bằng sin/cos cho giờ trong ngày và ngày trong tuần.

**Đặc trưng động tại mỗi segment**

`[velocity, LOS, sin(time), cos(time), sin(weekday), cos(weekday), ...]`

Tensor cuối cùng có kích thước **(14.049, 10.027, 7)**.

## Slide 7 — Tiền xử lý đặc trưng tĩnh

- Điền các thuộc tính OSM còn thiếu bằng giá trị mặc định phù hợp.
- One-hot encoding các biến phân loại.
- Đặc trưng `node`: tọa độ, giao lộ, đường sắt, vạch qua đường, đèn tín hiệu…
- Đặc trưng `way`: loại đường, bề mặt, một chiều, motorroad, số làn, tốc độ…
- Đặc trưng `segment`: chiều dài, cấp đường và loại địa điểm/đường liên quan.

| Loại nút | Số đối tượng | Số đặc trưng tĩnh |
|---|---:|---:|
| Node | 11.314 | 9 |
| Way | 1.967 | 12 |
| Segment | 10.027 | 26 |

## Slide 8 — Xây dựng đồ thị dị thể

Đồ thị gồm ba loại nút:

- `node`: điểm đầu/cuối hoặc nút giao trên bản đồ.
- `way`: một tuyến đường trong OpenStreetMap.
- `segment`: đoạn đường cần dự báo vận tốc.

Bốn loại quan hệ:

- `way → segment`: tuyến đường **chứa** đoạn đường.
- `node → segment`: node là **điểm bắt đầu** của segment.
- `node → segment`: node là **điểm kết thúc** của segment.
- `segment → segment`: hai đoạn đường **có thể đi tiếp** sang nhau.

**Lời nói gợi ý:**

Em không chỉ nối các segment theo khoảng cách. Quan hệ segment–segment được tạo theo tính liên thông thực tế: điểm cuối của đoạn trước trùng điểm đầu đoạn sau. Pipeline còn loại các hướng không hợp lệ trên đường một chiều và loại các cặp vi phạm quan hệ cấm rẽ trong OSM. Vì vậy, cạnh mang ý nghĩa giao thông rõ ràng hơn.

## Slide 9 — Biểu diễn đồ thị theo thời gian

- Với mỗi cửa sổ lịch sử, cấu trúc mạng đường được lặp lại theo từng timestamp `t0…tW−1`.
- Đặc trưng tĩnh mô tả hạ tầng; đặc trưng động mô tả trạng thái giao thông tại từng thời điểm.
- Đầu vào: `W` mốc lịch sử của toàn bộ đồ thị.
- Đầu ra: vận tốc của mỗi segment trong `Q` mốc tương lai.
- Thí nghiệm tốt nhất hiện có sử dụng **W = 24** và **Q = 12**: dùng 12 giờ lịch sử để dự báo 6 giờ tiếp theo.

## Slide 10 — Kiến trúc HTGNN

Luồng xử lý của một khối HTGNN:

1. **Input adapter:** đưa đặc trưng của từng loại nút về cùng không gian ẩn.
2. **Relation-specific GAT:** tổng hợp hàng xóm riêng cho từng loại quan hệ tại mỗi thời điểm.
3. **Relation attention:** học trọng số quan trọng giữa các quan hệ đi vào một loại nút.
4. **Temporal self-attention:** học phụ thuộc giữa các timestamp trong cửa sổ lịch sử.
5. **Residual gate:** kết hợp biểu diễn mới với thông tin đầu vào để ổn định huấn luyện.
6. **Node predictor:** dự báo đồng thời vận tốc của từng segment cho toàn bộ horizon.

**Lời nói gợi ý:**

GAT trả lời câu hỏi đoạn đường nên nhận bao nhiêu thông tin từ các hàng xóm. Attention giữa quan hệ giúp mô hình phân biệt ảnh hưởng từ tuyến đường, điểm đầu, điểm cuối và đoạn đường kế tiếp. Temporal attention tiếp tục xác định mốc lịch sử nào quan trọng đối với dự báo hiện tại.

## Slide 11 — Thiết kế huấn luyện và đánh giá

- Chia theo thời gian: **70% train – 20% validation – 10% test**.
- Bỏ một khoảng đệm giữa các tập bằng `window + horizon − 1`, tránh các cửa sổ dùng chung timestamp.
- Chuẩn hóa vận tốc theo từng segment, chỉ fit thống kê trên vùng train.
- Hàm mất mát: MSE có mask.
- Optimizer/hyperparameter của cấu hình tốt nhất: learning rate 0,005; weight decay 0,0001; hidden dimension 12; 1 lớp HTGNN; batch size 16.
- Tối đa 200 epoch, early stopping sau 10 epoch không cải thiện validation RMSE.
- Metric: RMSE, R² và MAPE.

**Giải thích metric**

- RMSE càng thấp càng tốt; phạt mạnh các sai số lớn.
- R² càng gần 1 càng tốt; thể hiện tỷ lệ biến thiên được mô hình giải thích.
- MAPE thể hiện sai số phần trăm trung bình, nhưng nhạy khi vận tốc thực gần 0.

## Slide 12 — Kết quả thực nghiệm

**Kết quả tốt nhất đã lưu trong dự án**

| Mô hình/cấu hình | RMSE | R² | MAPE |
|---|---:|---:|---:|
| HTGNN, W=24, Q=12, 1 lớp | **6,998** | **0,8453** | 27,72% |
| HTGNN, W=12, Q=12, 2 lớp | 7,031 | 0,8438 | **21,33%** |
| HTGNN, W=12, Q=12, 2 lớp (lần chạy khác) | 7,504 | 0,8221 | 21,75% |
| SE-HTGNN, W=24, Q=24 | 25,591 | −1,0689 | 60,17% |

**Nhận xét**

- HTGNN tốt nhất đạt RMSE khoảng **7 đơn vị vận tốc** và giải thích khoảng **84,5%** biến thiên trên tập test.
- Tăng độ sâu không bảo đảm cải thiện RMSE; cấu hình một lớp với cửa sổ 24 mốc đang tốt nhất về RMSE/R².
- SE-HTGNN hiện chưa hội tụ tốt trong pipeline này; R² âm cho thấy kết quả kém hơn dự báo bằng giá trị trung bình.
- Không nên tuyên bố SE-HTGNN kém về bản chất; kết quả có thể đến từ cách tích hợp embedding, hyperparameter hoặc quy trình huấn luyện hiện tại.

## Slide 13 — Kết quả thực tế và demo

**Luồng demo**

`Dữ liệu lịch sử → tiền xử lý → tạo heterograph → HTGNN → dự báo vận tốc → hiển thị trên bản đồ`

Khi demo nên trình bày:

- Chọn một thời điểm dự báo.
- Hiển thị vận tốc dự báo của các đoạn đường bằng màu sắc.
- Chọn một segment và so sánh đường dự báo với giá trị thực trong 12 bước tiếp theo.
- Chỉ ra một trường hợp dự báo tốt và một trường hợp sai số lớn.

**Câu nói an toàn:**

Kết quả cho thấy mô hình bám được xu hướng chung và phần lớn biến động của vận tốc. Tuy nhiên, tại các thay đổi đột ngột hoặc đoạn có ít quan sát, đường dự báo có xu hướng mượt và phản ứng chậm hơn dữ liệu thật.

## Slide 14 — Thảo luận và hạn chế

- Dữ liệu vận tốc thưa; forward fill có thể kéo dài trạng thái cũ quá lâu.
- MAPE thiếu ổn định khi vận tốc thật nhỏ.
- Chưa có đầy đủ biến ngoại sinh như mưa, tai nạn, sự kiện, ngày lễ hoặc công trình.
- Đồ thị hiện chủ yếu dựa trên topology tĩnh, trong khi mức ảnh hưởng giữa các đoạn đường có thể thay đổi theo thời gian.
- Chưa có bộ baseline đầy đủ như Persistence, GRU/LSTM, GCN thuần hoặc mô hình spatio-temporal khác.
- Chưa báo cáo sai số riêng theo horizon, khu vực và mức độ thiếu dữ liệu.

**Lời nói gợi ý:**

Điểm hạn chế quan trọng nhất của thực nghiệm hiện tại là chưa có baseline phi đồ thị đầy đủ. Vì vậy, kết quả chứng minh HTGNN dự báo tốt trên tập test, nhưng chưa đủ để định lượng chính xác graph cải thiện bao nhiêu so với LSTM hoặc phương pháp giữ nguyên vận tốc gần nhất.

## Slide 15 — Kết luận và hướng phát triển

**Kết luận**

- Đã hoàn thiện pipeline xử lý dữ liệu giao thông và OSM thành đồ thị dị thể theo thời gian.
- Đã mô hình hóa đồng thời quan hệ không gian, loại quan hệ và diễn biến thời gian.
- HTGNN tốt nhất đạt RMSE **6,998**, R² **0,8453** trên tập kiểm thử tương lai.
- Kết quả cho thấy cách tiếp cận đồ thị có tiềm năng cho dự báo giao thông quy mô lớn tại TP.HCM.

**Hướng phát triển**

- Bổ sung baseline và ablation study để đo đóng góp của từng thành phần.
- Thử graph động hoặc học trọng số cạnh thích nghi theo thời gian.
- Bổ sung thời tiết, ngày lễ, sự kiện và dữ liệu tai nạn.
- Thay forward fill bằng phương pháp imputation có xét không gian–thời gian.
- Tối ưu suy luận và triển khai cập nhật dự báo gần thời gian thực.

## Slide 16 — Q&A

**Xin cảm ơn thầy/cô và các bạn đã lắng nghe.**

---

## Các câu hỏi phản biện dễ gặp

### 1. Vì sao dùng GNN thay vì LSTM?

LSTM chủ yếu học diễn biến theo thời gian của chuỗi đầu vào. Trong giao thông, các đoạn đường có quan hệ liên thông và ảnh hưởng lẫn nhau. GNN giúp truyền thông tin theo cấu trúc mạng đường, còn temporal attention học phụ thuộc theo thời gian. Hai loại phụ thuộc được xử lý trong cùng mô hình.

### 2. Vì sao dùng đồ thị dị thể?

Mạng đường có nhiều thực thể và quan hệ khác nhau. Một segment thuộc một way, bắt đầu/kết thúc tại node và kết nối với segment khác. Gộp tất cả thành một loại nút sẽ làm mất ý nghĩa này; đồ thị dị thể cho phép mô hình học phép tổng hợp riêng cho từng quan hệ.

### 3. Làm sao tránh data leakage?

Dữ liệu được chia theo thứ tự thời gian, không xáo trộn ngẫu nhiên. Giữa train, validation và test có khoảng đệm bằng `window + horizon − 1` để các cửa sổ không dùng chung timestamp. Thống kê chuẩn hóa vận tốc cũng chỉ được fit trên vùng train.

### 4. Tại sao forward fill?

Forward fill giữ lại trạng thái gần nhất và đơn giản để triển khai trên chuỗi thưa. Tuy nhiên, phương pháp này có thể làm trạng thái cũ tồn tại quá lâu, nên dự án giữ mask quan sát để nhận biết dữ liệu gốc và xem đây là một hạn chế cần cải thiện.

### 5. R² bằng 0,845 có ý nghĩa gì?

Trên tập test, mô hình giải thích được khoảng 84,5% tổng biến thiên của vận tốc so với việc chỉ dùng giá trị trung bình. Đây là mức phù hợp khá tốt, nhưng không có nghĩa mọi đoạn đường hay mọi thời điểm đều đạt độ chính xác giống nhau.

### 6. Vì sao kết quả SE-HTGNN kém?

Trong thí nghiệm hiện tại, việc thêm semantic embedding chưa tạo ra cải thiện và mô hình chưa hội tụ tốt. Nguyên nhân có thể nằm ở chênh lệch không gian biểu diễn, cách fusion hoặc hyperparameter. Do chưa có ablation đầy đủ, chỉ nên kết luận cấu hình hiện tại kém hơn HTGNN, không kết luận kiến trúc SE-HTGNN luôn kém.

### 7. Graph có thực sự giúp không?

Thiết kế mô hình cho phép khai thác graph và kết quả HTGNN hiện tại tốt. Tuy nhiên, để chứng minh định lượng graph giúp bao nhiêu, cần so sánh cùng dữ liệu với Persistence, MLP, GRU/LSTM và mô hình bỏ phần GAT. Đây là thực nghiệm cần bổ sung.

### 8. Có thể triển khai thực tế không?

Có thể xây pipeline nhận dữ liệu vận tốc mới, cập nhật tensor theo bucket 30 phút, chạy suy luận và hiển thị kết quả trên bản đồ. Trước khi vận hành thật cần kiểm thử độ trễ, khả năng xử lý dữ liệu thiếu và độ ổn định khi phân phối dữ liệu thay đổi.

## Checklist trước khi báo cáo

- Thay tên sinh viên, giảng viên và tên môn học ở trang bìa.
- Chèn một hình bản đồ mạng đường và một hình kiến trúc HTGNN.
- Chèn biểu đồ train/validation loss từ file history.
- Chèn ít nhất một biểu đồ `ground truth vs prediction`.
- Nếu chưa chạy baseline, dùng đúng cách diễn đạt trong phần hạn chế; không khẳng định HTGNN tốt hơn LSTM.
- Kiểm tra đơn vị vận tốc trong nguồn dữ liệu trước khi ghi `km/h` trên slide; mã nguồn chỉ xác nhận giá trị `velocity`, chưa xác nhận metadata đơn vị.
- Khi nói “kết quả tốt nhất”, ghi rõ đây là kết quả tốt nhất trong các lần chạy đã lưu của dự án.
