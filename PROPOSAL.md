# Đề xuất đề tài

**Môn:** Học sâu
**Đánh giá khả năng tổng quát hóa của looped transformer trên các bài toán Sudoku, qua ba kiến trúc: LDT, TRM, HRM**

---

## 1. Vấn đề nghiên cứu

Kiến trúc transformer là một trong những cách mạng lớn. Gần đây, các mô hình nền tảng đang được đồn đoán là dùng kiến trúc transformer loop. Transformer thông thường muốn suy luận sâu thì ta chồng nhiều tầng. Một hướng khác đang nổi lên là giữ số tầng cố định, rồi áp dụng đúng khối transformer đó lên nhiều lần với cùng một trạng thái. Mỗi lần chạy lại là một bước suy luận. Nó giúp mô hình không đi sâu hơn mà đi lâu hơn trên cùng trạng thái. Kiến trúc này gọi là transformer loop.

Ba kiến trúc trong đề tài này là ba cách thực hiện khác nhau của cùng một ý tưởng đó. LDT giữ tập ứng viên trong không gian trạng thái và lan truyền ràng buộc qua từng vòng. TRM và HRM thu nhỏ dần tập ứng viên của từng ô rồi dừng khi mô hình tự tin là xong. Chúng khác nhau ở cách biểu diễn trạng thái, nhưng cùng một cơ chế: lặp một phép biến đổi trạng thái cho tới khi đạt.

## 2. Mục tiêu nghiên cứu

Mục tiêu chính là hiểu looped transformer: một vòng lặp gồm cấu trúc gì, biến
đổi trạng thái ra sao, quyết định dừng bằng tín hiệu nào, và điều gì làm một
vòng lặp có hiệu quả.

Mục tiêu cụ thể như sau:

* Tái lập thiết lập Sudoku-Extreme cho cả ba kiến trúc, để xác nhận pipeline
  đúng trước khi so sánh. Phần TRM và HRM được chạy sau cùng, tách hoàn toàn
  về đường dẫn khỏi pipeline scaled sudoku.
* Dựng corpus Sudoku 9×9 quy mô khoảng 100.000 câu, có nhãn độ khó theo
  phương pháp suy luận yếu nhất giải được, và chia theo nhóm câu gốc để các
  biến thể của một câu không rơi vào cả train lẫn test.
* Chạy ba kiến trúc trên corpus đó ở cùng ngân sách tính toán.
* Tách kết quả theo số clue, theo câu gốc và biến thể, theo nhóm độ khó
* Đo hiệu quả tăng theo ngân sách huấn luyện như thế nào.
* Ghi nhận thời gian huấn luyện, thời gian đánh giá, bộ nhớ đỉnh.
* Thử LDT trên Snowflake Sudoku và Killer Sudoku, để xem vòng lặp xử lý được
  ràng buộc ngoài Sudoku cơ bản không.

Tiêu chuẩn để kết luận: chỉ nói mô hình nào hơn khi đã loại được lý do do cách
chấm khác nhau. Nếu chưa loại được, khoảng cách đó chỉ nói về ngân sách
huấn luyện, không nói về kiến trúc.

## 3. Câu hỏi nghiên cứu

1. Chuyển sang corpus lớn hơn, khả năng giải thay đổi thế nào?
2. Hiệu năng thay đổi thế nào theo số clue, theo câu gốc và biến thể, theo nhóm
   độ khó?
3. Ở cùng số examples đã nhìn thấy, khác biệt giữa ba kiến trúc là bao nhiêu,
   và có phải do cách chấm điểm không?
4. Tăng số vòng suy luận lúc đánh giá có làm độ đúng tăng không?
5. LDT mở rộng được tới đâu khi bài toán có ràng buộc ngoài Sudoku cơ bản?

## 4. Phương pháp nghiên cứu

**Dữ liệu.** Sudoku-Extreme dùng để tái lập. Scaled Sudoku Instances, giới hạn
Sudoku 9×9 cho khớp với implementation của các mô hình, dùng làm corpus đánh
giá chính. Snowflake và Killer Sudoku cho phần mở rộng ràng buộc.

**Mô hình.** Lấy implementation chính thức của cả ba làm chuẩn, không sửa kiến
trúc reasoning. Chỉ thay phần đọc dữ liệu, cách chia tập và cách đánh giá.

**Cách so sánh.** Khóa ngân sách theo số ví dụ mỗi mô hình đã nhìn thấy, không
theo số bước. Ba điều kiện phải giữ: cùng corpus, cùng một file split đóng
băng, không rò rỉ giữa các nhóm câu gốc. Không so trực tiếp giá trị loss, vì
ba mô hình dùng objective khác nhau.

**Loại trừ yếu tố nhiễu.** Câu hỏi số 3 có nguy cơ bị hiểu sai: mô hình nào
được chấm nhiều vòng suy luận hơn thì dễ có điểm hơn. Nên phải đo độ đúng theo
chiều sâu suy luận. Nếu tăng ngân sách suy luận lên nhiều lần mà độ đúng đứng
yên, khoảng cách là do mô hình chứ không phải do cách chấm. Ngược lại, mô hình
chưa hội tụ ở ngân sách đã dùng thì kết quả chỉ ghi là hiệu năng dưới ngân
sách đó, không phải thứ hạng về phương pháp.

**Sản phẩm.** Báo cáo và mã nguồn. Thêm giao diện demo bằng Streamlit: nhập
một câu Sudoku, xem lời giải của mô hình cạnh lời giải của solver, và xem từng
bước lặp biến đổi trạng thái thế nào — điều mà bảng số không thể hiện.

## 5. Hạn chế

Chỉ làm Sudoku 9×9. Một số nhóm độ khó chỉ có vài chục mẫu, không suy rộng
được. Ba kiến trúc dùng cơ chế huấn luyện khác nhau nên không phải chỉ số nào
cũng so thẳng được. Số lần chạy lặp bị giới hạn bởi ngân sách GPU, nên các
khác biệt nhỏ chưa chắc là thật. Phần mở rộng ràng buộc chỉ làm với LDT, và
chỉ nhằm xem giới hạn của LDT chứ không phải dựng benchmark mới.
