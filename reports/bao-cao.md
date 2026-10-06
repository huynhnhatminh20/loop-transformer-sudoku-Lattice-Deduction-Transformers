# Báo cáo — LDT trên Killer Sudoku (Exp #1, Exp #2, Smoke, Full Train)

> Ngôn ngữ đơn giản, không chuyên sâu. Kết quả đo thật trên GPU Modal B200.

## 1. Smoke test là gì?

Trước khi huấn luyện tốn tiền, ta chạy thử **50 bước huấn luyện** (thay vì hàng nghìn
bước) để kiểm tra "đường ống" có thông suốt không: dữ liệu vào đúng ->
model chạy được -> loss không lỗi -> lưu/tải checkpoint được -> chấm điểm thử được.
Nếu bước nào hỏng thì **dừng**, không đốt tiền chạy tiếp.

## 2. Mô hình đem thử là gì?

- **LDT**: model giải Sudoku bằng cách "loại trừ ứng viên" (giống người chơi Sudoku
  gạch bỏ số không thể điền).
- Model này trước đây học trên **Sudoku thường**. Giờ ta muốn nó học thêm
  **Killer Sudoku** — loại Sudoku không cho số sẵn, chỉ cho các "lồng" (cage)
  kèm tổng phải đạt (ví dụ: 2 ô trong lồng phải cộng lại bằng 35).
- Vì Killer có thêm thông tin lồng/tổng, ta mở rộng đầu vào của model:
  **mỗi ô từ 9 con số thành 9 con số + 3 thông tin lồng/tổng** (mã lồng,
  tổng mục tiêu, vị trí trong lồng).
- Phần khung chính của model giữ nguyên từ checkpoint cũ (tiết kiệm);
  chỉ ~1.158 tham số mới (phần đọc 3 thông tin mới) khởi tạo từ đầu.

## 3. Kết quả: PASS (đạt)

| # | Hạng mục kiểm tra | Kết quả |
|---|---|---|
| 1 | Dữ liệu Killer đọc đúng, đủ 3 thông tin lồng/tổng | Đạt |
| 2 | Tận dụng checkpoint cũ (799.797 tham số giữ lại, 1.158 tham số mới) | Đạt |
| 3 | Model chạy ra đúng kích thước | Đạt |
| 4 | 50 bước huấn luyện, loss giảm 0.7459 → 0.6819, không lỗi | Đạt |
| 5 | Lưu checkpoint rồi tải lại dùng tiếp được | Đạt |
| 6 | Chấm thử 8 câu Killer (chạy được từ đầu đến cuối) | Đạt |

Checkpoint smoke: `/checkpoints/killer/smoke_seed0_50s_bs128_20261003_061300.pt`

## 4. Điều thú vị phát hiện được

Lần chạy đầu **bị lỗi** — và lỗi này có ích: hóa ra không chỉ "đầu đọc"
của model phải mở rộng, mà cả "đầu ra" cũng phải mở rộng theo
(9 hàng thành 12 hàng). Nhờ smoke test bắt được lỗi này khi mới tốn vài
phút GPU, thay vì phát hiện sau cả giờ huấn luyện đắt tiền.

## 5. Việc này KHÔNG chứng minh điều gì?

- **Không** chứng minh model giải được Killer Sudoku   (mới học 50 bước,
  bài chấm thử toàn hết giờ (timeout) là bình thường).
- **Không** chứng minh thông tin lồng/tổng giúp ích (muốn biết phải huấn
  luyện đầy đủ rồi so với bản không có thông tin lồng/tổng).
- Kết quả chấm thử 8 câu chỉ để xác nhận "máy chạy", không phải để khoe điểm.

## 6. Exp #1 — Bản gốc đoán mò trên Killer (không học thêm)

- Lấy model Sudoku cũ, đưa 144 câu Killer vào mà **không cho xem lồng/tổng**,
  không học thêm gì.
- Kết quả (chạy được 56/144 câu thì hết tiền phải dừng): **đúng 0 câu**.
- Dễ hiểu vì sao: Killer không cho số sẵn, nên 144 câu đưa vào model
  **trông y hệt nhau** (toàn lưới trắng) — model không có cách nào phân biệt.
  Đây là "đối chứng": chứng minh nếu không thấy lồng/tổng thì không giải được.

## 7. Exp #2 — Huấn luyện đầy đủ 2000 bước (đã chạy)

- Cho model **xem đầy đủ lồng/tổng** + học trên 672 câu Killer (giữ khung cũ,
  799.797 tham số giữ lại, 1.158 tham số mới).
- Học có vào (loss 0.82 → 0.38) nhưng **cao hơn hẳn** Sudoku thường (~0.27):
  model học chậm và ì.
- Kiểm tra giữa chừng (144 câu validation, sau mỗi 100 bước): **đúng 0 câu,
  lần nào cũng vậy** — đáng lo từ sớm.
- Checkpoint đã lưu:
  `/checkpoints/killer/seed0_2000s_bs512_20261003_071640_20261003_071642.pt`
- Quá trình học xem trực tiếp trên wandb (project `ldt-killer`).

## 8. Chấm điểm cuối — dở dang (không phải do ta dừng)

- Chấm 144 câu test thì tới câu **46/144: đúng 0, sai 12, hết giờ 34**,
  hệ thống Modal tự ngắt (kill) giữa chừng.
- Không có điểm chính thức. Chạy nốt 98 câu còn lại tốn thêm ~$4 nên chưa chạy.
- Nhưng bức tranh đã rõ: cho xem lồng/tổng + học 2000 bước **vẫn chưa giải được**.

## 9. Vì sao thêm thông tin mà vẫn không giải được?

Nói dễ hiểu: ta mới chỉ cho model **"nhìn thấy"** con số tổng (lồng này tổng 35),
nhưng chưa dạy nó **cách dùng** con số đó để suy luận (ví dụ: tổng 35 trên 2 ô
thì chỉ có thể là 8+27? không — phải liệt kê tổ hợp, loại trừ dần...).
Bộ "suy luận tổng" này (Sum Operator) nằm ngoài phạm vi đợt này, và kết quả
trên chính là bằng chứng vì sao nó cần thiết (future work).

## 10. Ba câu trả lời cuối

1. **Model Sudoku cũ có đoán được Killer không?** Không (0/56).
2. **Thêm thông tin lồng/tổng + học thêm có giúp không?** Với số liệu hiện có:
   **chưa thấy giúp** (0/144 validation, 0/46 test dở dang).
3. **Cái gì chưa kết luận được?** Điểm chính thức của #2 (thiếu 98 câu);
   phần giúp sức riêng của "thông tin" so với "học thêm"; và mọi thứ về
   suy luận tổng (chưa có Sum Operator).

## 11. Chi phí

| Giai đoạn | Tiền GPU (B200) |
|---|---|
| Smoke #2 (50 bước) | ~$0.15–0.20 |
| Full train #2 (2000 bước) | ~$1.09 |
| Chấm test dở dang (46/144) | nằm trong ~$1.09 trên |
| Chấm nốt 98 câu (chưa chạy) | ~$4 (chưa chi) |
