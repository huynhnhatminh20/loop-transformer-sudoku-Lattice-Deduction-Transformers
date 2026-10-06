# LDT trên Scaled Sudoku Instances — Báo cáo Full Experiment

> Ngôn ngữ đơn giản. Số liệu đo thật trên GPU Modal B200.

## 1. Experiment này là gì?

- **Model (LDT)**: model giải Sudoku bằng cách "loại trừ ứng viên" (giống người chơi
  gạch bỏ số không thể điền vào từng ô).
- **Dataset mới**: `zayenz/scaled-sudoku-instances` — 102,456 puzzle Sudoku 9×9 công khai,
  gồm 10,000 puzzle gốc (Base) + 92,456 biến thể dễ hơn (Variants, được tạo bằng cách
  cho thêm số gợi ý).
- **Cách chia train/val/test**: theo nhóm "gia đình" (ancestor) — toàn bộ puzzle gốc và
  các biến thể của nó luôn nằm chung một phía, để model không "nhìn trộm" đáp án.
  Chia sẵn từ phase audit (seed 42), experiment này dùng lại, không chia lại.
- **Cách train**: từ đầu (from scratch), 4,000 bước, không dùng checkpoint cũ,
  không sửa kiến trúc model, không sửa bộ giải.

## 2. Kết quả chính

| Tập | Đúng / Tổng | Tỉ lệ | Sai | Hết giờ |
|---|---|---|---|---|
| Validation (10,174) | 10,170 | **99.96%** | 0 | 4 |
| Test (10,239) | 10,233 | **99.94%** | 0 | 6 |

Đọc đúng: model **không trả lời sai câu nào** (wrong = 0). Vài câu "hết giờ" nghĩa là
chưa giải xong trong giới hạn 50 vòng, không phải trả lời sai.

## 3. Puzzle gốc (khó) vs biến thể (dễ)

Puzzle gốc là loại khó nhất (ít số gợi ý nhất, bớt 1 số là mất nghiệm duy nhất).

| Tập | Base (gốc) | Variant (dễ hơn) |
|---|---|---|
| Validation | 997/1000 (99.7%) | 9,173/9,174 (~100%) |
| Test | 996/1000 (99.6%) | 9,237/9,239 (~100%) |

Kết luận đúng hiện tượng: Base thấp hơn Variant một chút, không lấy số chung 99.9%
để che đi chỗ khó.

## 4. Càng nhiều số gợi ý càng dễ đúng (test)

| Số gợi ý | Đúng/Tổng |
|---|---|
| 21–23 | 311/311 (100%) |
| 24 | 997/999 |
| 25 | 1,822/1,824 |
| 26 | 1,702/1,704 |
| 27 trở lên | 100% toàn bộ |

6 câu miss đều nằm ở nhóm 24–26 gợi ý. Quan hệ rõ ràng: nhiều gợi ý hơn → đúng nhiều hơn.

## 5. Theo độ khó propagation (test)

| Nhóm | Đúng/Tổng |
|---|---|
| Val (dễ nhất) | 5,000/5,000 (100%) |
| ValBS | 5,000/5,004 |
| ValDS | 99/100 |
| ValChan | 22/22 |
| BndBS | 104/105 |
| DomBS / Bnd / Dom | 8/8 |

Lưu ý: dataset **lệch** mạnh (nhóm Val+ValBS chiếm ~98%), nên các nhóm hiếm chỉ có
vài mẫu — không kết luận gì từ 1–2 mẫu.

## 6. Đi dọc "hành trình" Base → Walk 1 → … → Walk 5 (test)

| Vị trí | Đúng/Tổng |
|---|---|
| Base (gốc) | 996/1,000 |
| Walk 1 | 1,880/1,880 |
| Walk 2 | 1,863/1,863 |
| Walk 3 | 1,852/1,853 |
| Walk 4 | 1,836/1,836 |
| Walk 5 | 1,806/1,807 |

Độ đúng tăng dần từ Base sang các biến thể (vì được cho thêm gợi ý) — đúng với
thiết kế của tác giả dataset.

## 7. Quá trình học (loss giảm đều, không lỗi)

| Step | 100 | 500 | 1000 | 2000 | 3000 | 4000 |
|---|---|---|---|---|---|---|
| Loss | 0.81 | 0.26 | 0.14 | 0.10 | 0.069 | **0.060** |

Không có NaN/Inf trong toàn bộ 4,000 bước.

## 8. Tốn bao nhiêu?

| Hạng mục | Số liệu thật |
|---|---|
| GPU | NVIDIA B200 (Modal) |
| Batch | 512 (vừa VRAM, đỉnh 26.3 GB) |
| Train 4,000 steps | 1,238 giây (~21 phút) |
| Chấm validation | 16 giây |
| Chấm test | 15 giây |
| Tổng | 1,278 giây (~21 phút) |
| Tiền GPU | = 1,278s × giá B200 trên billing Modal (không đoán giá; tham khảo ~$1–2) |

Để so sánh: smoke 50 steps trước đó chạy 9.5 giây (~5.3 steps/s).

## 9. Kiểm tra gian lận dữ liệu (leakage)

```text
Train ∩ Val (nhóm gia đình): 0
Train ∩ Test (nhóm gia đình): 0
Val ∩ Test (nhóm gia đình): 0
Trùng puzzle y hệt giữa các phía: 0
```

Test chỉ được dùng **sau khi** training xong, không dùng để chọn model hay chỉnh gì.

## 10. Ba câu trả lời cuối

1. **LDT có học được trên dataset mới không?** Có — loss 0.81 → 0.06, độ đúng ~99.9%.
2. **Puzzle gốc khó có giải được không?** Được 99.6–99.7%, hơi thấp hơn biến thể (đúng kỳ vọng).
3. **Cái gì chưa kết luận được?** Mọi thứ ngoài Sudoku 9×9 chuẩn (các size 16×16/25×25
   cần đổi kiến trúc, ngoài phạm vi). Kết quả này chứng minh LDT train tốt trên
   corpus ngoài, không phải "giỏi Sudoku nói chung".

## 11. File liên quan

| File | Nội dung |
|---|---|
| `scaled_sudoku_split.jsonl` | Split đóng băng theo ancestor (seed 42) |
| `scaled_sudoku_audit.md` | Báo cáo audit dataset + split |
| `src/lattice_diffusion/data/scaled_sudoku.py` | Adapter dữ liệu (CPU-only) |
| `experiments/scaled/train.py` | Pipeline train pool-based (copy pipeline chính thức, chỉ thay dataset) |
| `experiments/scaled/run.py` | Entry Modal: train + eval + phân tích |
| Volume `/checkpoints/scaled/` | Checkpoint final + 3 checkpoint giữa + `.eval.json` chi tiết |
| `scaled_eval.json` (local) | Kết quả eval đầy đủ đã tải về |
