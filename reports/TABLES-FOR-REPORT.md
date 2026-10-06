# Bộ bảng dùng cho đồ án (7 bảng)

> Chỉ 7 bảng này. Tất cả số đo thật trên B200, bf16, seed 0, cùng file
> split đóng băng. Bảng 3 là bảng bắt buộc có — thiếu nó thì kết luận
> bị hiểu sai (xem ghi chú dưới bảng).

---

## B1. Thiết lập và kết quả chính

| | LDT | TRM | HRM |
|---|---|---|---|
| Tham số | **799.635** | 5.027.842 | 27.274.242 |
| Steps × batch | 4.000 × 512 | 2.667 × 768 | 2.667 × 768 |
| Examples nhìn thấy | 2.048.000 | 2.048.256 | 2.048.256 |
| **Validation (10.174)** | **99,96%** | 31,02% | 34,72% |
| **Test (10.239)** | **99,94%** | 30,45% | **34,50%** |
| Greedy 1 đường | — | 17,86% | 18,14% |
| Câu trả lời sai | **0** | — | — |

Ba model nhìn cùng khoảng 2,05 triệu examples (compute-matched).
Greedy = cách chấm của paper; Vote 8 = 1 bản gốc + 7 biến thể bầu đa
số, dùng để cân test-time giữa ba bên. LDT `sai = 0`: vài câu "hết
giờ" là chưa giải xong trong 50 vòng, không phải trả lời sai.

---

## B2. Puzzle gốc khó vs biến thể dễ

| Tập | | LDT | TRM | HRM |
|---|---|---|---|---|
| Test | Base (gốc minimal) | **996/1.000** | **0/1.000** | 6/1.000 |
| | Variant | 9.237/9.239 | 3.118/9.239 | 3.526/9.239 |

Đây là phát hiện quan trọng nhất: LDT giải được puzzle gốc minimal,
hai model recursive gần như không giải được câu nào. Khoảng cách giữa
ba model **không** nằm ở nhóm biến thể dễ.

---

## B3. Độ đúng theo ngân sách huấn luyện — bắt buộc có

| Examples | LDT | TRM | HRM |
|---|---|---|---|
| 512.256 | — | **0,00%** | — |
| 1.023.744 | — | **0,00%** | — |
| 2.048.000 | **99,94%** | — | — |
| 2.048.256 | — | **30,45%** | **34,50%** |

```
0,00% ──┐
        │                        ┌─ 30,45%
0,00% ─┤                        │
        └────────────┬───────────┘
    512K    1,02M   2,05M  examples
```

**Ghi chú bắt buộc kèm theo bảng này.** Đây là so sánh compute-matched
hợp lệ: cùng split, cùng 2,05 triệu examples, cùng ngân sách eval (đã
kiểm ở `EVAL-SWEEP-AND-CURVE.md` §1). Nhưng ở một nửa ngân sách, TRM giải
được 0 trên 10.239 câu, dù loss đã giảm rõ (0,61) — tức model không bị
kẹt, chỉ là chưa giải trọn câu nào. Đường cong là **vực thẳm rồi nhảy**,
không dốc đều.

Hệ quả cần nói đúng, và **không nói quá**:

> Ở ngân sách compute-matched 2,05 triệu examples, LDT đạt 99,94%
> (996/1.000 base) trong khi TRM và HRM đạt 30,45% và 34,50%, gần như 0
> trên 1.000 puzzle khó nhất. Khoảng cách này là **thật** và không phải
> artifact chấm điểm. Tuy nhiên TRM/HRM **chưa hội tụ** tại điểm này, nên
> bảng **không** dùng để xếp hạng method hay kết luận về data efficiency.

Thiếu ghi chú này, người đọc sẽ hiểu 99,94% so với 34,50% là chênh lệch
3 lần về chất lượng method. Ngược lại, nếu dùng ghi chú để **loại** bảng
thì cũng sai — bảng này là phép đo đúng thiết kế, chỉ giới hạn ở phép suy ra.

---

## B4. Theo độ khó propagation (test)

**Cách đọc nhóm.** Dataset gắn mỗi câu vào nhóm theo *phương pháp suy
luận yếu nhất giải được nó không cần tìm kiếm* (nguồn:
`github.com/zayenz/scaled-sudoku-instances`). Tên nhóm ghép từ mô hình
cơ sở và phép cắt bổ sung:

| Thành phần | Nghĩa |
|---|---|
| `Val` / `Bnd` / `Dom` | Mô hình lan truyền giá trị / biên / miền — `Val` yếu nhất |
| `BS` | + bounds shaving (chỉ kiểm giá trị ở hai đầu miền) |
| `DS` | + domain shaving (kiểm mọi giá trị còn lại) |
| `Chan` | + ràng buộc channeling hàng/cột/vùng |

Nhóm yếu hơn = chỉ cần suy luận đơn giản hơn. **Điểm quan trọng: nhãn
là phương pháp *yếu nhất* giải được** — nên câu gắn `ValBS` là câu mà
lan truyền giá trị đơn thuần **không** giải nổi, phải thêm cắt biên mới
được.

| Nhóm (yếu → mạnh) | n | LDT | TRM | HRM |
|---|---|---|---|---|
| `Val` | 5.000 | **100,0%** | 61,5% | 68,1% |
| `ValBS` | 5.004 | **99,9%** | 0,8% | 2,5% |
| `ValDS` | 100 | **99,0%** | 1,0% | 1,0% |
| `ValChan` | 22 | **100,0%** | 0,0% | 0,0% |
| `Bnd` | 2 | **100,0%** | 0,0% | 0,0% |
| `BndBS` | 105 | **99,0%** | 0,0% | 0,0% |
| `Dom` | 2 | **100,0%** | 0,0% | 50,0% |
| `DomBS` | 4 | **100,0%** | 0,0% | 0,0% |

Cả ba gần như ngang ở nhóm yếu nhất, rồi LDT giữ ~100% ở mọi nhóm, còn
TRM và HRM **sụp từ 62–68% xuống 0,8–2,5% ngay ở nhóm kế tiếp**, rồi 0%
ở tất cả nhóm còn lại.

Đọc được: hai model recursive giải được những câu chỉ cần lan truyền
giá trị, nhưng **gần như không giải được câu nào cần thêm cắt biên**.
Vì nhãn là phương pháp yếu nhất, nên điểm gãy xảy ra đúng chỗ mà bước
suy luận phải mạnh lên một bậc — và LDT giải được chính bậc đó.

*Thận trọng:* `Val` + `ValBS` chiếm ~98% mẫu, các nhóm sau chỉ 2–105
mẫu nên không kết luận gì từ 1–2 mẫu.

---

## B5. Theo số gợi ý (test)

| Số gợi ý | n | LDT | TRM | HRM |
|---|---|---|---|---|
| ≤ 23 | 311 | **100,0%** | 0,0% | 0,0% |
| 24–26 | 4.527 | **99,9%** | 0,9% | 3,3% |
| 27–29 | 2.500 | **100,0%** | 27,9% | 37,8% |
| 30–32 | 1.929 | **100,0%** | 76,1% | 79,8% |
| 33–35 | 775 | **100,0%** | 92,4% | 91,7% |
| 36–38 | 179 | **100,0%** | 98,9% | 95,5% |
| ≥ 39 | 18 | **100,0%** | 94,4% | 94,4% |

Dễ đọc nhất trong bảng để dùng cho thuyết trình: **dưới 24 gợi ý thì
TRM và HRM bằng 0%, LDT bằng 100%**; trên 30 gợi ý thì cả ba gần như
ngang nhau.

HRM thắng TRM ở vùng 24–29 gợi ý (câu khó) và thua nhẹ ở vùng 30–38
(câu dễ), nên phần thắng của HRM đến từ vùng khó.

---

## B6. Tài nguyên và chi phí

| | LDT | TRM | HRM |
|---|---|---|---|
| Train time | 1.238s | 360s | 251s |
| Eval time | 31s | 243s | 103s |
| **Tổng wall clock** | **1.278s** | **603s** | **368s** |
| Peak VRAM | 26,29 GB | 31,93 GB | **11,62 GB** |
| GPU | B200 | B200 | B200 |

Nghịch lý đáng nói: **HRM lớn gấp 34 lần LDT nhưng nhanh hơn 3,5 lần
và nhẹ hơn 2,3 lần**. Lý do: LDT chạy solver 50 vòng tuần tự cho mỗi
câu, còn HRM học được nhóm câu dễ nhanh và dừng sớm.

---

## B7. Khoảng cách không phải do thiết lập chấm bài

TRM — tăng số vòng suy luận, 2.000 câu test con, không retrain:

| halt_max_steps | calls/câu | Vote | Base |
|---|---|---|---|
| 16 | 128 | 30,20% | 0/205 |
| 32 | 256 | 30,35% | 0/205 |
| 64 | 512 | 29,45% | 0/205 |
| 128 | 1.024 | **30,70%** | 1/205 |

LDT — tăng số nhánh, cùng tập câu:

| n_chains | Vote | Base | Timeout | eval_s |
|---|---|---|---|---|
| 1 | 99,85% | 203/205 | 3 | 5,2 |
| 4 | **100,00%** | 205/205 | 0 | 3,5 |
| 8 | 99,95% | 204/205 | 1 | 3,3 |

Cho TRM thêm 8 lượt suy luận thì độ đúng không đổi. LDT thì 1 nhánh
đã đạt 99,85%, và 4 nhánh đạt 100%. Hai điều này cùng loại trừ giả
thuyết "khoảng cách chỉ do cách chấm bài".

Chi tiết đáng kèm: thời gian eval của TRM **không tăng tuyến tính** khi
tăng `halt_max_steps`, vì đầu Q-halt vẫn dừng sớm — tức model *loop*
thay vì suy luận sâu hơn. Đây là bằng chứng cơ chế, mạnh hơn câu
"độ đúng không đổi".

---

## Phụ lục tuỳ chọn

Nếu thầy muốn thấy quá trình làm việc thì thêm một bảng kiểm kê các
run đã chạy, gồm cả run lỗi. Không thêm thì báo cáo gọn hơn.

| Run | steps × batch | Loss cuối | Test vote | Ghi chú |
|---|---|---|---|---|
| LDT | 4.000 × 512 | — | 99,94% | dùng trong báo cáo |
| TRM full4 | 2.667 × 768 | 0,4097 | 30,45% | dùng trong báo cáo |
| HRM full4 | 2.667 × 768 | 0,8852 | 34,50% | dùng trong báo cáo |
| TRM 1.333 bước | 1.333 × 768 | 0,5763 | 0,00% | điểm trên đường cong |
| TRM 667 bước | 667 × 768 | 0,6086 | 0,00% | điểm trên đường cong |
| TRM rút gọn | 4.000 × 512 | 0,0139 | 41,87% | loại — không phải method gốc |
| TRM full2 | 4.000 × 512 | **2,1978** | 0,00% | **lỗi carry-over** (ln 9 = 2,1978) |
| TRM smoke | 2 × 768 | 2,4586 | 0,00% | kiểm tra kỹ thuật |

Dùng phụ lục này thì nhớ nói kèm: run "rút gọn" cho 41,87% và 126/1.000
base nhưng **không phải TRM** (tự viết loss CE thường, không ACT
Q-loss, không EMA, không carry-over). Run "full2" có loss đúng bằng
ln(9) = 2,1978, dấu hiệu model chỉ học phân phối biên chứ không học
bài — đó là lỗi supervision trong carry-over, đã sửa.

---

## Câu hỏi thầy chắc hỏi, có sẵn câu trả lời

| Câu hỏi | Trả lời ngắn |
|---|---|
| Vì sao không so loss của LDT với TRM/HRM? | Ba hàm loss khác nhau (LDT dùng loss solver, TRM/HRM dùng stablemax CE + Q-halt). Chỉ so xu hướng, không so độ cao. |
| Một seed có đủ không? | Không đủ cho claim thống kê. Đây là điểm hạn chế đã ghi. |
| 34,50% có phải điểm cuối của HRM không? | Chưa chắc. Ở 1/2 ngân sách TRM đạt 0%, nên 2,05M có thể chưa đủ cho HRM hội tụ. HRM còn bị giới hạn bởi LR 7e-5 thấp hơn TRM và batch lớn hơn upstream. |
| Base 1.000 câu có quá nhỏ để kết luận? | Đúng, chỉ 1.000 câu nên không suy rộng. Nhưng khoảng cách 996 so với 0 và 6 quá lớn để là nhiễu. |
| So sánh công bằng không, batch khác nhau? | Cùng số examples. Batch 512 (LDT) và 768 (TRM/HRM) là do công thức gốc mỗi bên chỉ định; tổng examples mới là đại lượng so sánh, không phải batch. |
| HRM có phải chỉ do flash-attn bị thay bằng SDPA không? | Không. Stub cùng công thức softmax-attention, chỉ khác kernel. TRM không dùng flash-attn mà kết quả vẫn thấp hơn HRM. |
