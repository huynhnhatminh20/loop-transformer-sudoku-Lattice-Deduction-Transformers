# Baseline gộp ba model — LDT / TRM / HRM trên Scaled Sudoku

> Bảng để đưa vào báo cáo đồ án. Tất cả số **đo thật** trên GPU Modal
> B200, bf16, seed 0, cùng một file split đóng băng
> (`scaled_sudoku_split.jsonl`, seed 42).
>
> Nguồn: `eval.json` tải từ volume `lattice-diffusion-checkpoints`
> (`scaled/`, `scaled_trm/`, `scaled_hrm/`). Không có số nào chép tay.

---

## 1. Bảng chính — độ đúng tổng thể

Compute-matched: cả ba đều nhìn ~2,05 triệu examples.

| | LDT | TRM | HRM |
|---|---|---|---|
| Tham số | **799.635** | 5.027.842 | 27.274.242 |
| Steps × batch | 4.000 × 512 | 2.667 × 768 | 2.667 × 768 |
| Examples nhìn thấy | 2.048.000 | 2.048.256 | 2.048.256 |
| **Validation (10.174)** | **99,96%** | 31,02% | 34,72% |
| **Test (10.239)** | **99,94%** | 30,45% | **34,50%** |
| Greedy 1 đường | — | 17,86% | 18,14% |
| Câu trả lời sai (wrong) | **0** | — | — |

- Greedy = cách chấm của paper (1 vòng lặp suy luận, lấy argmax).
- Vote 8 = 1 bản gốc + 7 bản biến thể, giải riêng rồi bầu đa số,
  nhằm cân test-time với 8 chains của LDT.
- LDT `wrong = 0`: không câu nào trả lời sai; vài câu "hết giờ" là
  chưa giải xong trong 50 vòng.

## 2. Base (khó) vs Variant (dễ)

| Tập | | LDT | TRM | HRM |
|---|---|---|---|---|
| Validation | Base (gốc minimal) | **997/1.000** | 2/1.000 | 2/1.000 |
| | Variant | 9.173/9.174 | 3.154/9.174 | 3.530/9.174 |
| Test | Base (gốc minimal) | **996/1.000** | **0/1.000** | 6/1.000 |
| | Variant | 9.237/9.239 | 3.118/9.239 | 3.526/9.239 |

Đây là phân tách quan trọng nhất: LDT giải được puzzle gốc minimal
(996/1.000), hai model recursive gần như không giải được câu nào
(0 và 6). Khoảng cách giữa ba model **không** nằm ở variants.

## 3. Theo số gợi ý (test) — ba model dùng chung bucket

| Gợi ý | n | LDT | TRM | HRM |
|---|---|---|---|---|
| 21 | 9 | 100,0% | 0,0% | 0,0% |
| 22 | 99 | 100,0% | 0,0% | 0,0% |
| 23 | 825 | 100,0% | 0,0% | 0,0% |
| 24 | 2.997 | 99,8% | 0,0% | 0,8% |
| 25 | 5.472 | 99,9% | 0,5% | 1,8% |
| 26 | 5.112 | 99,9% | 2,0% | 6,3% |
| 27 | 3.033 | 100,0% | 8,1% | 16,6% |
| 28 | 2.157 | 100,0% | 27,8% | 40,6% |
| 29 | 2.310 | 100,0% | 53,9% | 62,9% |
| 30 | 2.223 | 100,0% | 65,9% | 75,2% |
| 31 | 1.944 | 100,0% | 79,8% | 83,6% |
| 32 | 1.620 | 100,0% | 85,7% | 81,7% |
| 33 | 1.161 | 100,0% | 91,2% | 90,2% |
| 34 | 768 | 100,0% | 92,6% | 91,8% |
| 35 | 396 | 100,0% | 95,5% | 96,2% |
| 36 | 282 | 100,0% | 97,9% | 94,7% |
| 37 | 183 | 100,0% | 100,0% | 96,7% |
| 38–43 | 126 | 100,0% | 96,8% | 95,2% |

Đọc bảng:

- Dưới 24 gợi ý: LDT 100%, TRM và HRM **0%**.
- Trên 30 gợi ý: cả ba gần như ngang nhau (~80–100%).
- **HRM thắng TRM ở vùng 24–30 gợi ý** (vùng khó), nhưng **thua nhẹ ở
  vùng 31–38** (vùng dễ). Nói cách khác HRM mạnh hơn ở câu khó, yếu
  hơn một chút ở đuôi dễ — nên thắng tổng là do vùng khó.

## 4. Theo độ khó propagation (test) — bảng sắc nhất

Nhãn `hardness_level` của dataset là **phương pháp suy luận yếu nhất
giải được câu đó không cần tìm kiếm** (nguồn:
`github.com/zayenz/scaled-sudoku-instances`). Thành phần tên nhóm:
`Val`/`Bnd`/`Dom` = mô hình lan truyền giá trị / biên / miền; `BS` =
+ bounds shaving; `DS` = + domain shaving; `Chan` = + channeling. Nhóm
yếu hơn = suy luận đơn giản hơn.

Vì nhãn là phương pháp *yếu nhất*, câu gắn `ValBS` là câu mà lan truyền
giá trị đơn thuần **không** giải nổi.

| Nhóm (yếu → mạnh) | n | LDT | TRM | HRM |
|---|---|---|---|---|
| Val | 5.000 | **100,0%** | 61,5% | 68,1% |
| ValBS | 5.004 | **99,9%** | 0,8% | 2,5% |
| ValDS | 100 | **99,0%** | 1,0% | 1,0% |
| ValChan | 22 | **100,0%** | 0,0% | 0,0% |
| Bnd | 2 | **100,0%** | 0,0% | 0,0% |
| BndBS | 105 | **99,0%** | 0,0% | 0,0% |
| Dom | 2 | **100,0%** | 0,0% | 50,0% |
| DomBS | 4 | **100,0%** | 0,0% | 0,0% |

Đây là phát hiện đáng ghi nhất trong toàn bộ so sánh:

- LDT **~100% ở mọi nhóm**, kể cả nhóm khó nhất.
- TRM và HRM giải được nhóm `Val` (lan truyền giá trị thuần) ở mức
  62–68%, nhưng **sụp xuống 0,8–2,5% ngay khi cần `ValBS`**, và 0% ở
  tất cả nhóm còn lại.

Nghĩa là: cả hai model recursive đều học được bước lan truyền giá trị,
nhưng **không học được bước kết hợp** — không lập được chuỗi lan truyền
nhiều tầng cần thiết cho nhóm khó. Và LDT giải được chính bước kết hợp
đó.

Lưu ý khi trích bảng này: nhóm `Val` + `ValBS` chiếm ~98% mẫu, nên
vài nhóm chỉ có 2–105 mẫu — **không kết luận gì từ 1–2 mẫu**.

## 5. Theo bước đi trong "hành trình" biến thể (LDT)

| Vị trí | Test |
|---|---|
| Base (gốc) | 996/1.000 |
| Walk 1 | 1.880/1.880 |
| Walk 2 | 1.863/1.863 |
| Walk 3 | 1.852/1.853 |
| Walk 4 | 1.836/1.836 |
| Walk 5 | 1.806/1.807 |

Độ đúng tăng dần từ Base sang các bước sau — đúng thiết kế dataset
(càng đi xa càng được thêm gợi ý). Chỉ LDT có bảng này vì TRM/HRM
trượt gần hết nên phân tích không có ý nghĩa.

## 6. Quá trình học

| Model | Loss đầu | Loss cuối | Ghi chú |
|---|---|---|---|
| LDT | 0,81 (step 100) | **0,060** (step 4000) | giảm đều, không NaN/Inf |
| TRM | 2,48 | **0,41** | xuống dưới marginal 2,197 |
| HRM | 2,43 | **0,89** | xuống dưới marginal từ step ~100 |

**Không so loss của LDT với TRM/HRM** — ba hàm loss khác nhau (LDT
dùng loss của solver, TRM/HRM dùng stablemax cross-entropy + Q-halt).
Chỉ so được *xu hướng giảm* trong từng cột.

Chi tiết đáng ghi ở HRM: loss **tăng lại** từ 0,58 (step 1500) lên
0,97 (step 2200), rồi đi ngang ~0,82–0,89. Có hỏng ngắn, hồi phục
một phần nhưng không về đáy.

## 7. Tài nguyên và chi phí

| | LDT | TRM | HRM |
|---|---|---|---|
| Train time | 1.238s | 360s | 251s |
| Eval time | 31s | 243s | 103s |
| **Tổng wall clock** | **1.278s** | **603s** | **368s** |
| Peak VRAM | 26,29 GB | 31,93 GB | 11,62 GB |
| Loss cuối | — | 0,41 | 0,89 |

Nghịch lý đáng nói: **HRM lớn gấp 34 lần LDT nhưng lại nhanh hơn 3,5
lần và nhẹ hơn 2,3 lần**. Lý do: LDT chạy solver 50 vòng tuần tự mỗi
câu, còn HRM học được câu dễ nhanh và halt sớm.

## 8. Hai bảng cần kèm khi trích bảng chính

Bảng 1 trên là số liệu quan sát được ở **ngân sách compute-matched** — cùng
split, cùng 2,05M examples, cùng ngân sách eval. Nó **không** phải bảng
xếp hạng method. Hai mục dưới là lý do.

### 8.1. Ngân sách 2,05M examples chưa đủ cho TRM/HRM hội tụ (test)

| Steps | Examples | Loss | Test vote | Base |
|---|---|---|---|---|
| 667 | 512.256 | 0,6086 | **0,00%** | 0/1.000 |
| 1.333 | 1.023.744 | 0,5763 | **0,00%** | 0/1.000 |
| 2.667 | 2.048.256 | 0,4097 | 30,45% | 0/1.000 |

Ở **một nửa** ngân sách compute-matched, TRM giải được 0 trên 10.239
câu, dù loss đã giảm (model không bị kẹt, chỉ là chưa giải trọn câu
nào). Đường cong là **vực thẳm**, không dốc đều: toàn bộ 30,45% xuất
hiện trong khoảng 1,02M–2,05M examples.

Hệ quả: điểm 2.667 bước nằm **ngay sau bậc nhảy**. Nên câu hỏi phụ mới
là **"TRM/HRM cần bao nhiêu examples để hội tụ"** — và đó là nghiên cứu
mới, chưa làm ở đây. Nó **không** bác bỏ bảng 1: khoảng cách 996/1.000 so
với 0/1.000 ở ngân sách này là thật, chỉ là chưa đủ để xếp hạng method.

### 8.2. Khoảng cách không do thiết lập chấm (test, 2.000 câu con)

Tăng số vòng suy luận của TRM lên gấp 8 lần:

| halt_max_steps | calls/câu | Vote 8 views | Base |
|---|---|---|---|
| 16 | 128 | 30,20% | 0/205 |
| 32 | 256 | 30,35% | 0/205 |
| 64 | 512 | 29,45% | 0/205 |
| 128 | 1.024 | **30,70%** | 1/205 |

Tăng suy luận 8× **không** cải thiện (biến động ±0,8 điểm, trong
nhiễu). Nên khoảng cách LDT–TRM không phải artifact chấm bài.

Trong khi đó LDT ở ngân sách *nhỏ hơn nhiều* vẫn gần 100%:

| n_chains | max_rounds | Vote | Base |
|---|---|---|---|
| 1 | 50 | 99,85% | 203/205 |
| 4 | 50 | 100,00% | 205/205 |

**LDT đạt 99,85% chỉ với 1 nhánh** — không cần vote 8 lần. Bất đối
xứng này mạnh hơn con số 99,94%.

Hạn chế phải disclose: `halt_max_steps = 128` nằm ngoài phạm vi huấn
luyện (ACT head học dừng trong 16 bước), nên kết luận đúng ở mức
*"ở ngân sách mà paper cho phép, TRM đứng yên"*, không phải *"TRM không
hưởng lợi từ suy luân sâu"* một cách tuyệt đối.

## 9. Cách phát biểu kết luận cho an toàn

| Nên nói | Không nên nói |
|---|---|
| Ở ~2,05M examples, LDT đạt ~100% còn TRM 34,5% và HRM 30,5%. | LDT data-efficient hơn TRM/HRM 3 lần. |
| LDT giải được puzzle gốc minimal (996/1.000), hai model kia gần như không (0 và 6). | LDT bị vượt về kiến trúc. |
| Khoảng cách không do thiết lập chấm (đã đo sweep 8×). | Khoảng cách là do LDT có nhiều lượt suy luận hơn. |
| Ở 1/2 ngân sách, TRM đạt 0% → 2,05M chưa phải ngân sách hội tụ. | 30,45% là độ chính xác cuối cùng của TRM. |
| HRM hơn TRM 4 điểm, nhưng nhận LR thấp hơn và batch lớn hơn upstream. | HRM bản chất vượt trội TRM. |

## 10. Hạn chế của toàn bộ so sánh

1. **Một seed duy nhất** (seed 0). Không có ước lượng sai số.
2. **Compute-matched theo examples, không theo FLOPs.** HRM 27,3M
   tham số / ~512 layer-pass mỗi bước suy luận; TRM 5,0M / 36; LDT là
   solver 50 vòng. Số examples bằng nhau không phải FLOPs bằng nhau.
   Đã kèm `train_s` và `peak_gb` ở bảng 7 để đọc được.
3. **Per-puzzle embedding 82k hàng**, mỗi puzzle gặp ~25 lần. Paper
   gốc chỉ có 1.000 puzzle nên bảng này học kỹ hơn nhiều — đây là điểm
   TRM/HRM bị thiệt ở quy mô data lớn.
4. **Hai LR khác nhau theo upstream** (TRM 1e-4, HRM 7e-5) và HRM
   dùng batch 768 gấp đôi lệnh sudoku gốc của họ (384). Giữ nguyên
   giá trị gốc là đúng nguyên tắc, nhưng nghĩa là không ai được tune
   lại cho run 2.667 bước.
5. **LDT dùng solver, không dùng model** ở phần "50 vòng" — hai hệ
   thống khác nhau về bản chất, đây là so sánh *hệ thống*, không thuần
   là so sánh *kiến trúc mạng*.

---

## 11. Kiểm kê toàn bộ run đã chạy (8 run)

Bảng này để trả lời "hôm qua chạy mấy bảng". Tất cả nằm trên volume
`lattice-diffusion-checkpoints`.

| Run | steps × batch | Examples | Loss cuối | Test vote | Base test | Trạng thái |
|---|---|---|---|---|---|---|
| LDT | 4.000 × 512 | 2.048.000 | — | **99,94%** | 996/1.000 | ✅ vào báo cáo |
| TRM full4 | 2.667 × 768 | 2.048.256 | 0,4097 | **30,45%** | 0/1.000 | ✅ vào báo cáo |
| HRM full4 | 2.667 × 768 | 2.048.256 | 0,8852 | **34,50%** | 6/1.000 | ✅ vào báo cáo |
| TRM curve 2 | 1.333 × 768 | 1.023.744 | 0,5763 | 0,00% | 0/1.000 | ✅ vào bảng 8.1 |
| TRM curve 1 | 667 × 768 | 512.256 | 0,6086 | 0,00% | 0/1.000 | ✅ vào bảng 8.1 |
| TRM rút gọn | 4.000 × 512 | 2.048.000 | 0,0139 | 41,87% | 126/1.000 | ⚠️ **loại** — xem 11.1 |
| TRM full2 | 4.000 × 512 | 2.048.000 | **2,1978** | 0,00% | 0/1.000 | ❌ **bug** — xem 11.2 |
| TRM smoke | 2 × 768 | 1.536 | 2,4586 | 0,00% | — | smoke kỹ thuật |

Ba điều đáng chú ý từ bảng này:

1. **Ta có run TRM đúng batch và đúng steps của LDT** (4.000 × 512 =
   2.048.000 examples, khớp tuyệt đối). Nhưng cả hai run ở cấu hình đó
   đều không dùng được: một cái là bug, một cái là công thức rút gọn.
   Nên bảng chính dùng 2.667 × 768 cho cả TRM và HRM.
2. **Bản rút gọn (41,87%) còn cao hơn bản đầy đủ (30,45%)** ở cùng
   ngân sách examples. Không dùng làm baseline vì không phải method
   gốc, nhưng là dữ liệu thật và có thể trích kèm caveat.
3. **Loss cuối 2,1978 = ln(9) chính xác** là dấu hiệu model chỉ học
   được phân phối biên của các con số, chứ không học được bài. Đây là
   bằng chứng máy cho lỗi carry-over.

### 11.1. Vì sao loại bản rút gọn (41,87%)

Bản này tự viết loss CE thường + carry mới mỗi step, không có ACT
Q-loss, không EMA, không carry-over. Tức là **không phải TRM** — là một
biến thể tự viết. Trích số nó rồi so với LDT sẽ là so sai phương pháp.

**Code của bản rút gọn đã bị xóa khỏi repo** (06/10): `experiments/scaled_trm_t4_smoke.py`
và `experiments/scaled_hrm_t4_smoke.py`, vì chúng dùng sai kiến trúc paper
và dễ bị nhầm là baseline hợp lệ. Bản đầy đủ đúng paper nằm ở
`experiments/scaled_trm/run.py` và `experiments/scaled_hrm/run.py` — hai
file này sinh ra 30,45% và 34,50% trong báo cáo. Con số 41,87% giữ lại
trong báo cáo chỉ như **log lịch sử**, không dùng làm căn cứ.

Ngoài ra bản này cho 126/1.000 base, tức **cao hơn cả LDT một bậc** ở
phần khó nhất. Đây là manh mối đáng điều tra: nếu ai đó chạy lại và
được 41,87% mà không biết bản full không được vì sao, rất dễ kết luận
sai rằng TRM chỉ kém LDT ở chỗ khác.

### 11.2. Vì sao full2 cho 0,00% (loss = 2,1978 = ln 9)

Đây là lần chạy cháy tiền: supervision đọc `labels` của batch mới cho
*cả* các ô đang mang trạng thái puzzle cũ, nên model nhận mâu thuẫn
"câu này vừa là câu A vừa là câu B". Chi phí ~$4-7 B200 (27 phút).

Đã sửa: đưa `labels` vào trong batch dict và đọc supervision từ
`carry.current_data["labels"]`.

HRM không thể mắc lỗi này vì `losses.py:58` và
`hrm_act_v1.py:246` đã làm đúng ngay trong code gốc.

## 12. Các đường cong có sẵn để vẽ

### 12.1. Accuracy theo ngân sách (đường cong quan trọng nhất)

| Examples | LDT | TRM | HRM |
|---|---|---|---|
| 512.256 | — | **0,00%** | — |
| 1.023.744 | — | **0,00%** | — |
| 2.048.000 | **99,94%** | — | — |
| 2.048.256 | — | **30,45%** | **34,50%** |

Hình dạng của TRM là **vực thẳm** rồi nhảy, không phải dốc đều:

```
0,00% ──┐
        │                        ┌─ 30,45%
0,00% ─┤                        │
        └────────────┬───────────┘
    512K    1,02M   2,05M  examples
```

Điểm 2,05M nằm **ngay sau bậc nhảy**, nên chưa đủ để xếp hạng method.
Đây là điểm cần nói rõ trong đồ án — nhưng nó thuộc phần *giới hạn phát
biểu*, không phủ định bảng số liệu.

*LDT và HRM mới chỉ có một điểm mỗi bên — chưa đủ vẽ đường cong.*

### 12.2. Loss theo step

LDT (batch 512, khác định nghĩa loss):

| Step | 100 | 500 | 1.000 | 2.000 | 3.000 | 4.000 |
|---|---|---|---|---|---|---|
| Loss | 0,81 | 0,26 | 0,14 | 0,10 | 0,069 | 0,060 |

HRM (batch 768, stablemax + Q-halt) — **đường cong đầy đủ 27 điểm**:

| Step | Loss | Step | Loss | Step | Loss |
|---|---|---|---|---|---|
| 1 | 2,4269 | 1.000 | 0,9248 | 1.900 | 0,7197 |
| 100 | 2,2018 | 1.100 | 0,8248 | 2.000 | 0,8180 |
| 200 | 1,6639 | 1.200 | 0,7576 | 2.100 | 0,7242 |
| 300 | 1,3880 | 1.300 | 0,7056 | 2.200 | 0,9690 |
| 400 | 1,2834 | 1.400 | 0,6496 | 2.300 | 0,8724 |
| 500 | 1,2717 | 1.500 | 0,5819 | 2.400 | 0,8877 |
| 600 | 1,2158 | 1.600 | 0,6316 | 2.500 | 0,8067 |
| 700 | 1,1760 | 1.700 | 0,6937 | 2.600 | 0,8162 |
| 800 | 1,0952 | 1.800 | 0,7075 | 2.667 | 0,8852 |
| 900 | 1,0124 | | | | |

TRM (batch 768) — chỉ 5 điểm, không lấy được log per-step vì app đã bị
xoá khỏi danh sách:

| Step | Loss |
|---|---|
| 1 | 2,48 |
| 667 | 0,6086 |
| 1.333 | 0,5763 |
| 2.667 | 0,4097 |

Hai điều đáng trích khi vẽ:

- **HRM có hỏng ngắn.** Loss giảm đều tới 0,5819 ở step 1500 rồi
  **tăng lên** 0,9690 ở step 2200, sau đó đi ngang ~0,82-0,89, không
  về lại đáy. Không có LDT hay TRM nào bị hiện tượng này.
- **Loss của LDT không so được với TRM/HRM** (khác hàm loss). Chỉ so
  hình dạng, không so độ cao.

### 12.3. Suy luận lúc eval: TRM điều chỉnh `halt_max_steps`

2.000 câu test con, seed 12345, không retrain:

| halt_max_steps | calls/câu | Vote | Greedy | Base | eval_s |
|---|---|---|---|---|---|
| 16 | 128 | 30,20% | 17,55% | 0/205 | 96,3 |
| 32 | 256 | 30,35% | 17,75% | 0/205 | 87,0 |
| 64 | 512 | 29,45% | 17,75% | 0/205 | 116,8 |
| 128 | 1.024 | 30,70% | 17,75% | 1/205 | — |

Phẳng. Tăng suy luận 8× không giúp. Nhưng để ý `eval_s` **không tăng
tuyến tính** (96s → 87s → 117s) — vì ACT head học dừng sớm ngay cả khi
cho phép nhiều bước hơn, tức model **loop** chứ không suy luận sâu hơn.

### 12.4. Số nhánh lúc eval: LDT điều chỉnh `n_chains`

2.000 câu test con, seed 12345, không retrain:

| n_chains | calls/câu* | Vote | Base | Timeout | eval_s |
|---|---|---|---|---|---|
| 1 | 0,025 | 99,85% | 203/205 | 3 | 5,2 |
| 2 | 0,0295 | 99,95% | 205/205 | 1 | 3,9 |
| 4 | 0,027 | **100,00%** | 205/205 | 0 | 3,5 |
| 8 | 0,0255 | 99,95% | 204/205 | 1 | 3,3 |

\* `calls_per_puzzle` của LDT không dùng để vẽ được: solver dừng sớm
khi giải xong nên cột này ra ~0. Khi so ngân sách phải dùng **ngân
sách cấp** (K × 50 vòng).

Phát hiện: **1 nhánh đã đạt 99,85%**, 4 nhánh đạt 100%. Và eval chỉ
mất 3-5 giây cho 2.000 câu — nhanh hơn TRM **~30 lần** dù LDT cho ra
kết quả tốt hơn nhiều.

### 12.5. Đường cong nào còn thiếu

| Đường cong | Hiện có | Thiếu gì để đủ |
|---|---|---|
| Accuracy vs ngân sách | TRM 3 điểm, LDT 1, HRM 1 | **điểm 1.333 bước cho HRM** (~3 phút, ~$0,4) |
| Loss vs step | HRM 27, LDT 6, TRM 5 | log per-step của TRM |
| Accuracy vs suy luận | TRM 4, LDT 4 | điểm cho HRM |

Chỉ cần chạy **HRM ở 1.333 bước** là có ngay điểm thứ hai cho đường
cong quan trọng nhất, và biết HRM có nằm trên bậc nhảy như TRM hay
không.


## 13. Đề xuất hướng cải thiện

> Mục này là **đề xuất**, dựa trên số đo đã có ở §2–§4. **Chưa kiểm
> chứng bằng thực nghiệm** và nằm ngoài ngân sách của phiên này.

### 13.1. Chẩn đoán: các model recursive hỏng ở *bước kết hợp*, không
phải ở *bước đơn*

Bảng §4 cho thấy TRM và HRM **không hỏng đồng loạt**, mà hỏng theo ranh
giới rất sắc:

| Phương pháp suy luận cần thiết | TRM | HRM |
|---|---|---|
| Lan truyền giá trị thuần (`Val`) | 61,5% | 68,1% |
| + bounds shaving (`ValBS`) | **0,8%** | **2,5%** |
| + domain shaving, channeling | 0% | 0% |

Trong khi LDT giữ 99,9–100% ở **mọi** nhóm.

*(Riêng nhóm `Dom` để HRM 50,0%, nhưng n = 2 mẫu — không kết luận được
gì từ đó, xem ghi chú cuối §4.)*

Cách đọc: cả hai model đã học được một bước lan truyền giá trị đơn
thuần, nhưng **không học được bước kế tiếp** — bước phải dùng thêm
thông tin biên/domain. Hệ quả là chúng không lập được chuỗi lan truyền
nhiều tầng, và chỉ giải được câu nào chỉ cần một tầng.

§3 và §4 khớp nhau: độ đúng tăng đơn điệu theo số gợi ý, và sụp gần
như thẳng về 0% ở nhóm đòi `ValBS`.

### 13.2. Nguyên nhân có khả năng cao

Cơ chế halving của TRM/HRM **loại ứng viên bằng phép chia đôi**. Thao
tác đó không dùng thông tin biên — nó chỉ hỏi "nhánh nào còn khả năng".
Trong không gian ứng viên, một tập lớn bị loại bởi điều kiện biên sẽ bị
chia đôi theo cách không phản ánh điều kiện đó, nên thông tin biên bị mất
sau một bước.

Giả thuyết tương ứng về LDT: nó giải bằng **loại trừ ứng viên** (mô tả
trong `reports/bao-cao.md`), tức mỗi bước áp dụng điều kiện lên tập ứng
viên thay vì loại bằng phép chia đôi. Nếu vậy thì khác biệt với TRM/HRM
nằm ở *cách biểu diễn trạng thái qua các bước*, chứ không chỉ ở *cách
chọn bước* — và đó mới là thứ quyết định việc chuỗi lan truyền nhiều tầng
có lập được hay không.

> **Cần kiểm chứng trước khi trích dẫn:** suy đoán này về kiến trúc LDT
> mới chỉ dựa trên mô tả hành vi trong báo cáo, **chưa đọc code**. Nếu
> không đọc `src/lattice_diffusion/` thì phải bỏ đoạn này, hoặc hạ
> xuống mức "một hướng giải thích khả dĩ" không khẳng định.

Hai giải thích cạnh tranh, chưa phân biệt được bằng dữ liệu hiện có:

1. **Biểu diễn trạng thái** — halving vô tình xoá thông tin biên (ở trên).
2. **Tín hiệu huấn luyện** — ACT chỉ nhận thưởng khi giải đúng; câu thuộc
   nhóm `ValBS` gần như không bao giờ được giải đúng trong quá trình
   huấn luyện, nên không có tín hiệu để học nhóm đó. Mỗi câu chỉ được
   gặp ~25 lần ở tập 82k, không đủ lặp lại một kỹ năng khó.

### 13.3. Đề xuất

**Giữ nguyên ràng buộc biên ở mỗi bước.** Thay bước "chia đôi rồi đoán"
bằng một primitive biến đổi tập ứng viên *theo điều kiện biên*, để một
lần bước có thể loại được tầng lan truyền thứ hai thay vì chỉ tầng thứ
nhất. Cụ thể hơn, ba hướng cụ thể hoá:

| Hướng | Nội dung | Bám theo |
|---|---|---|
| A | Thay halving bằng nhánh ứng viên có điều kiện biên chặt nhất | §4: nhóm `Bnd*`, `Dom*` là nơi hai model gần như 0% |
| B | Cho state mang thêm vector ứng viên *bị loại vì lý do*, không chỉ tập còn lại | §4: đây là thứ làm mất thông tin biên |
| C | Thay per-puzzle embedding bằng encoder dùng chung theo cấu trúc lưới | §13.2.2: mỗi câu chỉ gặp ~25 lần ở tập 82k |

Hướng C không liên quan đến bước kết hợp, nhưng cùng giải thích được
tại sao các model recursive suy giảm khi dữ liệu lớn lên, khi paper
gốc chỉ dùng 1.000 puzzle.

### 13.4. Vì sao tin rằng hướng A khả thi

LDT đạt 99,9% ở `ValBS` trong khi vẫn giữ 100% ở `Val`. Nghĩa là thao
tác xử lý biên **không phải thứ gì đắt đỏ** — nó rẻ hơn đáng kể so với
việc phải lập chuỗi lan truyền sâu hơn, và nó không làm hỏng nhóm dễ.
Thêm nữa, eval LDT chỉ mất 3–5 giây cho 2.000 câu (§12.4), nên một
primitive biên tương đương không phải là nút thắt về tính toán.

### 13.5. Công trình liên quan

* **RSM** — arXiv 2603.15641, cùng họ với TRM, báo cáo 97,5% trên
  Sudoku-Extreme.
* **FPRM** — arXiv 2606.18206, dùng chung harness với TRM.

Hai hướng này giải quyết cùng bài toán (recursive reasoning ở quy mô
lớn) theo cách khác. Chúng **chưa được đánh giá trong đồ án này** —
không phải vì không quan trọng, mà vì ngân sách không cho phép chạy
thêm ở quy mô 82k câu.

### 13.6. Trạng thái

Mục này **không kèo kết quả thực nghiệm**. Đề xuất ở §13.3 được rút ra
từ số đo ở §2–§4, chưa được kiểm chứng bằng một run nào. Khi trích
dẫn, phải ghi kèm giới hạn này.

---

## 14. Kế hoạch chạy Sudoku-Extreme cho TRM và HRM

> Đây là **kế hoạch**, chưa chạy. Viết ra trước để kết quả sau này có
> chỗ đứng và để chứng minh việc bảo vệ thành quả đã có sẵn.

### 14.1. Vì sao cần chạy thêm phần này

Cho tới nay TRM và HRM mới chỉ có kết quả trên scaled sudoku. Phần
Sudoku-Extreme chỉ có LDT. Đề tài nhắm tới ba kiến trúc, nên thiếu hai
dòng đó. Sudoku-Extreme **không** bão hoà: 4.000 puzzle, paper báo cáo
TRM khoảng 86,8% và HRM khoảng 80%, trong khi LDT đo được 99,88%. Khoảng
cách 13–20 điểm là có thật và đáng tái lập.

### 14.2. Tái lập thuần: chỉ viết một file chạy Modal

Cần phân biệt hai việc khác nhau, cho hai phát biểu khác nhau:

| | Tái lập | So sánh công bằng |
|---|---|---|
| Cách chạy | `upstream/pretrain.py` **nguyên bản** + config gốc | harness tự viết, khóa ngân sách theo examples |
| Đối chiếu với | con số trong paper gốc (~86,8% cho TRM) | LDT ở cùng ngân sách (§1) |
| Yêu cầu | **không sửa gì hết** | được phép khai báo sai lệch |

Phần tái lập chạy **upstream nguyên bản**. Việc duy nhất phải viết là một
file khởi chạy Modal: đóng gói image, gắn volume, đặt `PYTHONPATH`, rồi gọi
`pretrain.py`. Không viết adapter, không sửa model, không đụng `upstream/`.

Đây là ranh giới quan trọng: pipeline `experiments/scaled_trm/run.py` hiện có
là **harness tự viết**, nó import từ upstream nhưng tự quyết định steps, batch,
warmup và augmentation. Chạy Extreme bằng harness đó thì **không phải tái lập
thuần** — đó là so sánh compute-matched, khác câu hỏi.

### 14.3. Bằng chứng upstream chưa bị sửa

| Repo | So hash SHA-256 với GitHub | Commit |
|---|---|---|
| `upstream/TinyRecursiveModels/` (40 file) | **40 / 40 khớp, 0 khác biệt** | `c01103738605` (2026-03-31) |
| `upstream/HRM/` (26 file) | **26 / 26 khớp, 0 khác biệt** | `ac15626f8db0` (2026-04-01) |

Không dùng `git diff -- upstream/` làm bằng chứng: thư mục đó không được git
track nên lệnh luôn rỗng, dù đã sửa hay chưa. Cách đúng là clone bản gốc và
so hash từng file.

### 14.4. Nguyên tắc cô lập

Mọi run extreme nằm trên đường dẫn riêng, không giao với pipeline scaled:

| Lớp | Extreme (mới) | Scaled (đã có, không đụng) |
|---|---|---|
| Volume Modal | `extreme-checkpoints` | `lattice-diffusion-checkpoints` |
| Script | `experiments/extreme/` | `experiments/scaled_trm/` · `scaled_hrm/` |
| Đường dẫn checkpoint | `/checkpoints/extreme_trm/` · `extreme_hrm/` | `/checkpoints/scaled_trm/` · `scaled_hrm/` |
| Dataset | `data/sudoku-extreme-4000/` | `scaled_sudoku_split.jsonl` |
| Báo cáo | `reports/EXTREME.md` (tạo mới) | `TRM.md` · `HRM.md` (không sửa) |

Dùng chung repo, **không** clone riêng. Clone ra repo mới sẽ phải làm lại
dataset 2,5 GB, Modal image, `sys.path` — trong khi rủi ro xoá mất kết quả đến
từ việc ghi đè cùng một đường dẫn checkpoint, đã chặn được bằng đổi volume.

### 14.5. Thứ upstream đã có sẵn

Không phải viết mới:

| Cần | Có sẵn ở |
|---|---|
| Dựng bộ Sudoku-Extreme | `upstream/*/dataset/build_sudoku_dataset.py` (`source_repo = sapientinc/sudoku-extreme`) |
| Script train | `upstream/*/pretrain.py`, `config/cfg_pretrain.yaml`, `config/arch/*.yaml` |
| ACT Q-loss, carry-over, halving | `upstream/*/models/losses.py`, `models/recursive_reasoning/` |
| EMA (chỉ TRM có) | `upstream/*/models/ema.py` |

Hai thứ cần bổ sung vì B200 không có:

- `adam-atan2` cần nvcc. Dùng lại `experiments/scaled_trm/adam_atan2_torch.py`
  ở chế độ **chỉ đọc**, đã verify khớp float64 tới 2e-7.
- HRM cần `flash_attn`, Blackwell không có wheel. Chép đoạn SDPA stub sẵn có
  sang file mới, không import từ `scaled_hrm/run.py` để không tạo ràng buộc
  giữa hai pipeline.

### 14.6. Ràng buộc ngân sách

`upstream/TinyRecursiveModels/README.md:26` ghi recipe Sudoku-Extreme nhỏ nhất
cần **18 giờ trên một L40S** — tức mức ngân sách đã xác định là không khả thi
với ngân sách còn lại. Nên phải chọn một trong hai:

1. Chạy đúng `epochs` của paper → tốn vượt ngân sách.
2. Cắt `epochs`, giữ nguyên phần còn lại → phải ghi rõ là cắt, và kết luận
   phải gắn nhãn "dưới ngân sách của paper".

Không được trình bày phương án 2 như phương án 1.

### 14.7. Bẫy đã biết từ những lần chạy trước

| # | Bẫy | Hệ quả nếu lặp lại | Cách chặn |
|---|---|---|---|
| 1 | Thiếu `mkdir` trước khi lưu checkpoint trung gian | Chết ở `step % 1000 == 0` — đã xảy ra một lần với HRM | Có `mkdir`; smoke chạy **đúng số bước của run thật** |
| 2 | Chạy nhầm `scaled_hrm/run.py` | Ghi đè kết quả scaled | In đường dẫn ra log và xác nhận chứa `extreme` |
| 3 | `PYTHONPATH` thiếu `src` | Không import được `lattice_diffusion` | Thiết lập cả `src` và `upstream/<model>` |
| 4 | Ước lượng thời gian eval thay vì đọc log | Ngân sách bị chệch — đã ước 4–5 phút, thực tế 103 giây | Đo `eval_s` rồi mới nhân với rate |

### 14.8. Trình tự

1. Dựng `data/sudoku-extreme-4000/` bằng script gốc.
2. Viết file khởi chạy Modal cho TRM; smoke T4 với đúng số bước thật.
3. Viết file khởi chạy Modal cho HRM; smoke T4 có SDPA stub.
4. Chạy thật TRM.
5. Chạy thật HRM, **nếu còn ngân sách**.

Bỏ bước 5 nếu hết tiền. Chạy TRM trước vì đó là dòng có số để đối chiếu với
paper; HRM đã có số ở §1 rồi.

### 14.9. Kiểm chứng cô lập

Trước khi chạy, lưu bản ký hash của `experiments/scaled_trm/`,
`experiments/scaled_hrm/`, `upstream/` và mọi file `.eval.json` trên volume
scaled. Sau khi chạy xong hash lại và so byte. Khác một byte nghĩa là có gì
đã bị đụng.

---
