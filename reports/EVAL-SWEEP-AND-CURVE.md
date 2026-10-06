# TRM vs LDT — Test-time compute sweep + learning curve

> Ngày: 05/10/2026. Bổ sung cho `reports/TRM.md`.
> Repo: `C:\Users\ASUS\lattice-deduction-transformers` (bản chạy thật).
> Checkpoint: volume `lattice-diffusion-checkpoints`.
> Script: `experiments/scaled_trm/eval_sweep.py`, `experiments/scaled_trm/learning_curve.py`.

---

## 0. Tóm tắt

Hai câu hỏi cũ đã được trả lời bằng thực nghiệm, **không cần suy đoán**:

| Câu hỏi | Kết luận |
|---|---|
| Chênh lệch LDT 99,94% vs TRM 30,45% có phải do **test-time compute lệch** (50 vòng vs 16 bước) không? | **Không.** Tăng suy luận TRM gấp 8× không cải thiện. |
| 2.667 bước (2,05M examples) có đủ không? | **Không.** Ở **một nửa** ngân sách (1,02M) TRM đạt **0,00%** trên 10.239 câu. Đường cong là vực thẳm, không phải dốc đều. |

### Phát hiển quan trọng nhất

> **Con số 30,45% không cho phép suy ra thứ hạng method hay data efficiency.** Nó nói rằng ở 2,05M examples LDT đã hội tụ còn TRM thì chưa. Toàn bộ 30,45% xuất hiện trong khoảng 1,02M–2,05M examples.

**Phạm vi đúng của phát biểu.** Đây **là** một so sánh compute-matched hợp lệ — cùng split đóng băng, cùng ngân sách 2.048.256 examples, đúng như nguyên tắc đã chốt. Khoảng cách quan sát ở điểm này là **thật**, và §1 đã loại trừ giải thích thay thế rẻ nhất cho nó (thiếu lượt suy luận). Cái **không** làm được là **xếp hạng method**: vì TRM chưa hội tụ tại điểm này, ta không biết nó sẽ dừng ở đâu. Đây là giới hạn của phép suy luận, không phải lý do bác bỏ phép đo.

Hai quyết định phương pháp từ đây:
1. Báo cáo ghi rõ 2,05M là **ngân sách compute-matched mà TRM chưa hội tụ**: khoảng cách đo được là thật, nhưng là khoảng cách *dưới ngân sách*, không phải thứ hạng method.
2. Muốn kết luận về data efficiency thì phải tìm mức ngân sách mà TRM hội tụ — hướng nghiên cứu mới, **không** nằm trong phạm vi việc hiện tại, và không cần điều đó để bảo vệ kết luận chính ở bảng §0.

### Hai confound/rủi ro đã xử lý

| Vấn đề | Kết quả |
|---|---|
| Test-time compute lệch | **Đã loại bỏ** — đo cả đường cong, gap không nằm ở đây |
| 2.667 bước có đủ không | **Đã đo** — không đủ, và đây là phát hiển, không phải sự cố |

---

## 1. Confound: test-time compute

### Vấn đề

Run gốc (reports/TRM.md) so:

| | Số nhánh | Độ sâu suy luận |
|---|---|---|
| LDT | 8 | 50 vòng solver |
| TRM | 8 | 16 bước halving |

Cùng số nhánh, nhưng độ sâu lệch. Không tách được "method kém" khỏi "cho ít lượt".

### Cách sửa

Không so một điểm — **đo cả đường cong**, đưa về cùng đơn vị: *số lần gọi model mỗi câu*.

Điểm kiểm tra: file `upstream/TinyRecursiveModels/models/recursive_reasoning/trm.py:275` cho thấy ACT chỉ bật khi `self.training`. Ở eval, upstream **luôn chạy hết max steps**. Nên tăng `halt_max_steps` cho đúng nhiều bước suy luận hơn, không có early-exit.

### Kết quả — TRM (2.000 câu test con, seed 12345, 8 views)

| halt_max_steps | calls/câu | Vote 8 views | Greedy | Base |
|---|---|---|---|---|
| 16 | 128 | 30,20% | 17,55% | 0/205 |
| 32 | 256 | 30,35% | 17,75% | 0/205 |
| 64 | 512 | 29,45% | 17,75% | 0/205 |
| 128 | 1.024 | **30,70%** | 17,75% | 1/205 |

Biến động ±0,8 điểm — nằm trong nhiễu. Tăng **8 lượng tính không đổi độ đúng**.

### Kết quả — LDT (cùng tập con)

| n_chains | max_rounds | Vote | Base |
|---|---|---|---|
| 1 | 50 | 99,85% | 203/205 |
| 2 | 50 | 99,95% | 205/205 |
| 4 | 50 | **100,00%** | 205/205 |
| 8 | 50 | 99,95% | 204/205 |

### Kết luận

**Confound eval bị loại bỏ.** Nếu TRM thua vì thiếu lượt suy luận, 8× phải nhảy. Nó không nhảy. Khoảng cách còn lại là **thật**, không phải artifact thiết lập chấm.

Phát hiện phụ đáng ghi: **LDT đạt 99,85% chỉ với 1 nhánh** — không cần vote 8 lần, nó giải đúng ngay. Bất đối xứng này mạnh hơn con số 99,94%.

### Hai hạn chế phải disclose

1. **`halt_max_steps` 128 nằm ngoài phạm vi huấn luyện.** TRM được train với `halt_max_steps=16`; ACT head học dừng trong 16 bước. Ở 128, nó chạy 8× số bước chưa từng thấy. Giải thích thay thế cho đường cong phẳng: model **loop** thay vì suy luận sâu hơn. Kết quả vẫn đúng ở mức nghĩa *"ở ngân sách suy luận mà paper cho phép, TRM đứng yên"*, nhưng không được nói là "TRM không hưởng lợi từ suy luận sâu" một cách tuyệt đối.

2. **`calls/puzzle` của LDT không dùng được để vẽ.** Solver dừng sớm khi giải xong nên cột này ra ~0. Khi vẽ đường cong LDT phải dùng **ngân sách cấp** (K × 50 vòng), không phải ngân sách dùng.

---

## 2. Đường học: 2.667 bước có đủ không? **KHÔNG**

### Thiết kế

Gọi lại `experiments/scaled_trm/run.py` với `steps` khác — **không** copy vòng lặp huấn luyện, để ba điểm chỉ khác nhau ở ngân sách.

| Bước | Examples | Loss | Test vote | Test greedy | Base | Variant |
|---|---|---|---|---|---|---|
| 667 | 512.256 | 0,6086 | **0,00%** | 0,00% | 0/1.000 | 0/9.239 |
| 1.333 | 1.023.744 | 0,5763 | **0,00%** | 0,00% | 0/1.000 | 0/9.239 |
| 2.667 | 2.048.256 | — | 30,45% | 17,86% | 0/1.000 | 3.118/9.239 |

### Kết luận: khoảng cách quan sát được là artifact của ngân sách

Ở **1.023.744 examples** — một nửa ngân sách compute-matched — TRM giải được **0 trên 10.239 câu**, base 0/1.000.

Loss đã giảm (2,48 → 0,58) nên model **không** bị kẹt: nó học được phân phối. Nhưng không câu nào giải trọn vẹn.

Đường cong không dốc đều — nó là **vực thẳm**:

```
  512.256 examples  ──────────────►  0,00%
  1.023.744 examples ──────────────►  0,00%
  2.048.256 examples ──────────────► 30,45%
```

Toàn bộ 30,45% xuất hiện trong khoảng giữa 1,02M và 2,05M examples. Điểm 2.667 bước nằm **ngay sau bậc nhảy**, chưa đủ xa để khẳng định đã hội tụ.

**Hệ quả trực tiếp:** con số 30,45% **không** cho phép xếp hạng method
(LDT data-efficient hơn TRM, hay ngược lại). Nó chỉ nói: ở 2,05M
examples, LDT đã hội tụ còn TRM thì chưa. Kết luận về data efficiency
cần một điểm nữa ở ngân sách mà TRM đã hội tụ — chưa có.

**Cần nói rõ để khỏi hiểu nhầm:** điều này **không** làm mất hiệu lực
của phép so sánh compute-matched. Cùng split, cùng 2,05M examples, cùng
ngân sách eval (đã kiểm bằng §1) — đó là so sánh hợp lệ và đúng thiết
kế. Điều mất hiệu lực chỉ là **phép suy ra mạnh hơn**: xếp hạng method
và kết luận data efficiency.

### Việc này thay đổi gì

| Trước khi đo | Sau khi đo |
|---|---|
| "LDT thắng TRM 3,3× ở ngân sách chung" | **"Ở ngân sách compute-matched 2,05M, LDT đạt 99,94% (996/1000 base), TRM 30,45% (0/1000 base)"** |
| Ngầm hiểu là xếp hạng method | **Có khoảng cách thật ở ngân sách này, nhưng chưa xếp hạng được method** |
| Kết luận về data efficiency | **Không kết luận được gì về data efficiency** |
| Câu hỏi còn mở: budget hay method | Câu hỏi phụ: **TRM cần bao nhiêu examples nữa mới hội tụ?** |

Câu hỏi phụ đó nhiều khả năng cần lớn hơn 2,05M đáng kể, và là nghiên cứu
mới. Nó **không chặn** kết luận chính: bảng §0 vẫn đúng, vì nó được trình
bày đúng là số đo ở một ngân sách cụ thể, không phải thứ hạng method.

### Ghi chú thiết kế

Mỗi điểm là **một run anneal độc lập** (cosine theo tổng bước của chính nó), không phải checkpoint cắt từ run 2.667. Đây là phép so sánh đúng cho câu hỏi "2.667 có đủ không", nhưng nghĩa là các điểm không nằm trên cùng một quỹ đạo tối ưu hoá.

Chi phí điểm 1.333: train + eval ~8 phút B200.

---

## 3. Confound nào đã đóng, cái nào còn

| # | Yêu cầu công bằng | Trước §0 | Sau |
|---|---|---|---|
| 1 | Cùng dữ liệu, split đóng băng | ✅ | ✅ |
| 2 | Cùng ngân sách train (2.048.256 examples) | ✅ | ✅ |
| 3 | **Cùng ngân sách eval** | ❌ 50 vòng vs 16 bước | ✅ **đường cong đo rồi, confound không giải thích được gap** |
| 4 | Nhiều hạt giống | ❌ n=1 | ❌ **còn lại** |
| 5 | **Báo cáo tham số thật mỗi run** | ❌ | ✅ 5.027.842 / 799.635 đã ghi |

Đóng được **4/5**. Còn duy nhất #4 (nhiều hạt giống) — và lưu ý #4
chỉ ảnh hưởng phần HRM vs TRM, không ảnh hưởng khoảng cách LDT–TRM/HRM
(vốn lớn tới mức 0/1.000 vs 996/1.000, không thể là nhiễu).

Câu hỏi "ngân sách 2,05M đã đủ chưa" **đã trả lời xong** ở §2: đủ để đo,
chưa đủ để xếp hạng method.

---

## 4. Chi phí

| Việc | Thời gian | Ước tiền |
|---|---|---|
| Eval sweep (không retrain) | ~9 phút B200 | ~$0.5-1 |
| Learning curve 667 + 1.333 | ~25 phút B200 | ~$3-4 |
| **Tổng** | **~35 phút** | **~$4-5** |

Không còn tiền nào bị đốt cho giả định sai. (Có thêm ~$3-5 mất cho các lần chạy bị hủy giữa chừng do lỗi thao tác, không phải do thiết kế thí nghiệm.)

---

## 5. Việc còn lại

| Việc | Trạng thái | Ghi chú |
|---|---|---|
| Điểm đường học 1.333 | ✅ **Xong** — 0,00% | Xem §2 |
| Tìm ngân sách mà TRM hội tụ | ⬜ **Trở thành câu hỏi chính** | Cần >2,05M examples; đây là hướng nghiên cứu mới, không phải việc vá |
| TRM trên 1.000 câu, cùng 2.048.256 examples | ⬜ Chưa làm | Vẫn giá trị: tách câu hỏi "method yếu" khỏi "regime 82k câu". Nhưng kết luận chính giờ đã là về ngân sách |
| HRM full 82.043 câu | ⬜ Chưa làm | Cần smoke B200 trước — flash-attn đã fail một lần trên Blackwell. Chỉ cần một điểm ở 2.667 bước nếu chỉ dùng làm bối cảnh, **không cần đường học** |
| 2 seed thêm | ⬜ Chưa làm | Cần cho #4, nhưng hiện kết luận chính không dựa vào seed |

### Chi phí còn lại (ước tính)

| Việc | Tiền |
|---|---|
| Smoke HRM trên B200 | ~$0.5 |
| HRM full (1 điểm ở 2.667 bước) | ~$2-3 |
| TRM trên 1.000 câu | ~$2-3 |
| **Tổng nếu làm hết** | **~$5-7** |

Không đủ cả ba với ngân sách còn lại. Ưu tiên theo giá trị: **TRM 1.000 câu** (tách câu hỏi trung tâm) > **HRM full** (thêm bối cảnh) > smoke.

---

## 6. Quy tắc chốt trước (đã áp dụng)

> Điểm 1.333 bước đạt 0% trên 10.239 câu → ta **không** kết luận về data
> efficiency của TRM, và **không** xếp hạng method từ bộ số liệu này.
> Ta vẫn kết luận: ở ngân sách compute-matched 2.048.256 examples, LDT
> đạt 99,94% còn TRM 30,45% — khoảng cách này là **thật**, chỉ là khoảng
> cách **dưới ngân sách**.

Điều khoản này **đã được kích hoạt** bởi dữ liệu thực tế. Báo cáo chính
(`reports/TRM.md`) và `reports/TABLES-FOR-REPORT.md` đã được sửa theo §2:
số 30,45% vẫn được trình bày, nhưng kèm giới hạn phạm vi thay vì bị loại.

---

## 7. Bug đã bắt trong quá trình làm

| # | Lỗi | Cách phát hiện | Sửa |
|---|---|---|---|
| 1 | `env var` đọc **trong** container Modal luôn = None. Lần smoke "rẻ" hoá ra chạy sweep đầy đủ 4×8×2000 trên T4 | Log in ra `SMOKE=False` trong khi local set `=1` | Chuyển thành **function argument** |
| 2 | `torch.compile` đổi tên mọi key state_dict thành `_orig_mod.*`; smoke tắt compile nên lỗi chỉ lộ ở run thật | `RuntimeError: Missing key(s) ... model.inner.*` | Load **trước** khi compile; thêm `_strip_orig_mod()` |
| 3 | `local_entrypoint` trùng tên khi import `run.py` (file đó đã có entrypoint của riêng nó) | `Duplicate local entrypoint name` | `name="learning-curve"`, gọi bằng `::app.learning-curve` |
| 4 | `run()` thuộc app của `run.py`; khai báo app thứ hai thì Modal không hydrate | `the App it is defined on is not running` | Import và dùng chính `app` của `run.py` |
| 5 | Glob `*_*.pt` chọn nhầm checkpoint intermediate (`_step2000.pt` sort **sau** `_20261005_...`) | Rà tên file trên volume | `_final_ckpt()` lọc `_step\d+\.pt` rồi sort theo mtime |
| 6 | `SPLIT_LOCAL` dùng trước khi định nghĩa | `NameError` lúc build image | Đổi thứ tự khai báo |

Bài học chung: **smoke phải đi qua đúng đường code của run thật.** Lần smoke đầu tắt `torch.compile` để tiết kiệm thời gian, và vì thế **không** bắt được bug #2 — lỗi chỉ lộ ra khi đang đốt tiền B200. Sau đó smoke giữ nguyên compile (chỉ 16 câu, vẫn rất nhanh).

---

## 8. Bảng số liệu để đưa vào báo cáo chính

```
Test accuracy, 2.048.256 examples mọi dòng, GPU B200, bf16, seed 0
Đây là so sánh compute-matched hợp lệ. Giới hạn: TRM/HRM chưa hội tụ tại
điểm này, nên KHÔNG dùng để xếp hạng method hay kết luận data efficiency.

              | LDT      | TRM
--------------+----------+--------
Test accuracy | 99,94%   | 30,45%
Base (khó)    | 996/1000 | 0/1000
Variant (dễ)  | —        | 3118/9239
Tham số       | 799.635  | 5.027.842
Greedy 1 đg   | —        | 17,86%
```

Phân tách base/variant là phát hiện chính của báo cáo, và **được củng cố**
bởi §1: khoảng cách không nằm ở thiết lập chấm. §2 chỉ ràng buộc phép
suy ra, không bác bỏ phép đo.

*(Cập nhật 06/10: HRM đã chạy full — test 34,50%, base 6/1000, xem `reports/HRM.md`. Cả TRM và HRM đều chưa hội tụ ở ngân sách này.)*
