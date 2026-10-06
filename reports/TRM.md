# TRM trên Scaled Sudoku Instances — Báo cáo Baseline

> Ngôn ngữ đơn giản. Số liệu đo thật trên GPU Modal B200.
> Chỉ báo cáo bản full-recipe (upstream-faithful). Bản rút gọn thử
> nghiệm trước đây (CE + fresh carry, test 41.87%) đã loại, không dùng.

## 1. Experiment này là gì?

- **Model (TRM)**: Tiny Recursive Models — mạng nhỏ (~5M params) tự lặp
  lại nhiều vòng (H=3/L=6) để sửa dần đáp án, kèm đầu Q học "khi nào
  dừng suy nghĩ" (halt), đánh giá bằng EMA weights.
- **Dataset**: `zayenz/scaled-sudoku-instances`, dùng đúng file split
  đóng băng của LDT (`scaled_sudoku_split.jsonl`, seed 42):
  train 82,043 / val 10,174 / test 10,239, chia theo nhóm ancestor,
  leak = 0 (kiểm tra lại trong run, mục 8).
- **Compute-matched với LDT**: 768 x 2667 = 2,048,256 examples,
  ≈ LDT 512 x 4000 = 2,048,000 (lệch 0.01%). Cùng B200, from scratch,
  1 seed (seed0). Muốn claim cần thêm seeds.
- **Protocol**: train chỉ trên train → eval full val 1 lần → eval full
  test 1 lần. Test không bao giờ đụng tới trước khi train xong.

## 2. Kết quả chính

| Tập | Vote 8 views | Greedy 1 đường |
|---|---|---|
| Validation (10,174) | 3,156 (**31.02%**) | 1,930 (18.97%) |
| Test (10,239) | 3,118 (**30.45%**) | 1,829 (17.86%) |

- Greedy = cách chấm của gốc (1 forward loop, argmax) → số so với paper.
- Vote = 1 bản gốc + 7 bản biến thể, giải riêng rồi bầu đa số →
  số so công bằng test-time với LDT (8 chains).
- Vote hơn greedy ~12.6 điểm: đáp án đơn nhiễu, nhiều views vớt lại.

## 3. Base (khó) vs Variant (dễ) — chỗ phân biệt thật

| Tập | Base (gốc minimal) | Variant |
|---|---|---|
| Validation | 2/1000 (0.2%) | 3,154/9,174 (~34.4%) |
| Test | 0/1000 (0%) | 3,118/9,239 (~33.7%) |

TRM trượt gần hết base. LDT base test 996/1000. Khoảng cách hai
method nằm ở đây, không phải ở variants.

## 4. Càng nhiều gợi ý càng dễ đúng (validation, vote)

| Gợi ý 21–23 | 24 | 25 | 26 | 27 | 28–29 | 30+ |
|---|---|---|---|---|---|---|
| 0/409 (0%) | 2/1026 | 11/1688 | 41/1714 | 95/1000 | 592/1466 | ~88–100% |

Quan hệ rõ: nhiều gợi ý hơn → đúng nhiều hơn. 0% ở nhóm khó nhất.

## 5. Quá trình học

- Loss (stablemax + 0.5·Q-halt): 2.48 → 0.41. Train exact-acc lên
  ~0.88 cuối run nhưng val greedy lúc đó mới ~19% (hở generalization).
- In-train eval (val-200, EMA): 0/200 ở step 1000 và 2000 — điểm số
  chỉ bật lên ở đoạn cuối (khác LDT học đều từ sớm).
- Train 360s (~6 phút), peak VRAM 31.9GB, params 5,027,842.

## 6. So với LDT (cùng ~2.048M examples, cùng split)

| | LDT | TRM (bản này) |
|---|---|---|
| Test | 99.94% (solver) | 30.45% (vote) / 17.86% (greedy) |
| Base test | 996/1000 | 0/1000 |
| Câu trả lời sai (wrong) | 0 | (xem eval.json) |

**Đọc đúng con số này.** Đây là so sánh compute-matched hợp lệ: cùng
split, cùng 2.048.256 examples, cùng ngân sách eval. Khoảng cách ở đây
là **thật**. Nhưng ở **một nửa** ngân sách (1.023.744 examples) TRM đạt
**0,00%** trên cả 10.239 câu, nên điểm 2.667 bước nằm ngay sau bậc nhảy
và **chưa đủ xa để xếp hạng method**. Đường cong không dốc đều mà là
**vực thẳm**:

```
  1.023.744 examples ──────────────►  0,00%
  2.048.256 examples ──────────────► 30,45%
```

Toàn bộ 30,45% xuất hiện trong khoảng 1,02M–2,05M examples. Vì vậy
**không kết luận được gì về data efficiency của TRM** so với LDT từ bộ
số liệu này — nhưng khoảng cách 99,94% vs 30,45% tại ngân sách này thì
đứng được.

Giới hạn của claim này đã được đo, không phải suy đoán: `halt_max_steps`
tăng 8× không cải thiện (§1 `EVAL-SWEEP-AND-CURVE.md`), nên khoảng cách
**không** nằm ở thiết lập chấm.

Kết luận đứng được: **ở ngân sách compute-matched này, LDT giải được
996/1.000 puzzle khó nhất còn TRM 0/1.000.** Muốn xếp hạng method thật,
phải tìm mức ngân sách mà TRM hội tụ — xem
`reports/EVAL-SWEEP-AND-CURVE.md` §2.

Chi tiết: mục 6 của `EVAL-SWEEP-AND-CURVE.md` đã yêu cầu sửa lại chỗ
này từ trước; bản sửa nằm ở trên.

## 7. Khai báo khác biệt so với bản gốc (upstream)

Code gốc (`upstream/TinyRecursiveModels/`) không sửa 1 dòng. Đã kiểm bằng
cách so hash SHA-256 từng file với bản tải từ
`github.com/SamsungSAILMontreal/TinyRecursiveModels`: **40 / 40 file khớp,
0 khác biệt**. Commit gốc `c01103738605` (2026-03-31).
File chạy: `experiments/scaled_trm/run.py`. Dưới đây là mọi chỗ khác,
lý do, và cách sửa (nếu có).

### 7.1. Giữ nguyên gốc (import/dùng trực tiếp)

Model `TinyRecursiveReasoningModel_ACTV1`, `ACTLossHead`
(stablemax), `EMAHelper` (mu=0.999, ema_copy để eval),
`AdamATan2` + `SignSGD` cho puzzle_emb, cosine schedule, carry-over,
không clip grad, loss chia batch, `torch.compile`, halt config
(H=3/L=6, halt_max_steps=16, exploration 0.1), LR 1e-4/wd 1.0.

### 7.2. Chỗ khác gốc

1. **Nguồn data (adapter jsonl).** Gốc đọc npy `PuzzleDataset`
   build từ HF 1k-puzzle, không ăn được split 102k của ta.
   *Lý do:* bắt buộc (khác format). *Sửa:* không (đúng scope adapter).
2. **Steps 2667 + batch 768.** Công thức gốc (epochs × data) sinh
   cho regime 1k-data/50k-epochs; bê nguyên = ~500 giờ B200.
   *Lý do:* khả thi + compute-match LDT. *Sửa:* không.
3. **Sampler tự code (96 puzzles × 8 views).** Sampler gốc đọc index
   npy. Logic gom giữ nguyên (nhiều views cùng puzzle/batch).
   *Lý do:* không có npy. *Sửa:* không (muốn xóa thì convert npy).
4. **Aug on-the-fly.** aug-1000 lưu sẵn trên 82k = 82M files.
   Cùng họ transform (digit + transpose + band/stack).
   *Lý do:* disk. *Sửa:* không.
5. **Puzzle ids = row index; eval ids = 0.** Không có build metadata
   như gốc. Puzzle chưa thấy thì embedding nào cũng ngẫu nhiên;
   zeros deterministic dễ debug. *Sửa:* không.
6. **Bỏ code multi-GPU.** Chạy 1 GPU; nhánh dist của gốc cũng bị
   guard bỏ qua. Chạy y hệt. *Sửa:* không cần.
7. **In-train eval trên val-200.** Gốc eval trên test giữa chừng —
   ta từ chối để giữ protocol sạch (LDT cũng vậy). *Sửa:* không
   (đây là chỗ ta nghiêm hơn gốc).
8. **Vote 8 views + breakdowns.** Thêm ngoài greedy (greedy vẫn là
   số chính so gốc). Vote để cân test-time với 8 chains của LDT.
   *Sửa:* không.
9. **Infra (Modal image, wandb project, checkpoint path).**
   Môi trường chạy, không chạm toán học. *Sửa:* không.
10. **Warmup 10% (267 steps) thay vì absolute 2000.** Absolute 2000
    = 75% run ngắn → dìm TRM (bám máy móc mới là unfair).
    LDT cũng dùng fraction 0.1. *Đã sửa.*
11. **AdamATan2 pure-torch thay vì kernel nvcc.** Package PyPI thiếu
    backend biên dịch, image slim không có nvcc. Công thức replay
    y hệt `csrc/adam_atan2.cu`, test số local khớp 2e-7
    (`experiments/scaled_trm/adam_atan2_torch.py`). *Đã sửa.*

### 7.3. Bugs đã bắt và sửa trong quá trình làm (không ảnh hưởng số cuối)

- **Labels ngoài batch** → carry-over trộn nhầm labels cũ/mới,
  loss đứng yên ở ln(9), eval 0%. *Sửa:* labels vào trong batch dict
  (đúng kiểu gốc), supervision đọc `carry.current_data["labels"]`,
  verify alignment 200 steps trên CPU.
- **Gọi `.model` trên compiled wrapper** → trỏ vào TRM trần thiếu
  `return_keys`. *Sửa:* gọi wrapper trực tiếp + `return_keys=["preds"]`
  đúng protocol eval gốc.

### 7.4. Caveats phải disclose (tính chất, không phải bug)

- Per-puzzle embedding 82k rows, mỗi puzzle gặp ~25 lần (gốc 1k
  puzzles gặp đi gặp lại) — TRM thiệt hơn LDT ở điểm này.
- Halt-cap 16 của paper vs LDT 50 rounds mỗi view.
- Mới seed0. Aug paper cần train dài mới phát huy ở budget ngắn.

## 8. Tính toàn vẹn + files

- Leakage re-check trong run: cross-split ancestor = 0,
  cross-split puzzle = 0.
- Checkpoint: `/checkpoints/scaled_trm/full4_seed0_2667s_bs768_*.pt`
  (+ `.eval.json`, EMA weights trong file).
- wandb: project `scaled-sudoku-trm-hrm`, run `trm-full4-seed0-2667s`.
- Code: `experiments/scaled_trm/run.py` (full),
  `experiments/scaled_trm/t4_smoke.py` (smoke PASS 200 steps),
  `experiments/scaled_trm/adam_atan2_torch.py` (optimizer).

## 9. Chi phí (B200)

Train ~6 phút + eval vote ~4 phút + in-train eval/ckpt. Tổng run
~10 phút (~$1.5-3 tùy rate; số chính xác xem Billing Modal).
