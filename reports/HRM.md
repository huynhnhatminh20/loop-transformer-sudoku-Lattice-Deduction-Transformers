# HRM trên Scaled Sudoku Instances — Báo cáo Baseline

> Ngôn ngữ đơn giản. Số liệu đo thật trên GPU Modal B200.
> Chỉ có một bản chạy: full-recipe, upstream-faithful.
> Chạy ngày 06/10/2026, seed 0.

## 1. Experiment này là gì?

- **Model (HRM)**: Hierarchical Reasoning Model — 27.274.242 params.
  Kiến trúc hai tầng: tầng H (4 layers, trên toàn chuỗi) điều phối,
  tầng L (4 layers, trên lưới 9×9) làm việc cục bộ. Mỗi vòng suy luận
  chạy H_cycles=2 × L_cycles=2, tối đa 16 vòng, kèm đầu Q học
  "khi nào nên dừng" (halt). Không có EMA.
- **Dataset**: `zayenz/scaled-sudoku-instances`, dùng đúng file split
  đóng băng của LDT (`scaled_sudoku_split.jsonl`, seed 42):
  train 82.043 / val 10.174 / test 10.239, chia theo nhóm ancestor,
  leak = 0 (kiểm tra lại trong run, mục 8).
- **Compute-matched với LDT**: 768 × 2667 = 2.048.256 examples,
  ≈ LDT 512 × 4000 = 2.048.000 (lệch 0.01%). Cùng B200, from scratch,
  1 seed. Muốn claim cần thêm seeds.
- **Protocol**: train chỉ trên train → eval full val 1 lần → eval full
  test 1 lần. Test không bao giờ đụng tới trước khi train xong.

## 2. Kết quả chính

| Tập | Vote 8 views | Greedy 1 đường |
|---|---|---|
| Validation (10.174) | 3.532 (**34,72%**) | 1.896 (18,64%) |
| Test (10.239) | 3.532 (**34,50%**) | 1.857 (18,14%) |

- Greedy = cách chấm của gốc (1 forward loop, argmax) → số so với paper.
- Vote = 1 bản gốc + 7 bản biến thể, giải riêng rồi bầu đa số →
  số so công bằng test-time với LDT (8 chains).
- Vote hơn greedy ~16,4 điểm: đáp án đơn nhiễu, nhiều views vớt lại.

Lưu ý: val và test trùng đúng 3.532 câu đúng nhưng khác tập — đây là
trùng hợp, tỉ lệ 34,72% vs 34,50% là ngẫu nhiên.

## 3. Base (khó) vs Variant (dễ) — chỗ phân biệt thật

| Tập | Base (gốc minimal) | Variant |
|---|---|---|
| Validation | 2/1000 (0,2%) | 3.530/9.174 (~38,5%) |
| Test | 6/1000 (0,6%) | 3.526/9.239 (~38,2%) |

HRM trượt gần hết base. LDT base test 996/1000. Khoảng cách hai
method nằm ở đây, không phải ở variants.

## 4. Càng nhiều gợi ý càng dễ đúng (test, vote)

| Gợi ý ≤23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 33 | ≥38 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0/311 (0%) | 0,8% | 1,8% | 6,3% | 16,6% | 40,6% | 62,9% | 75,2% | 83,6% | 90,2% | ~95% |

Quan hệ gần như đơn điệu: nhiều gợi ý hơn → đúng nhiều hơn. Dưới
24 clue là 0%. Đây là toàn bộ lý do HRM trượt base.

## 5. Quá trình học

- Loss (stablemax + 0,5·Q-halt): 2,43 → 0,89. Xuống dưới ngưỡng
  marginal ln(9) = 2,197 ngay từ step ~100, tức là thật sự học chứ
  không chỉ học phân phối biên.
- Đáng chú ý: loss **tăng lại** từ step 1500 (0,58) lên step 2200
  (0,97), rồi đi ngang ~0,82-0,89. Có hiện tượng hỏng ngắn (catastrophic
  forgetting), hồi phục một phần nhưng không về lại đáy.
- Train exact-acc bắt đầu >0 từ step 1600, chạm 0,362 ở step 2600.
- In-train eval (val-200, greedy): 0/200 ở cả step 1000 và step 2000.
- Train 251s (~4,2 phút), eval 103s (55s val + 48s test),
  **tổng 368s = 6,1 phút**. Peak VRAM 11,62GB.

## 6. So với LDT và TRM (cùng ~2,048M examples, cùng split)

| | LDT | TRM | HRM (bản này) |
|---|---|---|---|
| Params | ~800K | 5.027.842 | 27.274.242 |
| Test (vote) | 99,94% | 30,45% | **34,50%** |
| Test (greedy) | — | 17,86% | 18,14% |
| Base test | 996/1000 | 0/1000 | 6/1000 |
| Train time | — | 360s | 251s |
| Peak VRAM | — | 31,9GB | 11,62GB |

Hai kết luận:

1. **HRM hơn TRM ở budget này** (+4,05 điểm vote). Greedy gần như bằng
   nhau (18,14% vs 17,86%), nên phần thắng nằm ở việc vote 8 views hợp
   hơn — tức đáp án của HRM ổn định hơn qua các biến thể.
2. **Cả TRM lẫn HRM đều sụp trên base**, dù HRM lớn gấp 5,4 lần và
   nhanh hơn 30%. Trên 1.000 puzzle minimal, LDT 996/1.000 còn
   TRM 0/1.000, HRM 6/1.000.

Điểm 2 mới là kết luận quan trọng, và nó **không giải thích được bằng
số lượng tham số**: mạng to hơn 5,4 lần, nhanh hơn 30%, nhẹ hơn 3× thì
vẫn không giúp gì trên câu ít gợi ý.

**Quan trọng — confound "test-time compute" đã được đo và loại bỏ
từ trước, không phải suy đoán.** `reports/EVAL-SWEEP-AND-CURVE.md` §1
đã chạy sweep trên chính TRM:

| halt_max_steps | calls/câu | Vote 8 views | Base |
|---|---|---|---|
| 16 | 128 | 30,20% | 0/205 |
| 32 | 256 | 30,35% | 0/205 |
| 64 | 512 | 29,45% | 0/205 |
| 128 | 1.024 | **30,70%** | 1/205 |

Tăng suy luận 8× không đổi độ đúng (biến động ±0,8 điểm, trong nhiễu).
Nên khoảng cách LDT–TRM **không** nằm ở thiết lập chấm.

Hệ quả cần nói đúng, và **không nói quá**: ở 2.048.256 examples LDT đạt
99,94% (996/1.000 base) còn TRM 30,45% và HRM 34,50% (0 và 6 trên 1.000
base). Đây là so sánh compute-matched hợp lệ và khoảng cách là **thật**.

Giới hạn: `EVAL-SWEEP` §2 cho thấy ở 1.023.744 examples (một nửa ngân
sách) TRM đạt **0,00%** trên cả 10.239 câu. Đường cong là vực thẳm, và
toàn bộ 30,45% của TRM xuất hiện trong khoảng 1,02M–2,05M examples. Điểm
2.667 bước nằm **ngay sau bậc nhảy**, chưa đủ xa để xếp hạng method hay
kết luận data efficiency. HRM chỉ có một điểm nên chưa vẽ được đường
cong của riêng nó.

Hạn chế của bằng chứng này phải disclose: `halt_max_steps = 128` nằm
ngoài phạm vi huấn luyện (ACT head học dừng trong 16 bước), nên kết
luận đúng ở mức *"ở ngân sách mà paper cho phép, TRM đứng yên"*, không
được nói là "TRM không hưởng lợi từ suy luận sâu" một cách tuyệt đối.

**Việc cần làm để *xếp hạng* method** (không chặn kết luận chính): tìm
ngân sách mà TRM/HRM hội tụ (>2,05M examples, nhiều khả năng lớn đáng
kể), hoặc chạy cả hai trên tập 1.000 puzzle của paper gốc để tách câu
hỏi "method yếu" khỏi "regime 82k câu". Cả hai đều ngoài phạm vi run hiện
tại.

## 7. Khai báo khác biệt so với bản gốc (upstream)

Code gốc (`upstream/HRM/`) không sửa 1 dòng. File chạy:
`experiments/scaled_hrm/run.py`.

**Cách chứng minh.** Không dùng `git diff -- upstream/` — thư mục đó không
được git track nên lệnh luôn rỗng bất kể đã sửa hay chưa, tức là bằng chứng
sai. Cách đúng là so hash từng file với bản gốc tải từ GitHub:

| So sánh | Kết quả |
|---|---|
| `upstream/HRM/` vs `github.com/sapientinc/HRM` | **26 / 26 file khớp SHA-256, 0 khác biệt** |
| Commit gốc | `ac15626f8db0` (2026-04-01) |

Dưới đây là mọi chỗ khác, lý do, và cách sửa (nếu có).

### 7.1. Giữ nguyên gốc (import/dùng trực tiếp)

Model `HierarchicalReasoningModel_ACTV1`, `ACTLossHead` (stablemax),
`AdamATan2` + `CastedSparseEmbeddingSignSGD` cho puzzle_emb, cosine
schedule (`pretrain.py:162-169` verbatim), carry-over, không clip
grad, loss chia batch, `torch.compile`, halt config (H=2/L=2,
H_layers=4, L_layers=4, halt_max_steps=16, exploration 0.1).
LR 7e-5 / puzzle_emb_lr 7e-5 / weight_decay 1.0 / betas (0,9; 0,95) —
đúng lệnh sudoku ở `HRM/README.md:79`.

### 7.2. Không có EMA — đã kiểm, không phải lựa chọn của tôi

HRM upstream **không có** EMA: không file `models/ema.py`, không cờ
`ema` trong bất kỳ config nào, không đoạn code nào. Grep toàn repo chỉ
khớp ở các chuỗi `stablemax` và `schema`. Nên eval dùng live weights,
giống hệt upstream. TRM thì có (và dùng `EMAHelper(0.999)`).

Đây là điểm fair quan trọng: TRM được hưởng lợi thế EMA còn HRM thì
không, nhưng đó là **đặc tính của method**, không phải thiếu sót của
harness. Giữ nguyên là đúng.

### 7.3. Chỗ khác gốc

1. **Nguồn data (adapter jsonl).** Gốc đọc npy `PuzzleDataset` build
   từ HF 1k-puzzle, không ăn được split 102k của ta.
   *Lý do:* bắt buộc (khác format). *Sửa:* không (đúng scope adapter).
2. **Steps 2667 + batch 768.** Công thức gốc (epochs × data) sinh
   cho regime 1k-data/20k-epochs; bê nguyên = ~500 giờ B200.
   *Lý do:* khả thi + compute-match LDT. *Sửa:* không.
   **Lưu ý batch:** lệnh sudoku gốc dùng `global_batch_size=384`; ta
   dùng 768 (giá trị mặc định trong `cfg_pretrain.yaml`, và là giá trị
   lệnh ARC 8-GPU của chính HRM). Cùng số examples như LDT/TRM.
3. **Sampler tự code (96 puzzles × 8 views).** Sampler gốc đọc index
   npy. Logic gom giữ nguyên (nhiều views cùng puzzle/batch).
   *Lý do:* không có npy. *Sửa:* không.
4. **Aug on-the-fly.** aug-1000 lưu sẵn trên 82k = 82M files.
   Cùng họ transform (digit + transpose + band/stack), và logic
   **giống hệt** file TRM dùng.
   *Lý do:* disk. *Sửa:* không.
5. **Puzzle ids = row index; eval ids = 0.** Không có build metadata
   như gốc. Puzzle chưa thấy thì embedding nào cũng ngẫu nhiên.
   *Sửa:* không.
6. **Bỏ code multi-GPU.** Chạy 1 GPU; nhánh dist của gốc cũng bị
   guard bỏ qua. *Sửa:* không cần.
7. **In-train eval trên val-200.** Gốc eval trên test giữa chừng —
   ta từ chối để giữ protocol sạch. *Sửa:* không (chỗ ta nghiêm hơn gốc).
8. **Vote 8 views + breakdowns.** Thêm ngoài greedy (greedy vẫn là
   số chính so gốc). Vote để cân test-time với 8 chains của LDT.
   *Sửa:* không.
9. **Infra (Modal image, wandb, checkpoint path).** Môi trường chạy,
   không chạm toán học. *Sửa:* không.
10. **Warmup 10% (266 steps) thay vì absolute 2000.** Absolute 2000
    = 75% run ngắn → dìm HRM. LDT cũng dùng fraction 0,1.
    *Đã sửa* — và TRM cùng giá trị, nên so sánh vẫn công bằng.
11. **AdamATan2 pure-torch thay vì kernel nvcc.** Package PyPI thiếu
    backend biên dịch, image slim không có nvcc. Công thức replay
    y hệt `csrc/adam_atan2.cu`, test số local khớp 2e-7. Dùng chung
    file với TRM. *Đã sửa.*
12. **Attention chạy SDPA thay vì flash-attn.** `models/layers.py:130`
    import `flash_attn_func`; B200 (Blackwell) không có wheel. Thay bằng
    stub cùng công thức softmax-attention, layout `[B,S,H,D]`, trả về
    `.contiguous()` để khớp `attn_output.view(...)` ở `layers.py:134`.
    **Cùng toán học, khác kernel; đoạn này dùng trên B200 luôn, không
    chỉ ở smoke.**

### 7.4. Bugs đã bắt và sửa trong quá trình làm

- **Thiếu `mkdir` trước khi lưu checkpoint trung gian** →
  `RuntimeError: Parent directory /checkpoints/scaled_hrm does not exist`,
  job chết ở step 1000/2667.
  *Nguyên nhân sâu hơn:* smoke T4 chỉ chạy 200 step nên nhánh
  `step % 1000 == 0` không bao giờ được kích hoạt — **smoke không cover
  nhánh đó**. Đây là lần thứ hai gặp bug "thiếu tạo thư mục checkpoint"
  (§4 của báo cáo công việc), lần đầu vá ở `ckpt_path`, quên `mid_path`.
  *Sửa:* thêm `Path(mid_path).parent.mkdir(parents=True, exist_ok=True)`.
  **Phát hiện thêm:** `experiments/scaled_trm/run.py:348` có **cùng bug
  tiềm ẩn** — chỉ sống sót vì `/checkpoints/scaled_trm/` đã được tạo từ
  lần chạy trước. Trên volume sạch thì TRM cũng sẽ chết. Đã vá cả hai.
- **Hai lỗi phía client, trước khi vào GPU (không tốn tiền):** WDAC
  chặn `torch\lib\shm.dll` lần import đầu; và `lattice_diffusion` nằm
  ở `src/` chứ không ở root nên phải set `PYTHONPATH=src`.

### 7.5. Caveats phải disclose (tính chất, không phải bug)

- **HRM có LR thấp hơn TRM** (7e-5 vs 1e-4) và **batch gấp đôi lệnh
  sudoku gốc** (768 vs 384). Cả hai giá trị upstream đều tuned cho
  20k-50k epoch trên 1k puzzle; ở run 2.667 step, HRM bị kép chậm học.
  Hệ quả: khi HRM vẫn thắng TRM thì kết luận "HRM ≥ TRM ở budget
  ngắn" được **đảo ngược thành công** — HRM có lẽ còn mạnh hơn con số
  báo cáo. Ngược lại nếu HRM thua ở một metric nào đó thì phải cẩn
  trọng với nguyên nhân recipe, không kết luận về method.
- **Compute-matched theo examples, không theo FLOPs.** HRM 27,3M params
  / ~512 layer-passes mỗi ACT step; TRM 5,0M / 36. Số examples bằng
  nhau không phải FLOPs bằng nhau. Vì vậy báo cáo kèm `train_s` và
  `peak_gb` (mục 6) — may mắn là HRM vừa nhanh hơn vừa nhẹ hơn.
- **Halt-cap 16 của paper vs LDT 50 vòng mỗi view.** Đây là caveat
  lớn nhất, **nhưng đã được đo và loại bỏ** — xem bảng sweep ở mục 6
  và `reports/EVAL-SWEEP-AND-CURVE.md` §1. Tăng lên 128 không giúp.
  Vì vậy 34,50% không nên được đọc là "sẽ cao hơn nếu cho nhiều lượt
  suy luận hơn".
- **Per-puzzle embedding 82k rows**, mỗi puzzle gặp ~25 lần (gốc 1k
  puzzles gặp đi gặp lại) — HRM thiệt hơn LDT ở điểm này.
- **Mới seed0.**
- **Loss không đơn điệu** (mục 5): nếu train dài hơn, HRM có thể hồi
  phục về loss thấp hơn nữa. Budget ngắn có thể chưa là điểm tối đa
  của nó.

## 8. Tính toàn vẹn + files

- Leakage re-check trong run: cross-split ancestor = 0, cross-split
  puzzle = 0 (6 cặp, tất cả 0). `assert` trong code fail nếu khác.
- Checkpoint: `/checkpoints/scaled_hrm/full_seed0_2667s_bs768_*.pt`
  (+ `.eval.json`, + 2 checkpoint trung gian `_step1000`, `_step2000`).
- wandb: project `scaled-sudoku-trm-hrm`, run `hrm-full-seed0-2667s`.
- Code: `experiments/scaled_hrm/run.py` (full),
  `experiments/scaled_hrm/t4_smoke.py` (smoke PASS, loss 2,66 → 1,50),
  `experiments/scaled_trm/adam_atan2_torch.py` (optimizer, dùng chung).

## 9. Chi phí (B200)

Tổng run **6,1 phút** (train 251s + eval 103s + overhead) — ước
lượng $0,7. Ba tiền tốn thêm trong phiên làm việc này:

| Hạng mục | Chi phí |
|---|---|
| Run chết ở step 1000 (bug mkdir) | ~$0,3-0,5 |
| Run thành công 6,1 phút | ~$0,7 |
| **Tổng phiên này** | **~$1-1,2** |

Ước lượng ban đầu của tôi là $2-4 — **sai**. Sai ở chỗ tôi giả
eval mất 4-5 phút; thực tế 103s, vì HRM halt khá sớm ở nhóm puzzle dễ.
