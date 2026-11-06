# Lattice Deduction Transformers — reproduction + đối chiếu baselines

Repo này gồm hai phần tách biệt:

1. **Tái lập LDT** (Lattice Deduction Transformers) trên hai benchmark Sudoku.
2. **Đóng góp của đồ án**: một benchmark Sudoku mở rộng (`scaled sudoku`) và
   hai baseline recursive reasoner (TRM, HRM) được cài vào cùng harness để so
   sánh công bằng với LDT.

> Phần (1) là mã nguồn của nhóm tác giả gốc, dưới giấy phép MIT —
> xem [`LICENSE`](LICENSE). Phần (2) là phần được bổ sung cho đồ án.

---

## 1. Nguồn gốc và ghi công

**Lattice Deduction Transformers** — Liam Davis, Leopold Haller,
Alberto Alfarano, Mark Santolucito. [arXiv:2605.08605](https://arxiv.org/abs/2605.08605).

Toàn bộ code LDT ở đây (`src/lattice_diffusion/`) là **bản dựng lại có
chọn lọc** từ codebase gốc, do nhóm trên viết cho các thí nghiệm trong
paper. Không sửa một dòng nào trong `upstream/`.

Mã của TRM và HRM giữ nguyên từ các repo gốc của chúng, đặt trong
`upstream/` và không bị chỉnh sửa:

| Model                               | Nguồn                            |
| ----------------------------------- | --------------------------------- |
| TRM — Tiny Recursive Models        | `upstream/TinyRecursiveModels/` |
| HRM — Hierarchical Recursive Model | `upstream/HRM/`                 |

Các câu hỏi mở, suy luận và số liệu trong `reports/` là phần đóng góp của
đồ án.

---

## 2. Những gì đã chạy thật

Mọi số dưới đây đo trên GPU Modal B200, bf16, seed 0.

### 2.1. Sudoku-Extreme (LDT)

Bộ 4.000 puzzle từ `sapientinc/sudoku-extreme`. Huấn luyện 4.000 bước ×
batch 512, eval 64 chains × 1.000 vòng.

| Chỉ số                       | Kết quả     |
| ------------------------------ | ------------- |
| Đúng                         | 3.995 / 4.000 |
| **Trả lời sai**        | **0**   |
| Hết giờ (chưa giải xong)   | 5             |
| Tỉ lệ suy luận không sound | 0,0026        |
| Thời gian train               | 926 s         |

|            | Giá trị                                                           |
| ---------- | ------------------------------------------------------------------- |
| Checkpoint | `/checkpoints/sudoku/seed0_4000s_bs512_aug1_20261003_040535_*.pt` |
| Log eval   | `logs/phase2_sudoku_B200.eval.json`                               |

### 2.2. Snowflake Sudoku (LDT)

1.000 puzzle do nhóm LDT sinh bằng solver CVC5. Đây là sân riêng của LDT —
paper gốc ghi rõ TRM, HRM và Sotaku **không** được huấn luyện trên bộ này.

| Chỉ số                | Kết quả               |
| ----------------------- | ----------------------- |
| Đúng                  | **1.000 / 1.000** |
| **Trả lời sai** | **0**             |
| Hết giờ               | 0                       |

|            | Giá trị                                                              |
| ---------- | ---------------------------------------------------------------------- |
| Checkpoint | `/checkpoints/snowflake/seed0_1000s_bs512_aug1_20261003_044719_*.pt` |
| Log eval   | `logs/phase3_snowflake_B200.eval.json`                               |

### 2.3. Killer Sudoku — thử nghiệm giới hạn (LDT)

Sinh bằng CVC5, huấn luyện tiếp từ checkpoint Sudoku-Extreme, đánh giá một
lần duy nhất trên tập test. Mục đích: xác định LDT còn chịu được bao xa khi
bài toán vượt quá cấu trúc mà nó sinh ra.

Code: `experiments/killer/`. Báo cáo đầy đủ: [`reports/bao-cao-tong-hop.md`](reports/bao-cao-tong-hop.md) (phần Killer Sudoku).

Kết quả chính: model LDT đã huấn luyện trên Sudoku-Extreme **không** giải
được câu Killer nào khi bị cấp lồng và tổng (0/56 ở Exp #1). Mở rộng đầu
vào để đọc được thông tin lồng/tổng vẫn không đủ — đây là bằng chứng LDT
cần thay đổi kiến trúc, không phải chỉ thêm thông tin đầu vào.

### 2.4. Scaled Sudoku — benchmark mới của đồ án

102.456 puzzle 9×9: 10.000 puzzle gốc tối thiểu + 92.456 biến thể dễ hơn.
Chia theo nhóm ancestor để không rò rỉ:

| Tập           | Số puzzle  |
| -------------- | ----------- |
| Train          | 82.043      |
| Validation     | 10.174      |
| Test           | 10.239      |
| **Leak** | **0** |

Nguồn corpus: Mikael Zayenz Lagerkvist, *Scaling Sudoku as a Constraint
Problem*. Độ khó gắn nhãn theo phương pháp suy luận yếu nhất giải được.

Mọi run dùng chung một file split đóng băng: `scaled_sudoku_split.jsonl`.

### 2.5. Ba model trên scaled sudoku — so sánh compute-matched

Nguyên tắc: **cả ba cùng nhìn khoảng 2,05 triệu examples.** "Step" của
các paper khác nhau không cùng định nghĩa, nên ta khóa theo số examples
chứ không theo số bước.

|                                  | LDT               | TRM          | HRM          |
| -------------------------------- | ----------------- | ------------ | ------------ |
| Tham số                         | **799.635** | 5.027.842    | 27.274.242   |
| Steps × batch                   | 4.000 × 512      | 2.667 × 768 | 2.667 × 768 |
| **Test**                   | **99,94%**  | 30,45%       | 34,50%       |
| — Base (1.000 câu tối thiểu) | **996**     | **0**  | 6            |
| — Variant (9.239 câu dễ)      | 9.237             | 3.118        | 3.526        |
| Suy luận trung bình/câu       | 12,8 vòng        | 128 call     | —           |

Phát hiển chính: khoảng cách giữa ba model nằm ở **nhóm puzzle tối thiểu**,
không phải ở nhóm biến thể dễ. Hai model recursive gần như không giải được
câu nào ở nhóm khó, trong khi đó giữ được 99,9–100% ở mọi nhóm độ khó
suy luận.

**Giới hạn phải đọc kèm:** ở một nửa ngân sách (1.023.744 examples) TRM
đạt **0,00%** trên cả 10.239 câu. Điểm 2.667 bước nằm ngay sau bậc nhảy,
nên bảng trên là phép đo ở một ngân sách cụ thể, **không** phải bảng xếp
hạng method. Chi tiết: `reports/bao-cao-tong-hop.md` (mục TRM vs LDT).

---

## 3. Cấu trúc thư mục

```
src/lattice_diffusion/     Mã LDT: model, data loader, tiện ích Modal
  models/
    looped_transformer.py    PowersetModel — kiến trúc lõi của LDT
    transformer2d.py         Khối transformer 2D
    weighted_bce.py          Loss BCE có trọng số
  data/
    sudoku_extreme.py        Loader Sudoku-Extreme
    scaled_sudoku.py         Loader scaled sudoku (viết cho đồ án)
  training/utils/           Checkpoint, scheduler
  modal/                    Định nghĩa image và volume cho Modal

experiments/
  sudoku/                  Sudoku-Extreme
  snowflake/               Sinh dữ liệu + train + eval Snowflake
  killer/                  Thử nghiệm giới hạn Killer Sudoku
  scaled/                  LDT trên scaled sudoku
  scaled_trm/              Baseline TRM (bổ sung cho đồ án)
  scaled_hrm/              Baseline HRM (bổ sung cho đồ án)

upstream/                  Mã gốc, không sửa
  TinyRecursiveModels/
  HRM/

reports/                   Báo cáo tổng hợp (đã gộp) + bảng số liệu
logs/                      Log eval thô (nguồn của mọi con số)
scaled_sudoku_split.jsonl  Split đóng băng, dùng chung cho cả ba model
```

### Quy tắc giữ ranh giới

`upstream/` **không sửa một dòng**, kiểm bằng cách so hash SHA-256 từng file
với bản gốc trên GitHub:

| Repo                              | Kết quả so hash                      | Commit           |
| --------------------------------- | -------------------------------------- | ---------------- |
| `TinyRecursiveModels` (40 file) | **40 / 40 khớp, 0 khác biệt** | `c01103738605` |
| `HRM` (26 file)                 | **26 / 26 khớp, 0 khác biệt** | `ac15626f8db0` |

Không dùng `git diff -- upstream/` để làm bằng chứng: thư mục đó không được
git track nên lệnh luôn rỗng dù đã sửa hay chưa.

Mọi sai lệch so với recipe gốc đều được liệt kê trong
`reports/bao-cao-tong-hop.md` (phần TRM và HRM). Chỉ có `adam_atan2_torch.py` là vendor lại (thay cho
package `adam-atan2` cần nvcc, không có wheel cho B200), đã kiểm khớp
reference float64 tới 2e-7.

---

## 4. Chạy

Tất cả run đều đi qua Modal, không có codepath GPU local.

```bash
pip install modal
modal setup

# Sudoku-Extreme
modal run experiments/sudoku/run.py

# Snowflake
modal run experiments/snowflake/run.py

# Scaled sudoku
modal run experiments/scaled/run.py
modal run experiments/scaled_trm/run.py
modal run experiments/scaled_hrm/run.py
```

Yêu cầu: đặt `PYTHONPATH=src`, vì `lattice_diffusion` nằm ở `src/` chứ
không ở thư mục gốc.

Mọi script chạy có tiền đều phải qua smoke T4 trước, và smoke phải chạm
tới đúng nhánh code của run thật — xem `reports/bao-cao-tong-hop.md` (mục TRM vs LDT, §7).

---

---
