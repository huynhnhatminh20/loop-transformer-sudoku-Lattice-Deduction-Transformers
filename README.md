# Lattice Deduction Transformers — reproduction + baselines

Repo gồm hai phần:

1. **Tái lập LDT** (Lattice Deduction Transformers) trên Sudoku-Extreme và
   Snowflake Sudoku.
2. **Đóng góp của đồ án**: benchmark `scaled sudoku` (102.456 puzzle 9×9) và
   hai baseline TRM/HRM chạy cùng harness để so sánh compute-matched với LDT.

Phần (1) là mã của nhóm tác giả gốc, giấy phép MIT — xem [`LICENSE`](LICENSE).
Phần (2) là phần bổ sung cho đồ án.

## Kết quả

Mọi số đo trên Modal B200, bf16, seed 0.

### Sudoku-Extreme — LDT

4.000 puzzle, train 4.000 bước × batch 512.

| Chỉ số         |     Kết quả |
| ---------------- | ------------: |
| Đúng           | 3.995 / 4.000 |
| Sai              |             0 |
| Timeout          |             5 |
| Thời gian train |         926 s |

### Snowflake Sudoku — LDT

1.000 puzzle hexagonal, 6 chữ số, lưới phủ 15×10, vocab 6.

| Chỉ số |     Kết quả |
| -------- | ------------: |
| Đúng   | 1.000 / 1.000 |
| Sai      |             0 |
| Timeout  |             0 |

### Scaled Sudoku — LDT / TRM / HRM

102.456 puzzle 9×9: 10.000 puzzle tối thiểu + 92.456 biến thể. Chia theo
ancestor group, leak = 0. Cả ba model nhìn khoảng 2,05 triệu examples.

|                      |              LDT |          TRM |          HRM |
| -------------------- | ---------------: | -----------: | -----------: |
| Tham số             |          799.635 |    5.027.842 |   27.274.242 |
| Steps × batch       |     4.000 × 512 | 2.667 × 768 | 2.667 × 768 |
| Test                 | **99,94%** |       30,45% |       34,50% |
| Base (1.000 câu)    |    **996** |            0 |            6 |
| Variant (9.239 câu) |            9.237 |        3.118 |        3.526 |

Khoảng cách giữa ba model nằm ở nhóm puzzle tối thiểu, không phải nhóm biến
thể dễ. Ở nửa ngân sách (1.023.744 examples), TRM đạt 0,00% trên cả 10.239
câu test; điểm 2.667 bước nằm ngay sau bậc nhảy. Bảng trên là phép đo ở một
ngân sách cụ thể, không phải bảng xếp hạng method.

### Killer Sudoku — giới hạn của LDT

LDT huấn luyện trên Sudoku-Extreme, zero-shot 0/56. Mở rộng input để đọc
lồng/tổng và fine-tune vẫn 0/144 validation, 0/46 test (đánh giá dở dang).
Kết luận: cần thay đổi kiến trúc, không chỉ thêm thông tin đầu vào.

## Cấu trúc

```
src/lattice_diffusion/
  models/looped_transformer.py   PowersetModel — kiến trúc lõi
  models/transformer2d.py        Transformer 2D
  models/weighted_bce.py         Weighted BCE
  data/sudoku_extreme.py         Loader Sudoku-Extreme
  data/scaled_sudoku.py          Loader scaled sudoku
  training/utils/                Checkpoint, scheduler
  modal/                         Modal image/volume

experiments/
  sudoku/                        Sudoku-Extreme
  snowflake/                     Sinh dữ liệu + train + eval Snowflake
  killer/                        Killer Sudoku
  scaled/                        LDT trên scaled sudoku
  scaled_trm/                    Baseline TRM
  scaled_hrm/                    Baseline HRM
  oneoff/                        Smoke, pilot, sweep — chạy một lần

upstream/
  TinyRecursiveModels/           Mã TRM gốc
  HRM/                           Mã HRM gốc

gradio_demo/                     Demo Gradio (9×9 và Snowflake)
```

## Chạy

Tất cả run đi qua Modal, không có codepath GPU local.

```bash
pip install modal
modal setup
export PYTHONPATH=src

modal run experiments/sudoku/run.py
modal run experiments/snowflake/run.py
modal run experiments/scaled/run.py
modal run experiments/scaled_trm/run.py
modal run experiments/scaled_hrm/run.py
```

Demo:

```bash
python gradio_demo/fetch_checkpoints.py --all   # tải checkpoint nếu chưa có
python gradio_demo/app.py
# hoặc
gradio_demo/run_demo.bat
```

## Dữ liệu và checkpoint

- `scaled_sudoku_split.jsonl`: 102.456 dòng, chứa `puzzle` + `solution`,
  dùng chung cho cả ba model. Không commit vì 57 MB; tải raw dataset từ
  `zayenz/scaled-sudoku-instances` rồi sinh lại split.
- `scaled_sudoku_smoke.jsonl`: tập smoke nhỏ (49 KB), có trong repo.
- `checkpoints/`, `logs/`: không commit (đã ignore).
- `scaled-sudoku-instances/`: raw dataset, không commit.

## Ghi công

- LDT: Liam Davis, Leopold Haller, Alberto Alfarano, Mark Santolucito.
  [arXiv:2605.08605](https://arxiv.org/abs/2605.08605). MIT.
- TRM: `upstream/TinyRecursiveModels/`, commit `c01103738605`.
- HRM: `upstream/HRM/`, commit `ac15626f8db0`.

`upstream/` giữ nguyên, kiểm bằng SHA-256 từng file: TRM 40/40, HRM 26/26
khớp. Chỉ `experiments/scaled_trm/adam_atan2_torch.py` là vendor lại (thay
package `adam-atan2` cần nvcc, không có wheel cho B200), đã kiểm khớp
reference float64 tới 2e-7.
