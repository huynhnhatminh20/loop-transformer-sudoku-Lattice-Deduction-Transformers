# `experiments/oneoff/` — script chạy một lần

Thư mục này để riêng các script chỉ dùng một lần. Chúng không nằm trong
luồng train/eval chính, nhưng vẫn giữ lại để xem lại cách đo và cách quyết
định trong báo cáo.

## Các file

| File | Việc nó làm |
|---|---|
| `scaled_smoke.py` | Chạy thử nhanh trên tập nhỏ (80 câu train + 20 câu val) |
| `scaled_b200_smoke.py` | Chạy thử 50 bước trên GPU B200 |
| `scaled_pilot.py` | Chạy thử 500 bước trên tập train |
| `balance_tables.py` | In 3 bảng so sánh ngân sách huấn luyện |
| `hrm_t4_smoke.py` | Chạy thử HRM trên GPU T4 |
| `trm_t4_smoke.py` | Chạy thử TRM trên GPU T4 |
| `eval_sweep.py` | Đo điểm theo số bước suy luận |
| `learning_curve.py` | Vẽ đường học của TRM |

## Cách chạy

Mở terminal ở **thư mục gốc của repo** rồi chạy:

```bash
modal run experiments/oneoff/scaled_smoke.py
modal run experiments/oneoff/scaled_b200_smoke.py
modal run experiments/oneoff/scaled_pilot.py
modal run experiments/oneoff/hrm_t4_smoke.py
modal run experiments/oneoff/trm_t4_smoke.py
modal run --detach experiments/oneoff/eval_sweep.py
modal run --detach experiments/oneoff/learning_curve.py
```

Chạy `balance_tables.py` thì chọn bảng bằng biến môi trường:

```bash
BT_TABLE=1 BT_SEEDS="0 1 2" modal run -m experiments.oneoff.balance_tables::ldt_entry
```

## Lưu ý

- Phải chạy từ thư mục gốc, vì các script đọc `scaled_sudoku_smoke.jsonl`,
  `scaled_sudoku_split.jsonl` và thư mục `upstream/` ở đó.
- Code train/eval chính vẫn nằm trong `experiments/scaled/`,
  `experiments/scaled_trm/` và `experiments/scaled_hrm/`.
