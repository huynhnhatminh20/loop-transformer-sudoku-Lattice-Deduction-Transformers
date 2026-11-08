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

Các script trong thư mục này được thiết kế cho runner GPU từ xa của nhóm;
repo không có codepath GPU local. Mở terminal ở **thư mục gốc của repo**,
đặt `PYTHONPATH=src`, cấu hình runner rồi chạy entry point tương ứng. Xem
docstring đầu mỗi file để biết tham số và cách chọn chế độ smoke/full.

`balance_tables.py` chọn bảng bằng biến môi trường `BT_TABLE` và `BT_SEEDS`.

## Lưu ý

- Phải chạy từ thư mục gốc, vì các script đọc `scaled_sudoku_smoke.jsonl`,
  `scaled_sudoku_split.jsonl` và thư mục `upstream/` ở đó.
- Code train/eval chính vẫn nằm trong `experiments/scaled/`,
  `experiments/scaled_trm/` và `experiments/scaled_hrm/`.
