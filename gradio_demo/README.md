# 🧩 Gradio Web Demo — LDT / HRM / TRM Sudoku Solver + Snowflake

Thư mục này cung cấp giao diện Web trực quan (viết bằng [Gradio](https://www.gradio.app/)) để so sánh các mô hình giải Sudoku:

- **LDT** — Lattice Deduction Transformers (arXiv:2605.08605), giải bằng vòng DPLL.
- **HRM** — Hierarchical Reasoning Model, suy luận lặp với cơ chế ACT tự dừng.
- **TRM** — Tiny Recursive Model, checkpoint lấy từ repo gốc **TinyRecursiveModels** (SamsungSAILMontreal).
- **❄️ Snowflake Sudoku (LDT)** — bàn cờ hexagonal 6 chữ số, huấn luyện riêng và đạt **1.000/1.000** trên test. Snowflake nằm ở **tab riêng** vì lưới phủ `15×10` và cách vẽ tam giác khác Sudoku 9×9.

Điểm quan trọng: **giao diện tự nạp checkpoint mặc định cho từng mô hình**. Khi chọn model trong dropdown, đường dẫn checkpoint được điền sẵn và model được nạp ngay — không cần dò file thủ công.

---

## 🚀 Các Tính Năng Chính

1. **Tự động nạp checkpoint theo mô hình**:
   - Chọn **LDT / HRM / TRM** ở dropdown → app tự nạp đúng checkpoint trong `checkpoints/`.
   - Tab **❄️ Snowflake Sudoku (LDT)** tự nạp `checkpoints/snowflake_sudoku.pt` khi khởi động.
   - Vẫn có thể ghi đè bằng cách nhập đường dẫn hoặc kéo-thả file `.pt`.
   - Nạp tự động trọng số EMA (`ema_state_dict`) nếu checkpoint LDT có EMA.
   - Nút **Tạo Model LDT Thử Nghiệm (Scratch Demo)** cho phép trải nghiệm UI khi chưa có checkpoint LDT.
   - Hỗ trợ chọn thiết bị: `auto` (tự nhận GPU CUDA), `cuda`, hoặc `cpu`.

2. **Giao diện bảng Sudoku trực quan**:
   - Bàn cờ 9x9 với đường phân cách khối 3x3.
   - 🟦 Số đề bài (givens) và 🟩 số do mô hình giải.
   - Snowflake: bàn cờ SVG hexagonal, mỗi hexagon chia 6 ô tam giác, cùng quy ước màu.
   - Cập nhật thời gian thực khi sửa chuỗi đề bài.

3. **Bộ đề mẫu phong phú**:
   - Các câu đố từ `scaled-sudoku` và benchmark chuẩn: Easy, Medium, Hard, Extreme, Near-Minimal (22 clues).
   - Snowflake: 10 puzzle mẫu cho `n = 4…8` hexagons, sinh bằng solver CVC5 của LDT.

4. **Tham số suy luận theo từng họ mô hình**:
   - **LDT**: `n_chains`, `max_rounds`, `threshold`, `temp_decide`, `augment`.
   - **HRM / TRM**: số view augmentation đem bỏ phiếu (`n_chains`) và `act_steps` (số bước ACT tối đa).
   - **Snowflake (LDT)**: `n_chains`, `max_rounds`, `threshold`, `temp_decide`, `cls_threshold`, `augment` (chỉ hoán vị chữ số).
   - UI tự ẩn/hiện nhóm tham số phù hợp khi đổi model.

5. **Kiểm tra hợp lệ & thống kê chi tiết**:
   - Sudoku: kiểm tra 9 hàng, 9 cột, 9 khối 3x3 và đối chiếu các số cho trước.
   - Snowflake: kiểm tra givens và mọi ràng buộc hexagon / điểm giao 6 tam giác.
   - Thống kê: số vòng/bước, số lần quay lui (LDT), số forward, thời gian giải.

---

## 📦 Checkpoint mặc định

| Model | File local | Nguồn | Độ chính xác tham khảo |
|---|---|---|---|
| LDT | `checkpoints/ldt_scaled_sudoku.pt` | Checkpoint nội bộ của đồ án — LDT, 4.000 bước × batch 512 | 99,94% (Scaled Sudoku) |
| HRM | `checkpoints/hrm_scaled_sudoku.pt` | Checkpoint nội bộ của đồ án — HRM, 2.667 bước × batch 768 | 34,50% (Scaled Sudoku) |
| TRM | `checkpoints/trm_sudoku_extreme.pt` | Checkpoint của code gốc TinyRecursiveModels — variant attention, 60k epochs, step 52080 | ~75–87% (Sudoku-Extreme, variant attention) |
| ❄️ Snowflake (LDT) | `checkpoints/snowflake_sudoku.pt` | Checkpoint nội bộ của đồ án — LDT, 1.000 bước × batch 512, dữ liệu CVC5 | 100% (Snowflake test, 1.000/1.000) |

Thư mục `checkpoints/` đã nằm trong `.gitignore`.

### Chuẩn bị checkpoint

Nếu môi trường đã cấu hình sẵn script tải và file chưa có (hoặc bị hỏng), chạy:

```bash
# Tải tất cả checkpoint còn thiếu
python -m gradio_demo.fetch_checkpoints --all

# Chỉ tải một model
python -m gradio_demo.fetch_checkpoints --model hrm

# Tải lại dù file đã tồn tại
python -m gradio_demo.fetch_checkpoints --model trm --force
```

Script tự chọn đúng nguồn cho từng checkpoint, kiểm tra file zip/torch sau khi tải, và bỏ qua file đã hợp lệ. Trên Windows, script dùng concurrency 1 vì tải song song có thể làm hỏng file lớn. Nếu không dùng script, chỉ cần chép các file `.pt` vào `checkpoints/` với đúng tên trong bảng trên.

---

## 📁 Cấu Trúc Thư Mục `gradio_demo/`

```text
gradio_demo/
├── __init__.py          # Khởi tạo package
├── app.py               # Gradio Web UI chính
├── registry.py          # Bảng model → checkpoint + metadata nguồn
├── solver.py            # Nạp checkpoint và suy luận LDT / HRM / TRM / Snowflake
├── snowflake.py         # Mã hoá, kiểm tra và vẽ bàn cờ Snowflake (SVG)
├── snowflake_examples.json # 10 puzzle Snowflake mẫu (n = 4…8)
├── fetch_checkpoints.py # Tải checkpoint còn thiếu về checkpoints/
├── run_demo.bat         # Launcher Windows (tự fallback sang WSL)
├── run_demo_wsl.bat     # Launcher dự phòng chạy trong WSL
├── utils.py             # Xử lý chuỗi đề bài, vẽ bảng HTML, kiểm tra luật
├── examples.py          # Câu đố Sudoku mẫu kèm nghiệm chuẩn
└── README.md            # Tài liệu này
```

---

## 🛠️ Hướng Dẫn Sử Dụng

### 1. Cài đặt thư viện

```bash
pip install gradio torch numpy
```

### 2. Khởi chạy ứng dụng Web

```bash
# Mặc định nạp LDT
python gradio_demo/app.py

# Chọn model khác ngay khi khởi động
python gradio_demo/app.py --model hrm --device cuda
python gradio_demo/app.py --model trm

# Ghi đè checkpoint mặc định
python gradio_demo/app.py --model ldt --checkpoint /duong/dan/khac.pt

# Đổi cổng hoặc tạo link public
python gradio_demo/app.py --port 8080 --share
```

Sau đó mở `http://127.0.0.1:7860`.

Trên Windows có thể dùng launcher:

```bat
gradio_demo\run_demo.bat
```

Launcher thử Python Windows trước; nếu DLL bị chặn (`WinError 4551`,
Application Control / Smart App Control) hoặc thiếu gói, nó tự chuyển sang WSL.
Có thể gọi thẳng bản WSL:

```bat
gradio_demo\run_demo_wsl.bat --model ldt
```

Script WSL tự kiểm tra/cài `torch gradio numpy`, chạy demo trong Ubuntu và vẫn
phục vụ ở `http://127.0.0.1:7860` cho trình duyệt Windows.

### 3. Quy trình trên giao diện

1. Mở accordion **⚙️ Quản Lý Mô Hình & Checkpoint**.
2. Chọn model — app tự điền checkpoint và nạp luôn.
3. Chọn đề mẫu hoặc nhập chuỗi 81 ký tự.
4. Nhấn **🚀 GIẢI SUDOKU**.
5. Xem bảng kết quả, trạng thái hợp lệ và thống kê.

**Snowflake:** chuyển sang tab **❄️ Snowflake Sudoku (LDT)**, chọn puzzle mẫu
`n = 4…8`, rồi nhấn **❄️ GIẢI SNOWFLAKE**. Chuỗi nhập có đúng `n×6` ô
(`1`–`6` = clue, `.`/`0` = trống). Tab tự nạp checkpoint Snowflake khi khởi động.

---

## 📋 Các Tuỳ Chọn Dòng Lệnh (CLI Arguments)

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--model` | `ldt` | Model tự nạp khi khởi động: `ldt`, `hrm`, `trm` |
| `--checkpoint` | `None` | Ghi đè checkpoint mặc định của model |
| `--device` | `auto` | Thiết bị tính toán: `auto`, `cuda`, `cpu` |
| `--port` | `7860` | Cổng mạng của Gradio Web Server |
| `--host` | `127.0.0.1` | Địa chỉ IP máy chủ (`0.0.0.0` để mở cho LAN) |
| `--share` | `False` | Tạo liên kết Gradio public qua Internet |

---

## 🧠 Cách suy luận của từng mô hình

- **LDT**: mã hoá mỗi ô thành vector 9 ứng viên (`[81, 9]`), chạy transformer 2D lặp `n_loops` vòng, rồi dùng DPLL (loại trừ theo `threshold`, phân nhánh theo `temp_decide`, reset khi xung đột). `n_chains` là số chuỗi song song.
- **HRM / TRM**: chạy vòng ACT tối đa `halt_max_steps` bước. Demo tạo thêm các view augmentation (hoán vị chữ số + đối xứng D4), chạy cùng lúc và bỏ phiếu đa số. `n_chains` là số view; nên dùng 8.
- **Snowflake (LDT)**: mỗi ô tam giác giữ powerset 6 ứng viên, puzzle nhúng vào lưới phủ `15×10 = 150` ô với mask `in_puzzle` (channel thứ 7). Vòng DPLL giống LDT nhưng augmentation chỉ hoán vị chữ số (không D4) để giữ topology hexagonal.

---

## 💡 Định Dạng Đề Bài Sudoku

Ứng dụng chấp nhận chuỗi 81 ký tự:

- Chữ số `1`–`9`: ô có sẵn (givens).
- Dấu chấm `.` hoặc số `0`: ô trống.
- Có thể nhập chuỗi 1 dòng hoặc ma trận 9 dòng. Nút **🔍 Chuẩn Hoá Định Dạng** tự trích xuất 81 ô hợp lệ.

**Ví dụ:**

```text
3..98.5...8.67...9..........3...........5..2.......41365..2.8..4..1.....8..5..94.
```
