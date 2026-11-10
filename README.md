# Lattice Deduction Transformers — demo, cách chơi và kết quả

Repo gồm ba phần:

1. **Demo Gradio** so sánh LDT / HRM / TRM trên Sudoku 9×9 và LDT trên
   Snowflake Sudoku. Đây là phần chạy được ngay, không cần huấn luyện.
2. **Tái lập LDT** (Lattice Deduction Transformers) trên Sudoku-Extreme và
   Snowflake Sudoku.
3. **Đóng góp của đồ án**: benchmark `scaled sudoku` (102.456 puzzle 9×9) và
   hai baseline TRM/HRM chạy cùng harness để so sánh compute-matched với LDT.

Phần (2) dùng mã của nhóm tác giả gốc, giấy phép MIT — xem
[`LICENSE`](LICENSE). Phần (1) và (3) là phần bổ sung cho đồ án.

> **Chạy nhanh:** cài `torch gradio numpy`, đặt 4 checkpoint vào `checkpoints/`,
> rồi chạy `python gradio_demo/app.py` và mở http://127.0.0.1:7860.

---

## 1. Chạy demo 

### 1.1. Cài đặt

Yêu cầu: Python 3.10+, `torch`, `gradio`, `numpy`.

```bash
# Đứng ở thư mục gốc của repo
python -m venv .venv

# Windows (cmd)
.venv\Scripts\activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1
# Linux / macOS
# source .venv/bin/activate

pip install --upgrade pip
pip install torch gradio numpy
```

Nếu máy có GPU NVIDIA, cài bản `torch` CUDA tương ứng với driver. Demo tự
chọn CUDA khi có, nếu không sẽ chạy CPU.

### 1.2. Khởi chạy

```bash
# Cách chuẩn, chạy từ thư mục gốc repo
python gradio_demo/app.py
```

Trên Windows có thể nhấp đúp hoặc chạy:

```bat
gradio_demo\run_demo.bat
```

Sau đó mở **http://127.0.0.1:7860**.

Một số cách chạy khác:

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

| Tham số         | Mặc định   | Ý nghĩa                                                  |
| ---------------- | ------------- | ---------------------------------------------------------- |
| `--model`      | `ldt`       | Model tự nạp khi khởi động: `ldt`, `hrm`, `trm` |
| `--checkpoint` | `None`      | Ghi đè checkpoint mặc định của model                 |
| `--device`     | `auto`      | Thiết bị tính toán: `auto`, `cuda`, `cpu`        |
| `--port`       | `7860`      | Cổng mạng của Gradio Web Server                         |
| `--host`       | `127.0.0.1` | Địa chỉ IP máy chủ (`0.0.0.0` để mở cho LAN)     |
| `--share`      | `False`     | Tạo liên kết Gradio public qua Internet                 |

### 1.3. Checkpoint cho demo

Checkpoint không được commit vì dung lượng. Đặt đúng 4 file sau vào thư mục
`checkpoints/` ở thư mục gốc repo:

| Model                | File local                            | Dùng cho            | Kết quả tham khảo                      |
| -------------------- | ------------------------------------- | -------------------- | ----------------------------------------- |
| LDT                  | `checkpoints/ldt_scaled_sudoku.pt`  | Sudoku 9×9          | 99,94% (Scaled Sudoku)                    |
| HRM                  | `checkpoints/hrm_scaled_sudoku.pt`  | Sudoku 9×9          | 34,50% (Scaled Sudoku)                    |
| TRM                  | `checkpoints/trm_sudoku_extreme.pt` | Sudoku 9×9          | ~75–87% (Sudoku-Extreme, bản attention) |
| ❄️ Snowflake (LDT) | `checkpoints/snowflake_sudoku.pt`   | Snowflake 6 chữ số | 100% (1.000/1.000)                        |

Nếu file chưa có và môi trường đã cấu hình sẵn script tải của repo:

```bash
python -m gradio_demo.fetch_checkpoints --all
# hoặc chỉ một model
python -m gradio_demo.fetch_checkpoints --model hrm
```

Script kiểm tra file sau khi tải và bỏ qua file đã hợp lệ. Nếu không dùng
script, chỉ cần chép 4 file `.pt` vào `checkpoints/` với đúng tên như bảng
trên. Giao diện cũng cho phép kéo-thả file `.pt` để ghi đè checkpoint.

### 1.4. Chạy demo trên Windows khi bị Application Control chặn

Một số máy Windows bật **Smart App Control** sẽ chặn các DLL chưa ký như
`torch\lib\torch.dll` hoặc `ujson`, khiến demo báo:

```text
OSError: [WinError 4551] An Application Control policy has blocked this file.
```

Có hai cách xử lý:

1. **Chạy trong WSL (khuyến nghị, không phải tắt bảo mật):**

   ```bat
   gradio_demo\run_demo_wsl.bat
   ```

   Script tự kiểm tra WSL, cài `torch gradio numpy` nếu còn thiếu, rồi chạy
   demo và vẫn mở được ở **http://127.0.0.1:7860** trên Windows. Có thể truyền
   tham số như bình thường, ví dụ:

   ```bat
   gradio_demo\run_demo_wsl.bat --model hrm --device cpu
   ```

   Nếu chưa có WSL: mở PowerShell quản trị và chạy
   `wsl --install -d Ubuntu-22.04`, sau đó khởi động lại máy.

2. **Tắt Smart App Control** (Windows 11): *Windows Security → App & browser
   control → Smart App Control → Off*. Cần quyền quản trị và khởi động lại.
   Chỉ nên làm nếu bạn hiểu rõ đánh đổi bảo mật.

Lưu ý: chạy trong WSL đọc file từ `/mnt/c/...` nên lần nạp checkpoint đầu tiên
chậm hơn; sau khi model đã nằm trong RAM, tốc độ giải không đổi đáng kể.

---

## 2. Hướng dẫn chơi

### 2.1. Giải Sudoku 9×9

1. Mở tab **🎯 Giải Sudoku (Interactive Solver)**.
2. Mở accordion **⚙️ Quản Lý Mô Hình & Checkpoint**.
3. Chọn **LDT**, **HRM** hoặc **TRM** ở dropdown. App tự điền đường dẫn
   checkpoint và nạp model. Khi thấy thẻ thông tin model (đường dẫn, số tham
   số, device, độ chính xác tham khảo) là model đã sẵn sàng.
4. Chọn một đề mẫu ở dropdown, hoặc dán chuỗi 81 ký tự vào ô nhập.
5. Nhấn **🔍 Chuẩn Hoá Định Dạng** nếu muốn app tự trích 81 ô hợp lệ.
6. Nhấn **🚀 GIẢI SUDOKU**.
7. Đọc kết quả ở cột bên phải:
   - 🟦 **Xanh dương**: số đề bài (givens).
   - 🟩 **Xanh lá**: số do model điền.
   - Dòng trạng thái cho biết giải xong, hết giờ hay không hợp lệ.
   - **Số vòng DPLL / ACT steps**, **Lần quay lui (Resets)**,
     **Lượt Forward**, **Thời gian**.
   - **Kiểm tra hợp lệ luật Sudoku**: kiểm tra 9 hàng, 9 cột, 9 khối 3×3
     và đối chiếu các số cho trước.
   - **Chuỗi kết quả 81 ký tự** có nút copy.

**Định dạng đề bài**

- Chuỗi 81 ký tự, hoặc ma trận 9 dòng.
- `1`–`9` là ô có sẵn; `.` hoặc `0` là ô trống.
- Có thể có khoảng trắng, xuống dòng; nút chuẩn hoá sẽ lọc ký tự hợp lệ.

Ví dụ:

```text
3..98.5...8.67...9..........3...........5..2.......41365..2.8..4..1.....8..5..94.
```

**Chọn model và tham số**

| Model         | Huấn luyện trên         | Tham số nên dùng                                                          | Ghi chú                                                             |
| ------------- | -------------------------- | ---------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| **LDT** | Scaled Sudoku              | `n_chains` 32, `max_rounds` 500, `threshold` 0,10, `temp_decide` 1,5 | Mạnh nhất trong demo; giải bằng loại trừ ứng viên + DPLL     |
| **HRM** | Scaled Sudoku              | `n_chains` 8 (số view bỏ phiếu), `act_steps` 16                       | So sánh baseline; tăng `act_steps` quá 16 không có tác dụng |
| **TRM** | Sudoku-Extreme (repo gốc) | `n_chains` 8, `act_steps` 16                                             | Checkpoint gốc, không huấn luyện lại trên scaled sudoku        |

- Với **LDT**, tăng `max_rounds` nếu gặp timeout; giảm `threshold` nếu model
  loại quá tay; `temp_decide` điều khiển độ ngẫu nhiên khi phân nhánh.
- Với **HRM / TRM**, `n_chains` là số view augmentation (hoán vị chữ số +
  đối xứng D4) đem bỏ phiếu. Nên dùng 8 view; tăng lên không cải thiện đáng kể.
- Nút **🧪 Tạo Model LDT Thử Nghiệm** cho phép mở giao diện khi chưa có
  checkpoint LDT. Model này chưa huấn luyện, chỉ để thử UI.
- Nếu chạy CPU, HRM/TRM sẽ chậm hơn nhiều; nên giảm `n_chains` hoặc dùng GPU.

### 2.2. Giải Snowflake Sudoku

Snowflake nằm ở tab riêng **❄️ Snowflake Sudoku (LDT)** vì bàn cờ hexagonal
6 chữ số có cách vẽ và cách mã hoá khác Sudoku 9×9.

1. Chuyển sang tab **❄️ Snowflake Sudoku (LDT)**.
2. Chọn puzzle mẫu `n = 4…8` hexagons. App tự nạp
   `checkpoints/snowflake_sudoku.pt` khi khởi động.
3. Chuỗi nhập có đúng `n × 6` ô, mỗi ký tự là một ô tam giác:
   - `1`–`6`: clue.
   - `.` hoặc `0`: ô trống.
4. Có thể chỉnh tham số trong accordion **🛠️ Tham Số Suy Luận Snowflake**:
   `n_chains` (mặc định 64), `max_rounds` (1000), `threshold` (0,10),
   `temp_decide` (1,5), `cls_threshold` (0,60), augmentation.
5. Nhấn **❄️ GIẢI SNOWFLAKE**.
6. Đọc bảng kết quả, chuỗi nghiệm và phần kiểm tra ràng buộc:
   mỗi hexagon và mỗi điểm giao của 6 tam giác phải chứa đủ 6 chữ số
   1–6 không trùng.

Lưu ý: Snowflake chỉ dùng augmentation hoán vị chữ số, không dùng D4, để giữ
topology hexagonal. `cls_threshold` là ngưỡng của conflict head; tăng lên nếu
muốn ít quay lui hơn, giảm xuống nếu muốn nhạy hơn với xung đột.

### 2.3. Lỗi thường gặp

| Hiện tượng                                                   | Cách xử lý                                                                                                                  |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `Không tìm thấy checkpoint`                                | Đặt đúng file vào `checkpoints/` hoặc nhập đường dẫn khác                                                        |
| `Không import được gradio`                                | `pip install gradio`                                                                                                         |
| Windows báo `DLL load failed` khi import `ujson`/`torch` | Chính sách Application Control đang chặn DLL; chạy `gradio_demo\run_demo_wsl.bat` hoặc xem mục 1.4 |
| CUDA không dùng được                                       | Chạy `--device cpu`; kiểm tra lại bản `torch` CUDA                                                                     |
| Giải quá lâu / timeout                                       | Tăng `max_rounds`, tăng `n_chains`, hoặc dùng GPU                                                                      |
| Cổng 7860 đang bận                                           | `python gradio_demo/app.py --port 8080`                                                                                      |
| HRM/TRM quá chậm trên CPU                                    | Giảm `n_chains` xuống 1–2 hoặc dùng GPU                                                                                 |

---

## 3. Kết quả

Các số đo huấn luyện/eval dưới đây là số đo thật trên GPU B200, bf16, seed 0
(trừ khi ghi khác). Audit dataset chạy trên CPU. Test chỉ được dùng sau khi
huấn luyện xong, không dùng để chọn model.

### 3.1. Tóm tắt

| Hạng mục                                | Kết quả                                                                  |
| ----------------------------------------- | -------------------------------------------------------------------------- |
| Sudoku-Extreme — LDT                     | **3.995 / 4.000** đúng, 0 sai, 5 timeout                           |
| Snowflake Sudoku — LDT                   | **1.000 / 1.000** đúng, 0 sai, 0 timeout                           |
| Scaled Sudoku — LDT                      | **99,94%** test (10.233 / 10.239)                                    |
| Scaled Sudoku — TRM                      | **30,45%** test (3.118 / 10.239)                                     |
| Scaled Sudoku — HRM                      | **34,50%** test (3.532 / 10.239)                                     |
| Puzzle gốc khó nhất (base, 1.000 câu) | LDT **996**, TRM **0**, HRM **6**                        |
| Killer Sudoku — LDT zero-shot            | **0 / 56**                                                           |
| Killer Sudoku — LDT fine-tune            | **0 / 144** validation, **0 / 46** test (đánh giá dở dang) |
| Audit dataset scaled sudoku               | **PASS**, leak = 0                                                   |

### 3.2. Sudoku-Extreme — LDT

4.000 puzzle, train 4.000 bước × batch 512, đánh giá với 64 chain.

| Chỉ số                            |               Kết quả |
| ----------------------------------- | ----------------------: |
| Đúng                              | **3.995 / 4.000** |
| Sai                                 |             **0** |
| Timeout                             |             **5** |
| Thời gian train (wall clock)       |         **926 s** |
| Số vòng trung bình để giải    |                   12,85 |
| Số lần reset trung bình / puzzle |                  134,98 |
| Tổng lượt gọi model             |                   8.113 |

926 s là wall clock của cả train: compile 92,5 s, eval trong lúc train 112,2 s,
train sau compile 721,6 s.

Chẩn đoán thêm trên cùng lần eval:

- **Độ đúng của phép loại suy luận**: 639.730 suy luận không hợp lệ trên
  246.056.406 phép đã thực hiện — tỉ lệ **0,26%**.
- **Conflict head** (so với ground truth sau khi loại ứng viên):
  precision **0,9975**, recall **0,1987**
  (`tp=580.942`, `fp=1.453`, `fn=2.342.122`, `tn=935.515`).
- 0 câu sai nghĩa là không có nghiệm điền xong nào sai; 5 timeout là chưa
  giải xong trong ngân sách vòng, không phải trả lời sai.

### 3.3. Snowflake Sudoku — LDT

1.000 puzzle hexagonal, 6 chữ số, lưới phủ 15×10, train 1.000 bước × batch
512, đánh giá với 64 chain.

| Chỉ số                            |               Kết quả |
| ----------------------------------- | ----------------------: |
| Đúng                              | **1.000 / 1.000** |
| Sai                                 |             **0** |
| Timeout                             |             **0** |
| Tổng lượt gọi model             |                     125 |
| Số vòng trung bình để giải    |                     0,0 |
| Số lần reset trung bình / puzzle |                     0,0 |

Chẩn đoán thêm:

- **Độ đúng của phép loại suy luận**: 460 suy luận không hợp lệ trên
  3.842.479 phép — tỉ lệ **0,012%**.
- **Conflict head**: precision **0,9118**, recall **0,6200**
  (`tp=31`, `fp=3`, `fn=19`, `tn=63.947`).
- Trên tập này solver kết thúc bằng lan truyền, không cần quay lui; vì vậy
  số vòng và số reset trung bình bằng 0.

### 3.4. Scaled Sudoku — LDT / TRM / HRM

**Dữ liệu.** 102.456 puzzle 9×9: 10.000 puzzle tối thiểu (base) + 92.456
biến thể (variant), thuộc 10.000 ancestor group. Chia theo ancestor, seed 42:

| Split      | Ancestor |  Tổng |  Base | Variant |
| ---------- | -------: | -----: | ----: | ------: |
| Train      |    8.000 | 82.043 | 8.000 |  74.043 |
| Validation |    1.000 | 10.174 | 1.000 |   9.174 |
| Test       |    1.000 | 10.239 | 1.000 |   9.239 |

- Leakage: **0** (cross-split ancestor = 0, cross-split puzzle = 0).
- Clue trung bình: 27,70 / 27,68 / 27,71 (train / val / test).
- Hardness: khoảng 98% mẫu thuộc nhóm `Val` hoặc `ValBS`; các nhóm hiếm chỉ
  có 2–105 mẫu.
- Kiểm tra toàn bộ 102.456 instance: 0 lỗi định dạng, 0 sai `clues_count`,
  0 clue lệch nghiệm, 0 nghiệm không hợp lệ, 0 puzzle trùng, 0 orphan
  variant, 0 ancestor bị gán nhiều split.

**Bảng chính.** Cả ba model nhìn khoảng 2,05 triệu example (compute-matched).

|                         |               LDT |          TRM |              HRM |
| ----------------------- | ----------------: | -----------: | ---------------: |
| Tham số                | **799.635** |    5.027.842 |       27.274.242 |
| Steps × batch          |      4.000 × 512 | 2.667 × 768 |     2.667 × 768 |
| Examples nhìn thấy    |         2.048.000 |    2.048.256 |        2.048.256 |
| Validation (10.174)     |  **99,96%** |       31,02% |           34,72% |
| **Test (10.239)** |  **99,94%** |       30,45% | **34,50%** |
| Greedy 1 đường       |                — |       17,86% |           18,14% |
| Câu trả lời sai      |       **0** |           — |               — |

- **Greedy 1 đường** = một vòng suy luận, lấy argmax — cách chấm của paper.
- **Vote 8 views** = 1 bản gốc + 7 bản biến thể, giải riêng rồi bầu đa số,
  dùng để cân test-time với 8 chain của LDT.
- LDT `sai = 0`: vài câu timeout là chưa giải xong trong 50 vòng, không phải
  trả lời sai.

**Base khó vs variant dễ (test).** Đây là phân tách quan trọng nhất.

| Nhóm               |                   LDT |                 TRM |                 HRM |
| ------------------- | --------------------: | ------------------: | ------------------: |
| Base (gốc minimal) | **996 / 1.000** | **0 / 1.000** | **6 / 1.000** |
| Variant             |         9.237 / 9.239 |       3.118 / 9.239 |       3.526 / 9.239 |

LDT giải được puzzle gốc minimal; hai model recursive gần như không giải được
câu nào. Khoảng cách giữa ba model **không** nằm ở nhóm biến thể dễ.

**Theo số gợi ý (test).**

| Số gợi ý |     n |              LDT |   TRM |   HRM |
| ----------- | ----: | ---------------: | ----: | ----: |
| ≤ 23       |   311 | **100,0%** |  0,0% |  0,0% |
| 24–26      | 4.527 |  **99,9%** |  0,9% |  3,3% |
| 27–29      | 2.500 | **100,0%** | 27,9% | 37,8% |
| 30–32      | 1.929 | **100,0%** | 76,1% | 79,8% |
| 33–35      |   775 | **100,0%** | 92,4% | 91,7% |
| 36–38      |   179 | **100,0%** | 98,9% | 95,5% |
| ≥ 39       |    18 | **100,0%** | 94,4% | 94,4% |

Đọc nhanh: dưới 24 gợi ý, TRM và HRM bằng 0%, LDT bằng 100%; trên 30 gợi ý,
cả ba gần như ngang nhau. HRM thắng TRM ở vùng 24–32 gợi ý (vùng khó) và thua
nhẹ ở vùng 33–38 (vùng dễ), nên phần thắng tổng của HRM đến từ vùng khó.

**Theo độ khó propagation (test).** Nhãn là phương pháp suy luận yếu nhất giải
được câu đó mà không cần tìm kiếm; `Val` yếu nhất, thêm `BS`/`DS`/`Chan` là
mạnh dần.

| Nhóm (yếu → mạnh) |     n |              LDT |            TRM |            HRM |
| --------------------- | ----: | ---------------: | -------------: | -------------: |
| `Val`               | 5.000 | **100,0%** |          61,5% |          68,1% |
| `ValBS`             | 5.004 |  **99,9%** | **0,8%** | **2,5%** |
| `ValDS`             |   100 |  **99,0%** |           1,0% |           1,0% |
| `ValChan`           |    22 | **100,0%** |           0,0% |           0,0% |
| `Bnd`               |     2 | **100,0%** |           0,0% |           0,0% |
| `BndBS`             |   105 |  **99,0%** |           0,0% |           0,0% |
| `Dom`               |     2 | **100,0%** |           0,0% |          50,0% |
| `DomBS`             |     4 | **100,0%** |           0,0% |           0,0% |

Phát hiện chính: LDT giữ ~100% ở mọi nhóm. TRM và HRM giải được nhóm `Val`
(62–68%) nhưng **sụp xuống 0,8–2,5% ngay khi cần `ValBS`**, rồi gần 0% ở các
nhóm còn lại. Nghĩa là hai model recursive học được bước lan truyền giá trị
nhưng không lập được chuỗi lan truyền nhiều tầng. Các nhóm sau `ValBS` chỉ có
2–105 mẫu nên không kết luận gì từ 1–2 mẫu; riêng `Dom` HRM được 50% nhưng
chỉ có n = 2.

**Độ đúng theo ngân sách huấn luyện (test).**

|  Examples |              LDT |              TRM |              HRM |
| --------: | ---------------: | ---------------: | ---------------: |
|   512.256 |               — |  **0,00%** |               — |
| 1.023.744 |               — |  **0,00%** |               — |
| 2.048.000 | **99,94%** |               — |               — |
| 2.048.256 |               — | **30,45%** | **34,50%** |

Ở **một nửa** ngân sách (1.023.744 examples), TRM đạt **0,00%** trên cả
10.239 câu test, dù loss đã giảm. Đường cong là **vực thẳm rồi nhảy**, không
dốc đều: toàn bộ 30,45% xuất hiện trong khoảng 1,02M–2,05M examples. Điểm
2.667 bước nằm ngay sau bậc nhảy.

**Tài nguyên và chi phí (B200).**

|                            |               LDT |             TRM |                HRM |
| -------------------------- | ----------------: | --------------: | -----------------: |
| Train time                 |           1.238 s |           360 s |              251 s |
| Eval time                  |              31 s |           243 s |              103 s |
| **Tổng wall clock** | **1.278 s** | **603 s** |    **368 s** |
| Peak VRAM                  |          26,29 GB |        31,93 GB | **11,62 GB** |

Nghịch lý đáng nói: HRM lớn gấp 34 lần LDT nhưng nhanh hơn 3,5 lần và nhẹ
hơn 2,3 lần. Lý do: LDT chạy solver 50 vòng tuần tự mỗi câu, còn HRM học được
câu dễ nhanh và halt sớm.

**Kiểm tra confound test-time compute.** Tăng số vòng suy luận của TRM trên
2.000 câu test con, không retrain:

| `halt_max_steps` | calls / câu |     Vote 8 views |    Base |
| -----------------: | -----------: | ---------------: | ------: |
|                 16 |          128 |           30,20% | 0 / 205 |
|                 32 |          256 |           30,35% | 0 / 205 |
|                 64 |          512 |           29,45% | 0 / 205 |
|                128 |        1.024 | **30,70%** | 1 / 205 |

Tăng suy luận 8× không cải thiện (biến động ±0,8 điểm, trong nhiễu). Trong khi
đó LDT ở ngân sách nhỏ hơn nhiều:

| `n_chains` |              Vote |      Base | Timeout | Eval time |
| -----------: | ----------------: | --------: | ------: | --------: |
|            1 |            99,85% | 203 / 205 |       3 |     5,2 s |
|            2 |            99,95% | 205 / 205 |       1 |     3,9 s |
|            4 | **100,00%** | 205 / 205 |       0 |     3,5 s |
|            8 |            99,95% | 204 / 205 |       1 |     3,3 s |

LDT đạt 99,85% chỉ với 1 nhánh; 4 nhánh đạt 100%. Bất đối xứng này mạnh hơn
con số 99,94%.

**Quá trình học.**

- LDT: loss 0,81 (step 100) → 0,26 (500) → 0,14 (1.000) → 0,10 (2.000) →
  0,069 (3.000) → **0,060** (4.000); không có NaN/Inf.
- TRM: loss 2,48 → **0,41**; xuống dưới mức marginal `ln(9) ≈ 2,197`.
- HRM: loss 2,43 → **0,89**; có hiện tượng tăng lại từ 0,58 (step 1.500) lên
  0,97 (step 2.200), rồi đi ngang ~0,82–0,89, không về lại đáy.
- Không so độ cao loss giữa ba model vì ba hàm loss khác nhau; chỉ so xu hướng
  giảm trong từng cột.

**Cách đọc kết quả cho đúng.**

> Ở ngân sách compute-matched 2,05 triệu examples, LDT đạt 99,94%
> (996/1.000 base) trong khi TRM và HRM đạt 30,45% và 34,50%, gần như 0 trên
> 1.000 puzzle khó nhất. Khoảng cách này là **thật** và không phải artifact
> chấm điểm. Tuy nhiên TRM/HRM **chưa hội tụ** tại điểm này, nên bảng **không**
> dùng để xếp hạng method hay kết luận về data efficiency.

Các giới hạn phải đọc kèm:

1. **Một seed duy nhất** (seed 0), không có ước lượng sai số.
2. **Compute-matched theo examples, không theo FLOPs.** HRM 27,3M tham số và
   ~512 layer-pass mỗi bước suy luận; TRM 5,0M và 36; LDT là solver 50 vòng.
   Số examples bằng nhau không phải FLOPs bằng nhau.
3. **Per-puzzle embedding 82k hàng**, mỗi puzzle gặp ~25 lần. Paper gốc chỉ có
   1.000 puzzle nên bảng này học kỹ hơn nhiều — đây là điểm TRM/HRM bị thiệt ở
   quy mô dữ liệu lớn.
4. **Hai learning rate khác nhau theo upstream** (TRM 1e-4, HRM 7e-5) và HRM
   dùng batch 768, gấp đôi lệnh Sudoku gốc của họ (384). Không ai được tune lại
   cho run 2.667 bước.
5. **LDT dùng solver, không dùng model thuần** ở phần 50 vòng. Đây là so sánh
   hệ thống, không thuần là so sánh kiến trúc mạng.
6. **`halt_max_steps = 128` nằm ngoài phạm vi huấn luyện** (ACT head học dừng
   trong 16 bước). Kết luận đúng ở mức "ở ngân sách mà paper cho phép, TRM
   đứng yên", không phải "TRM không hưởng lợi từ suy luận sâu" tuyệt đối.

### 3.5. Killer Sudoku — giới hạn của LDT

LDT huấn luyện trên Sudoku-Extreme, đem zero-shot sang Killer Sudoku:

- Chạy được 56/144 câu thì dừng; **đúng 0 câu**.
- Killer không cho số sẵn, nên nếu không đưa thông tin lồng/tổng, 144 câu
  trông giống nhau (toàn lưới trắng) — model không có cách phân biệt.

Mở rộng input để đọc lồng/tổng và fine-tune:

- **Smoke 50 bước**: PASS, loss 0,7459 → 0,6819, lưu/tải checkpoint được.
- **Full 2.000 bước** trên 672 câu: loss 0,82 → 0,38; vẫn cao hơn Sudoku
  thường (~0,27).
- **Validation 144 câu**, kiểm tra sau mỗi 100 bước: **0/144**, lần nào cũng
  vậy.
- **Test**: chấm tới câu 46/144 thì hệ thống dừng; **0 đúng, 12 sai, 34 hết
  giờ**. Không có điểm chính thức vì còn 98 câu chưa chấm.

Kết luận: chỉ thêm thông tin lồng/tổng vào input là **chưa đủ**; cần thay đổi
kiến trúc (ví dụ một `Sum Operator` để suy luận trên tổng của lồng), không chỉ
mở rộng đầu vào.

### 3.6. Audit dataset scaled sudoku

- Tổng: 434.201 instance ở mọi kích thước; lọc 9×9 còn **102.456**
  (10.000 base + 92.456 variant).
- 10.000 ancestor group; mỗi ancestor có 4–16 instance (trung bình 10,25).
- Kiểm tra từng instance: 0 lỗi format, 0 sai `clues_count`, 0 clue lệch
  nghiệm, 0 nghiệm không hợp lệ.
- 0 puzzle trùng hoàn toàn; 0 orphan variant; 0 instance gán nhiều ancestor;
  0 dangling reference trong index của base.
- Split theo ancestor (seed 42), `leakage = 0`.
- Kết luận audit: **PASS**, dataset sẵn sàng cho train/eval.

---

## 4. Cấu trúc

```text
src/lattice_diffusion/
  models/looped_transformer.py   PowersetModel — kiến trúc lõi
  models/transformer2d.py        Transformer 2D
  models/weighted_bce.py         Weighted BCE
  data/sudoku_extreme.py         Loader Sudoku-Extreme
  data/scaled_sudoku.py          Loader scaled sudoku
  training/utils/                Checkpoint, scheduler

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
  app.py                         Giao diện chính
  registry.py                    Model → checkpoint mặc định
  solver.py                      Nạp checkpoint và suy luận
  snowflake.py                   Mã hoá, kiểm tra, vẽ bàn cờ Snowflake
  examples.py                    Đề Sudoku mẫu
  fetch_checkpoints.py           Tải checkpoint còn thiếu
```

`checkpoints/`, `logs/` và `scaled-sudoku-instances/` không được commit.
`scaled_sudoku_smoke.jsonl` (49 KB) có trong repo;
`scaled_sudoku_split.jsonl` (57 MB) không commit.

---

## 5. Huấn luyện / tái lập

Demo chỉ cần checkpoint; **không cần huấn luyện**. Mã train/eval nằm trong
`experiments/`:

| Thư mục                   | Việc nó làm                        |
| --------------------------- | ------------------------------------- |
| `experiments/sudoku/`     | Train + eval LDT trên Sudoku-Extreme |
| `experiments/snowflake/`  | Sinh dữ liệu, train, eval Snowflake |
| `experiments/killer/`     | Zero-shot và fine-tune Killer Sudoku |
| `experiments/scaled/`     | LDT trên scaled sudoku               |
| `experiments/scaled_trm/` | Baseline TRM, compute-matched         |
| `experiments/scaled_hrm/` | Baseline HRM, compute-matched         |
| `experiments/oneoff/`     | Smoke, pilot, sweep, learning curve   |

Các script này được viết cho GPU rời (CUDA) và runner riêng, không có codepath
GPU local trong repo. Khi chạy lại, cần đặt `PYTHONPATH=src` và xem docstring
đầu mỗi file để biết tham số, checkpoint path và cách chọn split. Kết quả đầy
đủ của các lần chạy đã có ở mục 3.

---

## 6. Ghi công

- LDT: Liam Davis, Leopold Haller, Alberto Alfarano, Mark Santolucito.
  [arXiv:2605.08605](https://arxiv.org/abs/2605.08605). MIT.
- TRM: `upstream/TinyRecursiveModels/`, commit `c01103738605`.
- HRM: `upstream/HRM/`, commit `ac15626f8db0`.

`upstream/` giữ nguyên, kiểm bằng SHA-256 từng file: TRM 40/40, HRM 26/26
khớp. Chỉ `experiments/scaled_trm/adam_atan2_torch.py` là vendor lại (thay
package `adam-atan2` cần nvcc, không có wheel cho B200), đã kiểm khớp
reference float64 tới 2e-7.
