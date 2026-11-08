"""Lattice Deduction Transformers — Gradio Web Demo for Sudoku & Snowflake.

The UI auto-loads a checkpoint for the selected model:
  LDT  -> checkpoints/ldt_scaled_sudoku.pt
  HRM  -> checkpoints/hrm_scaled_sudoku.pt
  TRM  -> checkpoints/trm_sudoku_extreme.pt

Snowflake Sudoku (hexagonal, LDT) lives in its own tab and auto-loads:
  LDT  -> checkpoints/snowflake_sudoku.pt

Fetch missing checkpoints with:
    python -m gradio_demo.fetch_checkpoints --all

Launch:
    python gradio_demo/app.py
    python gradio_demo/app.py --model hrm --device cuda
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys

# Disable Gradio and HF analytics/telemetry (prevents slow network timeouts on startup/exit)
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "True"

from pathlib import Path
from typing import Any

# Ensure stdout/stderr handle UTF-8 cleanly on Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root and src/ are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
for p in [str(REPO_ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import gradio as gr

from gradio_demo.examples import EXAMPLES, get_example_by_name
from gradio_demo.registry import (
    DEFAULT_MODEL_KEY,
    MODEL_SPECS,
    SUDOKU_MODEL_KEYS,
    choice_labels,
    get_spec,
)
from gradio_demo.snowflake import (
    EXAMPLES as SNOWFLAKE_EXAMPLES,
    get_example_by_name as get_snowflake_example_by_name,
    parse_snowflake_input,
    render_snowflake_board,
    snowflake_example_info,
)
from gradio_demo.solver import (
    LoadedModel,
    SnowflakeSolveOutput,
    SolveOutput,
    load_model,
    make_scratch_loaded_model,
    solve_for_loaded_model,
    solve_snowflake,
)
from gradio_demo.utils import (
    format_sudoku_grid,
    parse_sudoku_input,
    render_board_html,
    validate_sudoku_solution,
)

# Global model state cache
_LOADED_MODEL: LoadedModel | None = None
_LOADED_META: dict[str, Any] | None = None

# Snowflake has its own tab/model cache (variable-size hex board).
_SF_LOADED_MODEL: LoadedModel | None = None
_SF_META: dict[str, Any] | None = None

CUSTOM_CSS = """
.gradio-container { max-width: 1200px !important; margin: auto !important; }
.stat-box { text-align: center; font-weight: 600; }
"""


def format_meta_markdown(meta: dict[str, Any] | None) -> str:
    """Format model metadata into a nice Markdown card."""
    if not meta:
        return (
            "> ⚠️ **Chưa nạp checkpoint.**\n"
            "> Chọn mô hình ở trên — giao diện sẽ tự nạp checkpoint mặc định.\n"
            "> Nếu file chưa có, chạy `python -m gradio_demo.fetch_checkpoints --all` "
            "để tải checkpoint."
        )

    status_icon = "🧪" if meta.get("is_scratch") else "✅"
    model_type = "Scratch / Demo (Untrained)" if meta.get("is_scratch") else "Trained Checkpoint"
    kind_label = meta.get("kind_label", "LDT")

    lines = [
        f"### {status_icon} Thông Tin Mô Hình — {kind_label} ({model_type})",
        f"- **Đường dẫn**: `{meta.get('checkpoint_path')}`",
        f"- **Kích thước file**: `{meta.get('checkpoint_size_mb', 0)} MB`",
        f"- **Thiết bị (Device)**: `{meta.get('device')}`",
        f"- **Tổng tham số (Parameters)**: `{meta.get('total_params', 0):,}`",
    ]

    if meta.get("config_summary"):
        lines.append(f"- **Cấu hình**: `{meta['config_summary']}`")
    else:
        lines.append(
            f"- **Cấu hình LDT**: `{meta.get('num_layers')} layers`, "
            f"`dim={meta.get('dim')}`, `{meta.get('n_heads')} heads`, "
            f"`{meta.get('n_loops')} recursive loops`"
        )

    if meta.get("source_note"):
        lines.append(f"- **Nguồn**: {meta['source_note']}")
    if meta.get("accuracy"):
        lines.append(f"- **Độ chính xác tham khảo**: {meta['accuracy']}")
    lines.append(f"- **EMA Weights**: `{'Có (Đã kích hoạt)' if meta.get('has_ema') else 'Không'}`")

    missing = meta.get("missing_keys", 0)
    unexpected = meta.get("unexpected_keys", 0)
    if missing or unexpected:
        lines.append(
            f"- **Cảnh báo state_dict**: `missing={missing}`, `unexpected={unexpected}`"
        )
    return "\n".join(lines)


def _load_and_describe(
    model_key: str,
    checkpoint: str | Path | None,
    device_choice: str,
) -> tuple[str, str]:
    """Load a model and return (markdown, toast message)."""
    global _LOADED_MODEL, _LOADED_META

    spec = get_spec(model_key)
    target = Path(checkpoint) if checkpoint else spec.checkpoint_path

    if not target.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint cho {spec.label}: {target}\n"
            f"Chạy: python -m gradio_demo.fetch_checkpoints --model {model_key}"
        )

    loaded = load_model(model_key, target, device=device_choice)
    loaded.meta.setdefault("source_note", spec.source_note)
    loaded.meta.setdefault("accuracy", spec.accuracy)
    _LOADED_MODEL = loaded
    _LOADED_META = loaded.meta
    return format_meta_markdown(loaded.meta), (
        f"Đã nạp {spec.label}: {loaded.meta['total_params']:,} tham số trên "
        f"{loaded.meta['device']}."
    )


def on_load_checkpoint(checkpoint_file, checkpoint_path_str: str, device_choice: str, model_key: str):
    """Handle loading model from uploaded file or path string."""
    global _LOADED_MODEL, _LOADED_META

    target_path = None
    if checkpoint_file is not None:
        target_path = getattr(checkpoint_file, "name", str(checkpoint_file))
    elif checkpoint_path_str and checkpoint_path_str.strip():
        target_path = checkpoint_path_str.strip()
    else:
        spec = get_spec(model_key)
        target_path = str(spec.checkpoint_path)

    try:
        meta_md, msg = _load_and_describe(model_key, target_path, device_choice)
        gr.Info(msg)
        return meta_md
    except Exception as e:
        _LOADED_MODEL = None
        _LOADED_META = None
        gr.Warning(f"Lỗi khi nạp checkpoint: {e}")
        return format_meta_markdown(None) + f"\n\n> ❌ **Lỗi:** `{e}`"


def on_create_scratch(device_choice: str):
    """Create untrained scratch LDT model for quick testing."""
    global _LOADED_MODEL, _LOADED_META
    try:
        loaded = make_scratch_loaded_model(device=device_choice)
        _LOADED_MODEL = loaded
        _LOADED_META = loaded.meta
        gr.Info("Đã khởi tạo mô hình LDT ngẫu nhiên (Scratch Demo) thành công!")
        return format_meta_markdown(_LOADED_META)
    except Exception as e:
        gr.Warning(f"Lỗi: {e}")
        return format_meta_markdown(_LOADED_META)


def on_select_model(model_key: str, device_choice: str):
    """Auto-load the default checkpoint when the user switches model."""
    global _LOADED_MODEL, _LOADED_META

    spec = get_spec(model_key)
    is_ldt = spec.is_ldt
    updates = (
        gr.update(value=str(spec.checkpoint_path)),
        gr.update(visible=is_ldt),
        gr.update(visible=not is_ldt),
        gr.update(visible=is_ldt),
    )

    try:
        meta_md, msg = _load_and_describe(model_key, spec.checkpoint_path, device_choice)
        gr.Info(msg)
        return (updates[0], meta_md, *updates[1:])
    except Exception as e:
        _LOADED_MODEL = None
        _LOADED_META = None
        gr.Warning(f"Chưa nạp được {spec.label}: {e}")
        meta_md = (
            f"### ⚠️ Chưa nạp được {spec.label}\n"
            f"- **Checkpoint mong đợi**: `{spec.checkpoint_path}`\n"
            f"- **Lỗi**: `{e}`\n\n"
            f"> Tải checkpoint bằng: `python -m gradio_demo.fetch_checkpoints --model {model_key}`"
        )
        return (updates[0], meta_md, *updates[1:])


def on_select_example(example_name: str):
    """Populate puzzle text and render preview when an example is picked."""
    ex = get_example_by_name(example_name)
    if not ex:
        return "", render_board_html("." * 81, title="Bảng Sudoku"), ""

    board_html = render_board_html(ex.puzzle, title=f"Đề bài: {ex.name} ({ex.clues_count} số cho trước)")
    desc = f"**Độ khó:** {ex.difficulty} | **Số clues:** {ex.clues_count}\n\n*{ex.description}*"
    return ex.puzzle, board_html, desc


def on_format_check(raw_puzzle: str):
    """Normalise the puzzle string and refresh the preview board."""
    try:
        clean = parse_sudoku_input(raw_puzzle)
    except Exception as e:
        gr.Warning(f"Không chuẩn hoá được: {e}")
        return raw_puzzle, render_board_html("." * 81, title="Bảng Sudoku (Định dạng chưa đúng)")
    gr.Info("Đã chuẩn hoá chuỗi 81 ký tự thành công!")
    return clean, render_board_html(clean, title="Bảng Đã Chuẩn Hoá")


def on_puzzle_text_change(raw_puzzle: str):
    """Update visual board when raw puzzle text changes."""
    if not raw_puzzle or not raw_puzzle.strip():
        return render_board_html("." * 81, title="Bảng Sudoku (Trống)")
    try:
        clean = parse_sudoku_input(raw_puzzle)
        givens = sum(1 for c in clean if c in "123456789")
        return render_board_html(clean, title=f"Bảng Sudoku Nhập Vào ({givens} số cho trước)")
    except Exception:
        return render_board_html("." * 81, title="Bảng Sudoku (Định dạng chưa đúng)")


def on_solve_sudoku(
    puzzle_str: str,
    n_chains: int,
    max_rounds: int,
    threshold: float,
    temp_decide: float,
    augment: bool,
    act_steps: int,
):
    """Main solver callback triggered by the Solve button."""
    global _LOADED_MODEL, _LOADED_META

    if _LOADED_MODEL is None:
        return (
            render_board_html("." * 81, title="Chưa nạp model"),
            "",
            "⚠️ **Vui lòng nạp checkpoint mô hình trước khi giải!** "
            "Chọn model ở trên hoặc nhấn 'Nạp Checkpoint'.",
            "",
            "",
            "",
            "",
            "",
        )

    try:
        clean_puzzle = parse_sudoku_input(puzzle_str)
    except Exception as e:
        return (
            render_board_html("." * 81, title="Lỗi định dạng đề bài"),
            "",
            f"❌ **Lỗi đề bài:** {str(e)}",
            "",
            "",
            "",
            "",
            "",
        )

    try:
        result: SolveOutput = solve_for_loaded_model(
            _LOADED_MODEL,
            clean_puzzle,
            n_chains=int(n_chains),
            max_rounds=int(max_rounds),
            threshold=float(threshold),
            temp_decide=float(temp_decide),
            augment=bool(augment),
            act_steps=int(act_steps),
        )

        title = (
            f"Kết quả: {result.status} "
            f"(Rounds/Steps: {result.rounds}, Resets: {result.resets})"
        )
        solved_html = render_board_html(clean_puzzle, result.solution_str, title=title)

        if result.status == "SOLVED":
            status_banner = (
                f'<div style="padding: 12px; background: #ecfdf5; border: 1px solid #10b981; '
                f'border-radius: 8px; color: #065f46; font-weight: 600; font-size: 15px;">'
                f'🎉 GIẢI THÀNH CÔNG! Hợp lệ 100% theo luật Sudoku trong {result.rounds} vòng/bước '
                f'({result.elapsed_sec} giây, {result.resets} lần quay lui / backtrack).'
                f'</div>'
            )
        elif result.status == "INVALID_SOLUTION":
            status_banner = (
                f'<div style="padding: 12px; background: #fef2f2; border: 1px solid #ef4444; '
                f'border-radius: 8px; color: #991b1b; font-weight: 600; font-size: 15px;">'
                f'⚠️ TÌM THẤY LỜI GIẢI NHƯNG KHÔNG HỢP LỆ '
                f'({len(result.validation.get("errors", []))} lỗi).'
                f'</div>'
            )
        else:
            status_banner = (
                f'<div style="padding: 12px; background: #fffbeb; border: 1px solid #f59e0b; '
                f'border-radius: 8px; color: #92400e; font-weight: 600; font-size: 15px;">'
                f'⏱️ CHƯA GIẢI XONG (sau {result.rounds} vòng/bước). '
                f'Đã điền được {result.filled_count}/81 ô.'
                f'</div>'
            )

        rounds_str = f"{result.rounds} vòng"
        resets_str = f"{result.resets} lần"
        calls_str = f"{result.model_calls} forwards"
        time_str = f"{result.elapsed_sec:.3f} s"
        valid_summary = result.validation.get("summary", "N/A")

        return (
            solved_html,
            result.solution_str,
            status_banner,
            rounds_str,
            resets_str,
            calls_str,
            time_str,
            valid_summary,
        )
    except Exception as e:
        err_banner = (
            f'<div style="padding: 12px; background: #fef2f2; border: 1px solid #ef4444; '
            f'border-radius: 8px; color: #991b1b; font-weight: 600;">'
            f'❌ Lỗi trong quá trình suy luận: {str(e)}'
            f'</div>'
        )
        return (
            render_board_html(clean_puzzle, title="Lỗi khi giải"),
            "",
            err_banner,
            "-",
            "-",
            "-",
            "-",
            f"Lỗi: {str(e)}",
        )


# ---------------------------------------------------------------------------
# Snowflake Sudoku handlers
# ---------------------------------------------------------------------------
def _load_snowflake_and_describe(
    checkpoint: str | Path | None,
    device_choice: str,
) -> tuple[str, str]:
    global _SF_LOADED_MODEL, _SF_META

    spec = get_spec("snowflake")
    target = Path(checkpoint) if checkpoint else spec.checkpoint_path
    if not target.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint Snowflake: {target}\n"
            "Chạy: python -m gradio_demo.fetch_checkpoints --model snowflake"
        )

    loaded = load_model("snowflake", target, device=device_choice)
    loaded.meta.setdefault("source_note", spec.source_note)
    loaded.meta.setdefault("accuracy", spec.accuracy)
    _SF_LOADED_MODEL = loaded
    _SF_META = loaded.meta
    return format_meta_markdown(loaded.meta), (
        f"Đã nạp {spec.label}: {loaded.meta['total_params']:,} tham số trên "
        f"{loaded.meta['device']}."
    )


def on_load_snowflake(checkpoint_path_str: str, device_choice: str):
    try:
        meta_md, msg = _load_snowflake_and_describe(checkpoint_path_str, device_choice)
        gr.Info(msg)
        return meta_md
    except Exception as e:
        gr.Warning(f"Lỗi khi nạp checkpoint Snowflake: {e}")
        return format_meta_markdown(None) + f"\n\n> ❌ **Lỗi:** `{e}`"


def on_select_snowflake_example(example_name: str):
    example = get_snowflake_example_by_name(example_name) or SNOWFLAKE_EXAMPLES[0]
    board = render_snowflake_board(
        example,
        title=f"Đề bài: {example.name} ({example.givens} clues)",
    )
    return example.puzzle_string, board, snowflake_example_info(example)


def on_snowflake_puzzle_change(raw_puzzle: str, example_name: str):
    example = get_snowflake_example_by_name(example_name) or SNOWFLAKE_EXAMPLES[0]
    try:
        digits = parse_snowflake_input(raw_puzzle, example.n_cells)
    except Exception as e:
        return (
            '<div style="padding: 12px; background: #fffbeb; border: 1px solid #f59e0b; '
            'border-radius: 8px; color: #92400e;">'
            f"⚠️ {e}</div>"
        )
    givens = sum(1 for d in digits if d != 7)
    return render_snowflake_board(
        example,
        puzzle=digits,
        title=f"Bảng nhập vào ({givens} clues / {example.n_cells} ô)",
    )


def on_solve_snowflake(
    example_name: str,
    puzzle_text: str,
    n_chains: int,
    max_rounds: int,
    threshold: float,
    temp_decide: float,
    cls_threshold: float,
    augment: bool,
):
    """Solve the Snowflake puzzle in the selected topology."""
    global _SF_LOADED_MODEL, _SF_META

    example = get_snowflake_example_by_name(example_name) or SNOWFLAKE_EXAMPLES[0]
    try:
        digits = parse_snowflake_input(puzzle_text, example.n_cells)
    except Exception as e:
        return (
            render_snowflake_board(example, title="Lỗi định dạng đề bài"),
            "",
            f"❌ **Lỗi đề bài:** {e}",
            "",
            "",
            "",
            "",
            "Lỗi định dạng",
        )

    if _SF_LOADED_MODEL is None:
        return (
            render_snowflake_board(example, puzzle=digits, title="Chưa nạp model Snowflake"),
            "",
            "⚠️ **Vui lòng nạp checkpoint Snowflake trước khi giải!** "
            "Mở accordion checkpoint trong tab này và nhấn 'Nạp Checkpoint'.",
            "",
            "",
            "",
            "",
            "Chưa nạp model",
        )

    try:
        result: SnowflakeSolveOutput = solve_snowflake(
            _SF_LOADED_MODEL.model,
            example,
            puzzle=digits,
            n_chains=int(n_chains),
            max_rounds=int(max_rounds),
            threshold=float(threshold),
            temp_decide=float(temp_decide),
            cls_threshold=float(cls_threshold),
            augment=bool(augment),
            device=_SF_LOADED_MODEL.device,
        )

        title = (
            f"Kết quả: {result.status} "
            f"(n={example.n}, {result.n_cells} ô, rounds={result.rounds})"
        )
        solved_html = render_snowflake_board(
            example,
            result.solution,
            title=title,
            puzzle=digits,
        )

        if result.status == "SOLVED":
            status_banner = (
                f'<div style="padding: 12px; background: #ecfdf5; border: 1px solid #10b981; '
                f'border-radius: 8px; color: #065f46; font-weight: 600; font-size: 15px;">'
                f'🎉 GIẢI THÀNH CÔNG! Hợp lệ 100% theo toàn bộ ràng buộc Snowflake '
                f'trong {result.rounds} vòng ({result.elapsed_sec} giây, '
                f'{result.resets} lần quay lui).'
                f'</div>'
            )
        elif result.status == "INVALID_SOLUTION":
            status_banner = (
                f'<div style="padding: 12px; background: #fef2f2; border: 1px solid #ef4444; '
                f'border-radius: 8px; color: #991b1b; font-weight: 600; font-size: 15px;">'
                f'⚠️ TÌM THẤY LỜI GIẢI NHƯNG KHÔNG HỢP LỆ '
                f'({len(result.validation.get("errors", []))} lỗi).'
                f'</div>'
            )
        else:
            status_banner = (
                f'<div style="padding: 12px; background: #fffbeb; border: 1px solid #f59e0b; '
                f'border-radius: 8px; color: #92400e; font-weight: 600; font-size: 15px;">'
                f'⏱️ CHƯA GIẢI XONG (sau {result.rounds} vòng). '
                f'Đã điền được {result.filled_count}/{result.n_cells} ô.'
                f'</div>'
            )

        return (
            solved_html,
            result.solution_string,
            status_banner,
            f"{result.rounds} vòng",
            f"{result.resets} lần",
            f"{result.model_calls} forwards",
            f"{result.elapsed_sec:.3f} s",
            result.validation.get("summary", "N/A"),
        )
    except Exception as e:
        err_banner = (
            f'<div style="padding: 12px; background: #fef2f2; border: 1px solid #ef4444; '
            f'border-radius: 8px; color: #991b1b; font-weight: 600;">'
            f'❌ Lỗi trong quá trình suy luận: {str(e)}'
            f'</div>'
        )
        return (
            render_snowflake_board(example, puzzle=digits, title="Lỗi khi giải"),
            "",
            err_banner,
            "-",
            "-",
            "-",
            "-",
            f"Lỗi: {str(e)}",
        )


def build_app(
    initial_checkpoint: str | None = None,
    initial_device: str = "auto",
    initial_model: str = DEFAULT_MODEL_KEY,
) -> gr.Blocks:
    """Construct Gradio UI Blocks."""
    global _LOADED_MODEL, _LOADED_META, _SF_LOADED_MODEL, _SF_META

    if initial_model not in SUDOKU_MODEL_KEYS:
        initial_model = DEFAULT_MODEL_KEY
    initial_spec = get_spec(initial_model)

    # Auto-load the default checkpoint on startup (unless a different one is passed).
    startup_checkpoint = Path(initial_checkpoint) if initial_checkpoint else initial_spec.checkpoint_path
    if startup_checkpoint.is_file():
        try:
            print(f"Preloading {initial_spec.label} from: {startup_checkpoint}...", flush=True)
            _LOADED_MODEL = load_model(initial_model, startup_checkpoint, device=initial_device)
            _LOADED_META = _LOADED_MODEL.meta
            _LOADED_META.setdefault("source_note", initial_spec.source_note)
            _LOADED_META.setdefault("accuracy", initial_spec.accuracy)
            print(f"Preloaded successfully on {_LOADED_META['device']}!", flush=True)
        except Exception as e:
            print(f"Failed to preload checkpoint: {e}", flush=True)
    else:
        print(
            f"Checkpoint not found: {startup_checkpoint}\n"
            f"Run: python -m gradio_demo.fetch_checkpoints --model {initial_model}",
            flush=True,
        )

    # Snowflake lives in its own tab.  Its checkpoint is tiny (~3 MB), so
    # preload it too and the tab works immediately.
    sf_spec = get_spec("snowflake")
    sf_checkpoint = sf_spec.checkpoint_path
    if sf_checkpoint.is_file():
        try:
            print(f"Preloading {sf_spec.label} from: {sf_checkpoint}...", flush=True)
            _SF_LOADED_MODEL = load_model("snowflake", sf_checkpoint, device=initial_device)
            _SF_META = _SF_LOADED_MODEL.meta
            _SF_META.setdefault("source_note", sf_spec.source_note)
            _SF_META.setdefault("accuracy", sf_spec.accuracy)
            print(f"Preloaded Snowflake on {_SF_META['device']}!", flush=True)
        except Exception as e:
            print(f"Failed to preload Snowflake checkpoint: {e}", flush=True)
    else:
        print(
            f"Snowflake checkpoint not found: {sf_checkpoint}\n"
            "Run: python -m gradio_demo.fetch_checkpoints --model snowflake",
            flush=True,
        )

    default_meta_md = format_meta_markdown(_LOADED_META)
    default_sf_meta_md = format_meta_markdown(_SF_META)
    default_ckpt = str(startup_checkpoint)
    default_sf_ckpt = str(sf_checkpoint)
    default_sf_example = SNOWFLAKE_EXAMPLES[0]

    with gr.Blocks(title="LDT / HRM / TRM Sudoku & Snowflake Solver", analytics_enabled=False) as demo:
        gr.Markdown(
            """
            # 🧩 Lattice Deduction Transformers (LDT) — Sudoku Solver
            **Demo so sánh LDT, HRM và TRM trên Sudoku** (LDT: arXiv:2605.08605).
            Tab **❄️ Snowflake Sudoku (LDT)** bổ sung bàn cờ hexagonal 6 chữ số.

            *Chọn mô hình — giao diện tự nạp checkpoint mặc định. Vẫn có thể ghi đè bằng
            đường dẫn hoặc kéo-thả file `.pt`.*
            """
        )

        # ------------------ TOP SECTION: CHECKPOINT MANAGER ------------------
        with gr.Accordion(
            "⚙️ Quản Lý Mô Hình & Checkpoint (tự động nạp — nhấn để đổi)",
            open=(_LOADED_MODEL is None),
        ):
            with gr.Row():
                with gr.Column(scale=3):
                    model_dropdown = gr.Dropdown(
                        label="Mô hình (tự động nạp checkpoint tương ứng)",
                        choices=choice_labels(keys=SUDOKU_MODEL_KEYS),
                        value=initial_model if initial_model in SUDOKU_MODEL_KEYS else DEFAULT_MODEL_KEY,
                    )
                    ckpt_path_input = gr.Textbox(
                        label="Đường dẫn file Checkpoint (.pt) trên máy",
                        placeholder="VD: checkpoints/ldt_scaled_sudoku.pt",
                        value=default_ckpt,
                    )
                    ckpt_file_input = gr.File(
                        label="Hoặc kéo thả file .pt tải lên (ghi đè checkpoint mặc định)",
                        file_types=[".pt", ".pth", ".ckpt"],
                    )
                with gr.Column(scale=1):
                    device_dropdown = gr.Dropdown(
                        label="Thiết bị tính toán (Device)",
                        choices=["auto", "cuda", "cpu"],
                        value=initial_device,
                    )
                    with gr.Row():
                        load_btn = gr.Button("📥 Nạp Checkpoint", variant="primary")
                    with gr.Row():
                        scratch_btn = gr.Button(
                            "🧪 Tạo Model LDT Thử Nghiệm",
                            variant="secondary",
                            visible=initial_spec.is_ldt,
                        )

            meta_display = gr.Markdown(value=default_meta_md)

            load_btn.click(
                fn=on_load_checkpoint,
                inputs=[ckpt_file_input, ckpt_path_input, device_dropdown, model_dropdown],
                outputs=[meta_display],
            )
            scratch_btn.click(
                fn=on_create_scratch,
                inputs=[device_dropdown],
                outputs=[meta_display],
            )

        # ------------------ MAIN TABS ------------------
        with gr.Tabs():
            with gr.Tab("🎯 Giải Sudoku (Interactive Solver)"):
                with gr.Row():
                    # Left column: Controls & Input
                    with gr.Column(scale=5):
                        gr.Markdown("### 1. Nhập Đề Bài Sudoku")

                        example_dropdown = gr.Dropdown(
                            label="Chọn câu đố mẫu từ Benchmark",
                            choices=[ex.name for ex in EXAMPLES],
                            value=EXAMPLES[0].name,
                        )
                        example_info = gr.Markdown(
                            value=f"**Độ khó:** {EXAMPLES[0].difficulty} | **Số clues:** {EXAMPLES[0].clues_count}\n\n*{EXAMPLES[0].description}*"
                        )

                        puzzle_textbox = gr.Textbox(
                            label="Chuỗi 81 ký tự Sudoku (dùng số 1-9 cho ô có sẵn, dấu chấm '.' hoặc '0' cho ô trống)",
                            value=EXAMPLES[0].puzzle,
                            lines=3,
                        )

                        with gr.Row():
                            clear_btn = gr.Button("🗑️ Xoá Trống")
                            format_check_btn = gr.Button("🔍 Chuẩn Hoá Định Dạng")

                        with gr.Accordion("🛠️ Tham Số Suy Luận Nâng Cao", open=False):
                            with gr.Row():
                                n_chains_slider = gr.Slider(
                                    label="Số chuỗi LDT / số view HRM-TRM",
                                    minimum=1,
                                    maximum=64,
                                    step=1,
                                    value=32,
                                    info=(
                                        "LDT: số chain DPLL song song. "
                                        "HRM/TRM: số view augmentation đem bỏ phiếu (nên 8)."
                                    ),
                                )
                            with gr.Group(visible=initial_spec.is_ldt) as ldt_param_group:
                                with gr.Row():
                                    max_rounds_slider = gr.Slider(
                                        label="Số vòng DPLL tối đa (max_rounds)",
                                        minimum=10,
                                        maximum=2000,
                                        step=10,
                                        value=500,
                                        info="Chỉ dùng cho LDT.",
                                    )
                                    threshold_slider = gr.Slider(
                                        label="Ngưỡng loại suy luận (threshold)",
                                        minimum=0.01,
                                        maximum=0.40,
                                        step=0.01,
                                        value=0.10,
                                        info="Chỉ dùng cho LDT: loại bit nếu sigma(bce) < threshold.",
                                    )
                                with gr.Row():
                                    temp_slider = gr.Slider(
                                        label="Nhiệt độ phân nhánh (temp_decide)",
                                        minimum=0.1,
                                        maximum=3.0,
                                        step=0.1,
                                        value=1.5,
                                        info="Chỉ dùng cho LDT: softmax temperature khi chọn ứng viên.",
                                    )
                            with gr.Group(visible=not initial_spec.is_ldt) as act_param_group:
                                act_steps_slider = gr.Slider(
                                    label="ACT max steps (HRM / TRM)",
                                    minimum=1,
                                    maximum=32,
                                    step=1,
                                    value=int(_LOADED_META.get("act_max_steps", 16)) if _LOADED_META else 16,
                                    info=(
                                        "Số bước suy luận ACT. Checkpoint gốc dừng ở 16 bước; "
                                        "tăng thêm không có tác dụng."
                                    ),
                                )
                            augment_toggle = gr.Checkbox(
                                label="Kích hoạt Data Augmentation (Digit Permutation + D4 Dihedral)",
                                value=True,
                                info=(
                                    "LDT: augment bên trong DPLL. "
                                    "HRM/TRM: augment các view đem bỏ phiếu."
                                ),
                            )

                        solve_btn = gr.Button("🚀 GIẢI SUDOKU", variant="primary", size="lg")

                    # Right column: Visual Board & Output
                    with gr.Column(scale=6):
                        gr.Markdown("### 2. Bảng Trực Quan & Kết Quả")
                        status_html = gr.HTML(
                            value='<div style="padding: 10px; color: #64748b; font-style: italic;">Sẵn sàng giải... Nhấn nút Giải Sudoku để bắt đầu.</div>'
                        )

                        board_display = gr.HTML(
                            value=render_board_html(EXAMPLES[0].puzzle, title="Bảng Đề Bài")
                        )

                        gr.Markdown("#### 📊 Thống Kê Hiệu Năng")
                        with gr.Row():
                            stat_rounds = gr.Textbox(label="Số vòng DPLL / ACT steps", interactive=False, elem_classes=["stat-box"])
                            stat_resets = gr.Textbox(label="Lần quay lui (Resets)", interactive=False, elem_classes=["stat-box"])
                            stat_calls = gr.Textbox(label="Lượt Forward", interactive=False, elem_classes=["stat-box"])
                            stat_time = gr.Textbox(label="Thời gian", interactive=False, elem_classes=["stat-box"])

                        stat_validation = gr.Textbox(label="Kiểm tra hợp lệ luật Sudoku", interactive=False)

                        tb_kwargs = {
                            "label": "Chuỗi kết quả 81 ký tự (Solution String)",
                            "interactive": False,
                        }
                        if "buttons" in inspect.signature(gr.Textbox.__init__).parameters:
                            tb_kwargs["buttons"] = ["copy"]
                        elif "show_copy_button" in inspect.signature(gr.Textbox.__init__).parameters:
                            tb_kwargs["show_copy_button"] = True
                        solution_string_out = gr.Textbox(**tb_kwargs)

            # ------------------------------------------------------------------
            # Snowflake Sudoku tab
            # ------------------------------------------------------------------
            with gr.Tab("❄️ Snowflake Sudoku (LDT)"):
                gr.Markdown(
                    """
                    ### ❄️ Snowflake Sudoku — LDT
                    Bàn cờ hexagonal: mỗi **hexagon** chia thành **6 ô tam giác** (một ô
                    cho mỗi hướng NE/E/SE/SW/W/NW). Mỗi hexagon và mỗi điểm giao của
                    6 tam giác là một ràng buộc "6 chữ số 1–6 không trùng".

                    LDT biểu diễn mỗi ô bằng powerset 6 ứng viên, nhúng toàn bộ puzzle
                    vào lưới phủ **15×10 = 150 ô**, rồi chạy cùng vòng DPLL như Sudoku
                    9×9 (thêm mask `in_puzzle`). Checkpoint trong demo đạt
                    **1.000/1.000** trên tập test Snowflake.
                    """
                )

                with gr.Accordion(
                    "⚙️ Checkpoint Snowflake (tự động nạp — nhấn để đổi)",
                    open=(_SF_LOADED_MODEL is None),
                ):
                    with gr.Row():
                        with gr.Column(scale=3):
                            sf_ckpt_path_input = gr.Textbox(
                                label="Đường dẫn file Checkpoint Snowflake (.pt)",
                                placeholder="VD: checkpoints/snowflake_sudoku.pt",
                                value=default_sf_ckpt,
                            )
                        with gr.Column(scale=1):
                            sf_device_dropdown = gr.Dropdown(
                                label="Thiết bị tính toán (Device)",
                                choices=["auto", "cuda", "cpu"],
                                value=initial_device,
                            )
                            sf_load_btn = gr.Button("📥 Nạp Checkpoint", variant="primary")
                    sf_meta_display = gr.Markdown(value=default_sf_meta_md)

                with gr.Row():
                    # Left column: input & parameters
                    with gr.Column(scale=5):
                        gr.Markdown("### 1. Chọn / Nhập Đề Bài Snowflake")

                        sf_example_dropdown = gr.Dropdown(
                            label="Chọn puzzle mẫu (n = 4…8 hexagons)",
                            choices=[ex.name for ex in SNOWFLAKE_EXAMPLES],
                            value=default_sf_example.name,
                        )
                        sf_example_info_md = gr.Markdown(
                            value=snowflake_example_info(default_sf_example)
                        )

                        sf_puzzle_textbox = gr.Textbox(
                            label=(
                                "Chuỗi ô tam giác (1–6 = clue, '.' hoặc '0' = trống). "
                                "Số ô phải khớp n×6 của puzzle đang chọn."
                            ),
                            value=default_sf_example.puzzle_string,
                            lines=3,
                        )

                        with gr.Accordion("🛠️ Tham Số Suy Luận Snowflake", open=False):
                            sf_n_chains = gr.Slider(
                                label="Số chuỗi DPLL song song (n_chains)",
                                minimum=1,
                                maximum=64,
                                step=1,
                                value=64,
                                info="Eval chính thức dùng 64 chains.",
                            )
                            sf_max_rounds = gr.Slider(
                                label="Số vòng DPLL tối đa (max_rounds)",
                                minimum=10,
                                maximum=2000,
                                step=10,
                                value=1000,
                            )
                            with gr.Row():
                                sf_threshold = gr.Slider(
                                    label="Ngưỡng loại suy luận (threshold)",
                                    minimum=0.01,
                                    maximum=0.40,
                                    step=0.01,
                                    value=0.10,
                                )
                                sf_temp = gr.Slider(
                                    label="Nhiệt độ phân nhánh (temp_decide)",
                                    minimum=0.1,
                                    maximum=3.0,
                                    step=0.1,
                                    value=1.5,
                                )
                            with gr.Row():
                                sf_cls = gr.Slider(
                                    label="Ngưỡng conflict head (cls_threshold)",
                                    minimum=0.10,
                                    maximum=0.95,
                                    step=0.05,
                                    value=0.60,
                                )
                            sf_augment = gr.Checkbox(
                                label="Kích hoạt augmentation (chỉ hoán vị chữ số)",
                                value=True,
                                info=(
                                    "Snowflake không dùng D4 dihedral vì lưới hex; "
                                    "chỉ hoán vị 6 chữ số."
                                ),
                            )

                        sf_solve_btn = gr.Button(
                            "❄️ GIẢI SNOWFLAKE", variant="primary", size="lg"
                        )

                    # Right column: boards & output
                    with gr.Column(scale=6):
                        gr.Markdown("### 2. Bảng Trực Quan & Kết Quả")
                        sf_status_html = gr.HTML(
                            value='<div style="padding: 10px; color: #64748b; '
                            'font-style: italic;">Sẵn sàng giải Snowflake...</div>'
                        )

                        sf_puzzle_board = gr.HTML(
                            value=render_snowflake_board(
                                default_sf_example,
                                title=(
                                    f"Đề bài: {default_sf_example.name} "
                                    f"({default_sf_example.givens} clues)"
                                ),
                            )
                        )
                        sf_result_board = gr.HTML(
                            value='<div style="padding: 10px; color: #94a3b8; '
                            'font-style: italic;">Bảng kết quả sẽ hiện ở đây.</div>'
                        )

                        gr.Markdown("#### 📊 Thống Kê Hiệu Năng")
                        with gr.Row():
                            sf_stat_rounds = gr.Textbox(
                                label="Số vòng DPLL", interactive=False,
                                elem_classes=["stat-box"],
                            )
                            sf_stat_resets = gr.Textbox(
                                label="Lần quay lui (Resets)", interactive=False,
                                elem_classes=["stat-box"],
                            )
                            sf_stat_calls = gr.Textbox(
                                label="Lượt Forward", interactive=False,
                                elem_classes=["stat-box"],
                            )
                            sf_stat_time = gr.Textbox(
                                label="Thời gian", interactive=False,
                                elem_classes=["stat-box"],
                            )

                        sf_validation = gr.Textbox(
                            label="Kiểm tra hợp lệ ràng buộc Snowflake",
                            interactive=False,
                        )

                        sf_tb_kwargs = {
                            "label": "Chuỗi kết quả Snowflake (mỗi ký tự = một ô tam giác)",
                            "interactive": False,
                        }
                        if "buttons" in inspect.signature(gr.Textbox.__init__).parameters:
                            sf_tb_kwargs["buttons"] = ["copy"]
                        elif "show_copy_button" in inspect.signature(gr.Textbox.__init__).parameters:
                            sf_tb_kwargs["show_copy_button"] = True
                        sf_solution_string = gr.Textbox(**sf_tb_kwargs)

                sf_load_btn.click(
                    fn=on_load_snowflake,
                    inputs=[sf_ckpt_path_input, sf_device_dropdown],
                    outputs=[sf_meta_display],
                )
                sf_example_dropdown.change(
                    fn=on_select_snowflake_example,
                    inputs=[sf_example_dropdown],
                    outputs=[sf_puzzle_textbox, sf_puzzle_board, sf_example_info_md],
                )
                sf_puzzle_textbox.change(
                    fn=on_snowflake_puzzle_change,
                    inputs=[sf_puzzle_textbox, sf_example_dropdown],
                    outputs=[sf_puzzle_board],
                )
                sf_solve_btn.click(
                    fn=on_solve_snowflake,
                    inputs=[
                        sf_example_dropdown,
                        sf_puzzle_textbox,
                        sf_n_chains,
                        sf_max_rounds,
                        sf_threshold,
                        sf_temp,
                        sf_cls,
                        sf_augment,
                    ],
                    outputs=[
                        sf_result_board,
                        sf_solution_string,
                        sf_status_html,
                        sf_stat_rounds,
                        sf_stat_resets,
                        sf_stat_calls,
                        sf_stat_time,
                        sf_validation,
                    ],
                )

            with gr.Tab("📖 Giới Thiệu Thuật Toán"):
                gr.Markdown(
                    """
                    ### Ba mô hình trong demo

                    - **LDT (Lattice Deduction Transformers)** — biểu diễn mỗi ô bằng tập
                      lũy thừa 9 ứng viên `[81, 9]`, chạy transformer 2D lặp `n_loops` vòng,
                      rồi dùng vòng DPLL (loại trừ tất định + phân nhánh + quay lui) để giải.
                      Checkpoint trong demo được huấn luyện trên **Scaled Sudoku**. Cùng
                      kiến trúc này cũng được huấn luyện cho **Snowflake Sudoku** (tab riêng).
                    - **HRM (Hierarchical Reasoning Model)** — hai mức suy luận H/L chạy lặp
                      với cơ chế ACT tự dừng. Checkpoint trong demo được huấn luyện trên
                      **Scaled Sudoku** (34,50% test).
                    - **TRM (Tiny Recursive Model)** — mạng nhỏ cập nhật lặp lại latent `z`
                      và đáp án `y`. Checkpoint trong demo lấy từ **repo gốc
                      TinyRecursiveModels** (SamsungSAILMontreal), variant attention,
                      huấn luyện trên **Sudoku-Extreme**.

                    #### Cơ chế LDT
                    1. **Powerset lattice**: ô trống giữ cả 9 ứng viên; ô cho trước giữ đúng 1.
                    2. **Looped transformer**: chạy lặp với outer LayerNorm và input reinjection.
                    3. **DPLL**: loại ứng viên dưới `threshold`, phân nhánh theo `temp_decide`,
                       reset chain khi conflict.

                    #### Cơ chế HRM/TRM
                    - Chạy ACT tối đa `halt_max_steps` bước; mỗi bước cập nhật latent và logits.
                    - Demo chạy nhiều view augmentation (digit permutation + D4) rồi bỏ phiếu
                       theo đa số — tương đương cách eval `n_chains=8` của LDT.

                    #### Snowflake Sudoku (LDT)
                    - Mỗi hexagon có 6 ô tam giác; mỗi hexagon và mỗi điểm giao 6 tam giác
                      là một ràng buộc hoán vị của 6 chữ số.
                    - Puzzle được nhúng vào lưới phủ `15×10 = 150` ô; ô ngoài puzzle bị
                      mask (channel thứ 7) nên không tham gia suy luận.
                    - Augmentation chỉ hoán vị chữ số (không D4) để giữ topology hex.
                    """
                )

        # Event bindings
        example_dropdown.change(
            fn=on_select_example,
            inputs=[example_dropdown],
            outputs=[puzzle_textbox, board_display, example_info],
        )

        puzzle_textbox.change(
            fn=on_puzzle_text_change,
            inputs=[puzzle_textbox],
            outputs=[board_display],
        )

        clear_btn.click(
            fn=lambda: ("." * 81, render_board_html("." * 81, title="Bảng Trống")),
            inputs=[],
            outputs=[puzzle_textbox, board_display],
        )

        format_check_btn.click(
            fn=on_format_check,
            inputs=[puzzle_textbox],
            outputs=[puzzle_textbox, board_display],
        )

        solve_btn.click(
            fn=on_solve_sudoku,
            inputs=[
                puzzle_textbox,
                n_chains_slider,
                max_rounds_slider,
                threshold_slider,
                temp_slider,
                augment_toggle,
                act_steps_slider,
            ],
            outputs=[
                board_display,
                solution_string_out,
                status_html,
                stat_rounds,
                stat_resets,
                stat_calls,
                stat_time,
                stat_validation,
            ],
        )

        # Model switch binding (added after parameter groups exist).
        model_dropdown.change(
            fn=on_select_model,
            inputs=[model_dropdown, device_dropdown],
            outputs=[ckpt_path_input, meta_display, ldt_param_group, act_param_group, scratch_btn],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="LDT / HRM / TRM Sudoku Solver Gradio Web App")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_KEY,
        choices=list(SUDOKU_MODEL_KEYS),
        help="Model to auto-load on startup (default: ldt)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override the default checkpoint path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device (default: auto)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for Gradio Web server (default: 7860)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Generate a public Gradio shareable link",
    )
    args = parser.parse_args()

    app = build_app(
        initial_checkpoint=args.checkpoint,
        initial_device=args.device,
        initial_model=args.model,
    )
    print(f"Khởi động Gradio App trên http://{args.host}:{args.port}...", flush=True)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
