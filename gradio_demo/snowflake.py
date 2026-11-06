"""Snowflake Sudoku helpers for the Gradio demo.

Snowflake Sudoku uses a hexagonal topology: every hexagon is split into six
triangular cells (one per direction), and every hexagon / meeting point is an
all-different constraint over digits 1..6.  LDT embeds each puzzle into a
fixed 15x10 covering grid (150 slots) and runs the same DPLL loop used for
9x9 Sudoku, with an extra ``in_puzzle`` mask channel.

This module bundles a handful of generated puzzles as JSON so the demo does
not need ``pyarrow`` at runtime, and renders the triangular board as an SVG.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from experiments.snowflake.data import (
    SEQ_LEN,
    VOCAB,
    cell_to_grid_idx,
    puzzle_to_state,
)

EXAMPLES_PATH = Path(__file__).resolve().parent / "snowflake_examples.json"

DIRECTIONS = ("NE", "E", "SE", "SW", "W", "NW")
DIRECTION_ANGLES = {"NE": 0, "E": 60, "SE": 120, "SW": 180, "W": 240, "NW": 300}
HEX_SIZE = 100.0
BLANK = 7


@dataclass
class SnowflakeExample:
    """One bundled Snowflake puzzle."""

    name: str
    n: int
    givens: int
    code: str
    puzzle: list[int]
    solution: list[int]
    topology: dict[str, Any]
    description: str

    @property
    def n_cells(self) -> int:
        return int(self.topology["n_cells"])

    @property
    def puzzle_string(self) -> str:
        return "".join("." if d == BLANK else str(d) for d in self.puzzle)

    @property
    def solution_string(self) -> str:
        return "".join(str(d) for d in self.solution)

    def as_record(self, puzzle: list[int] | None = None) -> dict[str, Any]:
        """Return the dict shape expected by ``experiments.snowflake.data``."""
        return {
            "puzzle": list(self.puzzle if puzzle is None else puzzle),
            "solution": list(self.solution),
            "topology": self.topology,
        }


def _load_examples() -> list[SnowflakeExample]:
    with open(EXAMPLES_PATH, encoding="utf-8") as fh:
        raw = json.load(fh)
    return [SnowflakeExample(**item) for item in raw]


EXAMPLES: list[SnowflakeExample] = _load_examples()


def get_example_by_name(name: str) -> SnowflakeExample | None:
    for example in EXAMPLES:
        if example.name == name:
            return example
    return None


def parse_snowflake_input(raw: str, n_cells: int) -> list[int]:
    """Parse a compact Snowflake puzzle string into per-cell digits.

    Accepts digits 1..6 for clues, '.' / '0' for blanks.  Any whitespace or
    separator characters are ignored.  The number of cells must match the
    selected topology exactly.
    """
    if not raw or not isinstance(raw, str):
        raise ValueError("Đề bài Snowflake đang trống.")

    digits: list[int] = []
    for char in raw:
        if char in "123456":
            digits.append(int(char))
        elif char in ".0":
            digits.append(BLANK)
        elif char.isdigit():
            raise ValueError(
                f"Snowflake chỉ dùng chữ số 1-6; gặp chữ số không hợp lệ: {char!r}."
            )
        # Whitespace and other separators are ignored.
    if len(digits) != n_cells:
        raise ValueError(
            f"Cần đúng {n_cells} ô cho Snowflake n={n_cells // 6}, "
            f"nhưng nhận được {len(digits)} ô."
        )
    return digits


# ---------------------------------------------------------------------------
# Tensor encoding / decoding
# ---------------------------------------------------------------------------
def puzzle_to_tensors(
    record: dict[str, Any],
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode one Snowflake record to the tensors expected by ``solve()``.

    Returns ``(state, ground_truth, given_mask, in_puzzle_mask)`` with shapes
    ``[1, 150, 7]``, ``[1, 150, 6]``, ``[1, 150]`` and ``[1, 150]``.
    """
    x, y, mask = puzzle_to_state(record)
    x_t = torch.from_numpy(x)[None]
    y_t = torch.from_numpy(y)[None]
    mask_t = torch.from_numpy(mask)[None]
    # Channel 6 = in-puzzle mask, locked at 1 for cells that belong to this
    # puzzle (the same 7-channel layout used during training).
    state = torch.cat([x_t, mask_t.unsqueeze(-1).float()], dim=-1)
    # Given cells are exactly the singleton rows of the input powerset.
    given = x_t.sum(dim=-1) == 1
    if device is not None:
        dev = torch.device(device)
        state, y_t, given, mask_t = (
            state.to(dev),
            y_t.to(dev),
            given.to(dev),
            mask_t.to(dev),
        )
    return state, y_t, given, mask_t


def decode_solution(record: dict[str, Any], solution_state: torch.Tensor) -> list[int]:
    """Decode a canonical ``[150, 7]`` state into per-cell digits.

    Cells whose powerset still has more than one live candidate decode to 0,
    which lets the UI show partially solved boards on a timeout.
    """
    topo = record["topology"]
    state = solution_state.detach().cpu()
    digits: list[int] = []
    for cell_id in range(int(topo["n_cells"])):
        cp = topo["cell_positions"][str(cell_id)]
        idx = cell_to_grid_idx(cp["q"], cp["r"], cp["direction"])
        alive = int((state[idx, :VOCAB] > 0.5).sum().item())
        if alive == 1:
            digits.append(int(state[idx, :VOCAB].argmax().item()) + 1)
        else:
            digits.append(0)
    return digits


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_snowflake_solution(
    record: dict[str, Any],
    digits: Iterable[int],
) -> dict[str, Any]:
    """Validate a full Snowflake assignment against givens and constraints."""
    digits = [int(d) for d in digits]
    topo = record["topology"]
    n_cells = int(topo["n_cells"])
    puzzle = record["puzzle"]

    errors: list[str] = []
    if len(digits) != n_cells:
        return {
            "is_valid": False,
            "errors": [f"Sai số ô: {len(digits)} (cần {n_cells})."],
            "complete": False,
            "givens_match": False,
            "summary": "Sai số ô",
        }

    missing = sum(1 for d in digits if d not in range(1, 7))
    if missing:
        errors.append(f"Còn {missing} ô chưa xác định (đa ứng viên hoặc trống).")

    mismatches = 0
    for cid, given in enumerate(puzzle):
        if given != BLANK and cid < len(digits) and digits[cid] != given:
            mismatches += 1
    if mismatches:
        errors.append(f"{mismatches} ô cho trước (givens) bị thay đổi.")

    violated = 0
    for group in topo.get("constraints", []):
        values = [digits[c] for c in group["cells"] if 0 < digits[c] < 7]
        if len(values) == len(group["cells"]) and len(set(values)) != len(values):
            violated += 1
        elif len(values) != len(group["cells"]):
            # incomplete groups are already reported as missing cells
            pass
    if violated:
        errors.append(f"{violated} ràng buộc (hexagon/điểm giao) bị trùng chữ số.")

    complete = missing == 0
    is_valid = complete and mismatches == 0 and violated == 0
    if is_valid:
        summary = "Hợp lệ (Đúng toàn bộ ràng buộc Snowflake)"
    elif complete:
        summary = f"Không hợp lệ ({len(errors)} vi phạm)"
    else:
        summary = f"Chưa hoàn tất ({missing}/{n_cells} ô còn mơ hồ)"

    return {
        "is_valid": is_valid,
        "errors": errors,
        "complete": complete,
        "givens_match": mismatches == 0,
        "violated_constraints": violated,
        "missing_cells": missing,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# SVG rendering
# ---------------------------------------------------------------------------
def _hex_center(q: int, r: int) -> tuple[float, float]:
    hx = HEX_SIZE * (1.5 * q)
    hy = HEX_SIZE * (math.sqrt(3) / 2 * q + math.sqrt(3) * r)
    return hx, hy


def _cell_polygon(q: int, r: int, direction: str) -> list[tuple[float, float]]:
    cx, cy = _hex_center(q, r)
    angle = math.radians(DIRECTION_ANGLES[direction])
    p1 = (cx + HEX_SIZE * math.cos(angle), cy + HEX_SIZE * math.sin(angle))
    p2 = (
        cx + HEX_SIZE * math.cos(angle + math.radians(60)),
        cy + HEX_SIZE * math.sin(angle + math.radians(60)),
    )
    return [(cx, cy), p1, p2]


def _hex_outline(q: int, r: int) -> list[tuple[float, float]]:
    cx, cy = _hex_center(q, r)
    points = []
    for k in range(6):
        angle = math.radians(60 * k)
        points.append((cx + HEX_SIZE * math.cos(angle), cy + HEX_SIZE * math.sin(angle)))
    return points


def _points_attr(points: Iterable[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _centroid(points: Iterable[tuple[float, float]]) -> tuple[float, float]:
    pts = list(points)
    return (
        sum(p[0] for p in pts) / len(pts),
        sum(p[1] for p in pts) / len(pts),
    )


def render_snowflake_board(
    example: SnowflakeExample,
    solution: Iterable[int] | None = None,
    title: str = "",
    puzzle: Iterable[int] | None = None,
) -> str:
    """Render a Snowflake board as an inline SVG inside a small HTML card."""
    topo = example.topology
    puzzle_digits = list(example.puzzle if puzzle is None else puzzle)
    solution_digits = None if solution is None else [int(d) for d in solution]

    # Collect every triangle and the drawing bounds.
    triangles: list[tuple[list[tuple[float, float]], float, float, int]] = []
    bounds: list[tuple[float, float]] = []
    for cell_id in range(example.n_cells):
        cp = topo["cell_positions"][str(cell_id)]
        pts = _cell_polygon(cp["q"], cp["r"], cp["direction"])
        cx, cy = _centroid(pts)
        triangles.append((pts, cx, cy, cell_id))
        bounds.extend(pts)

    xs = [p[0] for p in bounds]
    ys = [p[1] for p in bounds]
    pad = HEX_SIZE * 0.28
    min_x, max_x = min(xs) - pad, max(xs) + pad
    min_y, max_y = min(ys) - pad, max(ys) + pad
    width = max_x - min_x
    height = max_y - min_y
    font_size = HEX_SIZE * 0.30

    parts: list[str] = []
    parts.append(
        '<div style="font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, sans-serif; '
        'display: flex; flex-direction: column; align-items: center; padding: 10px;">'
    )
    if title:
        parts.append(
            f'<div style="font-size: 15px; font-weight: 600; margin-bottom: 8px; '
            f'color: #334155; text-align: center;">{html.escape(title)}</div>'
        )

    parts.append(
        f'<svg viewBox="{min_x:.1f} {min_y:.1f} {width:.1f} {height:.1f}" '
        f'width="100%" style="max-width: 620px; height: auto; '
        f'background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 12px;">'
    )

    for pts, cx, cy, cell_id in triangles:
        given = puzzle_digits[cell_id] if cell_id < len(puzzle_digits) else BLANK
        solved = None if solution_digits is None else solution_digits[cell_id]

        if given != BLANK:
            fill, color, weight = "#dbeafe", "#1d4ed8", "700"
            label = str(given)
        elif solved is not None and 1 <= solved <= 6:
            fill, color, weight = "#dcfce7", "#047857", "700"
            label = str(solved)
        else:
            fill, color, weight = "#ffffff", "#94a3b8", "400"
            label = ""

        parts.append(
            f'<polygon points="{_points_attr(pts)}" fill="{fill}" '
            f'stroke="#cbd5e1" stroke-width="1.2"/>'
        )
        if label:
            parts.append(
                f'<text x="{cx:.2f}" y="{cy:.2f}" text-anchor="middle" '
                f'dominant-baseline="central" font-size="{font_size:.1f}" '
                f'font-weight="{weight}" fill="{color}" '
                f'style="user-select: none;">{label}</text>'
            )

    # Hexagon outlines on top of the triangles.
    seen_hex: set[tuple[int, int]] = set()
    for cp in topo["cell_positions"].values():
        key = (int(cp["q"]), int(cp["r"]))
        if key in seen_hex:
            continue
        seen_hex.add(key)
        parts.append(
            f'<polygon points="{_points_attr(_hex_outline(*key))}" fill="none" '
            f'stroke="#334155" stroke-width="2.4" stroke-linejoin="round"/>'
        )

    parts.append("</svg>")

    parts.append(
        '<div style="display: flex; gap: 16px; margin-top: 8px; font-size: 13px; '
        'color: #475569; flex-wrap: wrap; justify-content: center;">'
        '  <div style="display: flex; align-items: center; gap: 6px;">'
        '    <span style="display: inline-block; width: 14px; height: 14px; '
        'background: #dbeafe; border: 1px solid #1d4ed8; border-radius: 3px;"></span>'
        '    <span><b>Số đề bài (Givens)</b></span>'
        '  </div>'
        '  <div style="display: flex; align-items: center; gap: 6px;">'
        '    <span style="display: inline-block; width: 14px; height: 14px; '
        'background: #dcfce7; border: 1px solid #047857; border-radius: 3px;"></span>'
        '    <span><b>Số mô hình giải (Solved)</b></span>'
        '  </div>'
        '</div>'
    )
    parts.append("</div>")
    return "".join(parts)


def snowflake_example_info(example: SnowflakeExample) -> str:
    """Short Markdown summary shown under the example dropdown."""
    return (
        f"**Kích thước:** n={example.n} hexagons · **{example.n_cells} ô tam giác** · "
        f"**{example.givens} clues** · mã `{example.code}`\n\n"
        f"*{example.description}*"
    )
