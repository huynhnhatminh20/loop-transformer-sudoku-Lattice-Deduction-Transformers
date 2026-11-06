"""Utility functions for Sudoku parsing, validation, and HTML rendering."""

from __future__ import annotations
import re
from typing import Any


def parse_sudoku_input(raw: str) -> str:
    """Parse raw user input into an 81-character puzzle string.
    
    Accepts:
    - 81-char one-line string (e.g. '3..98.5...8.67...')
    - Multi-line 9x9 grid text (space-separated, comma-separated, pipe-separated)
    - Replaces '0', '.', '_', '-' with '.'
    
    Returns:
        81-character string containing digits '1'-'9' and '.' for blanks.
    """
    if not raw or not isinstance(raw, str):
        raise ValueError("Input string is empty.")
    
    # Clean whitespace and comments if any
    cleaned = re.sub(r"[#//].*", "", raw)
    
    # Extract only valid characters: digits 0-9 and blank markers (. _ -)
    tokens = []
    for char in cleaned:
        if char in "123456789":
            tokens.append(char)
        elif char in "0._-":
            tokens.append(".")
    
    result = "".join(tokens)
    if len(result) != 81:
        raise ValueError(
            f"Expected exactly 81 puzzle cells, but found {len(result)} valid characters. "
            "Please provide a 9x9 grid or 81-character string."
        )
    return result


def format_sudoku_grid(puzzle_str: str) -> str:
    """Format an 81-char puzzle string into standard ASCII 9x9 grid format."""
    if len(puzzle_str) != 81:
        return puzzle_str
    
    lines = []
    lines.append("+-------+-------+-------+")
    for r in range(9):
        row_chars = []
        for c in range(9):
            idx = r * 9 + c
            ch = puzzle_str[idx] if puzzle_str[idx] != "." else "·"
            row_chars.append(ch)
            if c % 3 == 2 and c != 8:
                row_chars.append("|")
        line = "| " + " ".join(row_chars[:3]) + " | " + " ".join(row_chars[4:7]) + " | " + " ".join(row_chars[8:]) + " |"
        lines.append(line)
        if r % 3 == 2:
            lines.append("+-------+-------+-------+")
    return "\n".join(lines)


def validate_sudoku_solution(solution_str: str, givens_str: str = "") -> dict[str, Any]:
    """Validate whether solution_str is a fully-filled valid Sudoku solution.
    
    Checks:
    - 81 characters, all in '1'-'9' (no empty '.')
    - Every row contains 1-9 without duplicates
    - Every column contains 1-9 without duplicates
    - Every 3x3 block contains 1-9 without duplicates
    - Every given in givens_str matches the corresponding cell in solution_str
    """
    errors: list[str] = []
    
    if len(solution_str) != 81:
        return {
            "is_valid": False,
            "errors": [f"Invalid length: {len(solution_str)} (expected 81)"],
            "rows_valid": False,
            "cols_valid": False,
            "boxes_valid": False,
            "givens_match": False,
            "summary": "Invalid string length",
        }
    
    if "." in solution_str:
        blank_count = solution_str.count(".")
        errors.append(f"Grid contains {blank_count} unfilled cell(s).")
    
    grid = []
    for r in range(9):
        row = []
        for c in range(9):
            char = solution_str[r * 9 + c]
            row.append(int(char) if char.isdigit() else 0)
        grid.append(row)
    
    rows_valid = True
    cols_valid = True
    boxes_valid = True
    givens_match = True
    
    # 1. Row checks
    for r in range(9):
        row_vals = [grid[r][c] for c in range(9) if grid[r][c] != 0]
        if len(row_vals) != len(set(row_vals)) or len(row_vals) != 9:
            rows_valid = False
            missing = set(range(1, 10)) - set(row_vals)
            errors.append(f"Row {r+1} constraint violated (missing: {sorted(missing)})")
    
    # 2. Column checks
    for c in range(9):
        col_vals = [grid[r][c] for r in range(9) if grid[r][c] != 0]
        if len(col_vals) != len(set(col_vals)) or len(col_vals) != 9:
            cols_valid = False
            missing = set(range(1, 10)) - set(col_vals)
            errors.append(f"Column {c+1} constraint violated (missing: {sorted(missing)})")
            
    # 3. 3x3 Block checks
    for br in range(3):
        for bc in range(3):
            box_vals = []
            for r in range(br * 3, (br + 1) * 3):
                for c in range(bc * 3, (bc + 1) * 3):
                    val = grid[r][c]
                    if val != 0:
                        box_vals.append(val)
            if len(box_vals) != len(set(box_vals)) or len(box_vals) != 9:
                boxes_valid = False
                missing = set(range(1, 10)) - set(box_vals)
                errors.append(f"Block ({br+1},{bc+1}) violated (missing: {sorted(missing)})")
                
    # 4. Givens check
    if givens_str and len(givens_str) == 81:
        mismatches = 0
        for i in range(81):
            g = givens_str[i]
            s = solution_str[i]
            if g not in ".0" and g != s:
                mismatches += 1
        if mismatches > 0:
            givens_match = False
            errors.append(f"{mismatches} given clue(s) were altered in the solution.")
            
    is_valid = (
        rows_valid and cols_valid and boxes_valid and givens_match and ("." not in solution_str)
    )
    
    if is_valid:
        summary = "Hợp lệ (Đúng luật Sudoku 100%)"
    else:
        summary = f"Chưa hoàn tất / Lỗi ({len(errors)} vi phạm)"
        
    return {
        "is_valid": is_valid,
        "errors": errors,
        "rows_valid": rows_valid,
        "cols_valid": cols_valid,
        "boxes_valid": boxes_valid,
        "givens_match": givens_match,
        "summary": summary,
    }


def render_board_html(puzzle_str: str, solution_str: str | None = None, title: str = "") -> str:
    """Render a clean, modern, responsive 9x9 Sudoku board in HTML/CSS.
    
    - Given numbers: bold dark navy (#1e3a8a / dark: #60a5fa) with soft gray background.
    - Model filled numbers: bold vibrant emerald (#059669 / dark: #34d399) with light green background.
    - Blank cells: subtle placeholder dot.
    - 3x3 subgrid borders clearly emphasized.
    """
    if len(puzzle_str) != 81:
        puzzle_str = "." * 81
    if solution_str and len(solution_str) != 81:
        solution_str = None
        
    html = [
        '<div style="font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, sans-serif; '
        'display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 12px;">'
    ]
    
    if title:
        html.append(f'<div style="font-size: 16px; font-weight: 600; margin-bottom: 10px; color: #334155;">{title}</div>')
        
    html.append(
        '<div style="background: #ffffff; border: 3px solid #0f172a; border-radius: 8px; '
        'box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -2px rgba(0,0,0,0.1); '
        'overflow: hidden; max-width: 420px; width: 100%;">'
    )
    html.append('<table style="border-collapse: collapse; width: 100%; aspect-ratio: 1; table-layout: fixed;">')
    
    for r in range(9):
        html.append('<tr>')
        for c in range(9):
            idx = r * 9 + c
            orig_char = puzzle_str[idx]
            sol_char = solution_str[idx] if solution_str else "."
            
            is_given = (orig_char in "123456789")
            
            # Border styles for 3x3 grid delineation
            border_bottom = "3px solid #0f172a" if (r % 3 == 2 and r != 8) else "1px solid #cbd5e1"
            border_right = "3px solid #0f172a" if (c % 3 == 2 and c != 8) else "1px solid #cbd5e1"
            
            if is_given:
                val = orig_char
                bg_color = "#f1f5f9"
                text_color = "#1e40af"
                font_weight = "800"
            elif sol_char in "123456789":
                val = sol_char
                bg_color = "#ecfdf5"
                text_color = "#047857"
                font_weight = "700"
            else:
                val = '<span style="color: #cbd5e1; font-size: 14px;">·</span>'
                bg_color = "#ffffff"
                text_color = "#94a3b8"
                font_weight = "400"
                
            cell_style = (
                f"border-bottom: {border_bottom}; border-right: {border_right}; "
                f"background-color: {bg_color}; color: {text_color}; font-weight: {font_weight}; "
                f"text-align: center; vertical-align: middle; font-size: 20px; "
                f"height: 42px; width: 42px; user-select: none;"
            )
            html.append(f'<td style="{cell_style}">{val}</td>')
        html.append('</tr>')
        
    html.append('</table>')
    html.append('</div>')
    
    # Legend
    html.append(
        '<div style="display: flex; gap: 16px; margin-top: 10px; font-size: 13px; color: #475569;">'
        '  <div style="display: flex; align-items: center; gap: 6px;">'
        '    <span style="display: inline-block; width: 14px; height: 14px; background: #f1f5f9; border: 1px solid #1e40af; border-radius: 3px;"></span>'
        '    <span><b>Số đề bài (Givens)</b></span>'
        '  </div>'
        '  <div style="display: flex; align-items: center; gap: 6px;">'
        '    <span style="display: inline-block; width: 14px; height: 14px; background: #ecfdf5; border: 1px solid #047857; border-radius: 3px;"></span>'
        '    <span><b>Số mô hình giải (Solved)</b></span>'
        '  </div>'
        '</div>'
    )
    
    html.append('</div>')
    return "".join(html)

