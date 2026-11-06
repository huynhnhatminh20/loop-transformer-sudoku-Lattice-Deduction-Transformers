"""Curated benchmark Sudoku examples for testing the Gradio Demo."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class SudokuExample:
    name: str
    difficulty: str
    clues_count: int
    puzzle: str
    solution: str
    description: str


EXAMPLES: list[SudokuExample] = [
    SudokuExample(
        name="Classic Easy (34 clues)",
        difficulty="Easy",
        clues_count=34,
        puzzle="53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79",
        solution="534678912672195348198342567859761423426853791713924856961537284287419635345286179",
        description="Standard beginner puzzle. Mostly solvable by pure deduction in 1-2 DPLL rounds."
    ),
    SudokuExample(
        name="Scaled Sudoku - Medium (27 clues)",
        difficulty="Medium",
        clues_count=27,
        puzzle=".29..........2369.6....8..44...3.96....49.3....6.72...5.........31.4.72..4...7...",
        solution="729654813184723695653918274478135962215496387396872541567281439831549726942367158",
        description="Moderate puzzle from Scaled Sudoku benchmark (ValBS hardness level)."
    ),
    SudokuExample(
        name="Scaled Sudoku - Hard (25 clues)",
        difficulty="Hard",
        clues_count=25,
        puzzle="3.8.2....64..3.7.2.......85.7.....14....6.5..4....12.....8.9......6.34..15.......",
        solution="398725641645138792217946385576392814821467539439581276764859123982613457153274968",
        description="Challenging instance with 25 clues requiring stochastic branching and unit propagation."
    ),
    SudokuExample(
        name="Scaled Sudoku - Extreme #1 (24 clues)",
        difficulty="Extreme",
        clues_count=24,
        puzzle="3..98.5...8.67...9..........3...........5..2.......41365..2.8..4..1.....8..5..94.",
        solution="327981564184675239596243187238714695941356728765892413653429871479168352812537946",
        description="Instance 1 from scaled_sudoku_smoke.jsonl, 24 givens with non-trivial branching."
    ),
    SudokuExample(
        name="Scaled Sudoku - Minimal Clues (22 clues)",
        difficulty="Near-Minimal",
        clues_count=22,
        puzzle="3..2.5....8....4.7.6..3....2....31....3...68...8...........7..57...9...2.4......1",
        solution="374285916582916437169734528297863154453172689618459273821647395736591842945328761",
        description="Sparse 22-clue puzzle approaching the theoretical 17-clue minimum for unique Sudoku."
    ),
]


def get_example_by_name(name: str) -> SudokuExample | None:
    for ex in EXAMPLES:
        if ex.name == name:
            return ex
    return None

