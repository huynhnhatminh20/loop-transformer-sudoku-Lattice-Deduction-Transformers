"""CPU-only dataset adapter for scaled-sudoku-instances (9x9) -> LDT format.

Reads the frozen ancestor-group split metadata (`scaled_sudoku_split.jsonl`,
produced by the audit script) and exposes puzzles as LDT powerset tensors:

    blank `.` -> [1,1,1,1,1,1,1,1,1]   (all candidates alive)
    digit d   -> one-hot(d)            (given)

Output per sample: x [81,9] float32, y [81,9] float32, is_sat bool.
Row metadata (ancestor_id, is_base, walk/step, clues, hardness, split) is
carried alongside so the Base -> Walk -> Variant trajectory is never flattened
away.

No torch import at module level (numpy only) so this stays import-safe on CPU.
No training loop here -- the training phase reuses experiments/sudoku/train.py
with this adapter swapped in for SudokuExtremeDataset.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue

import numpy as np

DATA_ROOT = Path(__file__).resolve().parents[3] / "scaled-sudoku-instances" / \
    "scaled-sudoku-instances"
SPLIT_FILE = Path(__file__).resolve().parents[3] / "scaled_sudoku_split.jsonl"


@dataclass
class ScaledSudokuConfig:
    split: str = "train"  # "train" | "val" | "test"
    split_file: str = str(SPLIT_FILE)
    data_root: str = str(DATA_ROOT)
    seed: int = 42

    # Sample type weights (mirrors SudokuExtremeConfig semantics)
    zero_hint_weight: float = 0.20
    correct_hint_weight: float = 0.55
    error_hint_weight: float = 0.25
    error_fill_range: tuple[float, float] = (0.1, 1.0)
    error_rate_range: tuple[float, float] = (0.01, 0.30)
    correct_fill_range: tuple[float, float] = (0.0, 1.0)

    prefetch_batches: int = 2
    batch_size: int = 512


def encode_grid(puzzle: str) -> np.ndarray:
    """81-char puzzle string -> [81, 9] float32 powerset array."""
    x = np.zeros((81, 9), dtype=np.float32)
    for i, c in enumerate(puzzle):
        if c == ".":
            x[i] = 1.0
        else:
            x[i, int(c) - 1] = 1.0
    return x


def encode_solution(solution: str) -> np.ndarray:
    """81-char solution string -> [81, 9] float32 one-hot array."""
    y = np.zeros((81, 9), dtype=np.float32)
    for i, c in enumerate(solution):
        y[i, int(c) - 1] = 1.0
    return y


def _make_sample(puzzle: np.ndarray, solution: np.ndarray, rng: np.random.Generator,
                 sample_type: str, cfg: ScaledSudokuConfig):
    """Same hint-type semantics as sudoku_extreme._make_sample (pure numpy).

    `puzzle` is the real givens encoding ([81,9]: givens one-hot, blanks all-ones);
    hints are added only on blank (multi-alive) cells.
    """
    x = puzzle.astype(np.float32)
    y_sol = solution.astype(np.float32)
    if sample_type == "zero_hints":
        return x, y_sol, True
    blank_mask = x.sum(axis=1) > 1.5
    blank_indices = np.where(blank_mask)[0]
    n_blanks = len(blank_indices)
    if n_blanks == 0:
        return x, y_sol, True
    if sample_type == "correct_hints":
        lo, hi = cfg.correct_fill_range
        n_fill = int(rng.uniform(lo, hi) * n_blanks)
        if n_fill > 0:
            to_fill = rng.choice(blank_indices, size=n_fill, replace=False)
            x[to_fill] = y_sol[to_fill]
        return x, y_sol, True
    if sample_type == "error_hints":
        lo, hi = cfg.error_fill_range
        n_fill = max(1, int(rng.uniform(lo, hi) * n_blanks))
        to_fill = rng.choice(blank_indices, size=n_fill, replace=False)
        x[to_fill] = y_sol[to_fill]
        elo, ehi = cfg.error_rate_range
        n_corrupt = max(1, int(rng.uniform(elo, ehi) * n_fill))
        to_corrupt = rng.choice(to_fill, size=n_corrupt, replace=False)
        for idx in to_corrupt:
            correct = int(y_sol[idx].argmax())
            wrong = rng.choice([d for d in range(9) if d != correct])
            x[idx] = 0.0
            x[idx, wrong] = 1.0
        return x, np.zeros((81, 9), dtype=np.float32), False
    raise ValueError(f"Unknown sample_type: {sample_type}")


class ScaledSudokuDataset:
    """Streaming adapter over the frozen ancestor-group split.

    next_batch() -> (x [B,81,9] float, y [B,81,9] float, is_sat [B] bool).
    """

    def __init__(self, cfg: ScaledSudokuConfig):
        self.cfg = cfg
        self.rows = []
        with open(cfg.split_file) as fh:
            for line in fh:
                r = json.loads(line)
                if r["split"] == cfg.split:
                    self.rows.append(r)
        if not self.rows:
            raise ValueError(f"No rows for split={cfg.split} in {cfg.split_file}")
        self.rng = np.random.default_rng(cfg.seed)
        weights = np.array([cfg.zero_hint_weight, cfg.correct_hint_weight,
                            cfg.error_hint_weight])
        self.type_probs = weights / weights.sum()
        self.type_names = ["zero_hints", "correct_hints", "error_hints"]
        self._order = self.rng.permutation(len(self.rows))
        self._pos = 0
        self._queue: Queue = Queue(maxsize=cfg.prefetch_batches)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._thread.start()

    def _next_sample(self):
        if self._pos >= len(self.rows):
            self._order = self.rng.permutation(len(self.rows))
            self._pos = 0
        r = self.rows[self._order[self._pos]]
        self._pos += 1
        puzz = encode_grid(r["puzzle"])
        y_sol = encode_solution(r["solution"])
        sample_type = self.rng.choice(self.type_names, p=self.type_probs)
        return _make_sample(puzz, y_sol, self.rng, sample_type, self.cfg)

    def _prefetch_loop(self):
        while not self._stop.is_set():
            bx, by, bs = [], [], []
            for _ in range(self.cfg.batch_size):
                x, y, s = self._next_sample()
                bx.append(x)
                by.append(y)
                bs.append(s)
            try:
                import torch
                self._queue.put((
                    torch.from_numpy(np.stack(bx)),
                    torch.from_numpy(np.stack(by)),
                    torch.tensor(bs, dtype=torch.bool),
                ), timeout=1.0)
            except Exception:
                if self._stop.is_set():
                    break

    def next_batch(self):
        return self._queue.get()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=5.0)

    def __del__(self):
        self.close()
