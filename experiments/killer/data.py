"""Killer Sudoku dataset for Exp #2 (LDT + Cage/Sum Input, no Sum Operator).

Source: jackcai1206/killer-sudoku-puzzles (960 puzzles, all solution_count=1).
Format per row: puzzle_string = 81 cage-letters + '\\n' + 'A:35;B:6;...',
first_solution = 9x9 ground truth. Killer has NO givens.

Input representation (Exp #2):
  state [B, 81, 12] = 9 candidate channels + 3 locked auxiliary channels:
    ch  9: cage_id_norm    (per-puzzle remapped 0..K-1, divided by 41)
    ch 10: sum_norm        (cage target sum / 45)
    ch 11: pos_in_cage_norm (row-major index inside cage / max(size-1,1))
  StepConfig(vocab_dim=9): deduce/decide never touch channels 9..11.

IMPORTANT augmentation note: digit permutation is INVALID for Killer
(permuting digit labels breaks cage sums), so the trainer must use
StepConfig(permute_digits=False) with dihedral-only spatial aug
(rotation/reflection moves cages rigidly, sums stay glued to cells).

Fixed split 70/15/15 via np.random.default_rng(seed).permutation(960),
identical to experiments/killer/split_seed0.json (seed=0).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from queue import Queue

import numpy as np
import torch

SEQ_LEN = 81
VOCAB = 9
N_AUX = 3
N_CHANNELS = VOCAB + N_AUX  # 12

MAX_CAGES_DENOM = 41  # max observed num_cages (42) - 1
MAX_SUM_DENOM = 45    # max observed target sum


@dataclass
class KillerConfig:
    split: str = "train"  # "train" | "val" | "test"
    seed: int = 0         # split seed (70/15/15 permutation)
    n_puzzles: int | None = None  # cap puzzles in split (None = all)

    # Sample type weights (mirrors SudokuExtremeConfig semantics)
    zero_hint_weight: float = 0.20
    correct_hint_weight: float = 0.55
    error_hint_weight: float = 0.25
    error_fill_range: tuple[float, float] = (0.1, 1.0)
    error_rate_range: tuple[float, float] = (0.01, 0.30)
    correct_fill_range: tuple[float, float] = (0.0, 1.0)

    prefetch_batches: int = 2
    batch_size: int = 128


def download_killer_rows() -> list[dict]:
    """Download the killer parquet from HF (public, ~960 rows, tiny)."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(
        "jackcai1206/killer-sudoku-puzzles",
        "data/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    return pq.read_table(path).to_pylist()


def parse_killer(row: dict) -> tuple[np.ndarray, dict, np.ndarray]:
    """Return (cage_letters [81] str array, sums dict, solution [81] int 1..9)."""
    ps = row["puzzle_string"]
    grid = ps[:81]
    rest = ps[81:].strip()
    sums: dict[str, int] = {}
    for part in rest.split(";"):
        if ":" in part:
            k, v = part.split(":")
            sums[k] = int(v)
    sol2d = row["first_solution"]
    sol = np.array([int(d) for r in sol2d for d in r], dtype=np.int64)
    return np.array(list(grid)), sums, sol


def cage_features(letters: np.ndarray, sums: dict) -> np.ndarray:
    """Aux features [81, 3] float32: (cage_id_norm, sum_norm, pos_in_cage_norm).

    Cage letters are arbitrary per puzzle; remap to 0..K-1 in sorted-letter
    order (deterministic). Only the *equality* signal ("same cage") is
    meaningful, not the id values themselves.
    """
    uniq = sorted(set(letters.tolist()))
    remap = {c: i for i, c in enumerate(uniq)}
    aux = np.zeros((SEQ_LEN, N_AUX), dtype=np.float32)
    for s in range(SEQ_LEN):
        c = letters[s]
        cid = remap[c]
        tgt = sums[c]
        members = np.where(letters == c)[0]
        pos = int(np.where(members == s)[0][0])
        denom = max(len(members) - 1, 1)
        aux[s, 0] = cid / MAX_CAGES_DENOM
        aux[s, 1] = tgt / MAX_SUM_DENOM
        aux[s, 2] = pos / denom
    return aux


def _make_sample(
    answer: np.ndarray,
    rng: np.random.Generator,
    sample_type: str,
    cfg: KillerConfig,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Generate one powerset sample on an all-blank base (Killer has no givens).

    Returns (x_vocab [81,9] float32, y [81,9] float32, is_sat).
    """
    y_sol = np.zeros((81, 9), dtype=np.float32)
    y_sol[np.arange(81), answer - 1] = 1.0
    x = np.ones((81, 9), dtype=np.float32)  # blank base

    if sample_type == "zero_hints":
        return x, y_sol, True

    blank_indices = np.arange(81)
    if sample_type == "correct_hints":
        lo, hi = cfg.correct_fill_range
        n_fill = int(rng.uniform(lo, hi) * 81)
        if n_fill > 0:
            to_fill = rng.choice(blank_indices, size=n_fill, replace=False)
            x[to_fill] = y_sol[to_fill]
        return x, y_sol, True

    if sample_type == "error_hints":
        lo, hi = cfg.error_fill_range
        n_fill = max(1, int(rng.uniform(lo, hi) * 81))
        to_fill = rng.choice(blank_indices, size=n_fill, replace=False)
        x[to_fill] = y_sol[to_fill]
        elo, ehi = cfg.error_rate_range
        n_corrupt = max(1, int(rng.uniform(elo, ehi) * n_fill))
        to_corrupt = rng.choice(to_fill, size=n_corrupt, replace=False)
        for idx in to_corrupt:
            correct_digit = int(y_sol[idx].argmax())
            wrong_digit = rng.choice([d for d in range(9) if d != correct_digit])
            x[idx] = 0.0
            x[idx, wrong_digit] = 1.0
        return x, np.zeros((81, 9), dtype=np.float32), False

    raise ValueError(f"Unknown sample_type: {sample_type}")


class KillerDataset:
    """Streaming Killer dataset. Batch: (x_vocab, y, aux, is_sat)."""

    def __init__(self, cfg: KillerConfig):
        self.cfg = cfg
        rows = download_killer_rows()
        n = len(rows)
        perm = np.random.default_rng(cfg.seed).permutation(n).tolist()
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        parts = {
            "train": sorted(perm[:n_train]),
            "val": sorted(perm[n_train : n_train + n_val]),
            "test": sorted(perm[n_train + n_val :]),
        }
        sel = parts[cfg.split]
        self.rng = np.random.default_rng(cfg.seed + 1000 + hash(cfg.split) % 1000)
        if cfg.n_puzzles is not None and cfg.n_puzzles < len(sel):
            sub = self.rng.choice(len(sel), size=cfg.n_puzzles, replace=False)
            sub.sort()
            sel = [sel[i] for i in sub]

        self.answers: list[np.ndarray] = []
        self.auxes: list[np.ndarray] = []
        for i in sel:
            letters, sums, sol = parse_killer(rows[i])
            self.answers.append(sol)
            self.auxes.append(cage_features(letters, sums))
        self.n_puzzles = len(sel)

        weights = np.array([cfg.zero_hint_weight, cfg.correct_hint_weight,
                            cfg.error_hint_weight])
        self.type_probs = weights / weights.sum()
        self.type_names = ["zero_hints", "correct_hints", "error_hints"]

        self._order = self.rng.permutation(self.n_puzzles)
        self._pos = 0
        self._queue: Queue = Queue(maxsize=cfg.prefetch_batches)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._thread.start()

    def _next_sample(self):
        if self._pos >= self.n_puzzles:
            self._order = self.rng.permutation(self.n_puzzles)
            self._pos = 0
        idx = self._order[self._pos]
        self._pos += 1
        sample_type = self.rng.choice(self.type_names, p=self.type_probs)
        x, y, is_sat = _make_sample(self.answers[idx], self.rng, sample_type, self.cfg)
        return x, y, self.auxes[idx], is_sat

    def _prefetch_loop(self):
        while not self._stop.is_set():
            bx, by, ba, bs = [], [], [], []
            for _ in range(self.cfg.batch_size):
                x, y, a, s = self._next_sample()
                bx.append(x)
                by.append(y)
                ba.append(a)
                bs.append(s)
            try:
                self._queue.put((
                    torch.from_numpy(np.stack(bx)),
                    torch.from_numpy(np.stack(by)),
                    torch.from_numpy(np.stack(ba)),
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
