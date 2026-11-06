"""Exp #1 — strict zero-shot: Sudoku-Extreme checkpoint on Killer Sudoku givens-only.

Killer Sudoku (jackcai1206/killer-sudoku-puzzles) has NO givens: every puzzle
is an all-blank 9x9 grid constrained only by cages. Givens-only input is
therefore x = all-ones [81, 9] for every puzzle; the model never sees any
cage/sum information.

Usage:
    modal run --detach experiments/killer/eval_zero_shot.py \
        --checkpoint /checkpoints/sudoku/seed0_4000s_bs512_aug1_20261003_040535_20261003_040535.pt
"""

import json

import modal
import torch
from torch import nn

from lattice_diffusion.models.looped_transformer import LoopedTransformerConfig, PowersetModel
from lattice_diffusion.modal.image import (
    CHECKPOINT_MOUNT,
    checkpoint_volume,
    image,
)
from lattice_diffusion.training.utils.checkpoint import load_checkpoint

from experiments.sudoku.dpll import StepConfig
from experiments.sudoku.ema import swap_in_ema_if_present
from experiments.sudoku.solve import SolveConfig, solve

app = modal.App("killer-zero-shot")


def _load_killer_rows():
    """Download killer parquet from HF (public, tiny) and parse rows."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(
        "jackcai1206/killer-sudoku-puzzles",
        "data/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    return pq.read_table(path).to_pylist()


def _parse_killer(row):
    """Return (cage_grid [81] str, sums dict, solution [81] int)."""
    ps = row["puzzle_string"]
    grid = ps[:81]
    rest = ps[81:].strip()
    sums = {}
    for part in rest.split(";"):
        if ":" in part:
            k, v = part.split(":")
            sums[k] = int(v)
    sol2d = row["first_solution"]
    sol = [int(d) for r in sol2d for d in r]
    return grid, sums, sol


@app.function(
    image=image,
    gpu="B200",
    timeout=3600,
    volumes={CHECKPOINT_MOUNT: checkpoint_volume},
)
def run(
    checkpoint: str,
    split: str = "test",
    seed: int = 0,
    max_rounds: int = 1000,
    n_chains: int = 64,
    batch_size: int = 512,
    eval_dropout_p: float = 0.05,
):
    import time

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {checkpoint}", flush=True)
    ckpt = load_checkpoint(checkpoint)
    cfg_loaded = LoopedTransformerConfig(**ckpt["model_cfg"])
    print(f"  model_cfg: {ckpt['model_cfg']}", flush=True)
    model = PowersetModel(cfg_loaded)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    swap_in_ema_if_present(model, ckpt)
    if eval_dropout_p > 0.0:
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = eval_dropout_p
            elif isinstance(m, nn.MultiheadAttention):
                m.dropout = eval_dropout_p
        model.train()
        print(f"  eval dropout p={eval_dropout_p}, model in train() mode", flush=True)

    rows = _load_killer_rows()
    print(f"  killer rows: {len(rows)}", flush=True)

    # Fixed split 70/15/15 (must match experiments/killer/split_seed0.json,
    # built with np.random.default_rng(0).permutation(960)).
    import numpy as np

    idx = np.random.default_rng(seed).permutation(len(rows)).tolist()
    n_train = int(0.7 * len(rows))
    n_val = int(0.15 * len(rows))
    parts = {
        "train": sorted(idx[:n_train]),
        "val": sorted(idx[n_train : n_train + n_val]),
        "test": sorted(idx[n_train + n_val :]),
    }
    sel = parts[split]
    print(f"  split={split} seed={seed} n={len(sel)}", flush=True)

    # Givens-only input: all-blank state for every puzzle (Killer has no givens).
    n = len(sel)
    x = torch.ones(n, 81, 9)
    y = torch.zeros(n, 81, 9)
    for j, i in enumerate(sel):
        _, _, sol = _parse_killer(rows[i])
        for s, d in enumerate(sol):
            y[j, s, d - 1] = 1.0
    given_mask = torch.zeros(n, 81, dtype=torch.bool)
    x = x.to(device).float()
    y = y.to(device).float()

    step_cfg = StepConfig(
        threshold=0.10, temp_decide=1.5, cls_threshold=0.6, augment=False,
    )
    solve_cfg = SolveConfig(
        step=step_cfg, max_rounds=max_rounds, n_chains=n_chains,
        batch_size=batch_size,
    )
    res = solve(model, x, y, given_mask, solve_cfg)

    n_correct = int(res.correct.sum().item())
    n_wrong = int(res.wrong.sum().item())
    n_timeout = int(res.timeouts.sum().item())
    eval_secs = time.time() - t0

    print(f"\n{'=' * 60}\nEXP#1 RESULT (givens-only zero-shot, split={split})\n{'=' * 60}", flush=True)
    print(f"  correct={n_correct}/{n}  wrong={n_wrong}  timeouts={n_timeout}", flush=True)
    print(f"  FSR = {n_correct / max(n, 1):.4f}", flush=True)
    print(f"  model_calls={res.model_calls}  eval_secs={eval_secs:.1f}", flush=True)
    print(f"{'=' * 60}", flush=True)

    out_path = f"{CHECKPOINT_MOUNT}/killer/exp1_zeroshot_{split}_seed{seed}.eval.json"
    import pathlib

    pathlib.Path(f"{CHECKPOINT_MOUNT}/killer").mkdir(parents=True, exist_ok=True)
    pathlib.Path(out_path).write_text(json.dumps({
        "experiment": "exp1_zeroshot_givens_only",
        "checkpoint": checkpoint,
        "split": split,
        "seed": seed,
        "n": n,
        "correct": n_correct,
        "wrong": n_wrong,
        "timeouts": n_timeout,
        "fsr": n_correct / max(n, 1),
        "model_calls": res.model_calls,
        "eval_secs": eval_secs,
        "max_rounds": max_rounds,
        "n_chains": n_chains,
    }, indent=2))
    checkpoint_volume.commit()

    return {
        "split": split, "n": n, "correct": n_correct,
        "wrong": n_wrong, "timeouts": n_timeout,
        "fsr": n_correct / max(n, 1),
    }


@app.local_entrypoint()
def entrypoint(
    checkpoint: str,
    split: str = "test",
    seed: int = 0,
    max_rounds: int = 1000,
    n_chains: int = 64,
    batch_size: int = 512,
    eval_dropout_p: float = 0.05,
):
    result = run.remote(
        checkpoint=checkpoint, split=split, seed=seed,
        max_rounds=max_rounds, n_chains=n_chains,
        batch_size=batch_size, eval_dropout_p=eval_dropout_p,
    )
    print(f"\nFinal: {result}", flush=True)
