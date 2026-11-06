"""FULL: LDT from scratch on Scaled Sudoku Instances (4000 steps, B200).

Flow: train on TRAIN split -> final eval on full VAL split -> final eval on
full TEST split (once each) -> breakdowns (base/variant, clues, hardness,
walk) -> JSON artifacts. Test is used only after training completes.

Usage:
    modal run --detach experiments/scaled/run.py
"""

import dataclasses
import json
import time
from pathlib import Path

import modal
import torch
from torch import nn

from lattice_diffusion.models.looped_transformer import (
    LoopedTransformerConfig, PowersetModel)
from lattice_diffusion.modal.image import (
    CHECKPOINT_MOUNT, checkpoint_volume, image)
from lattice_diffusion.training.utils.checkpoint import load_checkpoint
from lattice_diffusion.data.scaled_sudoku import (
    ScaledSudokuConfig, encode_grid, encode_solution)

from experiments.sudoku.dpll import StepConfig
from experiments.sudoku.ema import swap_in_ema_if_present
from experiments.sudoku.solve import SolveConfig, solve
from experiments.scaled.train import TrainConfig, train

SPLIT_LOCAL = "scaled_sudoku_split.jsonl"
SPLIT_REMOTE = "/smoke/scaled_sudoku_split.jsonl"

full_image = image.add_local_file(SPLIT_LOCAL, SPLIT_REMOTE)

app = modal.App("scaled-sudoku-full")


def _load_rows(split):
    rows = []
    with open(SPLIT_REMOTE) as fh:
        for line in fh:
            r = json.loads(line)
            if r["split"] == split:
                rows.append(r)
    assert rows and all(r["split"] == split for r in rows)
    return rows


def _encode_rows(rows, device):
    import numpy as np
    X = np.stack([encode_grid(r["puzzle"]) for r in rows])
    Y = np.stack([encode_solution(r["solution"]) for r in rows])
    xt = torch.from_numpy(X).to(device).float()
    yt = torch.from_numpy(Y).to(device).float()
    return xt, yt, (xt.sum(dim=-1) == 1)


@app.function(image=full_image, gpu="B200", timeout=3600 * 4,
              volumes={CHECKPOINT_MOUNT: checkpoint_volume})
def run(steps: int = 4000, batch_size: int = 512, seed: int = 0):
    t_all = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    print(f"device={device} gpu={torch.cuda.get_device_name(0)}", flush=True)

    step_cfg = StepConfig(threshold=0.10, temp_decide=1.5, cls_threshold=0.5,
                          augment=True)
    model_cfg = LoopedTransformerConfig(cls_token=True)
    t0 = time.time()
    ckpt_path = train(TrainConfig(
        steps=steps, batch_size=batch_size, seed=seed,
        lr=3e-3, weight_decay=0.1,
        bce_pos_mult=4.0, bce_neg_mult=0.5,
        softmax_loss_weight=0.2, conflict_loss_weight=0.1,
        warmup_fraction=0.1, step=step_cfg, max_age=100,
        log_every=100, eval_every=100,
        eval_n_puzzles=200, eval_max_rounds=5, eval_n_chains=64,
        model=model_cfg,
        data=ScaledSudokuConfig(split="train", split_file=SPLIT_REMOTE,
                                batch_size=batch_size, seed=42),
        out_dir=f"{CHECKPOINT_MOUNT}/scaled",
        name=f"seed{seed}_{steps}s_bs{batch_size}",
    ))
    train_s = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"TRAIN done in {train_s:.0f}s peakVRAM={peak_gb:.2f}GB", flush=True)
    checkpoint_volume.commit()

    ckpt = load_checkpoint(str(ckpt_path))
    model = PowersetModel(LoopedTransformerConfig(**ckpt["model_cfg"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    swap_in_ema_if_present(model, ckpt)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.05
        elif isinstance(m, nn.MultiheadAttention):
            m.dropout = 0.05
    model.train()  # official dropout-noise eval mode
    print("eval dropout p=0.05 (official run.py default)", flush=True)

    eval_step_cfg = dataclasses.replace(step_cfg, cls_threshold=0.6)
    results = {}
    for split in ["val", "test"]:
        rows = _load_rows(split)
        x, y, gm = _encode_rows(rows, device)
        t1 = time.time()
        res = solve(model, x, y, gm,
                    SolveConfig(step=eval_step_cfg, max_rounds=50,
                                n_chains=8, batch_size=512),
                    verbose=False)
        es = time.time() - t1
        corr = res.correct.cpu().tolist()
        n = len(rows)
        summ = {"total": n, "terminated": n,
                "correct": int(res.correct.sum()),
                "wrong": int(res.wrong.sum()),
                "timeout": int(res.timeouts.sum()),
                "calls": res.model_calls, "eval_s": es}
        summ["accuracy"] = summ["correct"] / n
        # breakdowns
        bv = {"base": [0, 0], "variant": [0, 0]}
        clues, hard, walk = {}, {}, {}
        for r, c in zip(rows, corr):
            k = "base" if r["is_base"] else "variant"
            bv[k][1] += 1
            bv[k][0] += c
            for d, key in ((clues, r["clues_count"]),
                           (hard, r["hardness_level"]),
                           (walk, r["walk_index"])):
                d.setdefault(key, [0, 0])
                d[key][1] += 1
                d[key][0] += c
        summ["base_variant"] = bv
        summ["by_clues"] = {str(k): v for k, v in sorted(
            clues.items(), key=lambda kv: kv[0])}
        summ["by_hardness"] = {str(k): v for k, v in sorted(
            hard.items(), key=lambda kv: kv[0])}
        summ["by_walk"] = {str(k): v for k, v in sorted(
            walk.items(), key=lambda kv: str(kv[0]))}
        results[split] = summ
        print(f"EVAL {split}: correct={summ['correct']}/{n} "
              f"acc={summ['accuracy']:.4f} wrong={summ['wrong']} "
              f"timeout={summ['timeout']} calls={summ['calls']} "
              f"({es:.0f}s)", flush=True)
        print(f"  base={bv['base']} variant={bv['variant']}", flush=True)
        print(f"  by_hardness={summ['by_hardness']}", flush=True)

    # integrity re-check on frozen file
    anc = {"train": set(), "val": set(), "test": set()}
    puz = {"train": set(), "val": set(), "test": set()}
    with open(SPLIT_REMOTE) as fh:
        for line in fh:
            r = json.loads(line)
            anc[r["split"]].add(r["ancestor_id"])
            puz[r["split"]].add(r["puzzle"])
    integ = {
        "anc_tr_va": len(anc["train"] & anc["val"]),
        "anc_tr_te": len(anc["train"] & anc["test"]),
        "anc_va_te": len(anc["val"] & anc["test"]),
        "puz_tr_va": len(puz["train"] & puz["val"]),
        "puz_tr_te": len(puz["train"] & puz["test"]),
        "puz_va_te": len(puz["val"] & puz["test"]),
    }
    print(f"INTEGRITY: {integ}", flush=True)
    assert all(v == 0 for v in integ.values())

    total_s = time.time() - t_all
    # Budget accounting. LDT trains on a pool of exactly batch_size
    # distinct puzzles per step (train.py:335 randperm(pool)[:bs]),
    # each row under 1 augmentation view -> examples seen == unique
    # puzzle draws. Recorded explicitly so the report never has to
    # recompute it from (steps x batch_size).
    out = {"checkpoint": str(ckpt_path), "steps": steps,
           "batch_size": batch_size, "seed": seed,
           "examples_seen": steps * batch_size,
           "unique_puzzle_draws": steps * batch_size,
           "train_s": train_s, "peak_gb": peak_gb,
           "total_s": total_s, "integrity": integ,
           "val": results["val"], "test": results["test"]}
    p = Path(str(ckpt_path)).with_suffix(".eval.json")
    p.write_text(json.dumps(out, indent=2))
    checkpoint_volume.commit()
    print(f"WROTE {p} total={total_s:.0f}s", flush=True)
    return out


@app.local_entrypoint()
def entrypoint():
    print(json.dumps(run.remote(), indent=2)[:2000], flush=True)
