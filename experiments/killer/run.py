"""Exp #2 FULL: LDT + Cage/Sum Input, train on Killer, eval once on test.

Flow: partial-init from Sudoku-Extreme ckpt -> train on Killer train split
-> final eval (full protocol) on Killer test split, exactly once.

NO Sum Deduction Operator in this version (per spec).

Usage:
    modal run --detach experiments/killer/run.py \
        --init-from /checkpoints/sudoku/seed0_4000s_bs512_aug1_20261003_040535_20261003_040535.pt \
        --steps 2000 --batch-size 512
"""

import json
import time
from pathlib import Path

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
from experiments.killer.data import KillerConfig, KillerDataset
from experiments.killer.train import (
    SEQ_LEN, TrainConfig, _build_state, _given_mask, train,
)

app = modal.App("killer")

wandb_secret = modal.Secret.from_name("wandb-api")


@app.function(
    image=image,
    gpu="B200",
    timeout=3600 * 4,
    secrets=[wandb_secret],
    volumes={CHECKPOINT_MOUNT: checkpoint_volume},
)
def run(
    init_from: str,
    steps: int = 2000,
    batch_size: int = 512,
    seed: int = 0,
    lr: float = 3e-3,
    weight_decay: float = 0.1,
    bce_pos_mult: float = 4.0,
    bce_neg_mult: float = 0.5,
    softmax_loss_weight: float = 0.2,
    conflict_loss_weight: float = 0.1,
    max_age: int = 100,
    warmup_fraction: float = 0.1,
    eval_max_rounds: int = 1000,
    eval_n_chains: int = 64,
    eval_batch_size: int = 512,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts = time.strftime("%Y%m%d_%H%M%S")

    step_cfg = StepConfig(
        threshold=0.10, temp_decide=1.5, cls_threshold=0.5,
        augment=True, augment_dihedral=True,
        vocab_dim=9, permute_digits=False,  # digit-perm would break cage sums
    )

    t_train0 = time.time()
    ckpt_path = train(TrainConfig(
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        lr=lr,
        weight_decay=weight_decay,
        bce_pos_mult=bce_pos_mult,
        bce_neg_mult=bce_neg_mult,
        softmax_loss_weight=softmax_loss_weight,
        conflict_loss_weight=conflict_loss_weight,
        warmup_fraction=warmup_fraction,
        step=step_cfg,
        max_age=max_age,
        log_every=20,
        eval_every=100,
        eval_n_puzzles=144,  # val split size (70/15/15 of 960)
        eval_max_rounds=5,
        eval_n_chains=64,
        init_from=init_from,
        out_dir=f"{CHECKPOINT_MOUNT}/killer",
        name=f"seed{seed}_{steps}s_bs{batch_size}_{ts}",
        data=KillerConfig(split="train", seed=0),
        eval_data=KillerConfig(
            split="val", seed=0, n_puzzles=144, batch_size=144,
            zero_hint_weight=1.0, correct_hint_weight=0.0, error_hint_weight=0.0,
        ),
    ))
    train_secs = time.time() - t_train0
    checkpoint_volume.commit()

    print(f"\n{'=' * 60}\nFinal eval (Killer TEST split, zero-hint)\n{'=' * 60}", flush=True)
    ckpt = load_checkpoint(str(ckpt_path))
    model = PowersetModel(LoopedTransformerConfig(**ckpt["model_cfg"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    swap_in_ema_if_present(model, ckpt)

    test_ds = KillerDataset(KillerConfig(
        split="test", seed=0, batch_size=144,
        zero_hint_weight=1.0, correct_hint_weight=0.0, error_hint_weight=0.0,
    ))
    tx, ty, ta, _ = test_ds.next_batch()
    test_ds.close()
    tx = tx.to(device).float()
    ty = ty.to(device).float()
    ta = ta.to(device).float()
    test_state = _build_state(tx, ta)
    test_given = _given_mask(tx)
    test_ip = torch.ones(tx.shape[0], SEQ_LEN, dtype=torch.bool, device=device)
    n = tx.shape[0]
    print(f"  test puzzles: {n}", flush=True)

    eval_step_cfg = StepConfig(
        threshold=0.10, temp_decide=1.5, cls_threshold=0.6,
        augment=False, vocab_dim=9, permute_digits=False,
    )
    t_eval0 = time.time()
    res = solve(
        model, test_state, ty, test_given,
        SolveConfig(
            step=eval_step_cfg, max_rounds=eval_max_rounds,
            n_chains=eval_n_chains, batch_size=eval_batch_size,
        ),
        in_puzzle_mask=test_ip,
    )
    eval_secs = time.time() - t_eval0

    n_correct = int(res.correct.sum().item())
    n_wrong = int(res.wrong.sum().item())
    n_timeout = int(res.timeouts.sum().item())
    avg_rounds = float(
        res.round_solved[res.solved].float().mean().item()
        if int(res.solved.sum().item()) > 0 else 0.0
    )
    den = max(res.diag_total_deduced, 1)
    unsound_rate = res.diag_total_unsound_deductions / den

    print(f"\n{'=' * 60}\nEXP#2 RESULT\n{'=' * 60}", flush=True)
    print(f"  correct={n_correct}/{n}  wrong={n_wrong}  timeouts={n_timeout}", flush=True)
    print(f"  FSR = {n_correct / max(n, 1):.4f}", flush=True)
    print(f"  avg_rounds={avg_rounds:.1f}  unsound={unsound_rate:.4%}  "
          f"calls={res.model_calls}", flush=True)
    print(f"  train_secs={train_secs:.0f}  eval_secs={eval_secs:.0f}", flush=True)
    print(f"{'=' * 60}", flush=True)

    eval_json_path = Path(str(ckpt_path)).with_suffix(".eval.json")
    eval_json_path.write_text(json.dumps({
        "experiment": "exp2_killer_cagesum_input",
        "checkpoint": str(ckpt_path),
        "init_from": init_from,
        "steps": steps, "batch_size": batch_size, "seed": seed, "lr": lr,
        "n_test": n,
        "correct": n_correct, "wrong": n_wrong, "timeouts": n_timeout,
        "fsr": n_correct / max(n, 1),
        "avg_rounds_solved": avg_rounds,
        "unsound_rate": unsound_rate,
        "model_calls_total": res.model_calls,
        "train_secs": train_secs,
        "eval_secs": eval_secs,
        "max_rounds": eval_max_rounds,
        "n_chains": eval_n_chains,
    }, indent=2))
    checkpoint_volume.commit()

    return {
        "steps": steps, "correct": n_correct, "wrong": n_wrong,
        "timeouts": n_timeout, "fsr": n_correct / max(n, 1),
        "checkpoint": str(ckpt_path),
    }


@app.local_entrypoint()
def entrypoint(
    init_from: str,
    steps: int = 2000,
    batch_size: int = 512,
    seed: int = 0,
    lr: float = 3e-3,
    weight_decay: float = 0.1,
    bce_pos_mult: float = 4.0,
    bce_neg_mult: float = 0.5,
    softmax_loss_weight: float = 0.2,
    conflict_loss_weight: float = 0.1,
    max_age: int = 100,
    warmup_fraction: float = 0.1,
    eval_max_rounds: int = 1000,
    eval_n_chains: int = 64,
    eval_batch_size: int = 512,
):
    result = run.remote(
        init_from=init_from, steps=steps, batch_size=batch_size, seed=seed,
        lr=lr, weight_decay=weight_decay,
        bce_pos_mult=bce_pos_mult, bce_neg_mult=bce_neg_mult,
        softmax_loss_weight=softmax_loss_weight,
        conflict_loss_weight=conflict_loss_weight,
        max_age=max_age, warmup_fraction=warmup_fraction,
        eval_max_rounds=eval_max_rounds, eval_n_chains=eval_n_chains,
        eval_batch_size=eval_batch_size,
    )
    print(f"\nFinal: {result}", flush=True)
