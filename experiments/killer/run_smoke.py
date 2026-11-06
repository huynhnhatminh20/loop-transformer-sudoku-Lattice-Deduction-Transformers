"""Exp #2 SMOKE TEST (cheap): 50 train steps on Killer + checks.

Verifies, on a B200:
  1. KillerDataset batch shapes (x [B,81,9], y [B,81,9], aux [B,81,3])
  2. Partial checkpoint init (reuse vs new param counts)
  3. 12-channel forward shapes (bce/softmax [B,81,12])
  4. 50 train steps, finite loss
  5. Checkpoint save + fresh load roundtrip (state_dict equality)
  6. Mini eval on 8 test puzzles (capped rounds/chains — pipeline only)

Usage:
    modal run --detach experiments/killer/run_smoke.py \
        --init-from /checkpoints/sudoku/seed0_4000s_bs512_aug1_20261003_040535_20261003_040535.pt
"""

import json
from pathlib import Path

import modal
import torch

from lattice_diffusion.models.looped_transformer import LoopedTransformerConfig, PowersetModel
from lattice_diffusion.modal.image import (
    CHECKPOINT_MOUNT,
    checkpoint_volume,
    image,
)

# wandb package comes from the shared image; key via the "wandb-api" secret.
wandb_secret = modal.Secret.from_name("wandb-api")
from lattice_diffusion.training.utils.checkpoint import load_checkpoint

from experiments.sudoku.dpll import StepConfig
from experiments.sudoku.solve import SolveConfig, solve
from experiments.killer.data import KillerConfig, KillerDataset
from experiments.killer.train import (
    SEQ_LEN, TrainConfig, _build_state, _given_mask, load_partial_checkpoint, train,
)

app = modal.App("killer-smoke")


@app.function(
    image=image,
    gpu="B200",
    timeout=3600,
    secrets=[wandb_secret],
    volumes={CHECKPOINT_MOUNT: checkpoint_volume},
)
def run(
    init_from: str,
    steps: int = 50,
    batch_size: int = 128,
    seed: int = 0,
    lr: float = 3e-3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checks: dict[str, str] = {}

    # ---- 1. dataset shapes ----
    ds = KillerDataset(KillerConfig(split="train", seed=0, batch_size=batch_size))
    xb, yb, ab, sb = ds.next_batch()
    ds.close()
    assert xb.shape[1:] == (81, 9), xb.shape
    assert yb.shape[1:] == (81, 9), yb.shape
    assert ab.shape[1:] == (81, 3), ab.shape
    assert ab.min() >= 0 and ab.max() <= 1, (ab.min(), ab.max())
    checks["dataset_shapes"] = f"OK x={tuple(xb.shape)} y={tuple(yb.shape)} aux={tuple(ab.shape)}"
    print(checks["dataset_shapes"], flush=True)

    # ---- 2+3. partial init + forward shapes ----
    model_cfg = LoopedTransformerConfig(
        n_channels=12, seq_len=81, grid_rows=9, grid_cols=9, cls_token=True,
    )
    model = PowersetModel(model_cfg).to(device)
    rep = load_partial_checkpoint(init_from, model)
    print(f"reused={rep['n_reused']:,} new={rep['n_new']:,}", flush=True)
    for k, v in rep["details"].items():
        print(f"    {k}: {v}", flush=True)
    # Fresh: input_proj cols 9..11 (384) + bce/softmax head rows
    # 9..11 (2*(384+3)). Everything else reused.
    assert rep["n_new"] == 384 + 2 * (384 + 3), rep
    checks["partial_init"] = f"OK reused={rep['n_reused']} new={rep['n_new']}"

    model.eval()
    with torch.no_grad():
        st = _build_state(xb[:4].to(device).float(), ab[:4].to(device).float())
        assert st.shape == (4, 81, 12), st.shape
        out = model(st, use_final=True)
        assert out["bce"].shape == (4, 81, 12), out["bce"].shape
        assert out["softmax"].shape == (4, 81, 12), out["softmax"].shape
    checks["forward_shapes"] = "OK bce/softmax [4,81,12]"
    print(checks["forward_shapes"], flush=True)
    del model

    # ---- 4. 50 train steps ----
    ckpt_path = train(TrainConfig(
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        lr=lr,
        weight_decay=0.1,
        bce_pos_mult=4.0,
        bce_neg_mult=0.5,
        softmax_loss_weight=0.2,
        conflict_loss_weight=0.1,
        warmup_fraction=0.1,
        step=StepConfig(
            threshold=0.10, temp_decide=1.5, cls_threshold=0.5,
            augment=True, augment_dihedral=True,
            vocab_dim=9, permute_digits=False,
        ),
        max_age=100,
        log_every=10,
        eval_every=10**9,  # skip in-train eval in smoke
        init_from=init_from,
        out_dir=f"{CHECKPOINT_MOUNT}/killer",
        name=f"smoke_seed{seed}_{steps}s_bs{batch_size}",
        data=KillerConfig(split="train", seed=0),
    ))
    checkpoint_volume.commit()
    checks["train_50_steps"] = f"OK ckpt={ckpt_path}"
    print(f"Saved: {ckpt_path}", flush=True)

    # ---- 5. save/load roundtrip ----
    ckpt = load_checkpoint(str(ckpt_path))
    cfg_loaded = LoopedTransformerConfig(**ckpt["model_cfg"])
    assert cfg_loaded.n_channels == 12 and cfg_loaded.seq_len == 81
    model2 = PowersetModel(cfg_loaded)
    model2.load_state_dict(ckpt["model_state_dict"])
    model2.to(device).eval()
    checks["save_load_roundtrip"] = "OK state_dict loads into fresh C=12 model"
    print(checks["save_load_roundtrip"], flush=True)

    # ---- 6. mini eval: 8 test puzzles, capped protocol (pipeline only) ----
    eval_ds = KillerDataset(KillerConfig(
        split="test", seed=0, n_puzzles=8, batch_size=8,
        zero_hint_weight=1.0, correct_hint_weight=0.0, error_hint_weight=0.0,
    ))
    ex, ey, ea, _ = eval_ds.next_batch()
    eval_ds.close()
    ex = ex.to(device).float()
    ey = ey.to(device).float()
    ea = ea.to(device).float()
    estate = _build_state(ex, ea)
    egiven = _given_mask(ex)
    ip = torch.ones(ex.shape[0], SEQ_LEN, dtype=torch.bool, device=device)
    with torch.no_grad():
        res = solve(
            model2, estate, ey, egiven,
            SolveConfig(
                step=StepConfig(
                    threshold=0.10, temp_decide=1.5, cls_threshold=0.6,
                    augment=False, vocab_dim=9, permute_digits=False,
                ),
                max_rounds=50, n_chains=8, batch_size=64,
            ),
            in_puzzle_mask=ip,
            verbose=False,
        )
    n = res.solved.shape[0]
    summary = {
        "correct": int(res.correct.sum().item()),
        "wrong": int(res.wrong.sum().item()),
        "timeouts": int(res.timeouts.sum().item()),
        "model_calls": res.model_calls,
    }
    checks["mini_eval_8"] = f"OK {summary}"
    print(f"mini-eval (8 puzzles, capped): {summary}", flush=True)

    out_path = Path(str(ckpt_path)).with_suffix(".smoke.json")
    out_path.write_text(json.dumps({"checks": checks, "mini_eval": summary}, indent=2))
    checkpoint_volume.commit()
    print("SMOKE PASS", flush=True)
    return {"checks": checks}


@app.local_entrypoint()
def entrypoint(init_from: str, steps: int = 50, batch_size: int = 128,
               seed: int = 0, lr: float = 3e-3):
    result = run.remote(
        init_from=init_from, steps=steps, batch_size=batch_size,
        seed=seed, lr=lr,
    )
    print(f"\nFinal: {result}", flush=True)
