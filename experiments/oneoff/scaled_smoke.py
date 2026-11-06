"""SMOKE TEST: LDT from scratch on Scaled Sudoku Instances (tiny subset).

Verifies: adapter loader -> [B,81,9] input -> forward -> loss -> backward ->
optimizer -> checkpoint save/load -> solve loop -> metrics. No accuracy goal.

Scope (hard): steps<=100, 80 train + 20 val instances, 1 GPU (T4), 1 seed,
no sweep, no full eval. Uses ONLY train/val rows from scaled_sudoku_smoke.jsonl
(test rows never touched).

Usage:
    uv run modal run --detach experiments/oneoff/scaled_smoke.py
"""

import json
import time

import modal
import torch
import torch.nn.functional as F

from lattice_diffusion.models.looped_transformer import (
    LoopedTransformerConfig, PowersetModel)
from lattice_diffusion.models.weighted_bce import weighted_bce_with_logits
from lattice_diffusion.modal.image import (
    CHECKPOINT_MOUNT, checkpoint_volume, image)
from lattice_diffusion.training.utils.checkpoint import (
    load_checkpoint, save_checkpoint)

from experiments.sudoku.aug import aug_forward
from experiments.sudoku.dpll import StepConfig
from experiments.sudoku.solve import SolveConfig, solve

SMOKE_LOCAL = "scaled_sudoku_smoke.jsonl"
SMOKE_REMOTE = "/smoke/scaled_sudoku_smoke.jsonl"

smoke_image = image.add_local_file(SMOKE_LOCAL, SMOKE_REMOTE)

app = modal.App("scaled-sudoku-smoke")


def _encode(rows):
    import numpy as np
    X = np.zeros((len(rows), 81, 9), dtype=np.float32)
    Y = np.zeros((len(rows), 81, 9), dtype=np.float32)
    for i, r in enumerate(rows):
        for j, c in enumerate(r["puzzle"]):
            if c == ".":
                X[i, j] = 1.0
            else:
                X[i, j, int(c) - 1] = 1.0
        for j, c in enumerate(r["solution"]):
            Y[i, j, int(c) - 1] = 1.0
    return torch.from_numpy(X), torch.from_numpy(Y)


def _losses(out, state, orig_y, given_mask, is_sat, gt_conflict_target):
    B = state.shape[0]
    device = state.device
    bce_target = state * orig_y
    pos_w = torch.full((B, 1, 1), 4.0, device=device)
    neg_w = torch.full((B, 1, 1), 0.5, device=device)
    total = torch.zeros((), device=device)
    for i in range(len(out["bce"])):
        total = total + weighted_bce_with_logits(
            out["bce"][i], bce_target, pos_w, neg_w)
        sm_logits = out["softmax"][i]
        gt_idx = orig_y.argmax(dim=-1)
        ce_mask = ~given_mask & is_sat.unsqueeze(-1)
        if ce_mask.any():
            total = total + 0.2 * F.cross_entropy(
                sm_logits[ce_mask], gt_idx[ce_mask])
        if "conflict" in out:
            c_logits = out["conflict"][i].squeeze(-1)
            total = total + 0.1 * F.binary_cross_entropy_with_logits(
                c_logits, gt_conflict_target.float())
    return total / len(out["bce"])


@app.function(image=smoke_image, gpu="T4", timeout=1800,
              volumes={CHECKPOINT_MOUNT: checkpoint_volume})
def run(steps: int = 50, batch_size: int = 32, seed: int = 0, lr: float = 3e-3):
    t0 = time.time()
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    rows = [json.loads(l) for l in open(SMOKE_REMOTE)]
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    assert len(tr) == 80 and len(va) == 20, (len(tr), len(va))
    # leakage re-check inside container
    assert not ({r["ancestor_id"] for r in tr} & {r["ancestor_id"] for r in va})
    assert not ({r["puzzle"] for r in tr} & {r["puzzle"] for r in va})
    print(f"SMOKE subset: train={len(tr)} val={len(va)} leaks=0", flush=True)
    # no test rows allowed in this file
    assert not any(r["split"] == "test" for r in rows)

    model_cfg = LoopedTransformerConfig(cls_token=True)
    model = PowersetModel(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params:,}", flush=True)

    # --- input check: one batch [B,81,9] through forward ---
    xb, yb = _encode(tr[:batch_size])
    xb, yb = xb.to(device).float(), yb.to(device).float()
    assert xb.shape[1:] == (81, 9), xb.shape
    with torch.no_grad():
        probe = model(xb, use_final=True)
    assert probe["bce"].shape == (batch_size, 81, 9)
    assert probe["softmax"].shape == (batch_size, 81, 9)
    print(f"SMOKE input: X={tuple(xb.shape)} "
          f"bce={tuple(probe['bce'].shape)} OK", flush=True)

    step_cfg = StepConfig(threshold=0.10, temp_decide=1.5, cls_threshold=0.5,
                          augment=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    given_all = (xb.sum(dim=-1) == 1)

    # --- smoke training from scratch (direct batches, no pool) ---
    first_loss, last_loss = None, None
    model.train()
    for step in range(1, steps + 1):
        idx = torch.randperm(len(tr))[:batch_size]
        bx, by = _encode([tr[i] for i in idx.tolist()])
        bx, by = bx.to(device).float(), by.to(device).float()
        gm = (bx.sum(dim=-1) == 1)
        gt_conflict = torch.zeros(bx.shape[0], dtype=torch.bool, device=device)
        is_sat = torch.ones(bx.shape[0], dtype=torch.bool, device=device)
        opt.zero_grad(set_to_none=True)
        out, _ = aug_forward(model, bx, gm, orig_y=by, augment=False,
                             return_all=True)
        loss = _losses(out, bx, by, gm, is_sat, gt_conflict)
        assert torch.isfinite(loss).all(), f"non-finite loss at step {step}"
        if first_loss is None:
            first_loss = float(loss.item())
        last_loss = float(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 10 == 0:
            print(f"  step={step}/{steps} loss={last_loss:.4f}", flush=True)
    print(f"SMOKE train: first_loss={first_loss:.4f} "
          f"last_loss={last_loss:.4f} NaN/Inf=NO", flush=True)

    # --- checkpoint save/load roundtrip ---
    ckpt_path = f"{CHECKPOINT_MOUNT}/scaled_smoke/seed{seed}_{steps}s.pt"
    save_checkpoint(path=ckpt_path, model=model, model_cfg=model_cfg,
                    extra={"smoke": True})
    ckpt = load_checkpoint(ckpt_path)
    model2 = PowersetModel(LoopedTransformerConfig(**ckpt["model_cfg"]))
    model2.load_state_dict(ckpt["model_state_dict"])
    model2.to(device).eval()
    for k in ckpt["model_state_dict"]:
        assert torch.equal(ckpt["model_state_dict"][k],
                            model2.state_dict()[k].cpu())
    print(f"SMOKE checkpoint: saved+loaded OK ({ckpt_path})", flush=True)
    checkpoint_volume.commit()

    # --- smoke eval on 20 val (real givens, small protocol) ---
    ex, ey = _encode(va)
    ex, ey = ex.to(device).float(), ey.to(device).float()
    egiven = (ex.sum(dim=-1) == 1)
    with torch.no_grad():
        res = solve(model2, ex, ey, egiven,
                    SolveConfig(step=step_cfg, max_rounds=50,
                                n_chains=8, batch_size=64),
                    verbose=False)
    n = res.solved.shape[0]
    summary = {"correct": int(res.correct.sum()), "wrong": int(res.wrong.sum()),
               "timeouts": int(res.timeouts.sum()), "calls": res.model_calls}
    print(f"SMOKE eval ({n} val): {summary}", flush=True)

    wall = time.time() - t0
    print(f"SMOKE DONE wall={wall:.0f}s", flush=True)
    return {"first_loss": first_loss, "last_loss": last_loss,
            "eval": summary, "wall_s": wall, "params": n_params}


@app.local_entrypoint()
def entrypoint(steps: int = 50, batch_size: int = 32):
    print(run.remote(steps=steps, batch_size=batch_size), flush=True)
