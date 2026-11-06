"""PILOT: LDT from scratch, 500 steps on full Scaled-Sudoku TRAIN split.

Hard scope: steps=500, batch=512 (auto-fallback 256 on OOM), 1 seed, T4,
val subset 300 (seed 42, val split only, never test). No sweep, no pool
(direct fresh-sample batches -- documented pilot simplification; the full
experiment should wire the pool-based experiments/sudoku/train.py).

Usage:
    modal run experiments/oneoff/scaled_pilot.py
"""

import json
import random
import time

import modal
import torch

from lattice_diffusion.models.looped_transformer import (
    LoopedTransformerConfig, PowersetModel)
from lattice_diffusion.modal.image import (
    CHECKPOINT_MOUNT, checkpoint_volume, image)
from lattice_diffusion.training.utils.checkpoint import (
    load_checkpoint, save_checkpoint)

from experiments.oneoff.scaled_smoke import _encode, _losses
from experiments.sudoku.aug import aug_forward
from experiments.sudoku.dpll import StepConfig
from experiments.sudoku.solve import SolveConfig, solve

SPLIT_LOCAL = "scaled_sudoku_split.jsonl"
SPLIT_REMOTE = "/smoke/scaled_sudoku_split.jsonl"

pilot_image = image.add_local_file(SPLIT_LOCAL, SPLIT_REMOTE)

app = modal.App("scaled-sudoku-pilot")


@app.function(image=pilot_image, gpu="T4", timeout=3600,
              volumes={CHECKPOINT_MOUNT: checkpoint_volume})
def run(steps: int = 500, batch_size: int = 512, seed: int = 0,
        lr: float = 3e-3, val_n: int = 300):
    t_all = time.time()
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    rows = [json.loads(l) for l in open(SPLIT_REMOTE)]
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    assert not ({r["ancestor_id"] for r in tr} & {r["ancestor_id"] for r in va})
    assert not ({r["puzzle"] for r in tr} & {r["puzzle"] for r in va})
    val_sub = random.Random(42).sample(va, val_n)
    # guards: only train rows train, only val rows eval, test rows never touched
    assert all(r["split"] == "train" for r in tr)
    assert all(r["split"] == "val" for r in val_sub)
    print(f"PILOT data: train={len(tr)} val_pool={len(va)} "
          f"val_sub={len(val_sub)} leaks=0", flush=True)

    model_cfg = LoopedTransformerConfig(cls_token=True)
    model = PowersetModel(model_cfg).to(device)
    print(f"params={sum(p.numel() for p in model.parameters()):,}",
          flush=True)

    # batch-size probe with OOM fallback (512 -> 256 -> 128), reported.
    # Probe mirrors a real train step (return_all=True + backward), because a
    # no-grad use_final probe underestimates memory by a large factor.
    bs = batch_size
    opt_probe = torch.optim.AdamW(model.parameters(), lr=lr)
    while True:
        try:
            xb, yb0 = _encode(tr[:bs])
            xb = xb.to(device).float()
            yb0 = yb0.to(device).float()
            gm0 = (xb.sum(dim=-1) == 1)
            is0 = torch.ones(bs, dtype=torch.bool, device=device)
            gc0 = torch.zeros(bs, dtype=torch.bool, device=device)
            opt_probe.zero_grad(set_to_none=True)
            out0, _ = aug_forward(model, xb, gm0, orig_y=yb0, augment=False,
                                  return_all=True)
            l0 = _losses(out0, xb, yb0, gm0, is0, gc0)
            l0.backward()
            del out0, l0, xb, yb0
            opt_probe.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and bs > 64:
                bs //= 2
                torch.cuda.empty_cache()
                print(f"PILOT OOM probe -> batch={bs}", flush=True)
            else:
                raise
    print(f"PILOT batch_size={bs} (requested {batch_size})", flush=True)

    step_cfg = StepConfig(threshold=0.10, temp_decide=1.5, cls_threshold=0.5,
                          augment=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    print(f"PILOT lr={lr} constant (no scheduler -- pilot simplification)",
          flush=True)

    log_at = {1, 50, 100, 200, 300, 400, 500}
    losses = {}
    min_loss = float("inf")
    model.train()
    t_train = time.time()
    for step in range(1, steps + 1):
        idx = torch.randperm(len(tr))[:bs].tolist()
        bx, by = _encode([tr[i] for i in idx])
        bx, by = bx.to(device).float(), by.to(device).float()
        gm = (bx.sum(dim=-1) == 1)
        is_sat = torch.ones(bx.shape[0], dtype=torch.bool, device=device)
        gt_conflict = torch.zeros(bx.shape[0], dtype=torch.bool, device=device)
        opt.zero_grad(set_to_none=True)
        out, _ = aug_forward(model, bx, gm, orig_y=by, augment=False,
                             return_all=True)
        loss = _losses(out, bx, by, gm, is_sat, gt_conflict)
        if not torch.isfinite(loss).all():
            raise RuntimeError(f"STOP: non-finite loss at step {step}")
        v = float(loss.item())
        min_loss = min(min_loss, v)
        if step in log_at:
            losses[step] = v
            print(f"  step={step}/{steps} loss={v:.4f} lr={lr}", flush=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    train_s = time.time() - t_train
    print(f"PILOT train done: min_loss={min_loss:.4f} "
          f"{steps / train_s:.2f} steps/s ({train_s:.0f}s)", flush=True)

    ckpt_path = f"{CHECKPOINT_MOUNT}/scaled_pilot/seed{seed}_{steps}s.pt"
    save_checkpoint(path=ckpt_path, model=model, model_cfg=model_cfg,
                    extra={"pilot": True, "batch_size": bs})
    ckpt = load_checkpoint(ckpt_path)
    model2 = PowersetModel(LoopedTransformerConfig(**ckpt["model_cfg"]))
    model2.load_state_dict(ckpt["model_state_dict"])
    model2.to(device).eval()
    for k in ckpt["model_state_dict"]:
        assert torch.equal(ckpt["model_state_dict"][k],
                            model2.state_dict()[k].cpu())
    print("PILOT checkpoint save/load OK", flush=True)
    checkpoint_volume.commit()

    t_eval = time.time()
    ex, ey = _encode(val_sub)
    ex, ey = ex.to(device).float(), ey.to(device).float()
    egiven = (ex.sum(dim=-1) == 1)
    with torch.no_grad():
        res = solve(model2, ex, ey, egiven,
                    SolveConfig(step=step_cfg, max_rounds=50,
                                n_chains=8, batch_size=64),
                    verbose=False)
    eval_s = time.time() - t_eval
    n = res.solved.shape[0]
    summ = {"solved": int(res.solved.sum()), "correct": int(res.correct.sum()),
            "wrong": int(res.wrong.sum()),
            "timeout": int(res.timeouts.sum()), "calls": res.model_calls}
    print(f"PILOT eval ({n} val): {summ} ({eval_s:.0f}s)", flush=True)

    # breakdown by metadata (needs per-puzzle flags on CPU)
    import numpy as np
    corr = res.correct.cpu().numpy().tolist()
    by = {"base": [0, 0], "variant": [0, 0]}
    clues = {}
    hard = {}
    walk = {}
    for r, c in zip(val_sub, corr):
        k = "base" if r["is_base"] else "variant"
        by[k][1] += 1
        by[k][0] += c
        for d, key in ((clues, r["clues_count"]),
                       (hard, r["hardness_level"]),
                       (walk, r["walk_index"])):
            d.setdefault(key, [0, 0])
            d[key][1] += 1
            d[key][0] += c
    print(f"PILOT breakdown base={by['base']} variant={by['variant']}",
          flush=True)
    print(f"PILOT by_clues={dict(sorted(clues.items(), key=lambda kv: str(kv[0])))}",
          flush=True)
    print(f"PILOT by_hardness={dict(sorted(hard.items(), key=lambda kv: str(kv[0])))}",
          flush=True)
    print(f"PILOT by_walk={dict(sorted(walk.items(), key=lambda kv: str(kv[0])))}",
          flush=True)

    total_s = time.time() - t_all
    print(f"PILOT DONE train={train_s:.0f}s eval={eval_s:.0f}s "
          f"total={total_s:.0f}s", flush=True)
    return {"losses": losses, "min_loss": min_loss, "batch": bs,
            "eval": summ, "breakdown": {"base_variant": by, "clues": clues,
                                        "hardness": hard, "walk": walk},
            "train_s": train_s, "eval_s": eval_s, "total_s": total_s}


@app.local_entrypoint()
def entrypoint():
    print(run.remote(), flush=True)
