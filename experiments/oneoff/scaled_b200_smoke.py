"""B200 SMOKE: 50 steps, batch 512, from scratch on Scaled-Sudoku TRAIN split.

Measures: pipeline correctness on B200, batch-512 fit, steps/s, peak VRAM,
checkpoint save/load + output equality, mini-eval on 100 val.
HARD STOP after smoke. No full training here.

Usage:
    modal run experiments/oneoff/scaled_b200_smoke.py
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

b200_image = image.add_local_file(SPLIT_LOCAL, SPLIT_REMOTE)

app = modal.App("scaled-sudoku-b200-smoke")


@app.function(image=b200_image, gpu="B200", timeout=1800,
              volumes={CHECKPOINT_MOUNT: checkpoint_volume})
def run(steps: int = 50, batch_size: int = 512, seed: int = 0,
        lr: float = 3e-3, val_n: int = 100):
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"device={device} gpu={gpu_name}", flush=True)

    rows = [json.loads(l) for l in open(SPLIT_REMOTE)]
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    assert all(r["split"] == "train" for r in tr)
    val_sub = random.Random(42).sample(va, val_n)
    assert all(r["split"] == "val" for r in val_sub)
    assert not ({r["ancestor_id"] for r in tr} & {r["ancestor_id"] for r in va})
    assert not ({r["puzzle"] for r in tr} & {r["puzzle"] for r in va})
    print(f"SMOKE data: train={len(tr)} val_sub={len(val_sub)} "
          f"test_touched=NO leaks=0", flush=True)

    model_cfg = LoopedTransformerConfig(cls_token=True)
    model = PowersetModel(model_cfg).to(device)
    print(f"params={sum(p.numel() for p in model.parameters()):,}",
          flush=True)

    # forward probe: shape + finite, no NaN/Inf
    xb, _ = _encode(tr[:batch_size])
    xb = xb.to(device).float()
    assert xb.shape[1:] == (81, 9), xb.shape
    model.eval()
    with torch.no_grad():
        probe = model(xb, use_final=True)
    assert probe["bce"].shape == (batch_size, 81, 9)
    assert torch.isfinite(probe["bce"]).all() and \
        torch.isfinite(probe["softmax"]).all()
    print(f"SMOKE forward: X={tuple(xb.shape)} bce={tuple(probe['bce'].shape)} "
          f"finite=YES", flush=True)

    step_cfg = StepConfig(threshold=0.10, temp_decide=1.5, cls_threshold=0.5,
                          augment=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    torch.cuda.reset_peak_memory_stats()

    log_at = {1, 10, 25, 50}
    losses = {}
    model.train()
    t_train = time.time()
    for step in range(1, steps + 1):
        idx = torch.randperm(len(tr))[:batch_size].tolist()
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
        if step in log_at:
            losses[step] = float(loss.item())
            print(f"  step={step}/{steps} loss={losses[step]:.4f}",
                  flush=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    train_s = time.time() - t_train
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"SMOKE train: {steps / train_s:.2f} steps/s ({train_s:.0f}s) "
          f"peakVRAM={peak_gb:.2f}GB", flush=True)

    # checkpoint: save, load into fresh model, output equality in eval mode
    model.eval()
    with torch.no_grad():
        ref = model(xb, use_final=True)["bce"]
    ckpt_path = f"{CHECKPOINT_MOUNT}/scaled_b200_smoke/seed{seed}_{steps}s.pt"
    save_checkpoint(path=ckpt_path, model=model, model_cfg=model_cfg,
                    extra={"b200_smoke": True})
    ckpt = load_checkpoint(ckpt_path)
    model2 = PowersetModel(LoopedTransformerConfig(**ckpt["model_cfg"]))
    model2.load_state_dict(ckpt["model_state_dict"])
    model2.to(device).eval()
    for k in ckpt["model_state_dict"]:
        assert torch.equal(ckpt["model_state_dict"][k],
                            model2.state_dict()[k].cpu())
    with torch.no_grad():
        got = model2(xb, use_final=True)["bce"]
    eq = bool(torch.equal(ref.cpu(), got.cpu()))
    print(f"SMOKE checkpoint: save=PASS load=PASS output_equal={eq}",
          flush=True)
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
    summ = {"terminated": int(res.solved.sum()) + int(res.timeouts.sum()),
            "solved": int(res.solved.sum()),
            "correct": int(res.correct.sum()), "wrong": int(res.wrong.sum()),
            "timeout": int(res.timeouts.sum()), "calls": res.model_calls}
    assert summ["terminated"] == n
    print(f"SMOKE eval ({n} val): {summ} ({eval_s:.0f}s)", flush=True)
    print(f"SMOKE DONE train={train_s:.0f}s eval={eval_s:.0f}s", flush=True)
    return {"gpu": gpu_name, "losses": losses, "steps_per_s": steps / train_s,
            "train_s": train_s, "peak_gb": peak_gb, "ckpt_eq": eq,
            "eval": summ, "eval_s": eval_s}


@app.local_entrypoint()
def entrypoint():
    print(run.remote(), flush=True)
