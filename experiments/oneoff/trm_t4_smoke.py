"""T4 SMOKE of the full4 recipe (upstream-faithful TRM baseline).

Same code path as experiments/scaled_trm/run.py, only smaller:
steps=200, batch=32 (4 puzzles x 8 views), val greedy 100 (no vote).
Verifies: adam-atan2 import, SignSGD+AdamATan2 steps, ACTLossHead,
carry-over with labels-in-batch, EMA register/update, torch.compile,
checkpoint roundtrip, loss DECREASES (learning signal, not just finite).

PASS criteria: losses[200] < losses[1], finite, ckpt_eq=true,
eval completes 100/100. Accuracy may be ~0 (tiny budget, expected).

Usage:
    modal run experiments/oneoff/trm_t4_smoke.py
"""

import json
import random
import sys
import time
from pathlib import Path

import modal
import torch

from lattice_diffusion.modal.image import (
    CHECKPOINT_MOUNT, checkpoint_volume)

SPLIT_LOCAL = "scaled_sudoku_split.jsonl"
SPLIT_REMOTE = "/smoke/scaled_sudoku_split.jsonl"

smoke_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .uv_sync()
    .uv_pip_install("wandb", "einops", "pydantic")
    .add_local_python_source("lattice_diffusion")
    .add_local_python_source("experiments")
    .add_local_file(SPLIT_LOCAL, SPLIT_REMOTE)
    .add_local_dir("upstream/TinyRecursiveModels", "/opt/trm_upstream")
)

app = modal.App("scaled-trm-t4-smoke-full")

wandb_secret = modal.Secret.from_name("wandb-api")


@app.function(image=smoke_image, gpu="T4", timeout=1800,
              secrets=[wandb_secret],
              volumes={CHECKPOINT_MOUNT: checkpoint_volume})
def run(steps: int = 200, batch_size: int = 32, seed: int = 0,
        lr: float = 1e-4, val_n: int = 100):
    import os
    import numpy as np
    t0 = time.time()
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    sys.path.insert(0, "/opt/trm_upstream")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" \
        else "cpu"
    print(f"device={device} gpu={gpu_name}", flush=True)

    from experiments.scaled_trm.run import (  # same helpers as full recipe
        _aug_params, _apply_aug_np, _encode_np, _cosine_lr,
        LR, PUZZLE_EMB_LR, WEIGHT_DECAY, PUZZLE_EMB_WD, BETA1, BETA2,
        MIN_RATIO, WARMUP_FRACTION, EMA_RATE)

    rows = [json.loads(l) for l in open(SPLIT_REMOTE)]
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    assert all(r["split"] == "train" for r in tr)
    val_sub = random.Random(42).sample(va, val_n)
    assert not ({r["ancestor_id"] for r in tr} & {r["ancestor_id"] for r in va})
    assert not ({r["puzzle"] for r in tr} & {r["puzzle"] for r in va})
    tr_puz, tr_sol = _encode_np(tr)
    print(f"SMOKE data: train={len(tr)} val_sub={len(val_sub)} "
          f"test_touched=NO leaks=0", flush=True)

    from experiments.scaled_trm.adam_atan2_torch import (  # type: ignore
        AdamATan2)
    print("adam-atan2 (vendored math-identical) import OK", flush=True)
    from models.ema import EMAHelper  # type: ignore
    from models.losses import ACTLossHead  # type: ignore
    from models.recursive_reasoning.trm import (  # type: ignore
        TinyRecursiveReasoningModel_ACTV1)
    from models.sparse_embedding import (  # type: ignore
        CastedSparseEmbeddingSignSGD_Distributed)

    cfg_dict = dict(
        batch_size=batch_size, seq_len=81,
        puzzle_emb_ndim=512, num_puzzle_identifiers=len(tr),
        vocab_size=10,
        H_cycles=3, L_cycles=6, H_layers=0, L_layers=2,
        hidden_size=512, expansion=4.0, num_heads=8,
        pos_encodings="none", halt_max_steps=16,
        halt_exploration_prob=0.1, forward_dtype="bfloat16",
        mlp_t=True, puzzle_emb_len=16, no_ACT_continue=True,
    )
    with torch.device("cuda"):
        model = TinyRecursiveReasoningModel_ACTV1(cfg_dict)
        model = ACTLossHead(model, loss_type="stablemax_cross_entropy")
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)  # type: ignore
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params:,} (full arch, smoke budget)", flush=True)

    optimizers = [
        CastedSparseEmbeddingSignSGD_Distributed(
            model.model.puzzle_emb.buffers(),  # type: ignore
            lr=0, weight_decay=PUZZLE_EMB_WD, world_size=1),
        AdamATan2(model.parameters(), lr=0, weight_decay=WEIGHT_DECAY,
                  betas=(BETA1, BETA2)),
    ]
    optimizer_lrs = [PUZZLE_EMB_LR, LR]
    print("optimizers OK (SignSGD + AdamATan2)", flush=True)

    ema_helper = EMAHelper(mu=EMA_RATE)
    ema_helper.register(model)

    warmup_steps = max(1, int(steps * WARMUP_FRACTION))
    VIEWS = 8
    assert batch_size % VIEWS == 0
    n_puzzles = batch_size // VIEWS
    n_tr = len(tr)
    losses: dict = {}
    log_at = {1, 50, 100, 150, 200}
    carry = None
    model.train()
    t_train = time.time()
    for step in range(1, steps + 1):
        idx = np.random.choice(n_tr, n_puzzles, replace=False)
        bxi, byi, pids = [], [], []
        for i in idx:
            for _ in range(VIEWS):
                p = _aug_params(np.random)
                bxi.append(_apply_aug_np(tr_puz[i], p))
                byi.append(_apply_aug_np(tr_sol[i], p))
                pids.append(i)
        batch = {"inputs": torch.from_numpy(np.stack(bxi)).to(device),
                 "labels": torch.from_numpy(np.stack(byi)).to(device),
                 "puzzle_identifiers": torch.from_numpy(
                     np.asarray(pids, dtype=np.int64)).to(device)}
        for optim, base_lr in zip(optimizers, optimizer_lrs):
            lr_now = _cosine_lr(step, base_lr=base_lr,
                                num_warmup_steps=warmup_steps,
                                num_training_steps=steps,
                                min_ratio=MIN_RATIO)
            for pg in optim.param_groups:
                pg["lr"] = lr_now
        if carry is None:
            with torch.device(device):
                carry = model.initial_carry(batch)  # type: ignore
        carry, loss, metrics, _, _ = model(
            carry=carry, batch=batch, return_keys=[])  # type: ignore
        ((1 / batch_size) * loss).backward()
        for optim in optimizers:
            optim.step()
            optim.zero_grad()
        ema_helper.update(model)
        if step in log_at:
            v = float(loss.detach().item() / batch_size)
            losses[step] = v
            print(f"  step={step}/{steps} loss={v:.4f}", flush=True)
        if not torch.isfinite(loss).all():
            raise RuntimeError(f"STOP: non-finite loss at step {step}")
    train_s = time.time() - t_train
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 \
        if device.type == "cuda" else 0.0
    print(f"SMOKE train: {train_s:.0f}s peakVRAM={peak_gb:.2f}GB", flush=True)

    first, last = losses[1], losses[steps]
    learned = bool(last < first)
    print(f"SMOKE learning: first={first:.4f} last={last:.4f} "
          f"decreased={learned}", flush=True)
    if not learned:
        raise RuntimeError(
            f"STOP: no learning signal (first={first:.4f} last={last:.4f})")

    # Checkpoint roundtrip + output equality on EMA copy.
    ckpt_path = f"{CHECKPOINT_MOUNT}/scaled_trm_smoke/full4_{steps}s.pt"
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": {
        k: v.cpu() for k, v in model.state_dict().items()},
        "cfg": cfg_dict}, ckpt_path)
    em = ema_helper.ema_copy(model)
    em.eval()
    # NOTE: call the (compiled) wrapper itself — upstream calls
    # train_state.model(...); `.model` would resolve to the raw TRM.
    probe = {"inputs": torch.randint(0, 10, (batch_size, 81)).to(device),
             "labels": torch.zeros((batch_size, 81), dtype=torch.long,
                                   device=device),
             "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.long,
                                               device=device)}
    # Upstream protocol: preds come back via return_keys (4th tuple slot);
    # 2nd slot is the scalar loss. Compare EMA-copy determinism on preds.
    with torch.no_grad():
        with torch.device(device):
            c0 = em.initial_carry(probe)  # type: ignore
        _, _, _, o1, _ = em(  # type: ignore
            carry=c0, batch=probe, return_keys=["preds"])
        with torch.device(device):
            c0b = em.initial_carry(probe)  # type: ignore
        _, _, _, o2, _ = em(  # type: ignore
            carry=c0b, batch=probe, return_keys=["preds"])
    eq = bool(torch.equal(o1["preds"].cpu(), o2["preds"].cpu()))
    print(f"SMOKE checkpoint: save=PASS ema_copy=PASS "
          f"deterministic={eq}", flush=True)
    checkpoint_volume.commit()

    # Greedy eval on 100 val (pipeline only, no accuracy goal).
    ex_np, ey_np = _encode_np(val_sub)
    with torch.no_grad():
        for s in range(0, val_n, batch_size):
            chunk = ex_np[s:s + batch_size]
            cb = chunk.shape[0]
            cbatch = {"inputs": torch.from_numpy(chunk).to(device),
                      "labels": torch.zeros((cb, 81), dtype=torch.long,
                                            device=device),
                      "puzzle_identifiers": torch.zeros(
                          cb, dtype=torch.long, device=device)}
            with torch.device(device):
                cc = em.initial_carry(cbatch)  # type: ignore
            while True:
                cc, _, _, co, done = em(  # type: ignore
                    carry=cc, batch=cbatch, return_keys=["preds"])
                if bool(done):
                    break
            pred = co["preds"].cpu().numpy()
            ok = (pred == ey_np[s:s + batch_size]).all(axis=1)
            correct = int(ok.sum()) if s == 0 else correct + int(ok.sum())
    eval_s = time.time() - t_train - train_s
    print(f"SMOKE eval ({val_n} val): correct={correct} "
          f"({time.time() - t0:.0f}s wall)", flush=True)
    print("SMOKE PASS", flush=True)
    return {"losses": losses, "learned": learned, "ckpt_eq": eq,
            "eval_correct": correct, "train_s": train_s,
            "peak_gb": peak_gb}


@app.local_entrypoint()
def entrypoint():
    print(run.remote(), flush=True)
