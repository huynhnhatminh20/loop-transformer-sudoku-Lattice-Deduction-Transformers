"""FULL BASELINE: HRM on Scaled Sudoku (2667 steps x 768, B200).

Fidelity rule: ONLY data source + total steps differ from upstream.
Everything else is upstream-original code/values (HRM pretrain.py +
sudoku demo recipe):
  - arch: HierarchicalReasoningModel_ACTV1 (H_cycles=2, L_cycles=2,
    H_layers=4, L_layers=4, hidden=512, heads=8, rope, halt_max_steps=16)
  - loss: ACTLossHead(loss_type="stablemax_cross_entropy") [models.losses]
  - optim: SignSGD(puzzle_emb, 7e-5, wd=1.0) + AdamATan2(params, 7e-5,
    wd=1.0, betas=(0.9,0.95)); AdamATan2 via vendored math-identical
    pure-torch replay (see experiments/scaled_trm/adam_atan2_torch.py,
    verified 2e-7 vs csrc/adam_atan2.cu)
  - scheduler: upstream cosine_with_warmup (warmup=10% of steps=267,
    min_ratio=1.0)
  - NO EMA: HRM upstream has no EMA (no models/ema.py, no ema flag
    anywhere). Eval uses live weights, like upstream.
  - torch.compile(dynamic=False) (HRM style) unless DISABLE_COMPILE
  - train loop mirrors train_batch(): persistent carry, no grad clip,
    loss scaled by 1/batch (world=1)
  - eval ACT loop until halted.all, preds via return_keys (upstream eval)

ONE kernel-level deviation (disclosed): HRM layers import flash_attn,
whose wheels lack Blackwell (B200) builds. Attention here runs via an
SDPA stub (same softmax-attention math, different kernel), smoke-only
is NOT the case — stub is used on B200 too. Math-identical, speed differs.

Allowed differences (disclosed):
  - DATA: frozen scaled split (train 82043, seed 42) via local adapter;
    batch 768 = 96 puzzles x 8 aug views (grouped sampling mirrors
    upstream _sample_batch); puzzle_identifiers = row index;
    on-the-fly paper-style aug (shuffle_sudoku semantics).
  - STEPS: 2667 (768x2667=2,048,256 ~= LDT/TRM 2.048M, seed 0).
  - Eval adds breakdowns + majority vote over 8 views ALONGSIDE plain
    greedy (upstream-style). Vote mirrors LDT n_chains=8.
  - In-train eval every 1000 steps on first 200 VAL rows (upstream
    evaluates on test mid-training; ours never touches test).

Flow: train TRAIN -> eval full VAL -> eval full TEST (once each).

Usage:
    modal run --detach experiments/scaled_hrm/run.py
"""

import json
import math
import random
import sys
import time
import types
from pathlib import Path

import modal
import torch
import torch.nn.functional as F

from lattice_diffusion.modal.image import (
    CHECKPOINT_MOUNT, checkpoint_volume)

SPLIT_LOCAL = "scaled_sudoku_split.jsonl"
SPLIT_REMOTE = "/smoke/scaled_sudoku_split.jsonl"

# NOTE: build steps BEFORE add_local_* steps.
full_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .uv_sync()
    .uv_pip_install("wandb", "einops", "pydantic")
    .add_local_python_source("lattice_diffusion")
    .add_local_python_source("experiments")
    .add_local_file(SPLIT_LOCAL, SPLIT_REMOTE)
    .add_local_dir("upstream/HRM", "/opt/hrm_upstream")
)

app = modal.App("scaled-hrm-full")

wandb_secret = modal.Secret.from_name("wandb-api")

LR = 7e-5  # HRM sudoku demo recipe
PUZZLE_EMB_LR = 7e-5
WEIGHT_DECAY = 1.0
PUZZLE_EMB_WD = 1.0
BETA1, BETA2 = 0.9, 0.95
WARMUP_FRACTION = 0.10
MIN_RATIO = 1.0
N_VIEWS = 8


def _wandb_init(name: str, cfg_dict: dict):
    import os
    if not os.environ.get("WANDB_API_KEY"):
        print("wandb disabled (no WANDB_API_KEY env)", flush=True)
        return None
    try:
        import wandb
    except ImportError:
        print("wandb disabled (package not installed)", flush=True)
        return None
    wandb.init(project="scaled-sudoku-trm-hrm", name=name, config=cfg_dict)
    print(f"wandb tracking: {wandb.run.url}", flush=True)
    return wandb


def _install_flash_stub():
    """SDPA stand-in for flash_attn (B200 Blackwell has no FA wheels).
    q/k/v layout [B,S,H,D]; math-identical softmax attention.
    Returns contiguous (flash_attn parity; downstream .view() needs it)."""
    def _sdpa_flash(q, k, v, causal=False, **kwargs):
        q_ = q.transpose(1, 2)
        k_ = k.transpose(1, 2)
        v_ = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q_, k_, v_, is_causal=causal)
        return out.transpose(1, 2).contiguous()

    for name in ("flash_attn_interface", "flash_attn"):
        mod = types.ModuleType(name)
        mod.flash_attn_func = _sdpa_flash  # type: ignore
        sys.modules.setdefault(name, mod)


def _aug_params(rng):
    import numpy as np
    digit = np.random.permutation(9)
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3)
                               for b in bands])
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3)
                               for s in stacks])
    transpose = bool(rng.random() < 0.5)
    return {"digit": digit, "row": row_perm, "col": col_perm,
            "transpose": transpose}


def _apply_aug_np(arr81, p):
    import numpy as np
    g = np.asarray(arr81).reshape(9, 9)
    if p["transpose"]:
        g = g.T.copy()
    g = g[p["row"]][:, p["col"]]
    out = g.reshape(81).copy()
    m = out > 0
    out[m] = p["digit"][out[m] - 1] + 1
    return out


def _invert_aug_np(arr81, p):
    import numpy as np
    inv_digit = np.empty(9, dtype=np.int64)
    inv_digit[p["digit"]] = np.arange(9)
    g = np.asarray(arr81).reshape(9, 9).copy()
    m = g > 0
    g[m] = inv_digit[g[m] - 1] + 1
    inv_row = np.empty(9, dtype=np.int64)
    inv_row[p["row"]] = np.arange(9)
    inv_col = np.empty(9, dtype=np.int64)
    inv_col[p["col"]] = np.arange(9)
    g = g[inv_row][:, inv_col]
    if p["transpose"]:
        g = g.T.copy()
    return g.reshape(81)


def _encode_np(rows):
    import numpy as np
    inp = np.array([[0 if c == "." else int(c) for c in r["puzzle"]]
                    for r in rows], dtype=np.int64)
    lab = np.array([[int(c) for c in r["solution"]] for r in rows],
                   dtype=np.int64)
    return inp, lab


def _cosine_lr(current_step: int, base_lr: float, num_warmup_steps: int,
               num_training_steps: int, min_ratio: float = 0.0):
    """Verbatim upstream cosine_schedule_with_warmup_lr_lambda."""
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(
        0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))))


@app.function(image=full_image, gpu="B200", timeout=3600 * 4,
              secrets=[wandb_secret],
              volumes={CHECKPOINT_MOUNT: checkpoint_volume})
def run(steps: int = 2667, batch_size: int = 768, seed: int = 0):
    import os
    import numpy as np
    t_all = time.time()
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    _install_flash_stub()
    print("flash_attn -> SDPA stub (math-identical; no Blackwell wheels)",
          flush=True)
    sys.path.insert(0, "/opt/hrm_upstream")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    print(f"device={device} gpu={torch.cuda.get_device_name(0)}", flush=True)
    wb = _wandb_init(f"hrm-full-seed{seed}-{steps}s",
                     {"model": "hrm-full-upstream-faithful", "steps": steps,
                      "batch": batch_size, "lr": LR,
                      "puzzle_emb_lr": PUZZLE_EMB_LR,
                      "weight_decay": WEIGHT_DECAY, "seed": seed,
                      "gpu": "B200",
                      "warmup_fraction": WARMUP_FRACTION,
                      "min_ratio": MIN_RATIO, "flash": "sdpa-stub",
                      "arch": "H=2,L=2,Hl=4,Ll=4,hidden=512,rope"})

    rows = [json.loads(l) for l in open(SPLIT_REMOTE)]
    tr = [r for r in rows if r["split"] == "train"]
    assert len(tr) == 82043, len(tr)
    tr_puz, tr_sol = _encode_np(tr)
    print(f"FULL data: train={len(tr)} test_touched=NO", flush=True)

    from experiments.scaled_trm.adam_atan2_torch import (  # type: ignore
        AdamATan2)
    from models.hrm.hrm_act_v1 import (  # type: ignore
        HierarchicalReasoningModel_ACTV1)
    from models.losses import ACTLossHead  # type: ignore
    from models.sparse_embedding import (  # type: ignore
        CastedSparseEmbeddingSignSGD_Distributed)

    cfg_dict = dict(
        batch_size=batch_size, seq_len=81,
        puzzle_emb_ndim=512, num_puzzle_identifiers=len(tr),
        vocab_size=10,
        H_cycles=2, L_cycles=2, H_layers=4, L_layers=4,
        hidden_size=512, expansion=4.0, num_heads=8,
        pos_encodings="rope", halt_max_steps=16,
        halt_exploration_prob=0.1, forward_dtype="bfloat16",
    )
    with torch.device("cuda"):
        model = HierarchicalReasoningModel_ACTV1(cfg_dict)
        model = ACTLossHead(model, loss_type="stablemax_cross_entropy")
        if "DISABLE_COMPILE" not in os.environ:
            print("torch.compile(dynamic=False) …", flush=True)
            model = torch.compile(model, dynamic=False)  # type: ignore
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params:,} steps={steps} bs={batch_size}", flush=True)

    optimizers = [
        CastedSparseEmbeddingSignSGD_Distributed(
            model.model.puzzle_emb.buffers(),  # type: ignore
            lr=0, weight_decay=PUZZLE_EMB_WD, world_size=1),
        AdamATan2(model.parameters(), lr=0, weight_decay=WEIGHT_DECAY,
                  betas=(BETA1, BETA2)),
    ]
    optimizer_lrs = [PUZZLE_EMB_LR, LR]

    n_tr = len(tr)
    t0 = time.time()
    warmup_steps = max(1, int(steps * WARMUP_FRACTION))
    print(f"warmup={warmup_steps} steps ({WARMUP_FRACTION:.0%} of {steps})",
          flush=True)
    log_at = set(range(100, steps + 1, 100)) | {1}
    last_loss = float("nan")
    carry = None
    VIEWS_PER_PUZZLE = 8
    assert batch_size % VIEWS_PER_PUZZLE == 0
    n_puzzles = batch_size // VIEWS_PER_PUZZLE
    _val200 = []
    with open(SPLIT_REMOTE) as _fh:
        for _line in _fh:
            _r = json.loads(_line)
            if _r["split"] == "val":
                _val200.append(_r)
                if len(_val200) >= 200:
                    break
    _evx, _evy = _encode_np(_val200)
    model.train()
    for step in range(1, steps + 1):
        idx = np.random.choice(n_tr, n_puzzles, replace=False)
        idx = np.random.choice(n_tr, n_puzzles, replace=False)
        bxi, byi, pids = [], [], []
        for i in idx:
            for _ in range(VIEWS_PER_PUZZLE):
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
            with torch.device("cuda"):
                carry = model.initial_carry(batch)  # type: ignore
        carry, loss, metrics, _, _ = model(
            carry=carry, batch=batch, return_keys=[])  # type: ignore
        ((1 / batch_size) * loss).backward()
        for optim in optimizers:
            optim.step()
            optim.zero_grad()
        last_loss = float(loss.detach().item() / batch_size)
        if step in log_at:
            cnt = max(float(metrics["count"].detach().item()), 1.0)
            ex = float(metrics["exact_accuracy"].detach().item()) / cnt
            print(f"step={step:5d}/{steps} loss={last_loss:.4f} "
                  f"exact_acc={ex:.3f} "
                  f"lr={_cosine_lr(step, LR, warmup_steps, steps, MIN_RATIO):.2e}",
                  flush=True)
            if wb is not None:
                wb.log({"loss": last_loss, "step": step,
                        "train_exact_acc": ex})
        if step % 1000 == 0 and step != steps:
            # In-train eval on live weights (HRM has no EMA; VAL only).
            model.eval()
            _ev_batch = {"inputs": torch.from_numpy(_evx).to(device),
                         "labels": torch.zeros((200, 81), dtype=torch.long,
                                               device=device),
                         "puzzle_identifiers": torch.zeros(
                             200, dtype=torch.long, device=device)}
            with torch.no_grad():
                with torch.device("cuda"):
                    mc = model.initial_carry(_ev_batch)  # type: ignore
                while True:
                    mc, _, _, mo, done = model(  # type: ignore
                        carry=mc, batch=_ev_batch, return_keys=["logits"])
                    if bool(done):
                        break
                mid_pred = mo["logits"].argmax(-1).cpu().numpy()
            model.train()
            mid_ok = int((mid_pred == _evy).all(axis=1).sum())
            print(f"  [intrain-eval step={step}] val200 greedy "
                  f"{mid_ok}/200", flush=True)
            if wb is not None:
                wb.log({"intrain_val200": mid_ok, "step": step})
            mid_path = (f"{CHECKPOINT_MOUNT}/scaled_hrm/"
                        f"full_seed{seed}_{steps}s_bs{batch_size}"
                        f"_step{step}.pt")
            Path(mid_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": {
                k: v.cpu() for k, v in model.state_dict().items()},
                "cfg": cfg_dict}, mid_path)
            checkpoint_volume.commit()
            print(f"  saved intermediate {mid_path}", flush=True)
    train_s = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"TRAIN done in {train_s:.0f}s peakVRAM={peak_gb:.2f}GB", flush=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    ckpt_path = (f"{CHECKPOINT_MOUNT}/scaled_hrm/"
                 f"full_seed{seed}_{steps}s_bs{batch_size}_{ts}.pt")
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": {k: v.cpu() for k, v in
                                     model.state_dict().items()},
                "cfg": cfg_dict,
                "extra": {"steps": steps, "batch_size": batch_size,
                          "seed": seed, "train_s": train_s,
                          "peak_gb": peak_gb, "n_params": n_params,
                          "recipe": "hrm-full-upstream-faithful"}},
               ckpt_path)
    checkpoint_volume.commit()
    print(f"Saved: {ckpt_path}", flush=True)

    # ---- eval on live weights (HRM upstream has no EMA) ----
    model.eval()
    print("eval uses live weights (no EMA in HRM upstream)", flush=True)

    def _solve_view(inp_np):
        n = inp_np.shape[0]
        preds = np.zeros((n, 81), dtype=np.int64)
        with torch.no_grad():
            for s in range(0, n, batch_size):
                chunk = inp_np[s:s + batch_size]
                cb = chunk.shape[0]
                cbatch = {"inputs": torch.from_numpy(chunk).to(device),
                          "labels": torch.zeros((cb, 81), dtype=torch.long,
                                                device=device),
                          "puzzle_identifiers": torch.zeros(
                              cb, dtype=torch.long, device=device)}
                with torch.device("cuda"):
                    ccarry = model.initial_carry(cbatch)  # type: ignore
                while True:
                    ccarry, _, _, cout, done = model(  # type: ignore
                        carry=ccarry, batch=cbatch, return_keys=["logits"])
                    if bool(done):
                        break
                preds[s:s + batch_size] = (cout["logits"].argmax(-1)
                                           .cpu().numpy())
        return preds

    results = {}
    for split in ("val", "test"):
        srows = []
        with open(SPLIT_REMOTE) as fh:
            for line in fh:
                r = json.loads(line)
                if r["split"] == split:
                    srows.append(r)
        n = len(srows)
        ex_np, ey_np = _encode_np(srows)
        t1 = time.time()
        views = [_solve_view(ex_np)]
        rng_v = np.random.default_rng(seed + 777)
        for _ in range(N_VIEWS - 1):
            ps = [_aug_params(rng_v) for _ in range(n)]
            aug_in = np.stack([_apply_aug_np(ex_np[i], ps[i])
                               for i in range(n)])
            pr = _solve_view(aug_in)
            views.append(np.stack([_invert_aug_np(pr[i], ps[i])
                                  for i in range(n)]))
        views = np.stack(views)
        greedy_ok = (views[0] == ey_np).all(axis=1)
        vote_ok = np.zeros(n, dtype=bool)
        for i in range(n):
            cand, counts = np.unique(views[:, i, :], axis=0,
                                     return_counts=True)
            vote_ok[i] = bool((cand[int(np.argmax(counts))] == ey_np[i]).all())
        es = time.time() - t1
        bv = {"base": [0, 0], "variant": [0, 0]}
        clues: dict = {}
        hard: dict = {}
        for r, c in zip(srows, vote_ok.tolist()):
            k = "base" if r["is_base"] else "variant"
            bv[k][1] += 1
            bv[k][0] += c
            for d, key in ((clues, r["clues_count"]),
                           (hard, r["hardness_level"])):
                d.setdefault(key, [0, 0])
                d[key][1] += 1
                d[key][0] += c
        correct = int(vote_ok.sum())
        acc = correct / n
        gacc = float(greedy_ok.mean())
        # No timeout concept in ACT eval (always halts <= cap):
        # every non-correct prediction counts as wrong (LDT reports
        # wrong vs timeout separately; LDT wrong=0).
        results[split] = {"total": n, "correct": correct, "accuracy": acc,
                          "wrong": n - correct,
                          "greedy_acc": gacc,
                          "greedy_wrong": int((~greedy_ok).sum()),
                          "eval_s": es,
                          "base_variant": bv,
                          "by_clues": {str(k): v for k, v in sorted(
                              clues.items(), key=lambda kv: kv[0])},
                          "by_hardness": {str(k): v for k, v in sorted(
                              hard.items(), key=lambda kv: str(kv[0]))}}
        print(f"EVAL {split}: vote={correct}/{n} acc={acc:.4f} "
              f"greedy={gacc:.4f} ({es:.0f}s)", flush=True)
        print(f"  base={bv['base']} variant={bv['variant']}", flush=True)
        if wb is not None:
            wb.log({f"eval_{split}_acc": acc,
                    f"eval_{split}_greedy": gacc})

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
    out = {"checkpoint": ckpt_path, "steps": steps,
    # Budget accounting. HRM samples 8 augmentation views of each of
    # n_puzzles distinct puzzles per step, so examples seen (all rows)
    # is 8x the unique puzzle draws. Recorded explicitly so the report
    # never has to recompute it from (steps x batch_size).
    "examples_seen": steps * batch_size,
    "unique_puzzle_draws": steps * n_puzzles,
           "batch_size": batch_size, "seed": seed,
           "train_s": train_s, "peak_gb": peak_gb,
           "total_s": total_s, "integrity": integ,
           "n_params": n_params, "last_loss": last_loss,
           "val": results["val"], "test": results["test"]}
    p = Path(ckpt_path).with_suffix(".eval.json")
    p.write_text(json.dumps(out, indent=2, default=str))
    checkpoint_volume.commit()
    print(f"WROTE {p} total={total_s:.0f}s", flush=True)
    if wb is not None:
        wb.finish()
    return out


@app.local_entrypoint()
def entrypoint():
    print(json.dumps(run.remote(), indent=2, default=str)[:2000], flush=True)

