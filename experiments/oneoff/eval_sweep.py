"""EVAL SWEEP: how much of the TRM-vs-LDT gap is just test-time compute?

Turns the eval-compute confound into a measured axis. No retraining: both
models are loaded from their existing checkpoints on the Modal volume.

What it measures
----------------
  TRM: halt_max_steps in {16, 32, 64, 128} x 8 aug views, majority vote.
       At eval time upstream ACT always runs max steps (see trm.py:275
       `if self.training and ...`), so raising halt_max_steps gives
       strictly more reasoning steps per puzzle. checkpoint_steps lets us
       also log the ACT halt distribution (train-time ACT head intact).
  LDT: n_chains in {1, 2, 4, 8} at fixed max_rounds=50.

Common x-axis for plotting: `calls_per_puzzle` (model forward passes per
puzzle), the unit that is comparable across the two harnesses.

Subset by default (2000 test rows) to keep this ~$0.5. Set
SUBSET_ROWS=0 to run the full 10239-row test split.

Usage:
    modal run --detach experiments/oneoff/eval_sweep.py
"""

import json
import os
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

# NOTE: built from scratch rather than extending the shared `image`, because
# the shared one ends in add_local_python_source() and Modal forbids any build
# step after an add_local_* call. This sweep needs two extra pip installs
# (einops/pydantic for upstream TRM) on top of the shared base, so the whole
# chain is declared here in build-steps-first order, exactly as
# experiments/scaled_trm/run.py does. Values are identical to
# lattice_diffusion.modal.image (same debian_slim/py3.10/uv_sync/wandb).
full_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .uv_sync()
    # uv_pip_install (not pip_install): the uv venv has no pip module.
    .uv_pip_install("wandb", "einops", "pydantic")
    .add_local_python_source("lattice_diffusion")
    .add_local_python_source("experiments")
    .add_local_file(SPLIT_LOCAL, SPLIT_REMOTE)
    .add_local_dir("upstream/TinyRecursiveModels", "/opt/trm_upstream")
)

# Defaults for the real sweep. 2000 test rows keeps cost ~$0.5;
# subset_rows=0 runs the full 10239-row test split.
#
# Run the cheap validation first:
#     modal run experiments/oneoff/eval_sweep.py --smoke
# then the real sweep:
#     modal run --detach experiments/oneoff/eval_sweep.py
#
# NOTE: these are FUNCTION ARGUMENTS, not env vars. Modal does NOT forward
# local env vars into the container, so an env-var switch silently runs at
# full size — that is exactly what happened on the first smoke attempt
# (it billed a full 4x8x2000 sweep on a T4 while claiming to be a smoke).
SUBSET_ROWS = 2000
SUBSET_SEED = 12345
SMOKE = False
GPU = "B200"

TRM_HALT_STEPS = (16, 32, 64, 128)
LDT_N_CHAINS = (1, 2, 4, 8)
N_VIEWS = 8          # same 8-view budget the original TRM run reported
LDT_MAX_ROUNDS = 50  # same as the original LDT run


def smoke_overrides(smoke: bool):
    """Shrink every knob so the pipeline can be validated cheaply before
    spending B200 time (report 6.1: burn CPU minutes, not GPU minutes)."""
    if not smoke:
        return dict(subset_rows=SUBSET_ROWS, batch=768, views=N_VIEWS,
                    halt_steps=TRM_HALT_STEPS, n_chains=LDT_N_CHAINS,
                    max_rounds=LDT_MAX_ROUNDS)
    return dict(subset_rows=16, batch=8, views=2, halt_steps=(16,),
                n_chains=(1,), max_rounds=2)

app = modal.App("scaled-trm-eval-sweep")


def _pick(rows, n, seed):
    """Deterministic subset. Never reorders/duplicates: returns indices."""
    if not n or n >= len(rows):
        return list(range(len(rows)))
    rng = random.Random(seed)
    return sorted(rng.sample(range(len(rows)), n))


def _strip_orig_mod(sd):
    """The checkpoint was saved from a torch.compile'd wrapper, so every key
    is prefixed `_orig_mod.`. This run may load into either a compiled or an
    uncompiled module (smoke sets DISABLE_COMPILE), so normalise to the
    uncompiled ACTLossHead naming and drop nothing else."""
    if not any(k.startswith("_orig_mod.") for k in sd):
        return sd
    return {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
            for k, v in sd.items()}


def _final_ckpt(pattern):
    """Latest FINAL checkpoint for a pattern, skipping intermediate
    `_step<N>.pt` insurance checkpoints (train.py:571 / run.py:274).

    Sorting is not enough: `_20261005_...` sorts before `_step1000`, so a
    plain `[-1]` would silently pick an intermediate.
    """
    import re
    hits = [p for p in Path(CHECKPOINT_MOUNT).glob(pattern)
            if p.suffix == ".pt" and not re.search(r"_step\d+\.pt$", p.name)]
    if not hits:
        raise FileNotFoundError(
            f"no FINAL checkpoint matching {pattern} under {CHECKPOINT_MOUNT}. "
            f"present: "
            f"{[p.name for p in Path(CHECKPOINT_MOUNT).glob(pattern)][:10]}")
    hits.sort(key=lambda p: p.stat().st_mtime)
    return hits[-1]


# ---------------------------------------------------------------- TRM

@app.function(image=full_image, gpu=GPU, timeout=3600 * 4,
              volumes={CHECKPOINT_MOUNT: checkpoint_volume})
def sweep_trm(steps: int = 2667, batch_size: int = 768, seed: int = 0,
               smoke: bool = False):
    import numpy as np

    C = smoke_overrides(smoke)
    # NOTE: smoke deliberately does NOT disable torch.compile. The compile
    # path renames state_dict keys (`_orig_mod.` prefix) and is exactly what
    # the real B200 run uses, so a smoke that skips it would not cover the
    # failure mode we already hit once. 16 puzzles keeps it cheap anyway.
    if smoke:
        os.environ.pop("DISABLE_COMPILE", None)
    bs = min(batch_size, C["batch"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    print(f"smoke={smoke} device={device} batch={bs} "
          f"subset={C['subset_rows']} views={C['views']} "
          f"halt={C['halt_steps']}", flush=True)

    # The original run.py used /opt/trm_upstream for `models.*` and
    # experiments.scaled_trm.adam_atan2_torch for the optimizer; eval needs
    # neither, but the upstream path is required for the model classes.
    sys.path.insert(0, "/opt/trm_upstream")

    from models.losses import ACTLossHead  # type: ignore
    from models.recursive_reasoning.trm import (  # type: ignore
        TinyRecursiveReasoningModel_ACTV1)

    from experiments.scaled_trm.run import (
        _apply_aug_np, _aug_params, _encode_np, _invert_aug_np)

    ckpt_path = _final_ckpt(
        f"scaled_trm/full4_seed{seed}_{steps}s_bs{batch_size}_*.pt")
    print(f"TRM checkpoint: {ckpt_path}", flush=True)

    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg_dict = dict(ck["cfg"])
    n_params = ck.get("extra", {}).get("n_params")
    print(f"  n_params={n_params} train_s={ck.get('extra', {}).get('train_s')}",
          flush=True)

    rows = [json.loads(l) for l in open(SPLIT_REMOTE)]
    tr = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    keep = _pick(test, C["subset_rows"], SUBSET_SEED)
    sub = [test[i] for i in keep]
    print(f"  train={len(tr)} test={len(test)} subset={len(sub)}", flush=True)

    # Order matters: build -> load weights -> EMA -> eval mode -> compile.
    # torch.compile renames every state_dict key with an `_orig_mod.` prefix,
    # so loading into an already-compiled wrapper would need the reverse
    # mapping. Compiling LAST keeps this script's state_dict in the same
    # uncompiled namespace regardless of the DISABLE_COMPILE setting.
    with torch.device("cuda"):
        model = TinyRecursiveReasoningModel_ACTV1(cfg_dict)
        model = ACTLossHead(model, loss_type="stablemax_cross_entropy")

    model.load_state_dict(_strip_orig_mod(ck["model_state_dict"]))
    model.to(device)

    # EMA weights are the ones the original run evaluated with.
    if "ema_state_dict" in ck:
        from models.ema import EMAHelper  # type: ignore
        ema = EMAHelper(mu=0.999)
        ema.load_state_dict(_strip_orig_mod(ck["ema_state_dict"]))
        ema.ema(model)
        print("  loaded EMA weights (as original eval)", flush=True)
    else:
        print("  WARNING: no ema_state_dict, using live weights", flush=True)

    model.eval()
    if "DISABLE_COMPILE" not in os.environ:
        print("torch.compile …", flush=True)
        model = torch.compile(model)  # type: ignore

    n = len(sub)
    ex_np, ey_np = _encode_np(sub)
    base_mask = np.array([r["is_base"] for r in sub])

    def _set_halt_steps(H):
        """halt_max_steps is a Pydantic BaseModel field (trm.py:32), shared by
        reference across outer/inner/blocks. At eval time upstream always runs
        max steps (trm.py:275 gates ACT on self.training), so raising this
        gives strictly more reasoning steps with no ACT early-exit."""
        cfg_obj = None
        for path in ("model.config", "model.model.config",
                     "model._orig_mod.model.config",
                     "model._orig_mod.config"):
            obj = model
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                cfg_obj = obj
                break
            except AttributeError:
                continue
        if cfg_obj is None:
            raise AttributeError("could not reach TRM config object")
        cfg_obj.halt_max_steps = int(H)
        return cfg_obj

    def _solve(inp_np, halt_steps, collect_q=False):
        _set_halt_steps(halt_steps)
        m = n
        preds = np.zeros((m, 81), dtype=np.int64)
        q_halt_max = -np.inf
        with torch.no_grad():
            for s in range(0, m, bs):
                chunk = inp_np[s:s + bs]
                cb = chunk.shape[0]
                cbatch = {"inputs": torch.from_numpy(chunk).to(device),
                          "labels": torch.zeros((cb, 81), dtype=torch.long,
                                                device=device),
                          "puzzle_identifiers": torch.zeros(
                              cb, dtype=torch.long, device=device)}
                with torch.device("cuda"):
                    carry = model.initial_carry(cbatch)  # type: ignore
                while True:
                    carry, _, _, cout, done = model(  # type: ignore
                        carry=carry, batch=cbatch,
                        return_keys=["preds", "q_halt_logits"])
                    if collect_q:
                        q_halt_max = max(
                            q_halt_max,
                            float(cout["q_halt_logits"].max().item()))
                    if bool(done):
                        break
                preds[s:s + bs] = cout["preds"].cpu().numpy()
        return preds, (q_halt_max if collect_q else None)

    results = []
    for H in C["halt_steps"]:
        t0 = time.time()
        views = []
        for v in range(C["views"]):
            ps = None
            if v == 0:
                aug_in = ex_np
            else:
                rng_v = np.random.default_rng(seed * 1000 + 777 + v)
                ps = [_aug_params(rng_v) for _ in range(n)]
                aug_in = np.stack([_apply_aug_np(ex_np[i], ps[i])
                                   for i in range(n)])
            pr, qmax = _solve(aug_in, H, collect_q=(v == 0))
            if ps is None:
                views.append(pr)
            else:
                views.append(np.stack([_invert_aug_np(pr[i], ps[i])
                                       for i in range(n)]))
        views_arr = np.stack(views)
        greedy_ok = (views_arr[0] == ey_np).all(axis=1)
        vote_ok = np.zeros(n, dtype=bool)
        for i in range(n):
            cand, counts = np.unique(views_arr[:, i, :], axis=0,
                                     return_counts=True)
            vote_ok[i] = bool((cand[int(np.argmax(counts))] == ey_np[i]).all())
        el = time.time() - t0
        calls = H * C["views"]
        row = {
            "model": "trm",
            "halt_steps": H,
            "n_views": C["views"],
            "calls_per_puzzle": calls,
            "n": n,
            "vote_correct": int(vote_ok.sum()),
            "vote_acc": float(vote_ok.mean()),
            "greedy_acc": float(greedy_ok.mean()),
            "base_vote": [int((vote_ok & base_mask).sum()), int(base_mask.sum())],
            "variant_vote": [int((vote_ok & ~base_mask).sum()),
                             int((~base_mask).sum())],
            "eval_s": el,
            "train_halt_max": ck["cfg"]["halt_max_steps"],
            "eval_q_halt_max": qmax,
        }
        results.append(row)
        print(f"TRM H={H:4d} views={C['views']} calls/puzzle={calls:5d} "
              f"vote={row['vote_correct']}/{n} ({row['vote_acc']:.4f}) "
              f"greedy={row['greedy_acc']:.4f} "
              f"base={row['base_vote']} variant={row['variant_vote']} "
              f"({el:.0f}s)", flush=True)

    out = {"model": "trm", "checkpoint": str(ckpt_path),
           "n_params": n_params, "subset_rows": C["subset_rows"],
           "subset_seed": SUBSET_SEED, "smoke": smoke, "rows": results}
    if smoke:
        print(json.dumps(out, indent=2, default=str)[:1200], flush=True)
        return out
    p = Path(CHECKPOINT_MOUNT) / "scaled_trm" / "eval_sweep_trm.json"
    p.write_text(json.dumps(out, indent=2, default=str))
    checkpoint_volume.commit()
    print(f"WROTE {p}", flush=True)
    return out


# ---------------------------------------------------------------- LDT

@app.function(image=full_image, gpu=GPU, timeout=3600 * 4,
              volumes={CHECKPOINT_MOUNT: checkpoint_volume})
def sweep_ldt(steps: int = 4000, batch_size: int = 512, seed: int = 0,
               smoke: bool = False):
    import dataclasses

    from torch import nn

    from lattice_diffusion.models.looped_transformer import (
        LoopedTransformerConfig, PowersetModel)
    from lattice_diffusion.training.utils.checkpoint import load_checkpoint
    from lattice_diffusion.data.scaled_sudoku import (
        encode_grid, encode_solution)
    from experiments.sudoku.dpll import StepConfig
    from experiments.sudoku.ema import swap_in_ema_if_present
    from experiments.sudoku.solve import SolveConfig, solve

    C = smoke_overrides(smoke)
    ldt_batch = min(batch_size, C["batch"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    print(f"smoke={smoke} device={device} batch={ldt_batch} "
          f"subset={C['subset_rows']} chains={C['n_chains']} "
          f"rounds={C['max_rounds']}", flush=True)

    ckpt_path = _final_ckpt(
        f"scaled/seed{seed}_{steps}s_bs{batch_size}_*.pt")
    print(f"LDT checkpoint: {ckpt_path}", flush=True)

    ck = load_checkpoint(str(ckpt_path))
    model = PowersetModel(LoopedTransformerConfig(**ck["model_cfg"]))
    model.load_state_dict(_strip_orig_mod(ck["model_state_dict"]))
    model.to(device).eval()
    swap_in_ema_if_present(model, ck)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.05
        elif isinstance(m, nn.MultiheadAttention):
            m.dropout = 0.05
    model.train()  # official dropout-noise eval mode
    print("eval dropout p=0.05 (official run.py default)", flush=True)

    step_cfg = StepConfig(threshold=0.10, temp_decide=1.5, cls_threshold=0.5,
                          augment=True)
    eval_step_cfg = dataclasses.replace(step_cfg, cls_threshold=0.6)

    rows_all = [json.loads(l) for l in open(SPLIT_REMOTE)]
    test = [r for r in rows_all if r["split"] == "test"]
    keep = _pick(test, C["subset_rows"], SUBSET_SEED)
    sub = [test[i] for i in keep]
    n = len(sub)
    print(f"  test={len(test)} subset={n}", flush=True)

    import numpy as np
    X = np.stack([encode_grid(r["puzzle"]) for r in sub])
    Y = np.stack([encode_solution(r["solution"]) for r in sub])
    x = torch.from_numpy(X).to(device).float()
    y = torch.from_numpy(Y).to(device).float()
    gm = (x.sum(dim=-1) == 1)
    base_mask = np.array([r["is_base"] for r in sub])

    results = []
    for K in C["n_chains"]:
        t0 = time.time()
        res = solve(model, x, y, gm,
                    SolveConfig(step=eval_step_cfg, max_rounds=C["max_rounds"],
                                n_chains=K, batch_size=ldt_batch),
                    verbose=False)
        el = time.time() - t0
        corr = res.correct.cpu().numpy().astype(bool)
        calls = res.model_calls / max(n, 1)
        row = {
            "model": "ldt",
            "n_chains": K,
            "max_rounds": C["max_rounds"],
            "calls_per_puzzle": calls,
            "n": n,
            "correct": int(corr.sum()),
            "accuracy": float(corr.mean()),
            "wrong": int(res.wrong.sum()),
            "timeout": int(res.timeouts.sum()),
            "base": [int((corr & base_mask).sum()), int(base_mask.sum())],
            "variant": [int((corr & ~base_mask).sum()),
                        int((~base_mask).sum())],
            "eval_s": el,
        }
        results.append(row)
        print(f"LDT K={K} rounds={C['max_rounds']} calls/puzzle={calls:.1f} "
              f"acc={row['accuracy']:.4f} wrong={row['wrong']} "
              f"to={row['timeout']} base={row['base']} "
              f"variant={row['variant']} ({el:.0f}s)", flush=True)

    out = {"model": "ldt", "checkpoint": str(ckpt_path),
           "n_params": int(sum(p.numel() for p in model.parameters())),
           "subset_rows": C["subset_rows"], "subset_seed": SUBSET_SEED,
           "smoke": smoke, "rows": results}
    if smoke:
        print(json.dumps(out, indent=2, default=str)[:1200], flush=True)
        return out
    p = Path(CHECKPOINT_MOUNT) / "scaled" / "eval_sweep_ldt.json"
    p.write_text(json.dumps(out, indent=2, default=str))
    checkpoint_volume.commit()
    print(f"WROTE {p}", flush=True)
    return out


@app.local_entrypoint()
def entrypoint(smoke: bool = False):
    C = smoke_overrides(smoke)
    trm = sweep_trm.remote(smoke=smoke)
    ldt = sweep_ldt.remote(smoke=smoke)

    print("\n" + "=" * 72)
    print("TEST-TIME COMPUTE SWEEP (subset=%d rows%s)"
          % (C["subset_rows"], ", SMOKE" if smoke else ""))
    print("=" * 72)
    print(f"{'model':6s} {'budget':>18s} {'calls/puzzle':>13s} "
          f"{'acc':>8s}  base")
    for r in trm["rows"]:
        print(f"{'TRM':6s} {'H=%d x %d views' % (r['halt_steps'], r['n_views']):>18s} "
              f"{r['calls_per_puzzle']:>13d} {r['vote_acc']:>8.4f}  "
              f"{r['base_vote'][0]}/{r['base_vote'][1]}")
    for r in ldt["rows"]:
        print(f"{'LDT':6s} {'K=%d x %d rnds' % (r['n_chains'], r['max_rounds']):>18s} "
              f"{r['calls_per_puzzle']:>13.1f} {r['accuracy']:>8.4f}  "
              f"{r['base'][0]}/{r['base'][1]}")
    print("=" * 72)
    print("TRM trained with halt_max_steps=%d; eval sweep goes BEYOND that."
          % trm["rows"][0]["train_halt_max"])
