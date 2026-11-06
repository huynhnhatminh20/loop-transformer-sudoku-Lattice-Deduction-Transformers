"""LEARNING CURVE: is 2667 steps enough, or did we stop early?

Re-runs the SAME experiments/scaled_trm/run.py function at 667 and 1333
steps, then collects the three points of the curve:

    667 / 1333 / 2667 steps  x  batch 768  =  0.51M / 1.02M / 2.05M examples

Why this is a separate driver and not a modified run.py: `run()` already
takes `steps` as a parameter and owns the whole recipe (optimizer, EMA,
warmup, cosine schedule, aug). Driving it with different step counts
guarantees the curve points differ ONLY in budget. Copying the train loop
into a new file would risk silent recipe drift, which is exactly the class
of bug the reports keep hitting.

NOTE on the LR schedule: run.py's cosine is annealed over the run's own
total steps, so each point is a properly annealed run, NOT a checkpoint
truncated from the 2667-step run. That is the comparison we want for
"is 2667 enough" — but it means the 2667 point is not literally the same
optimisation trajectory extended. Disclosed in the output as
"independent annealed runs".

Saves a per-step curve checkpoint for each point, so the shape can be
re-plotted later without re-running.

Usage:
    modal run --detach experiments/oneoff/learning_curve.py
"""

import json
import os
from pathlib import Path

import modal

# 2667 is the already-trained reference run; re-running it would cost ~$2
# for a number we already have (eval_sweep_trm.json, and the original
# .eval.json on the volume).
# Read at MODULE level, i.e. on the local machine before anything ships to
# Modal — so this env var works (an env var read inside the remote function
# would silently see nothing and run at full size).
if os.environ.get("LC_SMOKE") == "1":
    CURVE_STEPS = (2,)
elif os.environ.get("LC_ONLY"):
    # Re-run a single curve point (e.g. after an aborted run left a
    # checkpoint without its .eval.json): LC_ONLY=1333
    CURVE_STEPS = (int(os.environ["LC_ONLY"]),)
else:
    CURVE_STEPS = (667, 1333)
REFERENCE_STEPS = 2667

# No subset here on purpose: run.py evaluates the FULL val+test split, which
# is what makes the three points directly comparable to the 2667-step
# reference already on the volume (30.45% vote / 17.86% greedy, base 0/1000).

# NOTE: run() is defined on run.py's own `app`, so this module must reuse that
# same app object rather than declaring its own — Modal refuses to hydrate a
# function whose defining app is not the running one ("the App it is defined on
# is not running"). Importing run.py also registers ITS own `entrypoint`, so
# ours must be renamed (local entrypoint names must be unique).
from experiments.scaled_trm.run import app  # noqa: F401  (app = trm_app)
from experiments.scaled_trm.run import run


@app.local_entrypoint(name="learning-curve")
def entrypoint():

    out = {"reference_steps": REFERENCE_STEPS,
           "reference_source": "reports/TRM.md (full test split)",
           "points": []}

    for steps in CURVE_STEPS:
        print(f"\n{'=' * 72}\nLEARNING CURVE POINT: steps={steps} "
              f"(batch 768 -> {steps * 768:,} examples)\n{'=' * 72}",
              flush=True)
        r = run.remote(steps=steps, batch_size=768, seed=0)
        point = {
            "steps": steps,
            "examples": steps * 768,
            "checkpoint": r["checkpoint"],
            "train_s": r["train_s"],
            "peak_gb": r["peak_gb"],
            "n_params": r["n_params"],
            "last_loss": r["last_loss"],
            "test_vote_acc": r["test"]["accuracy"],
            "test_greedy_acc": r["test"]["greedy_acc"],
            "test_base": r["test"]["base_variant"]["base"],
            "test_variant": r["test"]["base_variant"]["variant"],
            "val_vote_acc": r["val"]["accuracy"],
        }
        out["points"].append(point)
        print(f"  steps={steps} loss={r['last_loss']:.4f} "
              f"test_vote={point['test_vote_acc']:.4f} "
              f"test_greedy={point['test_greedy_acc']:.4f} "
              f"base={point['test_base']} variant={point['test_variant']}",
              flush=True)

    print(f"\n{'=' * 72}\nLEARNING CURVE (full test split, batch 768)\n"
          f"{'=' * 72}")
    print(f"{'steps':>8s} {'examples':>12s} {'loss':>8s} {'test vote':>11s} "
          f"{'test greedy':>12s}  base")
    for p in out["points"]:
        print(f"{p['steps']:>8d} {p['examples']:>12,d} {p['last_loss']:>8.4f} "
              f"{p['test_vote_acc']:>11.4f} {p['test_greedy_acc']:>12.4f}  "
              f"{p['test_base'][0]}/{p['test_base'][1]}")
    print(f"{REFERENCE_STEPS:>8d} {REFERENCE_STEPS * 768:>12,d} "
          f"{'see volume':>8s} {0.3045:>11.4f} {0.1786:>12.4f}  0/1000")
    print("=" * 72)
    print("NOTE: each point is an independently annealed run (cosine over its")
    print("own total steps), not a truncated prefix of the 2667-step run.")
    print("Reference row = original full-split number from reports/TRM.md.")
    print("All three rows are measured on the full 10239-row test split.")

    return out
