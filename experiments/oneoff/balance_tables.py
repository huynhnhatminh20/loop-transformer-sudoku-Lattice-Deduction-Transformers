"""DRIVER for the three balance tables. Adds no new training logic.

Each table fixes a different budget axis; this script only chooses the
`steps` argument passed to each model's existing `run()` and prints what
`.eval.json` recorded. Nothing in `scaled/run.py`, `scaled_trm/run.py` or
`scaled_hrm/run.py` is modified.

Budget axes
-----------
Table 1 - equal UNIQUE PUZZLE DRAWS (256,512)
Table 2 - equal TRAINING EXAMPLES (2,047,488)
Table 3 - equal WALL-CLOCK (~360s); LDT steps set after its first run

Why these step counts
---------------------
LDT draws batch_size=512 distinct puzzles per step, 1 view each.
TRM/HRM draw n_puzzles=96 distinct puzzles per step, 8 views each.
  Table 1: 501 x 512 = 2.672 x 96 = 256,512
           exact because 256,512 = 167 * lcm(512,96) = 167*1,536
  Table 2: 3.999 x 512 = 2.666 x 768 = 2,047,488   exact
The non-matched axis differs 8x in every table and is reported, not hidden.

Usage
-----
Options come from the environment because `modal run` consumes its own CLI
flags and rejects anything it does not recognise.

    BT_TABLE=1 BT_SEEDS="0 1 2" \\
        modal run -m experiments.oneoff.balance_tables::ldt_entry
    BT_TABLE=1 BT_SEEDS="0" \\
        modal run -m experiments.oneoff.balance_tables::trm_entry
    BT_TABLE=1 BT_SEEDS="0" \\
        modal run -m experiments.oneoff.balance_tables::hrm_entry

Each entrypoint is bound to its own model's App so Modal hydrates that
function; one entrypoint cannot drive all three apps at once.
"""

import json
import os

from experiments.scaled.run import app as ldt_app
from experiments.scaled.run import run as ldt_run
from experiments.scaled_hrm.run import app as hrm_app
from experiments.scaled_hrm.run import run as hrm_run
from experiments.scaled_trm.run import app as trm_app
from experiments.scaled_trm.run import run as trm_run

BATCH = {"ldt": 512, "trm": 768, "hrm": 768}

STEPS = {
    1: {"ldt": 501, "trm": 2672, "hrm": 2672},
    2: {"ldt": 3999, "trm": 2666, "hrm": 2666},
}
# Table 3: fixed from LDT's measured post-compile train rate.
# LDT 501 steps: post-compile train = 79.2s  ->  360/79.2*501 = 2277 steps
TABLE3 = {"ldt": 2277, "trm": 2672, "hrm": 2672}

RUNNER = {"ldt": ldt_run, "trm": trm_run, "hrm": hrm_run}

REPORT_KEYS = ["steps", "batch_size", "seed", "examples_seen",
               "unique_puzzle_draws", "train_s", "peak_gb", "last_loss"]


def steps_for(table, model):
    if table == 3:
        return TABLE3[model]
    return STEPS[table][model]


def go(model):
    table = int(os.environ.get("BT_TABLE", "1"))
    seeds = [int(s) for s in os.environ.get("BT_SEEDS", "0").split()]
    print(f"### TABLE {table} | {model.upper()} | seeds={seeds} ###",
          flush=True)
    for seed in seeds:
        steps = steps_for(table, model)
        bs = BATCH[model]
        print(f"\n{'='*70}")
        print(f"TABLE {table} | {model.upper()} | steps={steps} "
              f"batch={bs} seed={seed}", flush=True)
        print("="*70, flush=True)
        r = RUNNER[model].remote(steps=steps, batch_size=bs, seed=seed)
        got = {k: r.get(k) for k in REPORT_KEYS if k in r}
        if "test" in r:
            t = r["test"]
            got["test_acc"] = t.get("accuracy")
            got["test_correct"] = t.get("correct")
            got["test_total"] = t.get("total")
            got["test_base"] = (t.get("base_variant", {}).get("base", [None])[0])
        print(json.dumps(got, indent=2, default=str), flush=True)
        print(f"CHECKPOINT {r.get('checkpoint')}", flush=True)


@ldt_app.local_entrypoint()
def ldt_entry():
    go("ldt")


@trm_app.local_entrypoint()
def trm_entry():
    go("trm")


@hrm_app.local_entrypoint()
def hrm_entry():
    go("hrm")
