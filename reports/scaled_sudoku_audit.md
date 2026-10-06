# SCALED SUDOKU AUDIT (Base + Walk Variants, ancestor-group split)

- Dataset: `zayenz/scaled-sudoku-instances` (local: `scaled-sudoku-instances/scaled-sudoku-instances/`, READ-ONLY)
- LDT repo commit: `a1c534a`
- Scope: audit + split only. No training, no GPU, no Modal/Kaggle, no checkpoints.
- Method: `scaled_audit.py` (stdlib-only, CPU) over all 102,456 9×9 instances (204,912 files).

## 1. Dataset structure

```text
base/9x9/chunk-NNNN/puzzle-MMMMMM.sdk.{txt,json}        <- Base puzzles
walks/variants/9x9/chunk-NNNN/puzzle-MMMMMM.sdk/        <- Variant groups per Base
    walk-W/
        step-TTTT-<LEVEL>.sdk.{txt,json}                <- Variants (easier: base + added hints)
scripts/, LICENSE (MIT), README.md
```

`.sdk.txt` = header `# size=9 box=3x3` + 9 rows × 9 chars (`.` blank, `1-9` givens).
`.sdk.json` sidecar = `schema_version, id, identity{source_path, source_id, solution_hash},
size{order, box_width, box_height}, clues_count, solution_grid[81],
hardness{level, scheme}, minimality{is_minimal, status}, provenance{kind, walk{...}}`.

## 2. Base/Walk/Variant semantics (from real metadata, not filenames)

- **Base**: `provenance.kind = "base"`. Its sidecar lists `provenance.walk.transitions[]`
  (index of all walks: `walk_index, transition_index, from, to, variant_relative_path`).
- **Variant**: `provenance.kind = "walk-variant"` with `provenance.walk` =
  `{source_relative_path, source_id, source_solution_hash, walk_index, transition_index,
  parent_variant_relative_path, from, to, transition_hints[], added_hints[]}`.
- **Walk** = one `walk_index` chain; step order = `transition_index`; chain links =
  `parent_variant_relative_path` (null on first step; verified: 44,146 parent links resolve,
  48,310 first-steps have null parent AND `transition_index == 1`).
- Direction: variants ADD hints to the base (`added_hints` cumulative, e.g. step-0001 24→25 clues,
  step-0002 →29 clues; `from: BndBS → to: ValBS → to: Val`, i.e. weaker propagation suffices).
- **ancestor_id** = `identity.source_id` (base) = `provenance.walk.source_id` (variant),
  e.g. `chunk-0001-1`. Directory path is consistent with it but metadata is authoritative.

## 3. 9×9 filtering

```text
Total instances (all sizes):  434,201  (32,000 base + 402,201 variants)
9×9 Base:                      10,000
9×9 Variants:                  92,456
Other sizes excluded:         331,745  (6x6: 59,515 | 16x16: 177,026 | 25x25: 47,079 | 36x36: 48,125)
Number of ancestor groups:     10,000
```

Counts match the author's README exactly. LDT keeps `seq_len=81`, 9 channels — no architecture change.

## 4. Number of Base instances — 10,000 (all parsed, all valid)

## 5. Number of Variant instances — 92,456 (all parsed, all valid)

## 6. Number of ancestor groups — 10,000, instances/ancestor min=4 / max=16 / mean=10.25; variants/ancestor min=3 / max=15 / mean=9.25. Walks total = 49,922, max `transition_index` = 3.

## 7. Validation results (per-instance, 102,456 instances)

- Format: 9 lines × 9 chars of `[.1-9]`, header exact — 0 errors.
- `clues_count` == actual digit count — 0 mismatches.
- Every given matches `solution_grid` at that cell — 0 mismatches.
- `solution_grid`: 81 values in 1–9 AND valid Sudoku (rows/cols/3×3 boxes all 1–9) — 0 invalid.
- Metadata presence over all instances: `size, clues_count, solution_grid, hardness, provenance` = 102,456/102,456.
- ERRORS = 0. No repair performed on source data.

## 8. Duplicate results

- Exact duplicate puzzle grids: **0 groups** (every walk step adds hints, so even same-ancestor grids differ).
- Shared solutions: 10,000/10,000 distinct solutions each appear in >1 puzzle — EXPECTED (base + its variants share one solution by design). Reported, not treated as leakage (grouped by ancestor).

## 9. Ancestor integrity

- Orphan variants (ancestor without base): **0**.
- Instances linked to multiple ancestors: **0**.
- Base `transitions[]` index: 94,149 entries, **0 dangling** `variant_relative_path`.
- Walk chain reconstructable for every variant via `(walk_index, transition_index, parent_variant_relative_path)`.

## 10. Train/Val/Test statistics (seed=42, split by ancestor_id, 80/10/10)

```text
Split     Ancestors    Instances    Base    Variants
----------------------------------------------------
Train        8000        82043      8000     74043
Val          1000        10174      1000      9174
Test         1000        10239      1000      9239
```

Each split contains Base + Variants; every ancestor group lives in exactly one split.
Distribution is balanced (no skew forcing a protocol change): clues mean 27.70/27.68/27.71,
median 27/27/27; hardness dominated by Val/ValBS in all splits (~98%), rare levels (Bnd/Dom…)
present in all splits.

## 11. Leakage check

```text
Cross-split ancestor leakage = 0
Cross-split exact-puzzle leakage = 0
Every ancestor group belongs to exactly one split: YES
```

Criterion `leakage == 0`: SATISFIED. (Same-solution across base↔variants stays inside one split by construction.)

## 12. Clue distribution (per split)

- Train: min=20 max=44 mean=27.70 median=27
- Val:   min=21 max=45 mean=27.68 median=27
- Test:  min=21 max=43 mean=27.71 median=27

## 13. Hardness distribution (per split)

- Train: Val=40009 ValBS=39903 ValDS=857 ValChan=375 BndBS=823 BndDS=38 Bnd=1 DomBS=34 Dom=3
- Val:   Val=5002 ValBS=4919 ValDS=104 ValChan=35 BndBS=106 BndDS=4 Bnd=1 DomBS=2 Dom=1
- Test:  Val=5000 ValBS=5004 ValDS=100 ValChan=22 BndBS=105 Bnd=2 DomBS=4 Dom=2
- Note: levels are propagation buckets (Simonis hierarchy), not human difficulty; `Val*` ≈ 98%.

## 14. Trajectory statistics

- 49,922 walks; up to 3 transitions/walk; variants/ancestor 3–15 (mean 9.25).
- Split metadata preserves full trajectory: each row carries `ancestor_id, is_base, walk_index,
  transition_index, parent_variant_relative_path, from/to_level, clues_count, hardness_level,
  puzzle, solution, split` — any test instance traces back to its Base and walk position.
- Dataset is NOT flattened: ordering/trajectory recoverable from `scaled_sudoku_split.jsonl`.

## 15. LDT input compatibility (CPU spot-check, Base+Variant × Train/Val/Test)

`. → [1×9]`, `d → one-hot(d)`; all 6 spot samples produce `X=[81,9], Y=[81,9]` — OK.
`src/lattice_diffusion/data/scaled_sudoku.py` implements this mapping (+ split-aware streaming
reader with the same `next_batch() -> (x, y, is_sat)` interface as `SudokuExtremeDataset`).
Untouched: `transformer2d.py, looped_transformer.py, dpll.py, solve.py`.

## 16. Files created/modified

- CREATED `scaled_sudoku_split.jsonl` — 102,456 rows, frozen split (seed 42).
- CREATED `src/lattice_diffusion/data/scaled_sudoku.py` — CPU-only adapter (numpy; torch only
  inside prefetch thread, never exercised in this phase).
- CREATED `scaled_sudoku_audit.md` — this report.
- MODIFIED: none. Dataset folder: read-only, 0 writes. No LDT source modified (adapter is additive).
- Audit script kept outside repo (`Temp\opencode\scaled_audit.py`); re-runnable for verification.

## 17. Problems found

1. (Observation, non-blocking) One base sidecar index entry mislabels a path
   (`puzzle-000001`: walk 5 / transition 1 → `walk-0002/...` path). Path still resolves
   (0 dangling of 94,149), and grouping relies on variant sidecars, so no impact.
2. (By design) Variants share their base's solution — must never be split by instance
   (handled: ancestor-group split, leakage 0).
3. (By design) No official train/val/test split upstream — solved deterministically in-repo
   (`scaled_sudoku_split.jsonl`, seed 42).
4. Hardness is heavily skewed to Val/ValBS (~98%); stratified analysis on rare levels will have
   small n — flagged for the training phase, protocol unchanged.

## 18. Final verdict

```text
PASS — Dataset đã được audit, split theo ancestor đã tạo, không phát hiện leakage,
và dataset đã sẵn sàng cho phase training.
```

Safety: Training executed NO · GPU used NO · Modal/Kaggle used NO · Evaluation NO ·
Checkpoint touched NO · LDT source modified NO (1 additive adapter file) · Dataset modified NO.
Cost GPU: $0. HARD STOP here — no smoke training, no main training initiated.
