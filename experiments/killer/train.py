"""Pool-based trainer for Killer Sudoku (Exp #2: LDT + Cage/Sum Input).

Mirrors `experiments/snowflake/train.py` but adapts for Killer:
  - State [B, 81, 12]: `state[..., :9]` = vocab powerset (digits 1..9),
    `state[..., 9:]` = 3 locked auxiliary channels (cage_id_norm,
    sum_norm, pos_in_cage_norm). Model has `n_channels=12`.
  - `cfg.step.vocab_dim=9` so deduce/decide/loss never touch aux channels.
  - Digit permutation is INVALID for Killer (it breaks cage sums), so
    `cfg.step.permute_digits=False`: augmentation is dihedral-spatial
    only (rotation/reflection moves cages rigidly; sums stay glued).
    Implemented by passing `vocab_dim=0` (identity digit-perm) to
    `aug_forward`, mirroring `dpll_step`'s internal handling.
  - NO Sum Deduction Operator in this version (per spec).

Pool / discard policy identical to sudoku/snowflake. Same `dpll_step`.

Checkpoint reuse (`init_from`): Sudoku-Extreme checkpoint
(Linear(9->128) input) is partially loaded — every param with matching
shape is copied; `input_proj.weight` contributor columns [:, :9] are
copied and columns [:, 9:12] are freshly initialized (Normal 0.02).
Backbone + all three heads keep-compatible (shapes unchanged).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from lattice_diffusion.models.looped_transformer import LoopedTransformerConfig, PowersetModel
from lattice_diffusion.models.weighted_bce import weighted_bce_with_logits
from lattice_diffusion.training.utils.checkpoint import load_checkpoint, save_checkpoint
from lattice_diffusion.training.utils.scheduler import make_cosine_scheduler

from experiments.sudoku.aug import aug_forward, build_dihedral_cell_perms
from experiments.sudoku.dpll import StepConfig, dpll_step
from experiments.sudoku.ema import ModelEMA
from experiments.sudoku.solve import SolveConfig, solve

from experiments.killer.data import KillerConfig, KillerDataset

# Killer recipe constants (must stay consistent with data.py).
VOCAB = 9            # vocab channels (digits 1..9)
N_AUX = 3            # cage_id_norm, sum_norm, pos_in_cage_norm
N_CHANNELS = VOCAB + N_AUX  # 12
GRID_ROWS = 9
GRID_COLS = 9
SEQ_LEN = GRID_ROWS * GRID_COLS  # 81


def _default_step_cfg() -> StepConfig:
    """Killer StepConfig: vocab_dim=9, dihedral spatial aug, NO digit-perm
    (digit permutation would break cage sums)."""
    return StepConfig(vocab_dim=VOCAB, augment_dihedral=True, permute_digits=False)


def _default_model_cfg() -> LoopedTransformerConfig:
    return LoopedTransformerConfig(
        n_channels=N_CHANNELS,
        seq_len=SEQ_LEN,
        grid_rows=GRID_ROWS,
        grid_cols=GRID_COLS,
        cls_token=True,
    )


@dataclass
class TrainConfig:
    steps: int = 1000
    batch_size: int = 512
    lr: float = 3e-3
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_fraction: float = 0.1
    seed: int = 0

    bce_pos_mult: float = 4.0
    bce_neg_mult: float = 0.5
    softmax_loss_weight: float = 0.2
    conflict_loss_weight: float = 0.1

    step: StepConfig = field(default_factory=_default_step_cfg)
    max_age: int = 100
    compile: bool = True

    use_ema: bool = False
    ema_decay: float = 0.999

    log_every: int = 20
    eval_every: int = 100
    eval_n_puzzles: int = 200
    eval_max_rounds: int = 5
    eval_n_chains: int = 64

    # Sudoku-Extreme checkpoint for partial init (None = from scratch).
    init_from: str | None = None

    out_dir: str = "checkpoints/killer"
    name: str = ""

    model: LoopedTransformerConfig = field(default_factory=_default_model_cfg)
    data: KillerConfig = field(default_factory=KillerConfig)
    eval_data: KillerConfig | None = None


def _build_state(orig_x: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
    """Build [B, S, 12] state from [B, S, 9] vocab + [B, S, 3] aux features."""
    return torch.cat([orig_x, aux], dim=-1)


def _gt_conflict(
    state: torch.Tensor,            # [B, S, C] (vocab + aux)
    orig_y: torch.Tensor,           # [B, S, VOCAB]
    in_puzzle_mask: torch.Tensor,   # [B, S] (all-ones for Killer)
) -> torch.Tensor:
    state_v = state[..., :VOCAB]
    consistent = (state_v * orig_y).any(dim=-1)  # [B, S]
    return ~(consistent | ~in_puzzle_mask).all(dim=-1)


def _alpha_surviving(
    state: torch.Tensor,        # [B, S, C]
    solutions: torch.Tensor,    # [B, K, S, VOCAB]
    in_puzzle_mask: torch.Tensor,
    last_alpha: torch.Tensor | None = None,  # [B, S, VOCAB]
) -> torch.Tensor:
    sol_bool = solutions > 0.5
    state_v = state[..., :VOCAB] > 0.5
    consistent = (state_v.unsqueeze(1) & sol_bool).any(dim=-1)  # [B, K, S]
    consistent = consistent | ~in_puzzle_mask.unsqueeze(1)
    surviving = consistent.all(dim=-1)  # [B, K]
    any_surviving = surviving.any(dim=-1)
    alpha = (sol_bool & surviving.unsqueeze(-1).unsqueeze(-1)).any(dim=1)
    fallback = last_alpha if last_alpha is not None else sol_bool[:, 0].float()
    return torch.where(
        any_surviving.unsqueeze(-1).unsqueeze(-1), alpha.float(), fallback,
    )


def _given_mask(orig_x: torch.Tensor) -> torch.Tensor:
    """[B, S] bool: revealed-hint singletons in the vocab input."""
    return (orig_x.sum(dim=-1) == 1)


def _losses(
    out, state, orig_y, given_mask, is_sat, gt_conflict_target,
    bce_pos_mult, bce_neg_mult, softmax_w, conflict_w,
    in_puzzle_mask,
):
    """Losses over vocab channels only (aux channels get no gradient)."""
    B, S, C = state.shape
    device = state.device
    state_v = state[..., :VOCAB]
    orig_y_v = orig_y[..., :VOCAB]
    bce_target = state_v * orig_y_v
    pos_w = torch.full((B, 1, 1), bce_pos_mult, device=device)
    neg_w = torch.full((B, 1, 1), bce_neg_mult, device=device)
    ip_chan = in_puzzle_mask.unsqueeze(-1).float()
    multi_alive_target = orig_y_v.sum(dim=-1) > 1

    n_loops = len(out["bce"])
    total = torch.zeros((), device=device)
    for i in range(n_loops):
        bce_logits = out["bce"][i][..., :VOCAB]
        per_bit = weighted_bce_with_logits(
            bce_logits, bce_target, pos_w, neg_w, reduction="none",
        )
        total = total + (per_bit * ip_chan).mean()

        sm_logits = out["softmax"][i][..., :VOCAB]
        gt_idx = orig_y_v.argmax(dim=-1)
        ce_mask = ~given_mask & ~multi_alive_target & in_puzzle_mask & is_sat.unsqueeze(-1)
        if ce_mask.any():
            total = total + softmax_w * F.cross_entropy(
                sm_logits[ce_mask], gt_idx[ce_mask],
            )

        if "conflict" in out and conflict_w > 0:
            c_logits = out["conflict"][i].squeeze(-1)
            total = total + conflict_w * F.binary_cross_entropy_with_logits(
                c_logits, gt_conflict_target.float(),
            )
    return total / n_loops


def load_partial_checkpoint(ckpt_path: str, model: nn.Module) -> dict:
    """Partial init from a Sudoku-Extreme (C=9) checkpoint into a C=12 model.

    Returns {"n_reused": ..., "n_new": ..., "details": ...}.
    """
    ckpt = load_checkpoint(ckpt_path)
    src = ckpt["model_state_dict"]
    own = model.state_dict()
    n_reused = 0
    n_new = 0
    details: dict[str, str] = {}
    with torch.no_grad():
        for k, v in src.items():
            if k not in own:
                details[k] = "skipped (not in target)"
                continue
            if own[k].shape == v.shape:
                own[k].copy_(v)
                n_reused += v.numel()
                details[k] = f"reused {tuple(v.shape)}"
            elif k == "input_proj.weight" and v.shape == (own[k].shape[0], VOCAB):
                own[k][:, :VOCAB].copy_(v)
                nn.init.normal_(own[k][:, VOCAB:], std=0.02)
                n_reused += v.numel()
                n_new += own[k][:, VOCAB:].numel()
                details[k] = (f"partial: cols 0..8 reused {tuple(v.shape)}, "
                              f"cols 9..11 fresh Normal(0.02)")
            elif k in ("bce_head.weight", "softmax_head.weight") and \
                    v.shape == (VOCAB, own[k].shape[1]):
                # Output heads grow 9->12 rows; vocab-digit rows reused,
                # aux rows fresh (aux head outputs are inert: _losses slices
                # [:, :9] and dpll_step gates deduce/decide to vocab).
                own[k][:VOCAB].copy_(v)
                nn.init.normal_(own[k][VOCAB:], std=0.02)
                n_reused += v.numel()
                n_new += own[k][VOCAB:].numel()
                details[k] = (f"partial: rows 0..8 reused {tuple(v.shape)}, "
                              f"rows 9..11 fresh Normal(0.02)")
            elif k in ("bce_head.bias", "softmax_head.bias") and \
                    v.shape == (VOCAB,):
                own[k][:VOCAB].copy_(v)
                nn.init.zeros_(own[k][VOCAB:])
                n_reused += v.numel()
                n_new += own[k][VOCAB:].numel()
                details[k] = (f"partial: idx 0..8 reused {tuple(v.shape)}, "
                              f"idx 9..11 fresh zeros")
            else:
                details[k] = f"SHAPE MISMATCH src={tuple(v.shape)} dst={tuple(own[k].shape)}"
    # Fresh params not present in source (should be none beyond the 3 cols,
    # which were handled above) — verify.
    for k in own:
        if k not in src:
            n_new += own[k].numel()
            details[k] = f"fresh {tuple(own[k].shape)}"
    return {"n_reused": n_reused, "n_new": n_new, "details": details}


def _wandb_init(cfg: TrainConfig):
    """Init wandb iff WANDB_API_KEY is present AND wandb is installed.

    Returns the wandb module or None (training proceeds without tracking).
    """
    import os

    if not os.environ.get("WANDB_API_KEY"):
        print("wandb disabled (no WANDB_API_KEY env)", flush=True)
        return None
    try:
        import wandb
    except ImportError:
        print("wandb disabled (package not installed)", flush=True)
        return None
    wandb.init(
        project="ldt-killer",
        name=f"{cfg.name}",
        config={
            "steps": cfg.steps, "batch_size": cfg.batch_size, "lr": cfg.lr,
            "weight_decay": cfg.weight_decay, "seed": cfg.seed,
            "bce_pos_mult": cfg.bce_pos_mult, "bce_neg_mult": cfg.bce_neg_mult,
            "softmax_loss_weight": cfg.softmax_loss_weight,
            "conflict_loss_weight": cfg.conflict_loss_weight,
            "max_age": cfg.max_age, "init_from": cfg.init_from,
        },
    )
    print(f"wandb tracking: {wandb.run.url}", flush=True)
    return wandb


def train(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wb = _wandb_init(cfg)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    ckpt_path = out_dir / f"{cfg.name}_{ts}.pt"

    model = PowersetModel(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  params={n_params:,}  steps={cfg.steps}  bs={cfg.batch_size}",
          flush=True)

    if cfg.init_from:
        rep = load_partial_checkpoint(cfg.init_from, model)
        print(f"partial init from {cfg.init_from}: "
              f"reused={rep['n_reused']:,} new={rep['n_new']:,}", flush=True)
        for k, v in rep["details"].items():
            print(f"    {k}: {v}", flush=True)

    if cfg.compile and device.type == "cuda":
        print("torch.compile(dynamic=False) …", flush=True)
        model = torch.compile(model, dynamic=False)

    cfg.data.batch_size = cfg.batch_size
    dataset = KillerDataset(cfg.data)

    # In-train eval snapshot from the val split (zero-hint = test condition).
    eval_cfg = cfg.eval_data or KillerConfig(
        split="val", seed=cfg.data.seed, n_puzzles=cfg.eval_n_puzzles,
        batch_size=cfg.eval_n_puzzles,
        zero_hint_weight=1.0, correct_hint_weight=0.0, error_hint_weight=0.0,
    )
    eval_ds = KillerDataset(eval_cfg)
    ex_x, ex_y, ex_aux, ex_sat = eval_ds.next_batch()
    eval_ds.close()
    sat_mask = ex_sat.bool()
    eval_x = ex_x[sat_mask][:cfg.eval_n_puzzles].to(device).float()
    eval_y = ex_y[sat_mask][:cfg.eval_n_puzzles].to(device).float()
    eval_aux = ex_aux[sat_mask][:cfg.eval_n_puzzles].to(device).float()
    n_ev = eval_x.shape[0]
    eval_in_puzzle = torch.ones(n_ev, SEQ_LEN, dtype=torch.bool, device=device)
    eval_state = _build_state(eval_x, eval_aux)
    eval_given = _given_mask(eval_x)
    print(f"In-train eval set: {n_ev} puzzles "
          f"(every {cfg.eval_every} steps, "
          f"K={cfg.eval_n_chains}, max_rounds={cfg.eval_max_rounds})", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay, betas=cfg.betas)
    warmup_steps = max(1, int(cfg.steps * cfg.warmup_fraction))
    scheduler = make_cosine_scheduler(optimizer, cfg.steps, warmup_steps)

    ema: ModelEMA | None = None
    if cfg.use_ema:
        ema = ModelEMA(model, cfg.ema_decay)

    def fresh_batch(n: int):
        bx, by, ba = [], [], []
        while sum(t.shape[0] for t in bx) < n:
            x_b, y_b, a_b, _ = dataset.next_batch()
            bx.append(x_b)
            by.append(y_b)
            ba.append(a_b)
        x = torch.cat(bx, dim=0)[:n].to(device).float()
        y = torch.cat(by, dim=0)[:n].to(device).float()
        a = torch.cat(ba, dim=0)[:n].to(device).float()
        return x, y.unsqueeze(1), a  # solutions [n,1,S,9]

    pool_size = cfg.batch_size
    pool_orig_x, pool_solutions, pool_aux = fresh_batch(pool_size)
    pool_state = _build_state(pool_orig_x, pool_aux)
    pool_in_puzzle = torch.ones(pool_size, SEQ_LEN, dtype=torch.bool, device=device)
    pool_last_alpha = pool_solutions.bool().any(dim=1).float()
    pool_age = torch.zeros(pool_size, dtype=torch.long, device=device)
    print(f"Pool size: {pool_size}  augment={cfg.step.augment} "
          f"(dihedral-only; permute_digits={cfg.step.permute_digits})", flush=True)

    n_grid = GRID_ROWS
    cell_perms_train = build_dihedral_cell_perms(n=n_grid, device=device) \
        if cfg.step.augment else None
    # Digit-perm off for Killer (would break cage sums) -> identity digit perm.
    aug_vocab_dim = 0 if not cfg.step.permute_digits else VOCAB

    n_solved_total = 0
    n_tp_conflict_total = 0
    n_examples_seen = 0
    last_loss = float("nan")

    for step in range(1, cfg.steps + 1):
        sample_idx = torch.randperm(pool_size, device=device)[:cfg.batch_size]
        state = pool_state[sample_idx]
        solutions = pool_solutions[sample_idx]
        last_alpha = pool_last_alpha[sample_idx]
        orig_x = pool_orig_x[sample_idx]
        aux = pool_aux[sample_idx]
        in_puzzle_mask = pool_in_puzzle[sample_idx]
        age = pool_age[sample_idx]
        given_mask = _given_mask(orig_x)
        orig_y = _alpha_surviving(state, solutions, in_puzzle_mask, last_alpha)
        gt_conflict_pre = _gt_conflict(state, orig_y, in_puzzle_mask)
        is_sat_pre = ~gt_conflict_pre

        orig_y_padded = F.pad(orig_y, (0, N_CHANNELS - VOCAB))

        model.train()
        optimizer.zero_grad(set_to_none=True)
        out, aug_info = aug_forward(
            model, state, given_mask, orig_y=orig_y_padded,
            augment=cfg.step.augment,
            augment_dihedral=cfg.step.augment_dihedral,
            cell_perms=cell_perms_train,
            vocab_dim=aug_vocab_dim,
            return_all=True,
        )
        loss = _losses(
            out,
            aug_info["aug_state"], aug_info["aug_orig_y"], aug_info["aug_given_mask"],
            is_sat_pre, gt_conflict_pre,
            cfg.bce_pos_mult, cfg.bce_neg_mult,
            cfg.softmax_loss_weight, cfg.conflict_loss_weight,
            in_puzzle_mask,
        )
        assert torch.isfinite(loss).all(), f"non-finite loss at step {step}: {loss.item()}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        n_examples_seen += cfg.batch_size
        last_loss = float(loss.item())

        model.eval()
        want_info_stats = (step % cfg.log_every == 0)
        with torch.no_grad():
            new_state, detected_conflict, solved, _, info = dpll_step(
                model, state, given_mask, cfg.step, orig_y=orig_y_padded,
                in_puzzle_mask=in_puzzle_mask,
                want_stats=want_info_stats,
            )
        gt_conflict_post = _gt_conflict(new_state, orig_y, in_puzzle_mask)

        new_age = age + 1
        age_exceeded = new_age > cfg.max_age
        true_positive_conflict = detected_conflict & gt_conflict_post
        discard = solved | true_positive_conflict | age_exceeded

        n_solved_total += int(solved.sum().item())
        n_tp_conflict_total += int(true_positive_conflict.sum().item())

        n_to_replace = int(discard.sum().item())
        new_last_alpha = orig_y.clone()
        if n_to_replace > 0:
            fx, f_solutions, faux = fresh_batch(n_to_replace)
            new_state = new_state.clone()
            new_orig_x = orig_x.clone()
            new_solutions = solutions.clone()
            new_aux = aux.clone()
            new_age = new_age.clone()
            new_state[discard] = _build_state(fx, faux)
            new_orig_x[discard] = fx
            new_solutions[discard] = f_solutions
            new_aux[discard] = faux
            new_age[discard] = 0
            new_last_alpha[discard] = f_solutions.bool().any(dim=1).float()
        else:
            new_orig_x = orig_x
            new_solutions = solutions
            new_aux = aux

        pool_state[sample_idx] = new_state
        pool_orig_x[sample_idx] = new_orig_x
        pool_solutions[sample_idx] = new_solutions
        pool_aux[sample_idx] = new_aux
        pool_last_alpha[sample_idx] = new_last_alpha
        pool_age[sample_idx] = new_age

        if cfg.eval_every > 0 and step % cfg.eval_every == 0:
            model.eval()
            ema_backup = ema.swap_in(model) if ema is not None else None
            with torch.no_grad():
                eval_res = solve(
                    model, eval_state, eval_y, eval_given,
                    SolveConfig(
                        step=cfg.step, max_rounds=cfg.eval_max_rounds,
                        n_chains=cfg.eval_n_chains, batch_size=cfg.batch_size,
                    ),
                    in_puzzle_mask=eval_in_puzzle,
                    verbose=False,
                )
            if ema is not None and ema_backup is not None:
                ema.swap_out(model, ema_backup)
            n_e = eval_res.solved.shape[0]
            print(
                f"  [intrain-eval step={step}, max_rounds={cfg.eval_max_rounds}] "
                f"correct={int(eval_res.correct.sum())}/{n_e}  "
                f"wrong={int(eval_res.wrong.sum())}  to={int(eval_res.timeouts.sum())}  "
                f"calls={eval_res.model_calls}",
                flush=True,
            )

        if step % cfg.log_every == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"step={step:5d}/{cfg.steps}  loss={last_loss:.4f}  "
                f"lr={lr_now:.6f}",
                flush=True,
            )
            if wb is not None:
                wb.log({"step": step, "loss": last_loss, "lr": lr_now,
                        "n_solved_total": n_solved_total})

    extra = {
        "n_params": n_params, "n_examples_seen": n_examples_seen,
        "n_solved_total": n_solved_total, "n_tp_conflict_total": n_tp_conflict_total,
    }
    if ema is not None:
        extra["ema_state_dict"] = ema.state_dict()
        extra["ema_decay"] = cfg.ema_decay
    save_checkpoint(
        path=ckpt_path, model=model, model_cfg=cfg.model, train_cfg=cfg,
        extra=extra,
    )
    print(f"Saved: {ckpt_path}", flush=True)
    if wb is not None:
        wb.log({"final_loss": last_loss, "n_examples_seen": n_examples_seen})
        wb.finish()
    dataset.close()
    return ckpt_path
