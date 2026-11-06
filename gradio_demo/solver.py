"""Inference wrappers for the models in the demo.

- LDT  : Lattice Deduction Transformers, solved with the DPLL loop from
         ``experiments.sudoku.dpll`` (9x9 Sudoku and Snowflake Sudoku).
- HRM  : Hierarchical Reasoning Model, loaded from the *unmodified* upstream
         code in ``upstream/HRM``.
- TRM  : Tiny Recursive Model, loaded from the *unmodified* upstream code in
         ``upstream/TinyRecursiveModels``.

The upstream packages both expose a top-level ``models`` package, so the
loader below temporarily switches ``sys.path`` and clears ``models*`` from
``sys.modules`` before importing one of them.  Loaded class objects keep
working after the switch because they hold references to their own globals.
"""

from __future__ import annotations

import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure repository root and src/ are in sys.path for internal imports
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
for p in [str(REPO_ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gradio_demo.registry import ModelSpec, get_spec
from gradio_demo.snowflake import (
    SnowflakeExample,
    decode_solution as decode_snowflake_solution,
    puzzle_to_tensors,
    validate_snowflake_solution,
)
from gradio_demo.utils import parse_sudoku_input, validate_sudoku_solution
from lattice_diffusion.models.looped_transformer import (
    LoopedTransformerConfig,
    PowersetModel,
)
from lattice_diffusion.training.utils.checkpoint import load_checkpoint
from experiments.snowflake.data import VOCAB as SNOWFLAKE_VOCAB
from experiments.sudoku.dpll import StepConfig, dpll_step
from experiments.sudoku.ema import swap_in_ema_if_present
from experiments.sudoku.solve import SolveConfig, solve

HRM_ROOT = REPO_ROOT / "upstream" / "HRM"
TRM_ROOT = REPO_ROOT / "upstream" / "TinyRecursiveModels"


@dataclass
class SolveOutput:
    status: str             # "SOLVED", "TIMEOUT", "INVALID_SOLUTION", "ERROR"
    solution_str: str       # 81-character string
    rounds: int             # DPLL rounds / ACT steps
    resets: int             # Chain resets (LDT) / 0 for ACT models
    model_calls: int        # Number of forward passes
    elapsed_sec: float      # Time taken in seconds
    givens_count: int       # Number of initial clues
    filled_count: int       # Total filled cells in result
    validation: dict[str, Any]  # Validation output from validate_sudoku_solution
    error_message: str | None = None


@dataclass
class SnowflakeSolveOutput:
    """Result of solving one Snowflake puzzle (variable number of cells)."""

    status: str             # "SOLVED", "TIMEOUT", "INVALID_SOLUTION", "ERROR"
    solution: list[int]     # per-cell digits; 0 means still ambiguous
    rounds: int
    resets: int
    model_calls: int
    elapsed_sec: float
    givens_count: int
    filled_count: int
    n_cells: int
    validation: dict[str, Any]
    error_message: str | None = None

    @property
    def solution_string(self) -> str:
        return "".join("." if d == 0 else str(d) for d in self.solution)


@dataclass
class LoadedModel:
    """A model plus everything the UI needs to describe/dispatch it."""

    family: str
    model: Any
    meta: dict[str, Any]
    device: torch.device
    spec: ModelSpec | None = None
    token_offset: int = 0
    act_max_steps: int = 16

    @property
    def is_ldt(self) -> bool:
        return self.family == "ldt"

    @property
    def is_snowflake(self) -> bool:
        return self.family == "snowflake"


def resolve_device(device_name: str = "auto") -> torch.device:
    """Resolve device string to torch.device."""
    if device_name.lower() in ("cuda", "gpu") and torch.cuda.is_available():
        return torch.device("cuda")
    if device_name.lower() == "cpu":
        return torch.device("cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# LDT
# ---------------------------------------------------------------------------
def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: str = "auto",
) -> tuple[PowersetModel, dict[str, Any]]:
    """Load a trained PowersetModel from a .pt checkpoint.

    Supports:
    - Standard checkpoint format saved by lattice-deduction-transformers
    - Models saved with or without EMA
    - Models saved with torch.compile (_orig_mod. prefix)
    - Fallback default configs if model_cfg is not embedded
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    dev = resolve_device(device)

    # Load checkpoint dict
    ckpt = load_checkpoint(ckpt_path, device="cpu")

    # Extract or infer model config
    raw_cfg = ckpt.get("model_cfg", {})
    if isinstance(raw_cfg, dict) and raw_cfg:
        model_cfg = LoopedTransformerConfig(**raw_cfg)
    else:
        # Fallback default configuration for Sudoku LDT
        model_cfg = LoopedTransformerConfig(
            cls_token=True,
            n_loops=16,
            pre_norm=True,
        )

    model = PowersetModel(model_cfg)

    # Load state dict
    state_dict = ckpt.get("model_state_dict", ckpt)
    # Strip any compile prefixes
    clean_state = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(clean_state, strict=False)

    # Swap in EMA weights if present
    has_ema = bool(ckpt.get("ema_state_dict"))
    if has_ema:
        swap_in_ema_if_present(model, ckpt)

    model.to(dev).eval()

    total_params = sum(p.numel() for p in model.parameters())
    meta = {
        "family": "ldt",
        "kind_label": "LDT (Lattice Deduction Transformer)",
        "checkpoint_path": str(ckpt_path.resolve()),
        "checkpoint_size_mb": round(ckpt_path.stat().st_size / (1024 * 1024), 2),
        "device": str(dev),
        "total_params": total_params,
        "n_loops": model_cfg.n_loops,
        "dim": model_cfg.dim,
        "num_layers": model_cfg.num_layers,
        "n_heads": model_cfg.n_heads,
        "cls_token": model_cfg.cls_token,
        "has_ema": has_ema,
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "is_scratch": False,
    }
    return model, meta


def load_snowflake_from_checkpoint(
    checkpoint_path: str | Path,
    device: str = "auto",
) -> tuple[PowersetModel, dict[str, Any]]:
    """Load the LDT checkpoint trained on Snowflake Sudoku.

    The architecture is the same ``PowersetModel``; only the data layout
    differs (15x10 covering grid, 6 digits, 7 input channels).  The checkpoint
    stores its own ``model_cfg``, so the generic loader already rebuilds it.
    """
    model, meta = load_model_from_checkpoint(checkpoint_path, device=device)
    meta["family"] = "snowflake"
    meta["kind_label"] = "LDT — Snowflake Sudoku"
    meta["config_summary"] = (
        f"{meta.get('num_layers')} layers, dim={meta.get('dim')}, "
        f"{meta.get('n_heads')} heads, {meta.get('n_loops')} loops · "
        "lưới phủ 15×10, vocab 6, n_channels 7"
    )
    return model, meta


def create_scratch_model(device: str = "auto") -> tuple[PowersetModel, dict[str, Any]]:
    """Create a randomly initialized LDT model for testing when no checkpoint is available."""
    dev = resolve_device(device)
    model_cfg = LoopedTransformerConfig(
        cls_token=True,
        n_loops=8,
        dim=128,
        num_layers=4,
        pre_norm=True,
    )
    model = PowersetModel(model_cfg)
    model.to(dev).eval()

    total_params = sum(p.numel() for p in model.parameters())
    meta = {
        "family": "ldt",
        "kind_label": "LDT (Scratch / Untrained)",
        "checkpoint_path": "Demo Scratch Model (Untrained Random Weights)",
        "checkpoint_size_mb": 0.0,
        "device": str(dev),
        "total_params": total_params,
        "n_loops": model_cfg.n_loops,
        "dim": model_cfg.dim,
        "num_layers": model_cfg.num_layers,
        "n_heads": model_cfg.n_heads,
        "cls_token": model_cfg.cls_token,
        "has_ema": False,
        "missing_keys": 0,
        "unexpected_keys": 0,
        "is_scratch": True,
    }
    return model, meta


# ---------------------------------------------------------------------------
# Upstream HRM / TRM helpers
# ---------------------------------------------------------------------------
def _install_flash_stub() -> None:
    """SDPA stand-in for flash_attn (no Blackwell wheels on B200).

    HRM's ``models.layers`` imports flash_attn unconditionally.  This stub
    keeps the upstream file untouched while providing the same math.
    """
    if "flash_attn_interface" in sys.modules and "flash_attn" in sys.modules:
        return

    def _sdpa_flash(q, k, v, causal=False, **kwargs):
        q_ = q.transpose(1, 2)
        k_ = k.transpose(1, 2)
        v_ = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q_, k_, v_, is_causal=causal)
        return out.transpose(1, 2).contiguous()

    for name in ("flash_attn_interface", "flash_attn"):
        mod = types.ModuleType(name)
        mod.flash_attn_func = _sdpa_flash  # type: ignore[attr-defined]
        sys.modules.setdefault(name, mod)


def _clear_models_modules() -> None:
    for name in list(sys.modules):
        if name == "models" or name.startswith("models."):
            del sys.modules[name]


def _import_upstream(family: str):
    """Import the model/loss classes from one upstream package at a time."""
    _clear_models_modules()
    if family == "hrm":
        _install_flash_stub()
        root = HRM_ROOT
    elif family == "trm":
        root = TRM_ROOT
    else:
        raise ValueError(f"Unknown upstream family: {family}")

    root_str = str(root)
    sys.path.insert(0, root_str)
    try:
        if family == "hrm":
            from models.hrm.hrm_act_v1 import (  # type: ignore
                HierarchicalReasoningModel_ACTV1,
            )
            from models.losses import ACTLossHead  # type: ignore

            return HierarchicalReasoningModel_ACTV1, ACTLossHead
        from models.losses import ACTLossHead  # type: ignore
        from models.recursive_reasoning.trm import (  # type: ignore
            TinyRecursiveReasoningModel_ACTV1,
        )

        return TinyRecursiveReasoningModel_ACTV1, ACTLossHead
    finally:
        while root_str in sys.path:
            sys.path.remove(root_str)


def _state_dict_and_cfg(ckpt: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"], dict(ckpt.get("cfg") or {})
    if isinstance(ckpt, dict):
        return ckpt, {}
    raise ValueError("Unsupported checkpoint format: expected a state_dict or a dict with 'model_state_dict'.")


def load_hrm_from_checkpoint(
    checkpoint_path: str | Path,
    device: str = "auto",
) -> tuple[Any, dict[str, Any]]:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict, cfg = _state_dict_and_cfg(ckpt)
    if not cfg:
        raise ValueError(
            "HRM checkpoint is missing the embedded 'cfg' dict; cannot rebuild the architecture."
        )

    cfg = dict(cfg)
    # The eval batch is dynamic; the value only sizes a non-persistent buffer.
    cfg["batch_size"] = 1

    model_cls, loss_cls = _import_upstream("hrm")
    model = loss_cls(model_cls(cfg), loss_type="stablemax_cross_entropy")

    clean_state = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(clean_state, strict=False)

    dev = resolve_device(device)
    model.to(dev).eval()

    total_params = sum(p.numel() for p in model.parameters())
    meta = {
        "family": "hrm",
        "kind_label": "HRM (Hierarchical Reasoning Model)",
        "checkpoint_path": str(ckpt_path.resolve()),
        "checkpoint_size_mb": round(ckpt_path.stat().st_size / (1024 * 1024), 2),
        "device": str(dev),
        "total_params": total_params,
        "n_loops": cfg.get("H_cycles"),
        "num_layers": cfg.get("H_layers", 0) + cfg.get("L_layers", 0),
        "dim": cfg.get("hidden_size"),
        "n_heads": cfg.get("num_heads"),
        "has_ema": False,  # HRM upstream has no EMA
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "is_scratch": False,
        "config_summary": (
            f"H_cycles={cfg.get('H_cycles')}, L_cycles={cfg.get('L_cycles')}, "
            f"H_layers={cfg.get('H_layers')}, L_layers={cfg.get('L_layers')}, "
            f"hidden={cfg.get('hidden_size')}, halt_max_steps={cfg.get('halt_max_steps')}"
        ),
        "act_max_steps": int(cfg.get("halt_max_steps", 16)),
        "token_offset": 0,
        "cfg": cfg,
    }
    return model, meta


def load_trm_from_checkpoint(
    checkpoint_path: str | Path,
    device: str = "auto",
) -> tuple[Any, dict[str, Any]]:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict, stored_cfg = _state_dict_and_cfg(raw)
    clean_state = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    embed_key = next(
        (k for k in clean_state if k.endswith("inner.embed_tokens.embedding_weight")), None
    )
    puzzle_key = next(
        (k for k in clean_state if k.endswith("inner.puzzle_emb.weights")), None
    )
    vocab_size = (
        int(clean_state[embed_key].shape[0])
        if embed_key is not None
        else int(stored_cfg.get("vocab_size", 11))
    )
    num_ids = (
        int(clean_state[puzzle_key].shape[0])
        if puzzle_key is not None
        else int(stored_cfg.get("num_puzzle_identifiers", 1))
    )
    puzzle_emb_ndim = (
        int(clean_state[puzzle_key].shape[1])
        if puzzle_key is not None
        else int(stored_cfg.get("puzzle_emb_ndim", 512))
    )

    mlp_t = stored_cfg.get("mlp_t")
    if mlp_t is None:
        mlp_t = not any("self_attn" in k for k in clean_state)

    cfg = dict(
        batch_size=1,
        seq_len=int(stored_cfg.get("seq_len", 81)),
        puzzle_emb_ndim=puzzle_emb_ndim,
        num_puzzle_identifiers=num_ids,
        vocab_size=vocab_size,
        H_cycles=int(stored_cfg.get("H_cycles", 3)),
        L_cycles=int(stored_cfg.get("L_cycles", 6)),
        H_layers=int(stored_cfg.get("H_layers", 0)),
        L_layers=int(stored_cfg.get("L_layers", 2)),
        hidden_size=int(stored_cfg.get("hidden_size", 512)),
        expansion=float(stored_cfg.get("expansion", 4.0)),
        num_heads=int(stored_cfg.get("num_heads", 8)),
        pos_encodings=stored_cfg.get("pos_encodings", "rope"),
        halt_max_steps=int(stored_cfg.get("halt_max_steps", 16)),
        halt_exploration_prob=float(stored_cfg.get("halt_exploration_prob", 0.1)),
        forward_dtype=stored_cfg.get("forward_dtype", "bfloat16"),
        mlp_t=bool(mlp_t),
        puzzle_emb_len=int(stored_cfg.get("puzzle_emb_len", 16)),
        no_ACT_continue=bool(stored_cfg.get("no_ACT_continue", True)),
    )

    model_cls, loss_cls = _import_upstream("trm")
    model = loss_cls(model_cls(cfg), loss_type="stablemax_cross_entropy")
    missing, unexpected = model.load_state_dict(clean_state, strict=False)

    dev = resolve_device(device)
    model.to(dev).eval()

    # Official Sudoku-Extreme checkpoints use vocab 11 and shift tokens by +1
    # (0 = PAD, 1 = blank, 2..10 = digits).  Checkpoints trained by this repo
    # on scaled sudoku use vocab 10 and no shift.
    token_offset = 1 if vocab_size >= 11 else 0
    total_params = sum(p.numel() for p in model.parameters())
    meta = {
        "family": "trm",
        "kind_label": "TRM (Tiny Recursive Model)",
        "checkpoint_path": str(ckpt_path.resolve()),
        "checkpoint_size_mb": round(ckpt_path.stat().st_size / (1024 * 1024), 2),
        "device": str(dev),
        "total_params": total_params,
        "n_loops": cfg.get("L_cycles"),
        "num_layers": cfg.get("L_layers"),
        "dim": cfg.get("hidden_size"),
        "n_heads": cfg.get("num_heads"),
        "has_ema": False,
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "is_scratch": False,
        "config_summary": (
            f"{'MLP' if cfg.get('mlp_t') else 'Attention'} TRM, "
            f"H_cycles={cfg.get('H_cycles')}, L_cycles={cfg.get('L_cycles')}, "
            f"L_layers={cfg.get('L_layers')}, hidden={cfg.get('hidden_size')}, "
            f"halt_max_steps={cfg.get('halt_max_steps')}"
        ),
        "act_max_steps": int(cfg.get("halt_max_steps", 16)),
        "token_offset": token_offset,
        "cfg": cfg,
    }
    return model, meta


# ---------------------------------------------------------------------------
# Unified loading / dispatching
# ---------------------------------------------------------------------------
def load_model(
    model_key: str,
    checkpoint_path: str | Path | None = None,
    device: str = "auto",
) -> LoadedModel:
    """Load one of the registered models, using its default checkpoint by default."""
    spec = get_spec(model_key)
    target = Path(checkpoint_path) if checkpoint_path else spec.checkpoint_path

    if spec.family == "ldt":
        model, meta = load_model_from_checkpoint(target, device=device)
        meta["model_key"] = spec.key
        meta["source_note"] = spec.source_note
        meta["accuracy"] = spec.accuracy
        dev = next(model.parameters()).device
        return LoadedModel(
            family="ldt",
            model=model,
            meta=meta,
            device=dev,
            spec=spec,
            token_offset=0,
            act_max_steps=16,
        )

    if spec.family == "snowflake":
        model, meta = load_snowflake_from_checkpoint(target, device=device)
        meta["model_key"] = spec.key
        meta["source_note"] = spec.source_note
        meta["accuracy"] = spec.accuracy
        return LoadedModel(
            family="snowflake",
            model=model,
            meta=meta,
            device=next(model.parameters()).device,
            spec=spec,
            token_offset=0,
            act_max_steps=16,
        )

    if spec.family == "hrm":
        model, meta = load_hrm_from_checkpoint(target, device=device)
    elif spec.family == "trm":
        model, meta = load_trm_from_checkpoint(target, device=device)
    else:
        raise ValueError(f"Unsupported model family: {spec.family}")

    meta["model_key"] = spec.key
    meta["source_note"] = spec.source_note
    meta["accuracy"] = spec.accuracy
    return LoadedModel(
        family=spec.family,
        model=model,
        meta=meta,
        device=next(model.parameters()).device,
        spec=spec,
        token_offset=int(meta.get("token_offset", 0)),
        act_max_steps=int(meta.get("act_max_steps", 16)),
    )


def make_scratch_loaded_model(device: str = "auto") -> LoadedModel:
    model, meta = create_scratch_model(device=device)
    spec = get_spec("ldt")
    return LoadedModel(
        family="ldt",
        model=model,
        meta=meta,
        device=next(model.parameters()).device,
        spec=spec,
        token_offset=0,
        act_max_steps=16,
    )


# ---------------------------------------------------------------------------
# LDT inference
# ---------------------------------------------------------------------------
def encode_puzzle_to_tensor(puzzle_str: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode an 81-character puzzle string to:

    - puzzle_tensor: [1, 81, 9] float32 powerset lattice representation:
        - givens: one-hot at index (digit - 1)
        - blanks: all 9 digits alive (vector of 1.0)
    - given_mask: [1, 81] bool mask (True where clue is given)
    """
    puzzle_clean = parse_sudoku_input(puzzle_str)
    x = np.zeros((1, 81, 9), dtype=np.float32)
    given_mask = np.zeros((1, 81), dtype=bool)

    for i, c in enumerate(puzzle_clean):
        if c in "123456789":
            digit_idx = int(c) - 1
            x[0, i, digit_idx] = 1.0
            given_mask[0, i] = True
        else:
            x[0, i, :] = 1.0  # all 9 candidates alive
            given_mask[0, i] = False

    return torch.from_numpy(x), torch.from_numpy(given_mask)


def decode_state_to_puzzle(state_tensor: torch.Tensor) -> str:
    """Decode a [81, 9] state tensor back to an 81-character string."""
    chars = []
    alive_counts = (state_tensor > 0.5).sum(dim=-1)
    argmax_digits = state_tensor.argmax(dim=-1) + 1

    for i in range(81):
        if alive_counts[i] == 1:
            chars.append(str(int(argmax_digits[i].item())))
        else:
            chars.append(".")
    return "".join(chars)


def decode_state_argmax(state_tensor: torch.Tensor) -> str:
    """Decode a [81, 9] state tensor taking argmax for all cells (force full grid)."""
    argmax_digits = state_tensor.argmax(dim=-1) + 1
    return "".join(str(int(argmax_digits[i].item())) for i in range(81))


def solve_single_puzzle(
    model: PowersetModel,
    puzzle_str: str,
    *,
    n_chains: int = 32,
    max_rounds: int = 500,
    threshold: float = 0.10,
    temp_decide: float = 1.5,
    cls_threshold: float = 0.5,
    augment: bool = True,
    device: torch.device | None = None,
) -> SolveOutput:
    """Solve a single Sudoku puzzle using LDT's DPLL stochastic chains."""
    if device is None:
        device = next(model.parameters()).device

    clean_puzzle = parse_sudoku_input(puzzle_str)
    givens_count = sum(1 for c in clean_puzzle if c in "123456789")

    puz_tensor, gm_tensor = encode_puzzle_to_tensor(clean_puzzle)

    K = max(1, int(n_chains))
    state = puz_tensor.repeat(K, 1, 1).to(device)
    original = state.clone()
    given_mask = gm_tensor.repeat(K, 1).to(device)
    chain_done = torch.zeros(K, dtype=torch.bool, device=device)

    step_cfg = StepConfig(
        threshold=threshold,
        cls_threshold=cls_threshold,
        temp_decide=temp_decide,
        augment=augment,
    )

    total_resets = 0
    model_calls = 0
    winning_state: torch.Tensor | None = None
    solved_round = -1
    status = "TIMEOUT"

    start_time = time.perf_counter()

    for r in range(max_rounds):
        with torch.no_grad():
            new_state, conflict, just_solved_chain, _, info = dpll_step(
                model, state, given_mask, step_cfg, want_stats=False
            )
        model_calls += 1

        active_solved = just_solved_chain & ~chain_done
        if active_solved.any():
            win_idx = int(active_solved.nonzero(as_tuple=True)[0][0].item())
            winning_state = new_state[win_idx].clone()
            solved_round = r + 1
            status = "SOLVED"
            break

        active_conflict = conflict & ~chain_done
        if active_conflict.any():
            total_resets += int(active_conflict.sum().item())
            new_state[active_conflict] = original[active_conflict]

        state = new_state

    elapsed = round(time.perf_counter() - start_time, 4)

    if winning_state is not None:
        sol_str = decode_state_to_puzzle(winning_state)
        if "." in sol_str:
            sol_str = decode_state_argmax(winning_state)
    else:
        alive_per_chain = (state > 0.5).sum(dim=-1)
        singletons_per_chain = (alive_per_chain == 1).sum(dim=-1)
        best_k = int(singletons_per_chain.argmax().item())
        sol_str = decode_state_to_puzzle(state[best_k])

    validation = validate_sudoku_solution(sol_str, clean_puzzle)
    if status == "SOLVED" and not validation["is_valid"]:
        status = "INVALID_SOLUTION"

    filled_count = sum(1 for c in sol_str if c in "123456789")

    return SolveOutput(
        status=status,
        solution_str=sol_str,
        rounds=solved_round if solved_round > 0 else max_rounds,
        resets=total_resets,
        model_calls=model_calls,
        elapsed_sec=elapsed,
        givens_count=givens_count,
        filled_count=filled_count,
        validation=validation,
    )


# ---------------------------------------------------------------------------
# Snowflake inference (LDT on the 15x10 covering grid)
# ---------------------------------------------------------------------------
def solve_snowflake(
    model: PowersetModel,
    example: SnowflakeExample,
    *,
    puzzle: list[int] | None = None,
    n_chains: int = 64,
    max_rounds: int = 1000,
    threshold: float = 0.10,
    temp_decide: float = 1.5,
    cls_threshold: float = 0.6,
    augment: bool = True,
    device: torch.device | None = None,
) -> SnowflakeSolveOutput:
    """Solve one Snowflake puzzle with LDT's DPLL loop.

    ``puzzle`` optionally overrides the bundled givens (same length as
    ``example.n_cells``; blanks are 7).
    """
    if device is None:
        device = next(model.parameters()).device

    record = example.as_record(puzzle)
    givens_count = sum(1 for d in record["puzzle"] if d != 7)

    state, ground_truth, given_mask, in_puzzle_mask = puzzle_to_tensors(
        record, device=device
    )

    step_cfg = StepConfig(
        threshold=threshold,
        cls_threshold=cls_threshold,
        temp_decide=temp_decide,
        augment=augment,
        # The covering grid is hexagonal, so D4 dihedral augmentation would
        # break the topology; only digit permutations are used.
        augment_dihedral=False,
        vocab_dim=SNOWFLAKE_VOCAB,
    )
    K = max(1, int(n_chains))
    solve_cfg = SolveConfig(
        step=step_cfg,
        max_rounds=int(max_rounds),
        n_chains=K,
        batch_size=K,
    )

    start_time = time.perf_counter()
    with torch.no_grad():
        result = solve(
            model,
            state,
            ground_truth,
            given_mask,
            solve_cfg,
            in_puzzle_mask=in_puzzle_mask,
            verbose=False,
        )
    elapsed = round(time.perf_counter() - start_time, 4)

    solution = decode_snowflake_solution(record, result.solution[0])
    validation = validate_snowflake_solution(record, solution)

    solved = bool(result.solved[0].item())
    if validation["is_valid"]:
        status = "SOLVED"
    elif solved:
        status = "INVALID_SOLUTION"
    else:
        status = "TIMEOUT"

    rounds = int(result.round_solved[0].item()) if solved else int(max_rounds)
    if rounds < 0:
        rounds = int(max_rounds)
    resets = int(result.n_resets[0].item())
    calls = int(result.puzzle_calls[0].item())
    if calls < 0:
        calls = int(result.model_calls)
    filled_count = sum(1 for d in solution if 1 <= d <= 6)

    return SnowflakeSolveOutput(
        status=status,
        solution=solution,
        rounds=rounds,
        resets=resets,
        model_calls=calls,
        elapsed_sec=elapsed,
        givens_count=givens_count,
        filled_count=filled_count,
        n_cells=example.n_cells,
        validation=validation,
    )


# ---------------------------------------------------------------------------
# HRM / TRM inference (ACT loop, batched over augmented views)
# ---------------------------------------------------------------------------
def _aug_params(rng: np.random.Generator) -> dict[str, Any]:
    """Digit permutation + band/stack permutations + optional transpose."""
    digit = rng.permutation(9)
    bands = rng.permutation(3)
    row_perm = np.concatenate([b * 3 + rng.permutation(3) for b in bands])
    stacks = rng.permutation(3)
    col_perm = np.concatenate([s * 3 + rng.permutation(3) for s in stacks])
    transpose = bool(rng.random() < 0.5)
    return {"digit": digit, "row": row_perm, "col": col_perm, "transpose": transpose}


def _apply_aug(raw81: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    g = np.asarray(raw81).reshape(9, 9)
    if p["transpose"]:
        g = g.T.copy()
    g = g[p["row"]][:, p["col"]]
    out = g.reshape(81).copy()
    m = out > 0
    out[m] = p["digit"][out[m] - 1] + 1
    return out


def _invert_aug(raw81: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    inv_digit = np.empty(9, dtype=np.int64)
    inv_digit[p["digit"]] = np.arange(9)
    g = np.asarray(raw81).reshape(9, 9).copy()
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


def _solve_act_raw(
    model: Any,
    raw_inputs: np.ndarray,
    device: torch.device,
    *,
    token_offset: int,
    max_steps: int,
) -> tuple[np.ndarray, int]:
    """Run the ACT loop for a batch of raw 0..9 grids.

    Returns (predicted raw 0..9 digits [B, 81], steps executed).
    """
    inputs = torch.from_numpy(np.asarray(raw_inputs, dtype=np.int64)).to(device)
    if token_offset:
        inputs = inputs + token_offset

    batch = {
        "inputs": inputs,
        "labels": torch.zeros_like(inputs),
        "puzzle_identifiers": torch.zeros(inputs.shape[0], dtype=torch.long, device=device),
    }

    steps = 0
    with torch.no_grad():
        with torch.device(device):
            carry = model.initial_carry(batch)
        while True:
            carry, _, _, out, done = model(
                carry=carry, batch=batch, return_keys=["logits"]
            )
            steps += 1
            if bool(done) or steps >= max_steps:
                break
        preds = out["logits"].argmax(-1).cpu().numpy()

    if token_offset:
        preds = preds - token_offset
    return preds, steps


def solve_with_act_model(
    loaded: LoadedModel,
    puzzle_str: str,
    *,
    n_views: int = 8,
    max_steps: int | None = None,
    augment: bool = True,
    seed: int = 777,
) -> SolveOutput:
    """Solve with HRM/TRM; ``n_views`` augmented views are combined by majority vote."""
    clean_puzzle = parse_sudoku_input(puzzle_str)
    givens_count = sum(1 for c in clean_puzzle if c in "123456789")
    raw = np.array([0 if c == "." else int(c) for c in clean_puzzle], dtype=np.int64)

    views = max(1, min(int(n_views), 16))
    steps_limit = int(max_steps or loaded.act_max_steps)
    rng = np.random.default_rng(seed)

    if views == 1 or not augment:
        views = 1
        batch = raw[None, :]
        params: list[dict[str, Any]] = []
    else:
        params = [_aug_params(rng) for _ in range(views - 1)]
        batch = np.stack([raw] + [_apply_aug(raw, p) for p in params])

    start_time = time.perf_counter()
    preds, steps = _solve_act_raw(
        loaded.model,
        batch,
        loaded.device,
        token_offset=loaded.token_offset,
        max_steps=steps_limit,
    )
    elapsed = round(time.perf_counter() - start_time, 4)

    if params:
        out_views = [preds[0]] + [
            _invert_aug(preds[i + 1], p) for i, p in enumerate(params)
        ]
        stacked = np.stack(out_views)
        cand, counts = np.unique(stacked, axis=0, return_counts=True)
        best = cand[int(np.argmax(counts))]
    else:
        best = preds[0]

    sol_str = "".join(str(int(d)) if 1 <= int(d) <= 9 else "." for d in best)
    validation = validate_sudoku_solution(sol_str, clean_puzzle)
    if validation["is_valid"]:
        status = "SOLVED"
    elif "." not in sol_str:
        status = "INVALID_SOLUTION"
    else:
        status = "TIMEOUT"

    filled_count = sum(1 for c in sol_str if c in "123456789")
    return SolveOutput(
        status=status,
        solution_str=sol_str,
        rounds=steps,
        resets=0,
        model_calls=views * steps,
        elapsed_sec=elapsed,
        givens_count=givens_count,
        filled_count=filled_count,
        validation=validation,
    )


def solve_for_loaded_model(
    loaded: LoadedModel,
    puzzle_str: str,
    *,
    n_chains: int = 32,
    max_rounds: int = 500,
    threshold: float = 0.10,
    temp_decide: float = 1.5,
    augment: bool = True,
    act_steps: int | None = None,
) -> SolveOutput:
    """Dispatch to the LDT DPLL solver or the HRM/TRM ACT loop."""
    if loaded.is_ldt:
        return solve_single_puzzle(
            loaded.model,
            puzzle_str,
            n_chains=n_chains,
            max_rounds=max_rounds,
            threshold=threshold,
            temp_decide=temp_decide,
            augment=augment,
            device=loaded.device,
        )
    return solve_with_act_model(
        loaded,
        puzzle_str,
        n_views=n_chains,
        max_steps=act_steps or loaded.act_max_steps,
        augment=augment,
    )
