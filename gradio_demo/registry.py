"""Model registry for the Gradio demo.

Each entry maps a short model key to the checkpoint that should be loaded by
default, plus the storage location it can be fetched from.  Keeping this in one
place is what lets the UI "auto-fill" the checkpoint when the user switches
model instead of asking them to browse for a file every time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    family: str  # "ldt" | "hrm" | "trm" | "snowflake"
    local_path: str  # relative to the repository root
    accuracy: str
    source_note: str
    modal_profile: str
    modal_volume: str
    modal_path: str
    token_offset: int = 0
    act_max_steps: int = 16

    @property
    def checkpoint_path(self) -> Path:
        return (REPO_ROOT / self.local_path).resolve()

    @property
    def is_ldt(self) -> bool:
        return self.family == "ldt"

    @property
    def is_snowflake(self) -> bool:
        return self.family == "snowflake"


MODEL_SPECS: dict[str, ModelSpec] = {
    "ldt": ModelSpec(
        key="ldt",
        label="LDT — Scaled Sudoku (99,94%)",
        family="ldt",
        local_path="checkpoints/ldt_scaled_sudoku.pt",
        accuracy="99,94% test (10.233/10.239)",
        source_note="LDT huấn luyện: 4.000 bước × batch 512",
        modal_profile="huynhngoccung74",
        modal_volume="lattice-diffusion-checkpoints",
        modal_path="/scaled/seed0_4000s_bs512_20261003_131000.pt",
    ),
    "hrm": ModelSpec(
        key="hrm",
        label="HRM — Scaled Sudoku (34,50%)",
        family="hrm",
        local_path="checkpoints/hrm_scaled_sudoku.pt",
        accuracy="34,50% test (3.532/10.239)",
        source_note="HRM huấn luyện: 2.667 bước × batch 768",
        modal_profile="huynhngoccung74",
        modal_volume="lattice-diffusion-checkpoints",
        modal_path="/scaled_hrm/full_seed0_2667s_bs768_20261006_010536.pt",
        act_max_steps=16,
    ),
    "trm": ModelSpec(
        key="trm",
        label="TRM — Repo gốc (Sudoku-Extreme)",
        family="trm",
        local_path="checkpoints/trm_sudoku_extreme.pt",
        accuracy="~75–87% trên Sudoku-Extreme (bản attention, repo gốc)",
        source_note=(
            "Checkpoint của code gốc TinyRecursiveModels "
            "(SamsungSAILMontreal), variant attention, 60k epochs, step 52080"
        ),
        modal_profile="huynhnhatminh20102005",
        modal_volume="samsung-trm",
        modal_path="/checkpoints/attn_extreme_60k/step_52080",
        token_offset=1,
        act_max_steps=16,
    ),
    "snowflake": ModelSpec(
        key="snowflake",
        label="❄️ LDT — Snowflake Sudoku (100%)",
        family="snowflake",
        local_path="checkpoints/snowflake_sudoku.pt",
        accuracy="100% test (1.000/1.000, 0 sai, 0 timeout)",
        source_note="LDT huấn luyện: 1.000 bước × batch 512, dữ liệu CVC5",
        modal_profile="huynhnhatminh20102005",
        modal_volume="lattice-diffusion-checkpoints",
        modal_path=(
            "/snowflake/"
            "seed0_1000s_bs512_aug1_20261003_044719_20261003_044719.pt"
        ),
    ),
}

DEFAULT_MODEL_KEY = "ldt"

# Keys shown in the 9x9 Sudoku model dropdown.  Snowflake has its own tab
# because its board, input encoding (15x10 covering grid) and rendering are
# different, so it is not mixed into the Sudoku selector.
SUDOKU_MODEL_KEYS: tuple[str, ...] = ("ldt", "hrm", "trm")


def get_spec(key: str) -> ModelSpec:
    try:
        return MODEL_SPECS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown model key: {key!r}. Choose one of {list(MODEL_SPECS)}") from exc


def choice_labels(keys: tuple[str, ...] | None = None) -> list[tuple[str, str]]:
    """Return (label, key) pairs for a Gradio Dropdown.

    ``keys`` restricts the list (default: every registered model).
    """
    selected = MODEL_SPECS.keys() if keys is None else keys
    return [(MODEL_SPECS[key].label, key) for key in selected]

