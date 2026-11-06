"""Download the demo checkpoints from Modal.

Usage:
    python -m gradio_demo.fetch_checkpoints --all
    python -m gradio_demo.fetch_checkpoints --model ldt
    python -m gradio_demo.fetch_checkpoints --all --force

The files are stored in ``checkpoints/`` (gitignored).  Downloads use Modal's
Python client with concurrency 1: the Modal CLI (1.6.x) can corrupt large
files on Windows when it downloads blocks in parallel, while the
low-concurrency client path is reliable.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import zipfile
from pathlib import Path

# Ensure repository root is importable when run as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gradio_demo.registry import MODEL_SPECS, ModelSpec


def _zip_ok(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.testzip() is None
    except Exception:
        return False


def _activate_profile(profile: str) -> None:
    """Switch the Modal profile for the rest of this process.

    ``modal.config._set_profile`` changes the in-process default, but the SDK
    caches a single client built from whichever profile was used first.  Drop
    that cached client so the next call authenticates with the new profile.
    """
    import modal.config
    from modal.client import _Client

    modal.config._set_profile(profile)  # type: ignore[attr-defined]
    _Client._client_from_env = None


def _download_one(spec: ModelSpec, force: bool = False) -> Path:
    dest = spec.checkpoint_path
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        if _zip_ok(dest):
            print(f"[skip] {spec.key}: {dest} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
            return dest
        print(f"[redownload] {spec.key}: local file is corrupt, fetching again")

    import modal

    # Select the profile that owns the volume.  Modal's public CLI writes the
    # active profile to disk; the private helper only changes this process.
    _activate_profile(spec.modal_profile)
    vol = modal.Volume.from_name(spec.modal_volume)

    tmp = dest.with_suffix(dest.suffix + ".part")
    print(
        f"[fetch] {spec.key}: {spec.modal_profile}/{spec.modal_volume}{spec.modal_path} "
        f"-> {dest}"
    )

    original_cpu_count = multiprocessing.cpu_count
    multiprocessing.cpu_count = lambda: 1  # type: ignore[assignment]
    try:
        with open(tmp, "wb") as fh:
            total = vol.read_file_into_fileobj(spec.modal_path, fh)
        if total != tmp.stat().st_size:
            raise RuntimeError(f"short read: {total} != {tmp.stat().st_size}")
        if not _zip_ok(tmp):
            raise RuntimeError("downloaded file is not a valid torch zip archive")
        os.replace(tmp, dest)
    finally:
        multiprocessing.cpu_count = original_cpu_count  # type: ignore[assignment]
        if tmp.exists():
            tmp.unlink()

    print(f"[ok] {spec.key}: {dest} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch LDT/HRM/TRM/Snowflake demo checkpoints from Modal"
    )
    parser.add_argument(
        "--model",
        choices=[*MODEL_SPECS.keys(), "all"],
        default="all",
        help="Which checkpoint to fetch (default: all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Fetch every checkpoint (same as --model all)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if a valid file exists")
    args = parser.parse_args()

    keys = list(MODEL_SPECS) if (args.all or args.model == "all") else [args.model]
    failures = 0
    for key in keys:
        try:
            _download_one(MODEL_SPECS[key], force=args.force)
        except Exception as exc:
            failures += 1
            print(f"[fail] {key}: {exc}", file=sys.stderr)

    if failures:
        print(f"{failures} checkpoint(s) failed.", file=sys.stderr)
        return 1
    print("All requested checkpoints are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
