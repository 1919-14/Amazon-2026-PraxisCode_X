"""Canonical artifact paths for the L3-L5 blocking chain.

These helpers are the single source of truth for artifact naming. Layers 3, 4 and
5 all resolve paths through this module so an artifact written by one layer is
always found by the next one (previously ``main_l4`` owned the naming and
``main_l5`` imported it from there).
"""

from __future__ import annotations

from pathlib import Path

from config import PATH_ARTIFACTS_DIR

SCRATCH_DIRNAME = "_scratch"


def blocking_dir() -> Path:
    """Directory holding every blocking artifact of every run."""
    return PATH_ARTIFACTS_DIR / "blocking"


def l3_artifact_path(split: str, refs: str, country: str) -> Path:
    """Per-channel candidate lists written by Layer 3 for one country bucket."""
    return blocking_dir() / f"l3_{split}_{refs}_country={country}.parquet"


def l4_artifact_path(split: str, refs: str, country: str) -> Path:
    """Fused candidate ranking written by Layer 4 for one country bucket."""
    return blocking_dir() / f"l4_{split}_{refs}_country={country}.parquet"


def l3_artifact_run_key(split: str, refs: str, country: str) -> str:
    """Run key used for L3 report history entries."""
    return f"{split}_{refs}_{country}"


def l4_artifact_run_key(split: str, refs: str, country: str) -> str:
    """Run key used for L4 report history entries."""
    return f"{split}_{refs}_{country}"


def blocking_scratch_dir(split: str, refs: str, country: str) -> Path:
    """Scratch directory for the block-wise L3 index of one country bucket."""
    return blocking_dir() / SCRATCH_DIRNAME / f"{split}_{refs}_country={country}"
