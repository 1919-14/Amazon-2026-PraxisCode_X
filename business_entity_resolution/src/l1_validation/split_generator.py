"""Layer 1: Deterministic train/validation split generator and ID loader."""

import json
import random
import sys
from pathlib import Path
from typing import Any

# Ensure src/ is in sys.path
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
from config import PATH_ARTIFACTS_DIR, PATH_TRAIN_S1


def generate_split(seed: int = 42, val_fraction: float = 0.2) -> dict[str, Any]:
    """Generate a deterministic 80/20 grouped split of Source 1 entity IDs and save to JSON."""
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    splits_dir = PATH_ARTIFACTS_DIR / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_path = splits_dir / "train_ids.json"
    val_path = splits_dir / "val_ids.json"

    # Memory-safe chunked reading: concatenate only the string IDs
    id_list: list[str] = []
    for chunk in pd.read_csv(
        PATH_TRAIN_S1,
        sep="\t",
        usecols=["entity_id"],
        dtype=str,
        keep_default_na=False,
        chunksize=500_000,
    ):
        id_list.extend(chunk["entity_id"].tolist())

    total_count = len(id_list)
    if total_count == 0:
        raise ValueError(f"No records found in {PATH_TRAIN_S1}")

    # Deterministic sort then reproducible shuffle
    id_list.sort()
    rng = random.Random(seed)
    rng.shuffle(id_list)

    # Grouped split: 80% train, 20% validation
    val_count = int(total_count * val_fraction)
    train_count = total_count - val_count

    train_ids = id_list[:train_count]
    val_ids = id_list[train_count:]
    actual_val_fraction = val_count / total_count

    # Save IDs to JSON
    with train_path.open("w", encoding="utf-8") as f:
        json.dump(train_ids, f)

    with val_path.open("w", encoding="utf-8") as f:
        json.dump(val_ids, f)

    print(f"Total S1 entities: {total_count:,}")
    print(f"Train S1 count   : {train_count:,} ({(1 - actual_val_fraction) * 100:.2f}%)")
    print(f"Val S1 count     : {val_count:,} ({actual_val_fraction * 100:.2f}%)")
    print(f"Actual val fraction: {actual_val_fraction:.6f}")
    print(f"Saved: {train_path} & {val_path}")

    return {
        "total_count": total_count,
        "train_count": train_count,
        "val_count": val_count,
        "val_fraction": actual_val_fraction,
        "train_path": str(train_path),
        "val_path": str(val_path),
    }


def load_split_ids() -> dict[str, set[str]]:
    """Load pre-generated train and validation entity ID sets from JSON artifacts."""
    splits_dir = PATH_ARTIFACTS_DIR / "splits"
    train_path = splits_dir / "train_ids.json"
    val_path = splits_dir / "val_ids.json"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Split files not found in {splits_dir}. "
            "Please run generate_split() first to produce train_ids.json and val_ids.json."
        )

    with train_path.open("r", encoding="utf-8") as f:
        train_ids = set(json.load(f))

    with val_path.open("r", encoding="utf-8") as f:
        val_ids = set(json.load(f))

    return {"train_ids": train_ids, "val_ids": val_ids}
