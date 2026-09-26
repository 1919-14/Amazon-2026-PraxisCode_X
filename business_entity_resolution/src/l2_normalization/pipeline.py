"""Streaming pipeline: process full TSV files in 100K-row chunks → parquet shards."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

# Allow running as a module from project root
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import PATH_ARTIFACTS_DIR, DATASET_DIR
from l2_normalization.engine import normalize_chunk

CHUNK_SIZE = 100_000


def process_source(
    tsv_path: Path,
    output_dir: Path,
    source_label: str,
) -> tuple[int, int]:
    """Stream-normalize one TSV source file to parquet shards.

    Returns (shard_count, total_rows).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_idx = 0
    total_rows = 0
    t0 = time.time()

    for chunk in pd.read_csv(
        tsv_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):
        normalized = normalize_chunk(chunk)
        shard_path = output_dir / f"part_{shard_idx:05d}.parquet"
        normalized.to_parquet(shard_path, compression="snappy", index=False)
        total_rows += len(normalized)
        shard_idx += 1

        if shard_idx % 10 == 0 or shard_idx == 1:
            elapsed = time.time() - t0
            print(
                f"  [{source_label}] shard {shard_idx:3d} | "
                f"rows {total_rows:>8,} | elapsed {elapsed:.1f}s"
            )

    elapsed = time.time() - t0
    print(
        f"  [{source_label}] DONE — {shard_idx} shards, "
        f"{total_rows:,} rows in {elapsed:.1f}s"
    )
    return shard_idx, total_rows


def process_split(split: str) -> dict[str, dict[str, int]]:
    """Normalize all three sources for the given split (train or test).

    Returns dict mapping source label → {shards, rows}.
    """
    print(f"\n{'=' * 60}")
    print(f"  Processing split: {split.upper()}")
    print(f"{'=' * 60}")

    split_dir = DATASET_DIR / split
    results: dict[str, dict[str, int]] = {}

    for i in (1, 2, 3):
        source_label = f"S{i}"
        prefix = "train" if split == "train" else "test"
        tsv_path = split_dir / f"{prefix}_source{i}.tsv"
        output_dir = PATH_ARTIFACTS_DIR / "normalized" / f"{split}_s{i}"

        if not tsv_path.exists():
            print(f"  ⚠️  {tsv_path} not found — skipping {source_label}")
            continue

        print(f"\n  → Normalizing {source_label}: {tsv_path.name}")
        shards, rows = process_source(tsv_path, output_dir, source_label)
        results[source_label] = {"shards": shards, "rows": rows}

    return results
