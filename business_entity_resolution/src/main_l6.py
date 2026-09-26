"""Layer 6 entry point: training pair construction & hard-negative A/B datasets.

Consumes the L5 candidate set for the *training* reference entities, builds
positive / hard-negative / easy-negative pairs, and writes two datasets for the
L8 A/B test:

    artifacts/train_pairs/variant_a.parquet   (positives + hard + easy)
    artifacts/train_pairs/variant_b.parquet   (positives + easy)

The model-based A/B comparison (train two models, compare validation F0.5) is a
L7+L8 activity because it needs pairwise features and a matcher; this layer
produces the two datasets and the sampling diagnostics that feed it.

Usage
-----
# Build pairs from a train-split candidate set:
python business_entity_resolution/src/main_l6.py --candidates output/candidate_pairs_train.tsv

# Small dev run:
python business_entity_resolution/src/main_l6.py --candidates output/candidate_pairs_val.tsv --max-references 5000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
for _p in (str(SRC_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import (
    L6_EASY_POOL_SIZE,
    L6_MAX_REFERENCES,
    L6_SEED,
    L3_COUNTRIES,
    PATH_ARTIFACTS_DIR,
    PATH_OUTPUT_DIR,
)
from l3_l5_blocking.buckets import assign_country, iter_source_shards
from l6_l8_matching.pairs import (
    EASY,
    HARD,
    POSITIVE,
    SamplingConfig,
    build_easy_pool,
    iter_candidate_pairs,
    sample_reference_pairs,
    variant_b_rows,
)

# Reuse the L3 ground-truth loader so chunking/filtering stays consistent.
from main_l3 import load_ground_truth

PAIR_SCHEMA = pa.schema(
    [
        ("s1_id", pa.string()),
        ("cand_id", pa.string()),
        ("label", pa.int8()),
        ("neg_type", pa.string()),
        ("country", pa.string()),
        ("rank", pa.int32()),
    ]
)

WRITE_CHUNK = 500_000


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Layer 6 run."""
    parser = argparse.ArgumentParser(description="Layer 6 training pair construction")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument(
        "--candidates",
        type=str,
        default=str(PATH_OUTPUT_DIR / "candidate_pairs_train.tsv"),
        help="L5 candidate-pairs TSV for the training references.",
    )
    parser.add_argument("--countries", nargs="+", default=list(L3_COUNTRIES))
    parser.add_argument("--easy-pool-size", type=int, default=L6_EASY_POOL_SIZE)
    parser.add_argument("--max-references", type=int, default=L6_MAX_REFERENCES)
    parser.add_argument("--seed", type=int, default=L6_SEED)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["a", "b"],
        default=["a", "b"],
        help="Which variant datasets to write.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(PATH_ARTIFACTS_DIR / "train_pairs"),
    )
    return parser.parse_args()


class PairParquetWriter:
    """Streaming parquet writer for one variant dataset."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.writer = pq.ParquetWriter(str(path), PAIR_SCHEMA)
        self.buffer: list = []

    def add(self, samples) -> None:
        self.buffer.extend(samples)
        if len(self.buffer) >= WRITE_CHUNK:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        table = pa.table(
            {
                "s1_id": [s.s1_id for s in self.buffer],
                "cand_id": [s.cand_id for s in self.buffer],
                "label": [s.label for s in self.buffer],
                "neg_type": [s.neg_type for s in self.buffer],
                "country": [s.country for s in self.buffer],
                "rank": [s.rank for s in self.buffer],
            },
            schema=PAIR_SCHEMA,
        )
        self.writer.write_table(table)
        self.buffer.clear()

    def close(self) -> None:
        self.flush()
        self.writer.close()


def load_reference_countries(split: str, ref_ids: set[str]) -> dict[str, str]:
    """Map each reference id to its canonical country bucket."""
    wanted = set(ref_ids)
    countries: dict[str, str] = {}
    for shard in iter_source_shards(split, 1):
        df = pd.read_parquet(shard, columns=["entity_id", "country_norm"])
        df = df[df["entity_id"].isin(wanted)]
        if df.empty:
            continue
        for entity_id, country_norm in zip(df["entity_id"].tolist(), df["country_norm"].tolist()):
            countries[entity_id] = assign_country(country_norm)
    return countries


def main() -> None:
    """Run Layer 6 pair construction end to end."""
    args = parse_args()
    candidate_path = Path(args.candidates)
    out_dir = Path(args.out_dir)

    print("=" * 65)
    print("🚀 LAYER 6: TRAINING PAIR CONSTRUCTION & HARD-NEGATIVE A/B")
    print("=" * 65)

    if not candidate_path.exists():
        print(f"  ❌ candidate set not found: {candidate_path}")
        print("     generate it first, e.g.:")
        print("       python business_entity_resolution/src/main_l3.py --refs train")
        print("       python business_entity_resolution/src/main_l4.py --refs train")
        print("       python business_entity_resolution/src/main_l5.py --refs train")
        raise SystemExit(1)

    print(f"  candidates : {candidate_path}")
    print(f"  countries  : {args.countries}")
    print(f"  variants   : {args.variants}")

    # Pass 1: collect the reference ids present in the candidate set.
    t0 = time.time()
    reference_ids: set[str] = set()
    for s1_id, _ in iter_candidate_pairs(candidate_path):
        reference_ids.add(s1_id)
        if args.max_references is not None and len(reference_ids) >= args.max_references:
            break
    print(f"  references : {len(reference_ids):,}")

    # Ground truth and per-reference country for those references only.
    gt_map = load_ground_truth(reference_ids)
    ref_country = load_reference_countries(args.split, reference_ids)
    print(f"  ground truth loaded for {len(gt_map):,} references")

    # Easy-negative pools per country.
    easy_pools = {
        country: build_easy_pool(args.split, country, args.easy_pool_size, args.seed)
        for country in args.countries
    }
    for country, pool in easy_pools.items():
        print(f"  easy pool [{country}] : {len(pool):,}")

    rng = random.Random(args.seed)
    sampler_config = SamplingConfig()

    writers: dict[str, PairParquetWriter] = {}
    if "a" in args.variants:
        writers["a"] = PairParquetWriter(out_dir / "variant_a.parquet")
    if "b" in args.variants:
        writers["b"] = PairParquetWriter(out_dir / "variant_b.parquet")

    # Streaming accumulators for the report (counts only, never the samples).
    totals = {
        "a": {"pos": 0, "hard": 0, "easy": 0, "pairs": 0, "refs": 0},
        "b": {"pos": 0, "hard": 0, "easy": 0, "pairs": 0, "refs": 0},
    }
    per_country: dict[str, dict[str, int]] = {}

    # Pass 2: sample pairs and stream to parquet.
    processed = 0
    for s1_id, candidate_ids in iter_candidate_pairs(candidate_path):
        if s1_id not in reference_ids:
            continue
        country = ref_country.get(s1_id, "other")
        truth = gt_map.get(s1_id, set())
        samples = sample_reference_pairs(
            s1_id, candidate_ids, truth, country, easy_pools.get(country, []), rng, sampler_config
        )

        b_samples = variant_b_rows(samples)
        if "a" in writers:
            writers["a"].add(samples)
        if "b" in writers:
            writers["b"].add(b_samples)

        for key, batch in (("a", samples), ("b", b_samples)):
            totals[key]["pairs"] += len(batch)
            totals[key]["pos"] += sum(1 for s in batch if s.neg_type == POSITIVE)
            totals[key]["hard"] += sum(1 for s in batch if s.neg_type == HARD)
            totals[key]["easy"] += sum(1 for s in batch if s.neg_type == EASY)
            totals[key]["refs"] += 1

        bucket = per_country.setdefault(country, {"references": 0, "pairs_a": 0, "pairs_b": 0})
        bucket["references"] += 1
        bucket["pairs_a"] += len(samples)
        bucket["pairs_b"] += len(b_samples)

        processed += 1
        if processed % 200_000 == 0:
            print(f"    ... {processed:,} references sampled")

    for writer in writers.values():
        writer.close()

    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("📊 LAYER 6 PAIR SAMPLING REPORT")
    print("=" * 65)
    for key in ("a", "b"):
        if key not in writers:
            continue
        t = totals[key]
        ratio_h = t["hard"] / t["pos"] if t["pos"] else 0.0
        ratio_e = t["easy"] / t["pos"] if t["pos"] else 0.0
        label = "Variant A (with hard negs)" if key == "a" else "Variant B (no hard negs)"
        print(f"\n  {label}")
        print(f"    pairs: {t['pairs']:,} | pos: {t['pos']:,} | hard: {t['hard']:,} | easy: {t['easy']:,}")
        print(f"    achieved ratio: 1 pos : {ratio_h:.2f} hard : {ratio_e:.2f} easy")
    print(f"\n  elapsed: {elapsed:.1f}s")
    print("  ℹ️  model-based A/B (validation F0.5) runs in L8 once features exist.")

    report = {
        "config": {
            "candidates": str(candidate_path),
            "split": args.split,
            "countries": args.countries,
            "easy_pool_size": args.easy_pool_size,
            "max_references": args.max_references,
            "seed": args.seed,
            "variants": args.variants,
        },
        "totals": {k: totals[k] for k in totals if k in writers},
        "per_country": per_country,
        "elapsed_seconds": round(elapsed, 2),
    }
    PATH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = PATH_OUTPUT_DIR / "l6_pairs_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  saved report: {report_path}")
    print(f"  saved datasets: {out_dir}")
    print("=" * 65)
    print("🌟 LAYER 6 COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
