"""Layer 7 entry point: pairwise feature engineering.

Reads an L6 pair dataset (Variant A or B), joins each pair with its normalized
Source 1 and candidate records, computes the 27 pairwise features, and streams a
feature table to Parquet for the L8 matcher:

    artifacts/features/features_<variant>.parquet

Retrieval-signal features default to 0 unless ``--l4-dir`` / ``--l3-dir`` are
supplied, in which case the fused RRF score and channel-agreement count are
joined per pair.

Usage
-----
# Features for Variant A pairs:
python business_entity_resolution/src/main_l7.py --pairs artifacts/train_pairs/variant_a.parquet

# Small dev run:
python business_entity_resolution/src/main_l7.py --pairs artifacts/train_pairs/variant_a.parquet --max-pairs 200000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
for _p in (str(SRC_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import PATH_ARTIFACTS_DIR
from l3_l5_blocking.buckets import iter_source_shards
from l6_l8_matching.features import FEATURE_NAMES, build_idf, compute_features
from l6_l8_matching.signals import (
    MODE_SIDECAR,
    MODE_ZEROS,
    SignalsTooLargeError,
    load_signals_for_pairs,
    signals_path,
)

PAIR_COLUMNS = ["s1_id", "cand_id", "label", "neg_type", "rank"]
RECORD_COLUMNS = [
    "entity_id",
    "country_norm",
    "name_core",
    "name_norm",
    "name_tokens",
    "name_legal_suffix",
    "addr_norm",
    "addr_tokens",
    "addr_house_number",
    "addr_postal",
    "addr_state",
    "addr_city",
    "addr_digits",
    "is_missing_addr",
]

FEATURE_SCHEMA = pa.schema(
    [("s1_id", pa.string()), ("cand_id", pa.string()), ("label", pa.int8()), ("neg_type", pa.string())]
    + [(name, pa.float32()) for name in FEATURE_NAMES]
)

WRITE_CHUNK = 250_000


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Layer 7 run."""
    parser = argparse.ArgumentParser(description="Layer 7 pairwise feature engineering")
    parser.add_argument(
        "--pairs",
        type=str,
        default=str(PATH_ARTIFACTS_DIR / "train_pairs" / "variant_a.parquet"),
    )
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--signals",
        type=str,
        default=None,
        help="Retrieval-signal sidecar written by L5 (default: the sidecar for --signals-refs).",
    )
    parser.add_argument(
        "--signals-refs",
        choices=["val", "train", "all"],
        default="train",
        help="Run scope of the candidate set the pairs were sampled from.",
    )
    parser.add_argument(
        "--no-signals",
        action="store_true",
        help="Force retrieval features to 0 (only valid if inference does the same).",
    )
    parser.add_argument("--l4-dir", type=str, default=None, help="Legacy: L4 artifact dir for RRF scores.")
    parser.add_argument("--l3-dir", type=str, default=None, help="Legacy: L3 artifact dir for channel agreement.")
    return parser.parse_args()


def _default_out(pairs_path: Path) -> Path:
    variant = pairs_path.stem  # e.g. variant_a
    return PATH_ARTIFACTS_DIR / "features" / f"features_{variant}.parquet"


def iter_pairs(path: Path, columns: list[str], batch_size: int = 250_000):
    """Yield pandas batches of the pair dataset."""
    pf = pq.ParquetFile(str(path))
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pandas()


def load_record_lookup(split: str, source_indices: tuple[int, ...], ids: set[str]) -> dict[str, dict]:
    """Load normalized records for ``ids`` from the given sources into a dict."""
    lookup: dict[str, dict] = {}
    wanted = ids
    for source_idx in source_indices:
        for shard in iter_source_shards(split, source_idx):
            df = pd.read_parquet(shard, columns=RECORD_COLUMNS)
            df = df[df["entity_id"].isin(wanted)]
            if df.empty:
                continue
            for record in df.to_dict(orient="records"):
                # Parquet list columns arrive as numpy arrays; coerce to plain lists.
                for field in ("name_tokens", "addr_tokens", "addr_digits"):
                    raw = record.get(field)
                    record[field] = list(raw) if raw is not None else []
                lookup[record["entity_id"]] = record
    return lookup


def load_signals(
    needed: set[tuple[str, str]],
    l4_dir: str | None,
    l3_dir: str | None,
) -> dict[tuple[str, str], dict[str, float]]:
    """Optionally join RRF score (L4) and channel agreement (L3) for the pairs."""
    signals: dict[tuple[str, str], dict[str, float]] = {}
    if not needed:
        return signals

    if l4_dir:
        from main_l4 import load_l4_artifact  # local import keeps L7 standalone

        for path in sorted(Path(l4_dir).glob("l4_*_country=*.parquet")):
            reference_ids, fused = load_l4_artifact(path)
            for s1_id, row in zip(reference_ids, fused):
                for cand_id, score in row:
                    key = (s1_id, cand_id)
                    if key in needed:
                        signals.setdefault(key, {})["rrf_score"] = float(score)

    if l3_dir:
        from main_l4 import load_l3_artifact

        for path in sorted(Path(l3_dir).glob("l3_*_country=*.parquet")):
            reference_ids, channels = load_l3_artifact(path)
            n_channels = max(1, len(channels))
            for i, s1_id in enumerate(reference_ids):
                agreement: Counter[str] = Counter()
                for candidates in channels.values():
                    for cand_id in candidates[i]:
                        agreement[cand_id] += 1
                for cand_id, count in agreement.items():
                    key = (s1_id, cand_id)
                    if key in needed:
                        signals.setdefault(key, {})["channel_agreement"] = count / n_channels

    return signals


def main() -> None:
    """Run Layer 7 feature engineering end to end."""
    args = parse_args()
    pairs_path = Path(args.pairs)
    out_path = Path(args.out) if args.out else _default_out(pairs_path)

    print("=" * 65)
    print("🚀 LAYER 7: PAIRWISE FEATURE ENGINEERING")
    print("=" * 65)
    if not pairs_path.exists():
        print(f"  ❌ pair dataset not found: {pairs_path}")
        print("     run main_l6.py first to build Variant A/B pairs.")
        raise SystemExit(1)
    print(f"  pairs : {pairs_path}")
    print(f"  output: {out_path}")
    print(f"  features: {len(FEATURE_NAMES)} across 4 blocks")

    t0 = time.time()

    # Pass 1: collect the ids we need to load.
    s1_ids: set[str] = set()
    cand_ids: set[str] = set()
    n_pairs = 0
    needed: set[tuple[str, str]] = set()
    for batch in iter_pairs(pairs_path, PAIR_COLUMNS):
        if args.max_pairs is not None and n_pairs >= args.max_pairs:
            break
        s1_ids.update(batch["s1_id"].tolist())
        cand_ids.update(batch["cand_id"].tolist())
        for s1_id, cand_id, rank in zip(batch["s1_id"], batch["cand_id"], batch["rank"]):
            if int(rank) >= 0:
                needed.add((s1_id, cand_id))
        n_pairs += len(batch)
    if args.max_pairs is not None:
        n_pairs = min(n_pairs, args.max_pairs)
    print(f"  unique S1 ids={len(s1_ids):,} | candidate ids={len(cand_ids):,}")

    # Load normalized records + optional retrieval signals.
    print("  loading normalized records ...")
    s1_lookup = load_record_lookup(args.split, (1,), s1_ids)
    cand_lookup = load_record_lookup(args.split, (2, 3), cand_ids)
    print(f"  loaded S1 records={len(s1_lookup):,} | candidate records={len(cand_lookup):,}")

    # Retrieval signals must be identical at training and inference time. The
    # sidecar written by L5 is the single source of truth; the legacy --l3-dir /
    # --l4-dir join is still accepted but records a different mode.
    signals_mode = MODE_ZEROS
    signals_source = None
    if args.no_signals:
        signals: dict = {}
        print("  ⚠️  --no-signals: ret_rrf_score / ret_retriever_agreement will be 0")
        print("      inference must use the same setting or the model will see a skew")
    elif args.l3_dir or args.l4_dir:
        signals = load_signals(needed, args.l4_dir, args.l3_dir)
        signals_mode = "legacy_artifact_join"
        print(f"  retrieval signals joined from L3/L4 artifacts for {len(signals):,} pairs")
    else:
        sidecar = Path(args.signals) if args.signals else signals_path(args.split, args.signals_refs)
        if sidecar.exists():
            try:
                signals = load_signals_for_pairs(sidecar, needed)
            except SignalsTooLargeError as exc:
                print(f"  ❌ {exc}")
                raise SystemExit(1) from None
            signals_mode = MODE_SIDECAR
            signals_source = str(sidecar)
            print(f"  retrieval signals from sidecar: {sidecar.name} ({len(signals):,} pairs)")
        else:
            signals = {}
            print(f"  ⚠️  no signal sidecar at {sidecar}")
            print("      ret_rrf_score / ret_retriever_agreement will be 0 — run main_l5.py")
            print("      (it writes the sidecar) or pass --signals <path> to match inference")

    # IDF table from candidate name tokens.
    token_df: Counter[str] = Counter()
    for record in cand_lookup.values():
        tokens = record.get("name_tokens")
        if tokens:
            token_df.update(set(tokens))
    idf = build_idf(token_df, len(cand_lookup))

    # Pass 2: compute features and stream to parquet.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(str(out_path), FEATURE_SCHEMA)
    buffer: list[dict] = []
    written = 0
    skipped = 0
    missing_s1 = 0
    missing_cand = 0

    def flush() -> None:
        if not buffer:
            return
        columns: dict[str, list] = {name: [] for name in FEATURE_SCHEMA.names}
        for row in buffer:
            for name in FEATURE_SCHEMA.names:
                columns[name].append(row[name])
        writer.write_table(pa.table(columns, schema=FEATURE_SCHEMA))
        buffer.clear()

    processed = 0
    for batch in iter_pairs(pairs_path, PAIR_COLUMNS):
        for s1_id, cand_id, label, neg_type, rank in zip(
            batch["s1_id"], batch["cand_id"], batch["label"], batch["neg_type"], batch["rank"]
        ):
            if args.max_pairs is not None and processed >= args.max_pairs:
                break
            processed += 1

            s1 = s1_lookup.get(s1_id)
            cand = cand_lookup.get(cand_id)
            if s1 is None:
                missing_s1 += 1
            if cand is None:
                missing_cand += 1
            if s1 is None or cand is None:
                skipped += 1
                continue

            pair = {"rank": int(rank)}
            row = {
                "s1_id": s1_id,
                "cand_id": cand_id,
                "label": int(label),
                "neg_type": neg_type,
            }
            features = compute_features(s1, cand, pair, idf, signals.get((s1_id, cand_id)))
            for name, value in zip(FEATURE_NAMES, features):
                row[name] = float(value)
            buffer.append(row)
            written += 1

            if len(buffer) >= WRITE_CHUNK:
                flush()
        if args.max_pairs is not None and processed >= args.max_pairs:
            break

    flush()
    writer.close()
    elapsed = time.time() - t0

    print("\n" + "=" * 65)
    print("📊 LAYER 7 FEATURE REPORT")
    print("=" * 65)
    # Provenance: Layer 8 copies this into its report and Layer 11 refuses to run
    # a mismatch (a model trained with real signals but scored with zeros).
    provenance = {
        "split": args.split,
        "signals_mode": signals_mode,
        "signals_source": signals_source,
        "signals_refs": args.signals_refs,
        "signals_joined": len(signals),
        "pairs": str(pairs_path),
        "features": len(FEATURE_NAMES),
        "rows": written,
        "idf_documents": len(cand_lookup),
        "legacy_signal_join": bool(args.l3_dir or args.l4_dir),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print(f"  features written : {written:,}")
    print(f"  skipped (missing record) : {skipped:,} (s1 missing {missing_s1:,}, cand missing {missing_cand:,})")
    print(f"  feature count    : {len(FEATURE_NAMES)}")
    print(f"  retrieval signals: {signals_mode} ({len(signals):,} pairs)")
    print(f"  elapsed          : {elapsed:.1f}s")
    print(f"  output           : {out_path}")
    print(f"  provenance       : {meta_path}")
    print("=" * 65)
    print("🌟 LAYER 7 COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
