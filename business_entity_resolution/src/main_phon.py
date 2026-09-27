"""Memory-lean phonetic artifact generator (channel G).

The previous generator built its candidate pool with ``pd.concat`` over every
shard of a country, which needs multiple GB before the channel is even built -
that is what OOMed. This version:

* reads each normalized shard once and keeps only the columns the channel needs,
* appends plain Python lists (no pandas concat, no duplicate frame),
* honours ``--max-candidates`` so a country can be capped for a proof run,
* rebuilds the reference order from the matching L3 artifact so the output is
  row-aligned with the rest of the pipeline (required for L4x/L5 fusion).

Frozen output schema (one row per Source-1 entity), identical to the pipeline
contract::

    source1_entity_id   : string
    candidate_entity_ids: list<string>   (ranked best-first, up to --k)
    phon_scores         : list<float32>

Run::

    venv/Scripts/python.exe business_entity_resolution/src/main_phon.py \
      --split train --refs train --country india --k 200

    # bounded proof run
    venv/Scripts/python.exe business_entity_resolution/src/main_phon.py \
      --split test --refs all --country us --max-candidates 1500000
"""

from __future__ import annotations

import argparse
import glob
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

from config import PATH_ARTIFACTS_DIR
from l3_l5_blocking.phonetic_channel import PhoneticChannel

ARTIFACTS = PATH_ARTIFACTS_DIR
NORMALIZED = ARTIFACTS / "normalized"
BLOCKING = ARTIFACTS / "blocking"

CAND_COLUMNS = ["entity_id", "country_norm", "name_raw", "script_type", "addr_postal"]
REF_COLUMNS = ["entity_id", "name_raw", "script_type", "addr_postal"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Memory-lean phonetic channel generator")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--refs", choices=["val", "train", "all"], default="train")
    parser.add_argument("--country", required=True)
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-refs", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _paths(split: str, refs: str, country: str) -> tuple[Path, Path]:
    stem = f"phon_{split}_{refs}_country={country.lower()}"
    l3 = BLOCKING / f"l3_{split}_{refs}_country={country.lower()}.parquet"
    # Fall back to dense parquet when the merged L3 doesn't exist yet
    if not l3.exists():
        dense = BLOCKING / f"dense_{split}_{refs}_country={country.lower()}.parquet"
        if dense.exists():
            l3 = dense
    return BLOCKING / f"{stem}.parquet", l3


def load_references(split: str, refs: str, country: str, l3_path: Path, max_refs: int | None):
    """Load S1 refs in the exact L3/dense reference order (required for downstream fusion)."""
    if not l3_path.exists():
        raise FileNotFoundError(
            f"Reference artifact missing (tried l3 and dense parquet): {l3_path}"
        )
    print(f"  [phon] using reference order from: {l3_path.name}")
    ref_ids = [str(v) for v in pq.read_table(l3_path, columns=["source1_entity_id"]).column(0).to_pylist()]
    if max_refs is not None:
        ref_ids = ref_ids[:max_refs]
    wanted = set(ref_ids)
    source = f"{split}_s1"
    rows: dict[str, dict] = {}
    for shard in sorted(glob.glob(str(NORMALIZED / source / "part_*.parquet"))):
        df = pd.read_parquet(shard, columns=REF_COLUMNS + ["country_norm"])
        df = df[(df["country_norm"] == country.lower()) & (df["entity_id"].isin(wanted))]
        if df.empty:
            continue
        for rec in df.drop_duplicates("entity_id").to_dict(orient="records"):
            rows[rec["entity_id"]] = rec
        if len(rows) >= len(wanted):
            break
    missing = len(wanted) - len(rows)
    if missing:
        print(f"  ⚠️  {missing:,} references have no normalized record (kept as empty queries)")
    return ref_ids, rows


def load_candidates(split: str, country: str, max_candidates: int | None):
    """Stream S2+S3 candidate fields into lists (bounded by --max-candidates)."""
    ids: list[str] = []
    names: list[str] = []
    scripts: list[str] = []
    postals: list[str] = []
    for source in (f"{split}_s2", f"{split}_s3"):
        for shard in sorted(glob.glob(str(NORMALIZED / source / "part_*.parquet"))):
            df = pd.read_parquet(shard, columns=CAND_COLUMNS)
            df = df[df["country_norm"] == country.lower()]
            if df.empty:
                continue
            ids.extend(df["entity_id"].astype(str).tolist())
            names.extend(df["name_raw"].fillna("").astype(str).tolist())
            scripts.extend(df["script_type"].fillna("latin").astype(str).tolist())
            postals.extend(df["addr_postal"].fillna("").astype(str).tolist())
            if max_candidates is not None and len(ids) >= max_candidates:
                ids, names, scripts, postals = ids[:max_candidates], names[:max_candidates], scripts[:max_candidates], postals[:max_candidates]
                print(f"  candidate cap reached at {len(ids):,}")
                return ids, names, scripts, postals
    return ids, names, scripts, postals


def main() -> None:
    args = parse_args()
    country = args.country.lower()
    out_path, l3_path = _paths(args.split, args.refs, country)
    out_path = args.output or out_path

    print("=" * 65)
    print(f"🚀 PHONETIC CHANNEL (memory-lean) | split={args.split} refs={args.refs} country={country}")
    print("=" * 65)
    t0 = time.perf_counter()

    ref_ids, ref_rows = load_references(args.split, args.refs, country, l3_path, args.max_refs)
    print(f"  references: {len(ref_ids):,}")

    cand_ids, cand_names, cand_scripts, cand_postals = load_candidates(
        args.split, country, args.max_candidates
    )
    print(f"  candidates: {len(cand_ids):,}")

    channel = PhoneticChannel(max_posting=2000, k_retrieve=args.k)
    channel.build(cand_ids, cand_names, cand_scripts, cand_postals)
    print("  index built")

    table = channel.query_batch(
        ref_ids=ref_ids,
        ref_names=[str(ref_rows.get(r, {}).get("name_raw") or "") for r in ref_ids],
        ref_scripts=[str(ref_rows.get(r, {}).get("script_type") or "latin") for r in ref_ids],
        ref_postals=[str(ref_rows.get(r, {}).get("addr_postal") or "") for r in ref_ids],
        batch_size=5000,
        verbose=True,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="snappy")
    print(f"  saved {out_path} ({out_path.stat().st_size:,} bytes) in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
