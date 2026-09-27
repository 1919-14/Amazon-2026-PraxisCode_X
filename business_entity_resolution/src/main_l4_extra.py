"""Layer 4x entry point: fuse the L4 union with phonetic (G) / dense (H) channels.

Produces ``artifacts/blocking/l4x_<split>_<refs>_country=<c>.parquet`` with the
same schema as L4, so L5 -> L7 -> L11 consume it unchanged. This is an additive
step: it never rewrites the validated L4 artifact.

Usage
-----
# India train: fuse phonetic + dense into the union
python business_entity_resolution/src/main_l4_extra.py \
    --extra artifacts/blocking/phon_train_train_country=india.parquet \
    --extra artifacts/blocking/dense_train_train_country=india.parquet

# Weights: L4 weight first, then one per --extra (default 1.0 each)
python business_entity_resolution/src/main_l4_extra.py \
    --extra phon.parquet --extra dense.parquet --weights 2 1 1
"""

from __future__ import annotations

import argparse
import json
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

from config import L3_COUNTRIES, PATH_ARTIFACTS_DIR, PATH_OUTPUT_DIR
from l3_l5_blocking.artifacts import l4_artifact_path
from l3_l5_blocking.extra_channels import fuse_country_extra
from utils.reports import merge_json_report, write_text
from utils.coverage import normalize_countries


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the extra-channel fusion run."""
    parser = argparse.ArgumentParser(description="Layer 4x extra-channel fusion")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--refs", choices=["val", "train", "all"], default="train")
    parser.add_argument("--countries", nargs="+", default=list(L3_COUNTRIES))
    parser.add_argument(
        "--extra",
        action="append",
        default=None,
        help="Extra-channel artifact. Use {country} as a placeholder, or pass one "
        "per country with --country-extra.",
    )
    parser.add_argument(
        "--extra-template",
        action="append",
        default=None,
        help="Extra artifact path template with {country}, e.g. "
        "artifacts/blocking/phon_{split}_{refs}_country={country}.parquet",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="RRF weights: L4 first, then one per extra channel (default 1.0 each).",
    )
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--topn", type=int, default=None, help="Truncate the fused union.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(PATH_ARTIFACTS_DIR / "blocking"),
    )
    args = parser.parse_args()
    args.countries = normalize_countries(args.countries)
    return args


def resolve_extras(args: argparse.Namespace, country: str) -> list[Path]:
    """Resolve the extra artifacts for one country from --extra / --extra-template."""
    resolved: list[Path] = []
    for raw in args.extra or []:
        resolved.append(Path(raw.replace("{country}", country)))
    for template in args.extra_template or []:
        resolved.append(
            Path(
                template.format(
                    country=country, split=args.split, refs=args.refs
                )
            )
        )
    return resolved


def main() -> None:
    """Run extra-channel fusion for every requested country."""
    args = parse_args()
    out_dir = Path(args.out_dir)
    weights = args.weights

    print("=" * 65)
    print("🚀 LAYER 4x: EXTRA-CHANNEL FUSION (phonetic / dense)")
    print("=" * 65)
    print(f"  split={args.split} | refs={args.refs} | countries={args.countries}")
    if not args.extra and not args.extra_template:
        print("  ❌ pass at least one --extra or --extra-template")
        raise SystemExit(2)

    t0 = time.time()
    per_country: dict[str, dict] = {}
    for country in args.countries:
        l4_path = l4_artifact_path(args.split, args.refs, country)
        if not l4_path.exists():
            print(f"  ⚠️  missing L4 artifact for {country}: {l4_path} — skipping")
            per_country[country] = {"status": "missing_l4"}
            continue

        extras = [p for p in resolve_extras(args, country) if p.exists()]
        if not extras:
            print(f"  ⚠️  no extra-channel artifacts found for {country} — skipping")
            per_country[country] = {"status": "missing_extra"}
            continue

        out_path = out_dir / f"l4x_{args.split}_{args.refs}_country={country}.parquet"
        print(f"\n🌍 {country}: L4 + {', '.join(p.name for p in extras)}")
        rows = fuse_country_extra(
            l4_path,
            extras,
            out_path,
            country=country,
            weights=weights,
            batch_size=args.batch_size,
            rrf_k=args.rrf_k,
            topn=args.topn,
        )
        print(f"  saved {out_path} ({rows:,} references)")
        per_country[country] = {
            "status": "ok",
            "rows": rows,
            "extras": [str(p) for p in extras],
            "output": str(out_path),
        }

    elapsed = time.time() - t0
    report = {
        "config": {
            "split": args.split,
            "refs": args.refs,
            "countries": args.countries,
            "weights": weights,
            "rrf_k": args.rrf_k,
            "topn": args.topn,
        },
        "per_country": per_country,
        "elapsed_seconds": round(elapsed, 2),
    }
    PATH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merge_json_report(PATH_OUTPUT_DIR / "l4x_extra_report.json", f"{args.split}_{args.refs}", report)
    write_text(
        PATH_OUTPUT_DIR / "l4x_extra_report.md",
        "# Layer 4x Extra-Channel Fusion\n\n"
        f"- split `{args.split}` | refs `{args.refs}` | elapsed `{elapsed:.1f}s`\n"
        f"- per country: `{json.dumps(per_country)}`\n",
    )
    print(f"\n  report: {PATH_OUTPUT_DIR / 'l4x_extra_report.json'}")
    print("🌟 LAYER 4x COMPLETE")


if __name__ == "__main__":
    main()
