"""Layer 8 entry point: GBDT matcher training + hard-negative A/B comparison.

For each available variant (A = pairs with hard negatives, B = without) it:
  1. trains a grouped 5-fold LightGBM and collects out-of-fold probabilities,
  2. evaluates macro-F0.5 over a threshold grid using the L1 scorer (the same
     decision rule L10 will tune), restricted to retrieved candidates so the
     evaluation matches inference,
  3. fits a full model on all rows and exports it.

Both variants are compared on out-of-fold macro F0.5 and the winner is recorded
in ``output/l8_model_report.json`` / ``.md``.

Usage
-----
python business_entity_resolution/src/main_l8.py
python business_entity_resolution/src/main_l8.py --variant a --max-pairs 500000
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

# Load LightGBM before scikit-learn: on Windows their OpenMP runtimes conflict and
# importing sklearn first causes an access violation when LightGBM fits a dataset.
import lightgbm  # noqa: F401  (import order is deliberate)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import L8_N_FOLDS, PATH_ARTIFACTS_DIR, PATH_OUTPUT_DIR, SEED
from l6_l8_matching.features import FEATURE_NAMES
from l6_l8_matching.model import top_importances, train_full, train_oof

# Reuse the official scorer and the chunked ground-truth loader.
try:
    from src.l1_validation.metrics import evaluate_with_thresholds
except ImportError:  # pragma: no cover - path fallback
    from l1_validation.metrics import evaluate_with_thresholds

from main_l3 import load_ground_truth


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Layer 8 run."""
    parser = argparse.ArgumentParser(description="Layer 8 LightGBM matcher training")
    parser.add_argument("--variant", choices=["a", "b", "both"], default="both")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--n-folds", type=int, default=L8_N_FOLDS)
    parser.add_argument("--n-jobs", type=int, default=-1, help="LightGBM threads (-1 = all).")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--features-dir", type=str, default=str(PATH_ARTIFACTS_DIR / "features"))
    parser.add_argument("--out-dir", type=str, default=str(PATH_ARTIFACTS_DIR / "models"))
    parser.add_argument("--tau-match-grid", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--tau-s-grid", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--margin", type=float, default=0.05)
    return parser.parse_args()


def features_path(features_dir: str, variant: str) -> Path:
    return Path(features_dir) / f"features_variant_{variant}.parquet"


def tune_thresholds(scores: dict, gt_map: dict, tau_match_grid, tau_s_grid, margin: float) -> dict:
    """Grid search the (tau_match, tau_s) decision pair on out-of-fold scores."""
    best: dict | None = None
    for tau_match in tau_match_grid:
        for tau_s in tau_s_grid:
            result = evaluate_with_thresholds(scores, gt_map, tau_match, tau_s, margin)
            if best is None or result["macro_f05"] > best["macro_f05"]:
                best = result
    return best or {"macro_f05": 0.0, "tau_match": 0.0, "tau_s": 0.0, "margin": margin}


def feature_provenance(features_dir: str, variant: str) -> dict:
    """Read the Layer 7 provenance sidecar next to a feature table.

    Records whether the retrieval-signal features were real values or zeros, so
    Layer 11 can refuse to score the exported booster with the other convention.
    """
    meta_path = Path(features_dir) / f"features_variant_{variant}.meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def evaluate_variant(df: pd.DataFrame, args, out_dir: Path) -> dict:
    """Train, evaluate and export one variant; return its report entry."""
    feature_names = [name for name in FEATURE_NAMES if name in df.columns]
    provenance = feature_provenance(args.features_dir, args.variant)
    print(f"\n{'─' * 65}\n  Training variant ({len(df):,} pairs, {len(feature_names)} features)\n{'─' * 65}")
    if provenance:
        print(
            f"  retrieval signals: {provenance.get('signals_mode')} "
            f"({provenance.get('signals_joined', 0):,} pairs joined)"
        )

    t0 = time.time()
    oof, models, importances = train_oof(
        df, feature_names, n_folds=args.n_folds, seed=args.seed, params={"n_jobs": args.n_jobs}
    )
    train_time = time.time() - t0

    labels = df["label"].to_numpy()
    try:
        auc = float(roc_auc_score(labels, oof))
    except ValueError:
        auc = float("nan")

    # Evaluate on retrieved candidates only (matches inference-time inputs).
    retrieved = df["neg_type"].to_numpy() != "easy"
    ref_ids = set(df["s1_id"].tolist())
    gt_map = load_ground_truth(ref_ids)
    scores = {
        (s1_id, cand_id): float(prob)
        for s1_id, cand_id, prob in zip(
            df["s1_id"].to_numpy()[retrieved],
            df["cand_id"].to_numpy()[retrieved],
            oof[retrieved],
        )
    }
    print(f"  OOF AUC={auc:.4f} | scoring {len(scores):,} retrieved pairs over {len(gt_map):,} references")
    best = tune_thresholds(scores, gt_map, args.tau_match_grid, args.tau_s_grid, args.margin)

    # Export OOF probabilities (used by L9 calibration).
    oof_path = out_dir / f"oof_{args.variant}.parquet"
    pd.DataFrame(
        {
            "s1_id": df["s1_id"].to_numpy(),
            "cand_id": df["cand_id"].to_numpy(),
            "label": labels.astype("int8"),
            "neg_type": df["neg_type"].to_numpy(),
            "oof_prob": oof.astype("float32"),
        }
    ).to_parquet(oof_path, index=False)

    # Fit and export the full model.
    full_model = train_full(df, feature_names, seed=args.seed, params={"n_jobs": args.n_jobs})
    model_path = out_dir / f"lgbm_variant_{args.variant}.txt"
    full_model.booster_.save_model(str(model_path))

    top = top_importances(importances, 10)
    print(f"  macro F0.5 (OOF) = {best['macro_f05']:.4f} at tau_match={best['tau_match']} tau_s={best['tau_s']}")
    print("  top features:", ", ".join(f"{name}" for name, _ in top[:5]))

    return {
        "variant": args.variant,
        "pairs": int(len(df)),
        "references": len(ref_ids),
        "features": len(feature_names),
        "oof_auc": auc,
        "macro_f05": best["macro_f05"],
        "tau_match": best["tau_match"],
        "tau_s": best["tau_s"],
        "margin": best["margin"],
        "train_seconds": round(train_time, 2),
        "top_features": top,
        "oof_path": str(oof_path),
        "model_path": str(model_path),
        "signals_mode": provenance.get("signals_mode"),
        "signals_source": provenance.get("signals_source"),
        "signals_joined": provenance.get("signals_joined"),
        "feature_provenance": str(
            Path(args.features_dir) / f"features_variant_{args.variant}.meta.json"
        ),
    }


def main() -> None:
    """Run Layer 8 training and the A/B comparison."""
    args = parse_args()
    features_dir = Path(args.features_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = ["a", "b"] if args.variant == "both" else [args.variant]
    available: list[tuple[str, Path]] = []
    for variant in variants:
        path = features_path(features_dir, variant)
        if path.exists():
            available.append((variant, path))
        else:
            print(f"  ⚠️  missing feature table for variant {variant}: {path}")

    print("=" * 65)
    print("🚀 LAYER 8: GBDT MATCHER TRAINING + HARD-NEGATIVE A/B")
    print("=" * 65)
    print(f"  variants: {[v for v, _ in available]} | folds={args.n_folds} | seed={args.seed}")
    if not available:
        print("  ❌ no feature tables found. Run main_l7.py first.")
        raise SystemExit(1)

    report_entries: dict[str, dict] = {}
    for variant, path in available:
        df = pd.read_parquet(path)
        if args.max_pairs is not None and len(df) > args.max_pairs:
            df = df.sample(n=args.max_pairs, random_state=args.seed).reset_index(drop=True)
        args.variant = variant
        report_entries[variant] = evaluate_variant(df, args, out_dir)

    # A/B comparison
    winner = None
    if len(report_entries) > 1:
        a = report_entries.get("a", {}).get("macro_f05", -1.0)
        b = report_entries.get("b", {}).get("macro_f05", -1.0)
        winner = "a" if a >= b else "b"
        print("\n" + "=" * 65)
        print("🆚 A/B COMPARISON (out-of-fold macro F0.5)")
        print("=" * 65)
        print(f"  Variant A (with hard negatives): {a:.4f}")
        print(f"  Variant B (no hard negatives)  : {b:.4f}")
        print(f"  → winner: Variant {winner.upper()}")

    report = {
        "config": {
            "n_folds": args.n_folds,
            "seed": args.seed,
            "max_pairs": args.max_pairs,
            "tau_match_grid": args.tau_match_grid,
            "tau_s_grid": args.tau_s_grid,
            "margin": args.margin,
        },
        "variants": report_entries,
        "winner": winner,
    }
    PATH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = PATH_OUTPUT_DIR / "l8_model_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n  saved report: {report_path}")
    print("=" * 65)
    print("🌟 LAYER 8 COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
