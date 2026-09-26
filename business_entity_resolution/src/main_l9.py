"""Layer 9 entry point: conditional isotonic calibration of OOF probabilities.

Calibration is **conditional**: it is adopted only when it materially improves
probability reliability (ECE), because isotonic regression is monotonic and so
cannot change the best macro-F0.5 achievable by threshold tuning. We still
report F0.5 for every option to prove it does not degrade.

Options evaluated on the retrieved OOF pairs:
  * ``none``        - raw LightGBM probabilities.
  * ``global``      - one isotonic calibrator, cross-validated by ``s1_id``.
  * ``per_country`` - an isotonic calibrator per country bucket.

Outputs:
    artifacts/models/oof_calibrated_<variant>.parquet
    artifacts/models/isotonic_<variant>.json
    output/l9_calibration_report.json / .md

Usage
-----
python business_entity_resolution/src/main_l9.py
python business_entity_resolution/src/main_l9.py --variant a --min-ece-improvement 0.005
"""

from __future__ import annotations

import argparse
import json
import sys
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
from sklearn.metrics import roc_auc_score

from config import L8_N_FOLDS, L9_ENABLE_CALIBRATION, PATH_ARTIFACTS_DIR, PATH_OUTPUT_DIR, SEED
from l6_l8_matching.calibration import (
    apply_isotonic,
    apply_serialized,
    brier,
    calibrate_grouped,
    ece,
    fit_isotonic,
    log_loss,
    serialize_isotonic,
)

try:
    from src.l1_validation.metrics import evaluate_with_thresholds
except ImportError:  # pragma: no cover - path fallback
    from l1_validation.metrics import evaluate_with_thresholds

from main_l3 import load_ground_truth
from main_l6 import load_reference_countries


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Layer 9 run."""
    parser = argparse.ArgumentParser(description="Layer 9 conditional isotonic calibration")
    parser.add_argument("--variant", choices=["a", "b"], default="a")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--n-folds", type=int, default=L8_N_FOLDS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--min-ece-improvement", type=float, default=0.005)
    parser.add_argument("--tau-match-grid", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--tau-s-grid", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--models-dir", type=str, default=str(PATH_ARTIFACTS_DIR / "models"))
    return parser.parse_args()


def best_f05(scores: dict, gt_map: dict, tau_match_grid, tau_s_grid, margin: float) -> dict:
    """Grid search the best macro F0.5 for a set of pair scores."""
    best: dict | None = None
    for tau_match in tau_match_grid:
        for tau_s in tau_s_grid:
            result = evaluate_with_thresholds(scores, gt_map, tau_match, tau_s, margin)
            if best is None or result["macro_f05"] > best["macro_f05"]:
                best = result
    return best or {"macro_f05": 0.0, "tau_match": 0.0, "tau_s": 0.0}


def metrics(probs: np.ndarray, labels: np.ndarray, n_bins: int) -> dict:
    """Reliability metrics for one probability vector."""
    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = float("nan")
    return {
        "auc": auc,
        "brier": brier(probs, labels),
        "log_loss": log_loss(probs, labels),
        "ece": ece(probs, labels, n_bins),
    }


def scores_from(df: pd.DataFrame, probs: np.ndarray) -> dict:
    """Build the ``(s1_id, cand_id) -> prob`` score dict for the scorer."""
    return {
        (s1_id, cand_id): float(prob)
        for s1_id, cand_id, prob in zip(df["s1_id"].to_numpy(), df["cand_id"].to_numpy(), probs)
    }


def main() -> None:
    """Run Layer 9 conditional calibration."""
    args = parse_args()
    models_dir = Path(args.models_dir)
    oof_path = models_dir / f"oof_{args.variant}.parquet"

    print("=" * 65)
    print("🚀 LAYER 9: CONDITIONAL ISOTONIC CALIBRATION")
    print("=" * 65)
    print(f"  variant={args.variant} | folds={args.n_folds} | n_bins={args.n_bins}")

    if not oof_path.exists():
        print(f"  ❌ OOF file not found: {oof_path}")
        print("     run main_l8.py first to produce out-of-fold probabilities.")
        raise SystemExit(1)

    df = pd.read_parquet(oof_path)
    # Calibrate on retrieved candidates only: those are the only inputs at inference.
    df = df[df["neg_type"] != "easy"].reset_index(drop=True)
    if args.max_pairs is not None and len(df) > args.max_pairs:
        df = df.sample(n=args.max_pairs, random_state=args.seed).reset_index(drop=True)

    probs = df["oof_prob"].to_numpy(dtype=np.float64)
    labels = df["label"].to_numpy(dtype=np.int32)
    groups = df["s1_id"].to_numpy()
    print(f"  retrieved pairs: {len(df):,}")

    # Country for per-country calibration.
    ref_ids = set(df["s1_id"].tolist())
    ref_country = load_reference_countries(args.split, ref_ids)
    countries = np.array([ref_country.get(s1, "other") for s1 in df["s1_id"].tolist()])

    # Uncalibrated baseline.
    base_metrics = metrics(probs, labels, args.n_bins)

    # Global cross-validated calibration.
    global_cal = calibrate_grouped(probs, labels, groups, n_folds=args.n_folds, seed=args.seed)

    # Per-country cross-validated calibration.
    per_country_cal = probs.copy()
    for country in np.unique(countries):
        mask = countries == country
        if mask.sum() < 20:
            continue
        per_country_cal[mask] = calibrate_grouped(
            probs[mask], labels[mask], groups[mask], n_folds=args.n_folds, seed=args.seed
        )

    gt_map = load_ground_truth(ref_ids)

    options = {
        "none": (probs, base_metrics),
        "global": (global_cal, metrics(global_cal, labels, args.n_bins)),
        "per_country": (per_country_cal, metrics(per_country_cal, labels, args.n_bins)),
    }
    for name, (cal_probs, m) in options.items():
        m["macro_f05"] = best_f05(
            scores_from(df, cal_probs), gt_map, args.tau_match_grid, args.tau_s_grid, args.margin
        )["macro_f05"]

    # ------------------------------------------------------------------
    # Conditional decision: adopt calibration only if it improves ECE enough
    # and does not degrade the best achievable macro F0.5.
    # ------------------------------------------------------------------
    baseline_ece = options["none"][1]["ece"]
    baseline_f05 = options["none"][1]["macro_f05"]
    eligible = [
        name
        for name in ("global", "per_country")
        if options[name][1]["ece"] <= baseline_ece - args.min_ece_improvement
        and options[name][1]["macro_f05"] >= baseline_f05 - 1e-6
    ]
    if not L9_ENABLE_CALIBRATION or not eligible:
        chosen = "none"
    else:
        chosen = min(eligible, key=lambda name: options[name][1]["ece"])

    print("\n" + "=" * 65)
    print("📊 LAYER 9 CALIBRATION REPORT")
    print("=" * 65)
    print(f"{'option':<14}{'ECE':>9}{'Brier':>9}{'LogLoss':>9}{'AUC':>8}{'F0.5':>8}")
    for name in ("none", "global", "per_country"):
        m = options[name][1]
        print(
            f"{name:<14}{m['ece']:>9.4f}{m['brier']:>9.4f}{m['log_loss']:>9.4f}"
            f"{m['auc']:>8.4f}{m['macro_f05']:>8.4f}"
        )
    print(f"\n  chosen calibration: {chosen}")
    if chosen == "none":
        print("  → calibration skipped (ECE improvement below threshold or disabled)")

    # ------------------------------------------------------------------
    # Persist the chosen calibrator (fit on all retrieved rows for inference)
    # and the calibrated OOF table.
    # ------------------------------------------------------------------
    models_dir.mkdir(parents=True, exist_ok=True)
    calibrator_payload: dict = {"variant": args.variant, "chosen": chosen, "split": args.split}
    if chosen == "global":
        full_iso = fit_isotonic(probs, labels)
        calibrator_payload["global"] = serialize_isotonic(full_iso)
    elif chosen == "per_country":
        per_country_payload: dict[str, dict] = {}
        for country in np.unique(countries):
            mask = countries == country
            if mask.sum() < 20:
                continue
            per_country_payload[str(country)] = serialize_isotonic(
                fit_isotonic(probs[mask], labels[mask])
            )
        calibrator_payload["per_country"] = per_country_payload

    cal_path = models_dir / f"isotonic_{args.variant}.json"
    with cal_path.open("w", encoding="utf-8") as f:
        json.dump(calibrator_payload, f, indent=2)

    calibrated_probs = options[chosen][0]
    out_oof = models_dir / f"oof_calibrated_{args.variant}.parquet"
    pd.DataFrame(
        {
            "s1_id": df["s1_id"].to_numpy(),
            "cand_id": df["cand_id"].to_numpy(),
            "label": labels.astype("int8"),
            "neg_type": df["neg_type"].to_numpy(),
            "country": countries,
            "oof_prob": probs.astype("float32"),
            "calibrated_prob": calibrated_probs.astype("float32"),
        }
    ).to_parquet(out_oof, index=False)

    report = {
        "config": {
            "variant": args.variant,
            "split": args.split,
            "n_folds": args.n_folds,
            "n_bins": args.n_bins,
            "min_ece_improvement": args.min_ece_improvement,
            "enabled": L9_ENABLE_CALIBRATION,
        },
        "options": {name: options[name][1] for name in options},
        "baseline_ece": baseline_ece,
        "baseline_macro_f05": baseline_f05,
        "eligible": eligible,
        "chosen": chosen,
        "calibrator_path": str(cal_path),
        "calibrated_oof_path": str(out_oof),
    }
    PATH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = PATH_OUTPUT_DIR / "l9_calibration_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_path = PATH_OUTPUT_DIR / "l9_calibration_report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Layer 9 Calibration Report\n\n")
        f.write(f"- variant: `{args.variant}` | folds: `{args.n_folds}` | bins: `{args.n_bins}`\n")
        f.write(f"- chosen calibration: **{chosen}**\n\n")
        f.write("| option | ECE | Brier | LogLoss | AUC | macro F0.5 |\n|---|---|---|---|---|---|\n")
        for name in ("none", "global", "per_country"):
            m = options[name][1]
            f.write(
                f"| {name} | {m['ece']:.4f} | {m['brier']:.4f} | {m['log_loss']:.4f} | "
                f"{m['auc']:.4f} | {m['macro_f05']:.4f} |\n"
            )
        f.write(
            "\nIsotonic regression is monotonic, so it preserves ranking and cannot "
            "change the best achievable macro F0.5; calibration is adopted only for "
            "probability reliability.\n"
        )

    print(f"\n  saved calibrator: {cal_path}")
    print(f"  saved calibrated OOF: {out_oof}")
    print(f"  saved report: {report_path}")
    print("=" * 65)
    print("🌟 LAYER 9 COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
