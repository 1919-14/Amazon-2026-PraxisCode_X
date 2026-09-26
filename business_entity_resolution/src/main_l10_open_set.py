"""Layer 10 open-set tuner: make the France veto measurable.

France appears only in the test set, so there is no ground truth to validate the
open-set policy against - yet the policy decides the fate of ~259k test entities
(the veto sends any reference whose top score is below ``veto_min_confidence`` to
an empty prediction, and a boosted ``tau_s`` does the same for weaker scores).

This script creates the missing evidence by running a **pseudo-open-set**
experiment on a country that does have ground truth:

1. tune ``(tau_match, tau_s)`` on the *seen* countries, exactly as production does;
2. hold one country out as if it were unseen, and grid search
   ``(open_set_boost, veto_min_confidence)`` to maximise macro F0.5 **on that
   country's references** - which is precisely what the open-set parameters control;
3. write the winner to ``output/l10_open_set_policy.json``, which Layer 10 and
   Layer 11 load as their default (CLI args still override).

Optionally, ``--preview-scores`` points at the per-pair score table Layer 11 writes
for the test split, so the script can also report how many France entities a given
veto threshold would throw away before you submit.

Usage
-----
python business_entity_resolution/src/main_l10_open_set.py --pseudo-open-country india
python business_entity_resolution/src/main_l10_open_set.py --open-country us --boosts 0 0.05 0.1 --vetoes 0 0.2 0.4 0.5
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

import pandas as pd

from config import (
    L10_MARGIN,
    L10_OPEN_SET_TAU_BOOST,
    L10_OPEN_SET_VETO_MIN_CONFIDENCE,
    PATH_ARTIFACTS_DIR,
    PATH_OUTPUT_DIR,
    TAU_MATCH_GRID,
    TAU_S_GRID,
)
from l10_decision.decision import tune_thresholds
from l10_decision.open_set import (
    DEFAULT_BOOST_GRID,
    DEFAULT_VETO_GRID,
    policy_path,
    save_open_set_policy,
    score_quantiles,
    tune_open_set_policy,
    veto_impact_preview,
)
from utils.reports import merge_json_report, write_text

from main_l3 import load_ground_truth
from main_l6 import load_reference_countries


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the open-set policy run."""
    parser = argparse.ArgumentParser(description="Tune the L10 open-set policy")
    parser.add_argument("--variant", choices=["a", "b"], default="a")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument(
        "--pseudo-open-country",
        "--open-country",
        dest="open_country",
        default="india",
        help="Country bucket held out as if it were unseen (needs ground truth).",
    )
    parser.add_argument("--scores", type=str, default=None, help="Score parquet override.")
    parser.add_argument("--prob-col", type=str, default=None)
    parser.add_argument(
        "--use-calibrated",
        action="store_true",
        help="Prefer the L9 calibrated OOF table (default: any available, calibrated first).",
    )
    parser.add_argument("--models-dir", type=str, default=str(PATH_ARTIFACTS_DIR / "models"))
    parser.add_argument("--margin", type=float, default=L10_MARGIN)
    parser.add_argument("--tau-match-grid", type=float, nargs="+", default=list(TAU_MATCH_GRID))
    parser.add_argument("--tau-s-grid", type=float, nargs="+", default=list(TAU_S_GRID))
    parser.add_argument("--boosts", type=float, nargs="+", default=list(DEFAULT_BOOST_GRID))
    parser.add_argument("--vetoes", type=float, nargs="+", default=list(DEFAULT_VETO_GRID))
    parser.add_argument("--out", type=str, default=str(policy_path()))
    parser.add_argument(
        "--preview-scores",
        type=str,
        default=str(PATH_ARTIFACTS_DIR / "scores" / "test_a.parquet"),
        help="Per-pair score table (Layer 11) used for the France impact preview.",
    )
    parser.add_argument(
        "--preview-country",
        type=str,
        default="france",
        help="Open-set country whose veto impact should be previewed.",
    )
    return parser.parse_args()


def resolve_scores(args: argparse.Namespace) -> tuple[Path, str]:
    """Pick the OOF score table (calibrated L9 first, then raw L8)."""
    if args.scores:
        return Path(args.scores), (args.prob_col or "prob")

    models_dir = Path(args.models_dir)
    calibrated = models_dir / f"oof_calibrated_{args.variant}.parquet"
    raw = models_dir / f"oof_{args.variant}.parquet"
    if args.use_calibrated and calibrated.exists():
        return calibrated, (args.prob_col or "calibrated_prob")
    if calibrated.exists():
        return calibrated, (args.prob_col or "calibrated_prob")
    return raw, (args.prob_col or "oof_prob")


def load_scores(path: Path, prob_col: str) -> dict[tuple[str, str], float]:
    """Load ``(s1_id, cand_id) -> probability`` from a score parquet."""
    df = pd.read_parquet(path, columns=["s1_id", "cand_id", prob_col])
    scores = {
        (str(s1), str(cand)): float(prob)
        for s1, cand, prob in zip(df["s1_id"], df["cand_id"], df[prob_col])
    }
    del df
    return scores


def preview_open_set_impact(path: str, country: str) -> dict | None:
    """Quantiles + veto impact of an open-set country's top scores (no ground truth)."""
    preview_path = Path(path)
    if not preview_path.exists():
        return None
    df = pd.read_parquet(preview_path, columns=["s1_id", "prob", "country"])
    df = df[df["country"] == country]
    if df.empty:
        return None
    top = df.groupby("s1_id")["prob"].max()
    result = {
        "country": country,
        "references": int(len(top)),
        "top_score_quantiles": score_quantiles(top.tolist()),
        "veto_impact": veto_impact_preview(top.to_dict()),
    }
    del df, top
    return result


def main() -> None:
    """Run the pseudo-open-set policy tuning end to end."""
    args = parse_args()
    scores_path, prob_col = resolve_scores(args)

    print("=" * 65)
    print("🚀 LAYER 10 OPEN-SET POLICY TUNER (pseudo-open country)")
    print("=" * 65)
    print(f"  variant={args.variant} | pseudo-open country='{args.open_country}'")

    if not scores_path.exists():
        print(f"  ❌ score table not found: {scores_path}")
        print("     run main_l8.py (and optionally main_l9.py) first.")
        raise SystemExit(1)

    scores = load_scores(scores_path, prob_col)
    print(f"  scores: {scores_path.name} ({len(scores):,} pairs, prob='{prob_col}')")

    reference_ids = sorted({s1_id for s1_id, _ in scores})
    countries = load_reference_countries(args.split, set(reference_ids))
    open_ids = [s1 for s1 in reference_ids if countries.get(s1) == args.open_country]
    seen_ids = [s1 for s1 in reference_ids if countries.get(s1) != args.open_country]
    print(
        f"  references: {len(reference_ids):,} | pseudo-open: {len(open_ids):,} | "
        f"seen: {len(seen_ids):,}"
    )

    gt_map = load_ground_truth(set(reference_ids))
    if not gt_map:
        print("  ❌ no ground truth available — the policy cannot be tuned without it.")
        raise SystemExit(1)
    if not open_ids:
        print(f"  ❌ pseudo-open country '{args.open_country}' has no scored references.")
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # Step 1: thresholds on the *seen* countries (production behaviour)
    # ------------------------------------------------------------------
    seen_scores = {
        key: value for key, value in scores.items() if countries.get(key[0]) != args.open_country
    }
    seen_gt = {s1_id: gt_map[s1_id] for s1_id in seen_ids if s1_id in gt_map}
    t0 = time.time()
    seen_tuned = tune_thresholds(
        seen_scores, seen_gt, args.tau_match_grid, args.tau_s_grid, args.margin
    )
    print(
        f"\n  step 1 — thresholds tuned on seen countries: "
        f"tau_match={seen_tuned['tau_match']} tau_s={seen_tuned['tau_s']} "
        f"→ macro F0.5={seen_tuned['macro_f05']:.4f}"
    )

    # ------------------------------------------------------------------
    # Step 2: grid search the open-set parameters on the held-out country
    # ------------------------------------------------------------------
    open_scores = {
        key: value for key, value in scores.items() if countries.get(key[0]) == args.open_country
    }
    open_gt = {s1_id: gt_map[s1_id] for s1_id in open_ids if s1_id in gt_map}
    tuning = tune_open_set_policy(
        open_scores,
        open_gt,
        [s1_id for s1_id in open_ids if s1_id in open_gt],
        tau_match=seen_tuned["tau_match"],
        tau_s=seen_tuned["tau_s"],
        margin=args.margin,
        boost_grid=args.boosts,
        veto_grid=args.vetoes,
    )
    best = tuning["best"]
    default = tuning["default"]
    print(
        f"\n  step 2 — open-set policy on '{args.open_country}' "
        f"({best['references']:,} references):"
    )
    print(
        f"    config default (boost={L10_OPEN_SET_TAU_BOOST}, "
        f"veto<{L10_OPEN_SET_VETO_MIN_CONFIDENCE}): macro F0.5={default['macro_f05']:.4f} "
        f"({default['predicted_empty']:,} empties)"
    )
    print(
        f"    tuned          (boost={best['open_set_boost']}, "
        f"veto<{best['veto_min_confidence']}): macro F0.5={best['macro_f05']:.4f} "
        f"({best['predicted_empty']:,} empties)"
    )
    print(f"    improvement: {tuning['improvement']:+.4f} macro F0.5 points")

    # ------------------------------------------------------------------
    # Optional: France impact preview from real inference scores
    # ------------------------------------------------------------------
    preview = preview_open_set_impact(args.preview_scores, args.preview_country)
    if preview:
        print(
            f"\n  preview — '{preview['country']}' top-score distribution "
            f"({preview['references']:,} references):"
        )
        print(f"    quantiles: {preview['top_score_quantiles']}")
        for row in preview["veto_impact"]:
            print(
                f"    veto<{row['veto_min_confidence']:<4} would empty "
                f"{row['vetoed_references']:>8,} references "
                f"({row['vetoed_fraction'] * 100:5.1f}%)"
            )
    else:
        print(
            f"\n  (no '{args.preview_country}' score preview available — run main_l11.py, "
            f"then re-run with --preview-scores)"
        )

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    grid = sorted(tuning["grid"], key=lambda row: row["macro_f05"], reverse=True)
    payload = {
        "chosen": {
            "open_set_boost": best["open_set_boost"],
            "veto_min_confidence": best["veto_min_confidence"],
            "pseudo_open_macro_f05": best["macro_f05"],
            "default_macro_f05": default["macro_f05"],
            "improvement": tuning["improvement"],
        },
        "pseudo_open_country": args.open_country,
        "tuned_thresholds": {
            "tau_match": seen_tuned["tau_match"],
            "tau_s": seen_tuned["tau_s"],
            "margin": args.margin,
            "macro_f05_seen": seen_tuned["macro_f05"],
        },
        "scores": str(scores_path),
        "variant": args.variant,
        "split": args.split,
        "grid_top": grid[:20],
        "grid_size": len(grid),
        "preview": preview,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    out_path = Path(args.out)
    save_open_set_policy(out_path, payload)
    run_key = f"{args.split}_{args.variant}_{args.open_country}"
    merged = merge_json_report(PATH_OUTPUT_DIR / "l10_open_set_report.json", run_key, payload)

    lines = [
        "# Layer 10 Open-Set Policy",
        "",
        "The France veto cannot be validated directly (France has no training ground",
        "truth), so it is tuned on a *pseudo-open* country that does.",
        "",
        "| run | pseudo-open country | boost | veto | tuned F0.5 | default F0.5 | gain |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, run in sorted((merged.get("runs") or {}).items()):
        chosen = run.get("chosen") or {}
        lines.append(
            f"| `{key}`{' (latest)' if key == merged.get('latest_run') else ''} | "
            f"`{run.get('pseudo_open_country')}` | {chosen.get('open_set_boost')} | "
            f"{chosen.get('veto_min_confidence')} | "
            f"{chosen.get('pseudo_open_macro_f05', 0):.4f} | "
            f"{chosen.get('default_macro_f05', 0):.4f} | "
            f"{chosen.get('improvement', 0):+.4f} |"
        )
    lines.extend(
        [
            "",
            f"## Adopted policy: boost=`{best['open_set_boost']}`, "
            f"veto_min_confidence=`{best['veto_min_confidence']}`",
            "",
            f"- tuned on: `{args.open_country}` (held out as unseen), variant `{args.variant}`",
            f"- thresholds from the seen countries: `tau_match={seen_tuned['tau_match']}`, "
            f"`tau_s={seen_tuned['tau_s']}`",
            "- Layers 10 and 11 load this file automatically; CLI flags still override it.",
        ]
    )
    if preview:
        lines.extend(["", f"## `{preview['country']}` veto impact preview", ""])
        lines.append(f"- references: `{preview['references']:,}`")
        lines.append(f"- top-score quantiles: `{preview['top_score_quantiles']}`")
        for row in preview["veto_impact"]:
            lines.append(
                f"- veto<{row['veto_min_confidence']}: empties "
                f"`{row['vetoed_references']:,}` ({row['vetoed_fraction'] * 100:.1f}%)"
            )
    write_text(PATH_OUTPUT_DIR / "l10_open_set_report.md", "\n".join(lines) + "\n")

    print(f"\n  saved policy: {out_path}")
    print(f"  saved report: {PATH_OUTPUT_DIR / 'l10_open_set_report.json'} (history preserved)")
    print("=" * 65)
    print("🌟 OPEN-SET POLICY TUNING COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
