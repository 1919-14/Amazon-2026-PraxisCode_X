"""Layer 10 entry point: decision engine with singleton + open-set protection.

Turns candidate scores into the submission decision:

  * **joint threshold search** — ``(tau_match, tau_s)`` are grid searched with the
    official L1 scorer on any reference set that has ground truth;
  * **singleton guard** — a reference whose top candidate score is below
    ``tau_s`` is predicted empty (the scorer rewards correctly-empty singletons);
  * **open-set fallback veto** — references from a country never seen in training
    (France is test-only) get a boosted singleton threshold and a minimum-
    confidence veto, because a spurious match there costs more than a miss.

Writes the submission decision file plus a report:

    output/matching_results.tsv        (test split — the scored deliverable)
    output/matching_results_val.tsv    (train split dev output)
    output/l10_decision_report.json / .md

Scores come from the L9 calibrated OOF table when present, otherwise the L8 OOF
table. At real test time the same script consumes model probabilities over the
L5 ``candidate_pairs.tsv`` (produced by L11 inference).

Usage
-----
# Tune + decide on the validation/train OOF probabilities:
python business_entity_resolution/src/main_l10.py --split train

# Decide with fixed thresholds on a scored candidate set (inference):
python business_entity_resolution/src/main_l10.py --split test \\
    --scores artifacts/scores/test_scores.parquet --candidate-pairs output/candidate_pairs.tsv
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

import pandas as pd

from config import (
    L10_MARGIN,
    L10_OPEN_SET_COUNTRIES,
    L10_OPEN_SET_TAU_BOOST,
    L10_OPEN_SET_VETO_MIN_CONFIDENCE,
    L10_SEEN_COUNTRIES,
    PATH_ARTIFACTS_DIR,
    PATH_OUTPUT_DIR,
    TAU_MATCH_GRID,
    TAU_S_GRID,
)
from l10_decision.decision import (
    apply_decision_rule,
    build_reference_ids,
    group_scores,
    tune_thresholds,
    write_id_list_tsv,
)
from l10_decision.open_set import resolve_open_set_policy
from l10_decision.decision_v2 import (
    DEFAULT_RECALL_HAT,
    apply_decision_v2,
    tune_decision_v2,
)

from main_l3 import load_ground_truth
from main_l6 import load_reference_countries

MATCHING_HEADER = ("source1_entity_id", "matched_entity_ids")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Layer 10 run."""
    parser = argparse.ArgumentParser(description="Layer 10 decision engine")
    parser.add_argument("--variant", choices=["a", "b"], default="a")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument(
        "--scores",
        type=str,
        default=None,
        help="Parquet with s1_id, cand_id, <prob> columns (default: L8/L9 OOF table).",
    )
    parser.add_argument("--prob-col", type=str, default=None, help="Probability column name.")
    parser.add_argument(
        "--use-calibrated",
        action="store_true",
        help="Prefer the L9 calibrated OOF table when it exists.",
    )
    parser.add_argument(
        "--candidate-pairs",
        type=str,
        default=None,
        help="Optional candidate_pairs TSV; defines the full reference coverage.",
    )
    parser.add_argument("--tau-match", type=float, default=None)
    parser.add_argument("--tau-s", type=float, default=None)
    parser.add_argument("--margin", type=float, default=L10_MARGIN)
    parser.add_argument("--tau-match-grid", type=float, nargs="+", default=list(TAU_MATCH_GRID))
    parser.add_argument("--tau-s-grid", type=float, nargs="+", default=list(TAU_S_GRID))
    parser.add_argument("--seen-countries", nargs="+", default=list(L10_SEEN_COUNTRIES))
    parser.add_argument("--open-set-countries", nargs="+", default=list(L10_OPEN_SET_COUNTRIES))
    parser.add_argument(
        "--open-set-boost",
        type=float,
        default=None,
        help="Override the tuned open-set boost (default: tuned policy, then config).",
    )
    parser.add_argument(
        "--veto-min-confidence",
        type=float,
        default=None,
        help="Override the tuned open-set veto floor (default: tuned policy, then config).",
    )
    parser.add_argument(
        "--open-set-policy",
        type=str,
        default=None,
        help="Tuned open-set policy JSON (default: output/l10_open_set_policy.json).",
    )
    parser.add_argument("--models-dir", type=str, default=str(PATH_ARTIFACTS_DIR / "models"))
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output TSV path (default: matching_results[_val].tsv).",
    )
    parser.add_argument("--no-write-output", action="store_true", help="Report only, skip the TSV.")
    parser.add_argument(
        "--decision-v2",
        action="store_true",
        help="Use entity-level expected-F0.5 set selection (per-country).",
    )
    parser.add_argument(
        "--recall-hat",
        type=float,
        default=DEFAULT_RECALL_HAT,
        help="Fallback true-count estimator for entity-level selection (test split).",
    )
    parser.add_argument(
        "--tau-floor",
        type=float,
        default=0.0,
        help="Probability floor for the entity-level rule.",
    )
    parser.add_argument(
        "--recall-hats",
        type=float,
        nargs="+",
        default=[0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.0],
        help="Grid searched on the train split.",
    )
    parser.add_argument(
        "--decision-v2-policy",
        type=str,
        default=None,
        help="Tuned decision-v2 policy JSON (default: output/l10_decision_v2_policy.json).",
    )
    return parser.parse_args()


def resolve_scores_path(args: argparse.Namespace) -> tuple[Path, str]:
    """Pick the score parquet and its probability column.

    Resolution order: explicit ``--scores``; the per-pair score table Layer 11
    writes for the test split (``artifacts/scores/<split>_<variant>.parquet``);
    then the L9 calibrated / L8 raw out-of-fold tables used for training-split runs.
    """
    if args.scores:
        return Path(args.scores), (args.prob_col or "prob")

    inference_scores = PATH_ARTIFACTS_DIR / "scores" / f"{args.split}_{args.variant}.parquet"
    if args.split == "test" and inference_scores.exists():
        return inference_scores, (args.prob_col or "prob")

    models_dir = Path(args.models_dir)
    calibrated = models_dir / f"oof_calibrated_{args.variant}.parquet"
    raw = models_dir / f"oof_{args.variant}.parquet"

    if args.use_calibrated and calibrated.exists():
        return calibrated, (args.prob_col or "calibrated_prob")
    if raw.exists():
        return raw, (args.prob_col or "oof_prob")
    if calibrated.exists():
        return calibrated, (args.prob_col or "calibrated_prob")
    # Default to the raw path so the error message is the useful one.
    return raw, (args.prob_col or "oof_prob")


def load_reference_tsv(path: Path) -> list[str]:
    """Read the reference column of a candidate/results TSV."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return [str(v) for v in df.iloc[:, 0].tolist()]


def thresholds_from_l8_report(variant: str) -> tuple[float, float] | None:
    """Read the tuned thresholds for a variant from the Layer 8 report, if any."""
    path = PATH_OUTPUT_DIR / "l8_model_report.json"
    if not path.exists():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    entry = report.get("variants", {}).get(variant)
    if not entry:
        return None
    return float(entry.get("tau_match", 0.5)), float(entry.get("tau_s", 0.3))


def main() -> None:
    """Run Layer 10 decision making end to end."""
    args = parse_args()
    scores_path, prob_col = resolve_scores_path(args)

    print("=" * 65)
    print("🚀 LAYER 10: DECISION ENGINE + SINGLETON PROTECTION")
    print("=" * 65)
    print(f"  variant={args.variant} | split={args.split} | margin={args.margin}")

    if not scores_path.exists():
        print(f"  ❌ score table not found: {scores_path}")
        print("     run main_l8.py (and optionally main_l9.py) first.")
        raise SystemExit(1)

    df = pd.read_parquet(scores_path, columns=["s1_id", "cand_id", prob_col])
    print(f"  scores: {scores_path.name} ({len(df):,} pairs, prob='{prob_col}')")

    scores = {
        (str(s1), str(cand)): float(prob)
        for s1, cand, prob in zip(df["s1_id"], df["cand_id"], df[prob_col])
    }
    del df

    # Reference coverage: the candidate file is authoritative when supplied.
    candidate_refs = None
    if args.candidate_pairs:
        candidate_refs = load_reference_tsv(Path(args.candidate_pairs))
        print(f"  candidate pairs: {len(candidate_refs):,} references")
    reference_ids = build_reference_ids((s1 for s1, _ in scores), candidate_refs)
    print(f"  references to decide: {len(reference_ids):,}")

    # ------------------------------------------------------------------
    # Thresholds: tune on ground truth, else reuse L8's tuned pair.
    # ------------------------------------------------------------------
    gt_map = load_ground_truth(set(reference_ids)) if args.split == "train" else {}
    tau_match, tau_s = args.tau_match, args.tau_s
    tuned: dict | None = None

    if gt_map:
        tuned = tune_thresholds(
            scores, gt_map, args.tau_match_grid, args.tau_s_grid, args.margin
        )
        print(
            f"  tuned thresholds: tau_match={tuned['tau_match']} tau_s={tuned['tau_s']} "
            f"→ macro F0.5={tuned['macro_f05']:.4f}"
        )
        if tau_match is None:
            tau_match = tuned["tau_match"]
        if tau_s is None:
            tau_s = tuned["tau_s"]
    else:
        print("  no ground truth — using fixed thresholds")

    if tau_match is None or tau_s is None:
        fallback = thresholds_from_l8_report(args.variant)
        if fallback is not None:
            tau_match = tau_match if tau_match is not None else fallback[0]
            tau_s = tau_s if tau_s is not None else fallback[1]
            print(f"  thresholds from L8 report: tau_match={tau_match} tau_s={tau_s}")
        else:
            tau_match = 0.5 if tau_match is None else tau_match
            tau_s = 0.3 if tau_s is None else tau_s
            print(f"  thresholds defaulted: tau_match={tau_match} tau_s={tau_s}")

    # ------------------------------------------------------------------
    # Open-set policy: tuned on a pseudo-open country when available, because
    # France itself has no ground truth to tune against.
    # ------------------------------------------------------------------
    open_set_boost, veto_min_confidence, policy_source = resolve_open_set_policy(
        args.open_set_boost,
        args.veto_min_confidence,
        args.open_set_policy,
    )
    print(
        f"  open-set policy: boost={open_set_boost} veto<{veto_min_confidence} "
        f"(from {policy_source})"
    )

    # ------------------------------------------------------------------
    # Open-set detection: unseen countries get the fallback veto.
    # ------------------------------------------------------------------
    ref_country = load_reference_countries(args.split, set(reference_ids))
    seen = set(args.seen_countries)
    open_set_configured = set(args.open_set_countries)
    open_set_ids = {
        s1_id
        for s1_id in reference_ids
        if ref_country.get(s1_id, "other") in open_set_configured
        or ref_country.get(s1_id, "other") not in seen
    }
    print(
        f"  open-set references (veto): {len(open_set_ids):,} "
        f"(boost={open_set_boost}, veto<{veto_min_confidence})"
    )

    # ------------------------------------------------------------------
    # Apply the decision rule (v1 global thresholds, or v2 entity-level F0.5).
    # ------------------------------------------------------------------
    grouped = group_scores(scores)
    predictions = apply_decision_rule(
        grouped,
        reference_ids,
        tau_match=tau_match,
        tau_s=tau_s,
        margin=args.margin,
        open_set_ids=open_set_ids,
        open_set_boost=open_set_boost,
        veto_min_confidence=veto_min_confidence,
    )

    decision_params = None
    if args.decision_v2:
        if gt_map:
            decision_params = tune_decision_v2(
                scores,
                gt_map,
                reference_ids,
                country_by_ref=ref_country,
                recall_hats=args.recall_hats,
                tau_floors=sorted({0.0, args.tau_floor, 0.1, 0.2, 0.3, 0.5}),
                open_set_ids=open_set_ids,
            )
            v1_score = tuned["macro_f05"] if tuned else None
            print(
                f"  decision v2 tuned: recall_hat={decision_params['recall_hat']} "
                f"tau_floor={decision_params['tau_floor']} "
                f"per_country={decision_params['per_country']} → "
                f"macro F0.5={decision_params['macro_f05']:.4f}"
                + (f" (v1 was {v1_score:.4f})" if v1_score is not None else "")
            )
            policy_path = (
                Path(args.decision_v2_policy)
                if args.decision_v2_policy
                else PATH_OUTPUT_DIR / "l10_decision_v2_policy.json"
            )
            policy_path.parent.mkdir(parents=True, exist_ok=True)
            policy_path.write_text(json.dumps(decision_params, indent=2), encoding="utf-8")
            print(f"  saved decision v2 policy: {policy_path}")
        else:
            policy_path = (
                Path(args.decision_v2_policy)
                if args.decision_v2_policy
                else PATH_OUTPUT_DIR / "l10_decision_v2_policy.json"
            )
            if policy_path.exists():
                decision_params = json.loads(policy_path.read_text(encoding="utf-8"))
                print(f"  decision v2 policy: {policy_path.name} (from the train run)")
            else:
                decision_params = {
                    "recall_hat": args.recall_hat,
                    "tau_floor": args.tau_floor,
                    "per_country": {},
                }
                print(
                    f"  decision v2: no policy on disk; recall_hat={args.recall_hat} "
                    "(run main_l10.py --split train --decision-v2 to tune it)"
                )

        predictions = apply_decision_v2(
            grouped,
            reference_ids,
            recall_hat=float(decision_params.get("recall_hat", args.recall_hat)),
            per_country=decision_params.get("per_country", {}),
            country_by_ref=ref_country,
            tau_floor=float(decision_params.get("tau_floor", args.tau_floor)),
            open_set_ids=open_set_ids,
            open_set_recall_hat=float(decision_params.get("recall_hat", args.recall_hat)),
            veto_min_confidence=veto_min_confidence,
        )

    n_non_empty = sum(1 for ids in predictions.values() if ids)
    n_singletons = len(predictions) - n_non_empty
    n_matched_pairs = sum(len(ids) for ids in predictions.values())

    # Hold-out macro F0.5 for the applied predictions (train split only).
    applied_macro = None
    if gt_map:
        from l1_validation.metrics import macro_f05

        gt_only = {s1: gt_map[s1] for s1 in reference_ids if s1 in gt_map}
        applied_macro = macro_f05(predictions, gt_only)["macro_f05"]

    print("\n" + "=" * 65)
    print("📊 LAYER 10 DECISION REPORT")
    print("=" * 65)
    print(f"  thresholds        : tau_match={tau_match} tau_s={tau_s} margin={args.margin}")
    print(f"  references        : {len(predictions):,}")
    print(f"  non-empty (match) : {n_non_empty:,}")
    print(f"  singletons (empty): {n_singletons:,}")
    print(f"  matched pairs     : {n_matched_pairs:,}")
    if applied_macro is not None:
        print(f"  applied macro F0.5: {applied_macro:.4f}")

    # ------------------------------------------------------------------
    # Output TSV.
    # ------------------------------------------------------------------
    out_path = args.out
    if out_path is None:
        name = "matching_results.tsv" if args.split == "test" else "matching_results_val.tsv"
        out_path = str(PATH_OUTPUT_DIR / name)
    rows_written = 0
    if not args.no_write_output:
        rows_written = write_id_list_tsv(out_path, predictions, reference_ids, MATCHING_HEADER)
        print(f"\n  saved decisions: {out_path} ({rows_written:,} rows)")

    # ------------------------------------------------------------------
    # Report.
    # ------------------------------------------------------------------
    report = {
        "config": {
            "variant": args.variant,
            "split": args.split,
            "scores": str(scores_path),
            "prob_col": prob_col,
            "tau_match": tau_match,
            "tau_s": tau_s,
            "margin": args.margin,
            "tau_match_grid": args.tau_match_grid,
            "tau_s_grid": args.tau_s_grid,
            "seen_countries": args.seen_countries,
            "open_set_countries": args.open_set_countries,
            "open_set_boost": open_set_boost,
            "veto_min_confidence": veto_min_confidence,
            "open_set_policy_source": policy_source,
            "decision_v2": bool(args.decision_v2),
        },
        "tuned": tuned,
        "decision_v2": decision_params,
        "open_set_references": len(open_set_ids),
        "references": len(predictions),
        "non_empty": n_non_empty,
        "singletons": n_singletons,
        "matched_pairs": n_matched_pairs,
        "applied_macro_f05": applied_macro,
        "output": out_path if not args.no_write_output else None,
        "rows_written": rows_written,
    }
    PATH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = PATH_OUTPUT_DIR / "l10_decision_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_path = PATH_OUTPUT_DIR / "l10_decision_report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Layer 10 Decision Report\n\n")
        f.write(f"- variant: `{args.variant}` | split: `{args.split}`\n")
        f.write(f"- thresholds: `tau_match={tau_match}`, `tau_s={tau_s}`, `margin={args.margin}`\n")
        if tuned is not None:
            f.write(
                f"- tuned (joint grid search): macro F0.5 = `{tuned['macro_f05']:.4f}`\n"
            )
        f.write(f"- open-set references (veto): `{len(open_set_ids):,}`\n\n")
        f.write("## Decisions\n\n")
        f.write(f"- references: `{len(predictions):,}`\n")
        f.write(f"- non-empty (match): `{n_non_empty:,}`\n")
        f.write(f"- singletons (empty): `{n_singletons:,}`\n")
        f.write(f"- matched pairs: `{n_matched_pairs:,}`\n")
        if applied_macro is not None:
            f.write(f"- applied macro F0.5: `{applied_macro:.4f}`\n")
        f.write(
            "\nThe singleton guard keeps the empty prediction when the top candidate "
            "score is below `tau_s`; open-set references additionally require a "
            "boosted threshold and a minimum top confidence.\n"
        )

    print(f"  saved report: {report_path}")
    print("=" * 65)
    print("🌟 LAYER 10 COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
