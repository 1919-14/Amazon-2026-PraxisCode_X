"""Layer 11 entry point: test inference, submission generation & validation.

The end-to-end inference run:

  1. load the L5 ``candidate_pairs.tsv`` (the exact set the matcher scores);
  2. stream the 27 L7 features per pair over reference batches;
  3. score with the L8 booster, optionally corrected by the L9 calibrator;
  4. apply the L10 decision engine (singleton guard + open-set veto);
  5. write ``output/matching_results.tsv`` and ``output/candidate_pairs.tsv``;
  6. run the official ``validate_submission.py``.

No ground truth is needed: thresholds come from the L10 report, then the L8
report, then conservative defaults.

Usage
-----
python business_entity_resolution/src/main_l11.py --variant a
python business_entity_resolution/src/main_l11.py --variant a --max-references 5000 --no-validate
"""

from __future__ import annotations

import os
# Must be set BEFORE lightgbm / numpy are imported to prevent the Windows
# OpenMP DLL conflict (lib_lightgbm.dll vs vcomp.dll) that causes:
# OSError: exception: access violation reading 0x0000000000000000
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LIGHTGBM_NUM_THREADS", "1")

import argparse
import json
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
for _p in (str(SRC_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from config import (
    DATASET_DIR,
    L10_MARGIN,
    L10_OPEN_SET_COUNTRIES,
    L10_OPEN_SET_TAU_BOOST,
    L10_OPEN_SET_VETO_MIN_CONFIDENCE,
    L10_SEEN_COUNTRIES,
    PATH_ARTIFACTS_DIR,
    PATH_OUTPUT_DIR,
    WORKSPACE_ROOT,
)
from l10_decision.open_set import resolve_open_set_policy
from l11_inference.inference import (
    read_candidate_pairs,
    run_inference,
    write_candidates,
    write_predictions,
)
from l11_inference.scorer import load_booster, load_calibrator
from l6_l8_matching.signals import MODE_SIDECAR, MODE_ZEROS, signals_path
from l6_l8_matching.extra_signals import extra_signals_path
from utils.coverage import CoverageError, check_reference_coverage
from utils.reports import load_json_report, merge_json_report, render_markdown_table, write_text


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Layer 11 run."""
    parser = argparse.ArgumentParser(description="Layer 11 test inference")
    parser.add_argument("--variant", choices=["a", "b"], default="a")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument(
        "--candidate-pairs",
        type=str,
        default=str(PATH_OUTPUT_DIR / "candidate_pairs.tsv"),
        help="L5 candidate pairs TSV (input).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Booster path (default: artifacts/models/lgbm_variant_<variant>.txt).",
    )
    parser.add_argument(
        "--model-kind",
        choices=["lgbm", "stack"],
        default="lgbm",
        help="Single LightGBM booster (default) or the GBDT stack ensemble.",
    )
    parser.add_argument(
        "--stack-dir",
        type=str,
        default=None,
        help="Stack directory (default: artifacts/models/stack_<variant>).",
    )
    parser.add_argument(
        "--calibrator",
        type=str,
        default=None,
        help="L9 calibrator JSON (default: artifacts/models/isotonic_<variant>.json).",
    )
    parser.add_argument("--no-calibration", action="store_true", help="Ignore the L9 calibrator.")
    parser.add_argument("--tau-match", type=float, default=None)
    parser.add_argument("--tau-s", type=float, default=None)
    parser.add_argument("--margin", type=float, default=L10_MARGIN)
    parser.add_argument("--seen-countries", nargs="+", default=list(L10_SEEN_COUNTRIES))
    parser.add_argument("--open-set-countries", nargs="+", default=list(L10_OPEN_SET_COUNTRIES))
    parser.add_argument(
        "--open-set-boost",
        type=float,
        default=None,
        help="Override the tuned open-set boost (default: l10_open_set_policy.json, then config).",
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
    parser.add_argument(
        "--signals",
        type=str,
        default=None,
        help="Retrieval-signal sidecar written by L5 (default: derived from --signals-refs).",
    )
    parser.add_argument(
        "--signals-refs",
        choices=["val", "train", "all"],
        default=None,
        help="Run scope of the candidate set (default: all for test, val for train).",
    )
    parser.add_argument(
        "--no-signals",
        action="store_true",
        help="Force retrieval features to 0 (must match how the booster was trained).",
    )
    parser.add_argument(
        "--extra-signals",
        type=str,
        default=None,
        help="Extra-channel sidecar (phonetic/dense); default: derived from --signals-refs.",
    )
    parser.add_argument(
        "--no-extra-signals",
        action="store_true",
        help="Force the extra-channel retrieval features to 0.",
    )
    parser.add_argument(
        "--allow-signal-mismatch",
        action="store_true",
        help="Run even when the booster was trained with signals the sidecar cannot supply.",
    )
    parser.add_argument(
        "--coverage-refs",
        choices=["val", "train", "all"],
        default=None,
        help="Reference scope the candidate file is expected to cover (default: derived).",
    )
    parser.add_argument(
        "--allow-partial-coverage",
        action="store_true",
        help="Submit even when the candidate set misses references of the split.",
    )
    parser.add_argument(
        "--scores-out",
        type=str,
        default=None,
        help="Per-pair score parquet (default: artifacts/scores/<split>_<variant>.parquet).",
    )
    parser.add_argument("--no-scores", action="store_true", help="Skip the per-pair score parquet.")
    parser.add_argument("--ref-batch-size", type=int, default=50_000)
    parser.add_argument("--max-references", type=int, default=None, help="Dev cap (disables full coverage).")
    parser.add_argument("--max-shards", type=int, default=None, help="Dev cap on shards per source.")
    parser.add_argument("--idf-max-shards", type=int, default=None, help="Dev cap for the IDF pass.")
    parser.add_argument(
        "--matching-out",
        type=str,
        default=None,
        help="Output matching TSV (default: output/matching_results[_val].tsv).",
    )
    parser.add_argument(
        "--candidate-out",
        type=str,
        default=None,
        help="Output candidate TSV (default: output/candidate_pairs[_val].tsv).",
    )
    parser.add_argument("--no-validate", action="store_true", help="Skip the official validator.")
    parser.add_argument(
        "--decision-v2",
        action="store_true",
        help="Apply the entity-level expected-F0.5 decision (needs a tuned policy).",
    )
    parser.add_argument(
        "--decision-v2-policy",
        type=str,
        default=None,
        help="Tuned decision-v2 policy JSON (default: output/l10_decision_v2_policy.json).",
    )
    return parser.parse_args()


def default_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    """Resolve model, calibrator, matching and candidate output paths."""
    models_dir = PATH_ARTIFACTS_DIR / "models"
    model_path = Path(args.model) if args.model else models_dir / f"lgbm_variant_{args.variant}.txt"
    calibrator_path = (
        Path(args.calibrator) if args.calibrator else models_dir / f"isotonic_{args.variant}.json"
    )

    suffix = "" if args.split == "test" else "_val"
    matching_path = (
        Path(args.matching_out)
        if args.matching_out
        else PATH_OUTPUT_DIR / f"matching_results{suffix}.tsv"
    )
    candidate_path = (
        Path(args.candidate_out)
        if args.candidate_out
        else PATH_OUTPUT_DIR / f"candidate_pairs{suffix}.tsv"
    )
    return model_path, calibrator_path, matching_path, candidate_path


def resolve_thresholds(args: argparse.Namespace) -> tuple[float, float, str]:
    """Pick (tau_match, tau_s): CLI → L10 report → L8 report → defaults."""
    if args.tau_match is not None and args.tau_s is not None:
        return args.tau_match, args.tau_s, "cli"

    l10_report = PATH_OUTPUT_DIR / "l10_decision_report.json"
    if l10_report.exists():
        try:
            config = json.loads(l10_report.read_text(encoding="utf-8")).get("config", {})
            if config.get("tau_match") is not None and config.get("tau_s") is not None:
                return float(config["tau_match"]), float(config["tau_s"]), "l10_report"
        except (OSError, json.JSONDecodeError):
            pass

    l8_report = PATH_OUTPUT_DIR / "l8_model_report.json"
    if l8_report.exists():
        try:
            entry = json.loads(l8_report.read_text(encoding="utf-8")).get("variants", {}).get(args.variant)
            if entry:
                return float(entry.get("tau_match", 0.5)), float(entry.get("tau_s", 0.3)), "l8_report"
        except (OSError, json.JSONDecodeError):
            pass

    tau_match = args.tau_match if args.tau_match is not None else 0.5
    tau_s = args.tau_s if args.tau_s is not None else 0.3
    return tau_match, tau_s, "default"


def trained_signals_mode(variant: str) -> tuple[str | None, str | None]:
    """Read how the exported booster was trained (from the L8 report)."""
    report = load_json_report(PATH_OUTPUT_DIR / "l8_model_report.json")
    entry = ((report or {}).get("variants") or {}).get(variant) or {}
    return entry.get("signals_mode"), entry.get("signals_source")


def resolve_signals(
    args: argparse.Namespace,
) -> tuple[Path | None, str]:
    """Decide whether to score with the retrieval-signal sidecar.

    The booster was trained with either real signals or zeros; scoring it with the
    other convention is a silent train/serve skew, so a sidecar that the model did
    not see is ignored and a missing sidecar the model *does* need is an error.
    """
    refs = args.signals_refs or ("all" if args.split == "test" else "val")
    candidate_sidecar = Path(args.signals) if args.signals else signals_path(args.split, refs)
    model_mode, model_source = trained_signals_mode(args.variant)

    if args.no_signals:
        print("  signals: disabled by --no-signals (retrieval features = 0)")
        return None, MODE_ZEROS

    if model_mode == MODE_SIDECAR:
        if candidate_sidecar.exists():
            print(f"  signals: sidecar {candidate_sidecar.name} (matches training)")
            return candidate_sidecar, MODE_SIDECAR
        message = (
            f"the booster for variant '{args.variant}' was trained WITH retrieval signals "
            f"({model_source or 'sidecar'}) but no sidecar was found at {candidate_sidecar}. "
            "Scoring without it would feed the model zeros for two features it relies on. "
            "Fix: run main_l5.py for this candidate set, pass --signals <path>, or "
            "pass --allow-signal-mismatch to proceed anyway."
        )
        if args.allow_signal_mismatch:
            print(f"  ⚠️  {message}")
            return None, MODE_ZEROS
        print(f"  ❌ {message}")
        raise SystemExit(2)

    if model_mode == MODE_ZEROS:
        if candidate_sidecar.exists():
            print(
                "  signals: zeros (the booster was trained without signals; the sidecar "
                f"at {candidate_sidecar.name} is ignored to keep train/serve parity)"
            )
        else:
            print("  signals: zeros (matches training)")
        return None, MODE_ZEROS

    # Unknown provenance (e.g. an older L8 report): prefer the sidecar when present,
    # but say so, because the parity guarantee cannot be verified.
    if candidate_sidecar.exists():
        print(
            f"  signals: sidecar {candidate_sidecar.name} — ⚠️  the L8 report records no "
            "signal provenance, so train/serve parity is unverified"
        )
        return candidate_sidecar, MODE_SIDECAR
    print("  signals: zeros — ⚠️  the L8 report records no signal provenance (unverified)")
    return None, MODE_ZEROS


def run_validator(matching_path: Path, candidate_path: Path) -> int:
    """Invoke the official submission validator; return its exit code."""
    validator = WORKSPACE_ROOT / "DATA SET" / "student_resource" / "utils" / "validate_submission.py"
    test_dir = DATASET_DIR / "test"
    if not validator.exists():
        print(f"  ⚠️  validator not found: {validator}")
        return 1
    cmd = [
        sys.executable,
        str(validator),
        "--matching",
        str(matching_path),
        "--candidate",
        str(candidate_path),
        "--test-dir",
        str(test_dir),
    ]
    print("  running validator:", " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd))
    return subprocess.run(cmd).returncode


def main() -> None:
    """Run Layer 11 inference end to end."""
    args = parse_args()
    model_path, calibrator_path, matching_path, candidate_path = default_paths(args)
    candidate_in = Path(args.candidate_pairs)

    print("=" * 65)
    print("🚀 LAYER 11: TEST INFERENCE + SUBMISSION GENERATION")
    print("=" * 65)
    print(f"  variant={args.variant} | split={args.split}")
    print(f"  candidates: {candidate_in}")
    print(f"  model     : {model_path}")

    stack_dir = Path(args.stack_dir) if args.stack_dir else (PATH_ARTIFACTS_DIR / "models" / f"stack_{args.variant}")
    if args.model_kind == "stack":
        if not (stack_dir / "stack_meta.json").exists():
            print(f"  ❌ stack not found: {stack_dir}")
            print("     run main_l8.py --model-kind stack first.")
            raise SystemExit(1)
    elif not model_path.exists():
        print(f"  ❌ model not found: {model_path}")
        print("     run main_l8.py first to train and export the matcher.")
        raise SystemExit(1)
    if not candidate_in.exists():
        print(f"  ❌ candidate pairs not found: {candidate_in}")
        print("     run main_l5.py --split test --refs all first.")
        raise SystemExit(1)

    tau_match, tau_s, source = resolve_thresholds(args)
    print(f"  thresholds: tau_match={tau_match} tau_s={tau_s} margin={args.margin} (from {source})")

    booster = None
    stack_scorer = None
    if args.model_kind == "stack":
        # Lazy import: keeps scikit-learn out of the LightGBM-only inference
        # process (Windows OpenMP access violation otherwise).
        from l6_l8_matching.stack import load_stack

        stack_scorer = load_stack(stack_dir)
        print(f"  model kind: stack ({stack_scorer.bases}) from {stack_dir}")
    else:
        booster = load_booster(model_path)
        print(f"  model kind: lgbm ({model_path.name})")
    calibrator = None
    if not args.no_calibration:
        calibrator = load_calibrator(calibrator_path)
    print(f"  calibrator: {calibrator['mode'] if calibrator else 'none'}")

    open_set_boost, veto_min_confidence, policy_source = resolve_open_set_policy(
        args.open_set_boost,
        args.veto_min_confidence,
        args.open_set_policy,
    )
    print(
        f"  open-set policy  : boost={open_set_boost} veto<{veto_min_confidence} "
        f"(from {policy_source})"
    )
    print(
        "  ℹ️  run main_l10_open_set.py to tune that policy on a pseudo-open country "
        "(France has no ground truth to tune against)."
    )

    signals_file, signals_mode = resolve_signals(args)

    # Extra-channel sidecar (phonetic/dense). Optional; absent -> features stay 0.
    extra_signals_file = None
    if not args.no_signals and not args.no_extra_signals:
        refs_for_signals = args.signals_refs or ("all" if args.split == "test" else "val")
        candidate_extra = (
            Path(args.extra_signals)
            if args.extra_signals
            else extra_signals_path(args.split, refs_for_signals)
        )
        if candidate_extra.exists():
            extra_signals_file = candidate_extra
            print(f"  extra signals: sidecar {candidate_extra.name}")
        else:
            print(f"  extra signals: none at {candidate_extra.name} (phon/dense features = 0)")

    decision_v2_params = None
    if args.decision_v2:
        policy_path = (
            Path(args.decision_v2_policy)
            if args.decision_v2_policy
            else PATH_OUTPUT_DIR / "l10_decision_v2_policy.json"
        )
        if policy_path.exists():
            decision_v2_params = json.loads(policy_path.read_text(encoding="utf-8"))
            print(
                f"  decision v2 policy: {policy_path.name} "
                f"(recall_hat={decision_v2_params.get('recall_hat')}, "
                f"per_country={decision_v2_params.get('per_country')})"
            )
        else:
            print(f"  ⚠️  --decision-v2 set but no policy at {policy_path}; using v1 rule")

    if args.no_scores:
        scores_out = None
    elif args.scores_out:
        scores_out = Path(args.scores_out)
    else:
        scores_out = PATH_ARTIFACTS_DIR / "scores" / f"{args.split}_{args.variant}.parquet"

    # Load candidates once so the map can be re-emitted (completeness-checked).
    refs_file, cand_map = read_candidate_pairs(candidate_in)

    # Refuse to submit a partial candidate set: the references it misses would be
    # written as empty predictions, which is format-valid and scores ~0.
    coverage_refs = args.coverage_refs or ("all" if args.split == "test" else "val")
    coverage = check_reference_coverage(
        args.split,
        coverage_refs,
        refs_file,
        allow_partial=args.allow_partial_coverage,
        remediation="run main_l3/4/5 for every country bucket, then re-run this layer",
    )
    print(
        f"  candidate coverage: {coverage['coverage'] * 100:.2f}% of the {args.split} "
        f"reference set ({coverage['present']:,} references in the candidate file)"
    )

    predictions, stats = run_inference(
        split=args.split,
        candidate_pairs_path=candidate_in,
        booster=booster,
        stack_scorer=stack_scorer,
        calibrator=calibrator,
        tau_match=tau_match,
        tau_s=tau_s,
        margin=args.margin,
        seen_countries=args.seen_countries,
        open_set_countries=args.open_set_countries,
        open_set_boost=open_set_boost,
        veto_min_confidence=veto_min_confidence,
        all_references=True,
        ref_batch_size=args.ref_batch_size,
        max_references=args.max_references,
        max_shards=args.max_shards,
        idf_max_shards=args.idf_max_shards,
        signals_path=signals_file,
        extra_signals=extra_signals_file,
        decision_v2_params=decision_v2_params,
        scores_out=scores_out,
    )

    # Emit with the same (complete) reference ordering as the predictions.
    reference_ids = list(predictions)
    matching_path.parent.mkdir(parents=True, exist_ok=True)
    n_match = write_predictions(matching_path, predictions, reference_ids)
    n_cand = write_candidates(candidate_path, cand_map, reference_ids)

    print("\n" + "=" * 65)
    print("📊 LAYER 11 INFERENCE REPORT")
    print("=" * 65)
    print(f"  references written : {n_match:,}")
    print(f"  candidates written : {n_cand:,}")
    print(f"  pairs scored       : {stats['pairs_scored']:,}")
    print(f"  pairs missing rec. : {stats['pairs_missing_record']:,}")
    print(f"  non-empty (match)  : {stats['non_empty']:,}")
    print(f"  matched pairs      : {stats['matched_pairs']:,}")
    print(f"  open-set refs      : {stats['open_set_references']:,}")
    print(f"  retrieval signals  : {stats['signals_mode']} ({stats['signals_rows']:,} rows)")
    if scores_out is not None:
        print(f"  per-pair scores    : {scores_out} ({stats['scores_written']:,} pairs)")
    print(f"  saved: {matching_path}")
    print(f"  saved: {candidate_path}")

    run_key = f"{args.split}_{args.variant}"
    report = {
        "config": {
            "variant": args.variant,
            "split": args.split,
            "model": str(model_path),
            "calibrator": calibrator["mode"] if calibrator else "none",
            "candidate_pairs": str(candidate_in),
            "tau_match": tau_match,
            "tau_s": tau_s,
            "margin": args.margin,
            "threshold_source": source,
            "open_set_boost": open_set_boost,
            "veto_min_confidence": veto_min_confidence,
            "open_set_policy_source": policy_source,
            "signals_mode": signals_mode,
            "signals_path": str(signals_file) if signals_file else None,
            "coverage_refs": coverage_refs,
            "max_references": args.max_references,
            "run": run_key,
        },
        "stats": stats,
        "coverage": coverage,
        "outputs": {
            "matching": str(matching_path),
            "candidate": str(candidate_path),
            "scores": str(scores_out) if scores_out else None,
        },
    }
    PATH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = PATH_OUTPUT_DIR / "l11_inference_report.json"
    merged = merge_json_report(report_path, run_key, report)
    md_path = PATH_OUTPUT_DIR / "l11_inference_report.md"
    lines = [
        "# Layer 11 Inference Report",
        "",
        "Every run is kept: `runs` in `l11_inference_report.json` holds the full history.",
        "",
        render_markdown_table(
            merged.get("runs", {}),
            {
                "split": "split",
                "variant": "variant",
                "references": "references",
                "coverage": "coverage",
                "signals": "signals_mode",
            },
            lambda key, run: {
                "split": (run.get("config") or {}).get("split"),
                "variant": (run.get("config") or {}).get("variant"),
                "references": (run.get("stats") or {}).get("references"),
                "coverage": (
                    f"{(run.get('coverage') or {}).get('coverage', 0) * 100:.2f}%"
                    if run.get("coverage")
                    else "n/a"
                ),
                "signals": (run.get("config") or {}).get("signals_mode"),
            },
            latest=merged.get("latest_run"),
        ),
        "",
        f"## Latest run: `{run_key}`",
        "",
        f"- thresholds: `tau_match={tau_match}` `tau_s={tau_s}` `margin={args.margin}` (from `{source}`)",
        f"- open-set policy: `boost={open_set_boost}` `veto<{veto_min_confidence}` (from `{policy_source}`)",
        f"- retrieval signals: `{signals_mode}`",
        f"- candidate coverage: `{coverage['coverage'] * 100:.2f}%`",
        f"- non-empty (match): `{stats['non_empty']:,}` | matched pairs: `{stats['matched_pairs']:,}`",
    ]
    write_text(md_path, "\n".join(lines) + "\n")
    print(f"  saved report: {report_path} (history preserved)")
    print(f"  saved report: {md_path}")

    if not args.no_validate:
        code = run_validator(matching_path, candidate_path)
        report["validation_exit_code"] = code
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  validator exit code: {code}")
        if code != 0:
            print("  ⚠️  validator reported issues — see output above")

    print("=" * 65)
    print("🌟 LAYER 11 COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    try:
        main()
    except CoverageError as error:
        print(f"\n❌ COVERAGE GUARD: {error}\n")
        raise SystemExit(2) from None
