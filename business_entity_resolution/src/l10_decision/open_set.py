"""L10 open-set policy: make the France veto a tuned, evidence-backed rule.

The decision engine treats entities from a country that never appears in the
training ground truth (France) conservatively: the singleton threshold is raised
by ``open_set_boost`` and any top score below ``veto_min_confidence`` is vetoed to
empty. Both numbers used to be hand-picked constants applied to ~15% of the test
set with **no way to check them**: France has no ground truth by construction, so
nothing in the repository could tell whether the veto was protecting precision or
destroying fifteen percent of the score.

This module makes the policy measurable with a *pseudo-open-set* experiment:

  1. take a country that *does* have ground truth (US or India);
  2. tune ``(tau_match, tau_s)`` on the remaining ("seen") countries exactly as
     production does;
  3. hold the chosen country out as if it were unseen, and grid search
     ``(open_set_boost, veto_min_confidence)`` to maximise macro F0.5 *on that
     country's references only* - which is precisely what the policy controls;
  4. persist the winner as ``output/l10_open_set_policy.json``; Layer 10 and
     Layer 11 load it as their default.

The same module also reports the score-quantile table of any open-set country, so
the impact of a candidate threshold on France ("how many entities would this veto
throw away?") is visible before submitting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from config import (
    L10_MARGIN,
    L10_OPEN_SET_TAU_BOOST,
    L10_OPEN_SET_VETO_MIN_CONFIDENCE,
    PATH_OUTPUT_DIR,
)
from l10_decision.decision import apply_decision_rule

try:  # pragma: no cover - path fallback when run outside the package
    from src.l1_validation.metrics import macro_f05
except ImportError:  # pragma: no cover
    from l1_validation.metrics import macro_f05

DEFAULT_QUANTILES: tuple[float, ...] = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
DEFAULT_BOOST_GRID: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
DEFAULT_VETO_GRID: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)

POLICY_FILENAME = "l10_open_set_policy.json"


def policy_path() -> Path:
    """Canonical path of the tuned open-set policy."""
    return PATH_OUTPUT_DIR / POLICY_FILENAME


def top_scores_by_reference(scores: Mapping[tuple[str, str], float]) -> dict[str, float]:
    """Collapse per-pair scores into each reference's best candidate score."""
    top: dict[str, float] = {}
    for (s1_id, _), score in scores.items():
        value = float(score)
        if value > top.get(s1_id, float("-inf")):
            top[s1_id] = value
    return top


def score_quantiles(
    values: Sequence[float],
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> dict[str, float]:
    """Quantiles of a score distribution as ``{"q50": 0.83, ...}``."""
    import numpy as np

    if len(values) == 0:
        return {f"q{int(q * 100):02d}": 0.0 for q in quantiles}
    array = np.asarray(list(values), dtype=np.float64)
    return {f"q{int(q * 100):02d}": float(np.quantile(array, q)) for q in quantiles}


def macro_f05_over_references(
    predictions: Mapping[str, Sequence[str]],
    ground_truth: Mapping[str, set[str]],
    reference_ids: Iterable[str],
) -> float:
    """Macro F0.5 restricted to a subset of references (e.g. only the open country)."""
    subset = {s1_id: ground_truth[s1_id] for s1_id in reference_ids if s1_id in ground_truth}
    if not subset:
        return 0.0
    return float(macro_f05({k: list(v) for k, v in predictions.items()}, subset)["macro_f05"])


def evaluate_open_set_policy(
    scores: Mapping[tuple[str, str], float],
    ground_truth: Mapping[str, set[str]],
    reference_ids: Sequence[str],
    *,
    tau_match: float,
    tau_s: float,
    margin: float = L10_MARGIN,
    open_set_boost: float = L10_OPEN_SET_TAU_BOOST,
    veto_min_confidence: float = L10_OPEN_SET_VETO_MIN_CONFIDENCE,
    treated_as_open: bool = True,
) -> dict:
    """Score one ``(boost, veto)`` policy on a reference set treated as open-set."""
    grouped: dict[str, list[tuple[str, float]]] = {}
    for (s1_id, cand_id), score in scores.items():
        grouped.setdefault(s1_id, []).append((cand_id, float(score)))

    predictions = apply_decision_rule(
        grouped,
        reference_ids,
        tau_match=tau_match,
        tau_s=tau_s,
        margin=margin,
        open_set_ids=reference_ids if treated_as_open else (),
        open_set_boost=open_set_boost if treated_as_open else 0.0,
        veto_min_confidence=veto_min_confidence if treated_as_open else 0.0,
    )

    n_empty = sum(1 for ids in predictions.values() if not ids)
    return {
        "macro_f05": macro_f05_over_references(predictions, ground_truth, reference_ids),
        "open_set_boost": float(open_set_boost),
        "veto_min_confidence": float(veto_min_confidence),
        "references": len(reference_ids),
        "predicted_empty": n_empty,
        "matched_pairs": sum(len(ids) for ids in predictions.values()),
    }


def tune_open_set_policy(
    scores: Mapping[tuple[str, str], float],
    ground_truth: Mapping[str, set[str]],
    open_reference_ids: Sequence[str],
    *,
    tau_match: float,
    tau_s: float,
    margin: float = L10_MARGIN,
    boost_grid: Sequence[float] = DEFAULT_BOOST_GRID,
    veto_grid: Sequence[float] = DEFAULT_VETO_GRID,
) -> dict:
    """Grid search ``(open_set_boost, veto_min_confidence)`` on a pseudo-open country.

    Both parameters only affect open-set references, so maximising macro F0.5 over
    the held-out country's references is exactly the objective the France policy
    optimises - and it is measurable, unlike the real France set.

    Returns:
        ``{"best": {...}, "default": {...}, "grid": [...], "improvement": float}``
    """
    default = evaluate_open_set_policy(
        scores,
        ground_truth,
        open_reference_ids,
        tau_match=tau_match,
        tau_s=tau_s,
        margin=margin,
        open_set_boost=L10_OPEN_SET_TAU_BOOST,
        veto_min_confidence=L10_OPEN_SET_VETO_MIN_CONFIDENCE,
    )

    grid: list[dict] = []
    best: dict | None = None
    for boost in boost_grid:
        for veto in veto_grid:
            result = evaluate_open_set_policy(
                scores,
                ground_truth,
                open_reference_ids,
                tau_match=tau_match,
                tau_s=tau_s,
                margin=margin,
                open_set_boost=boost,
                veto_min_confidence=veto,
            )
            grid.append(result)
            if best is None or result["macro_f05"] > best["macro_f05"]:
                best = result

    assert best is not None
    return {
        "best": best,
        "default": default,
        "grid": grid,
        "improvement": best["macro_f05"] - default["macro_f05"],
    }


def veto_impact_preview(
    top_scores: Mapping[str, float],
    thresholds: Sequence[float] = DEFAULT_VETO_GRID,
) -> list[dict]:
    """How many references each candidate veto threshold would force to empty."""
    scores = list(top_scores.values())
    n = len(scores)
    return [
        {
            "veto_min_confidence": float(threshold),
            "vetoed_references": int(sum(1 for value in scores if value < threshold)),
            "vetoed_fraction": (sum(1 for value in scores if value < threshold) / n) if n else 0.0,
        }
        for threshold in thresholds
    ]


def save_open_set_policy(path: str | Path, payload: Mapping) -> Path:
    """Persist the tuned policy (and the evidence behind it)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")
    return out_path


def load_open_set_policy(path: str | Path | None = None) -> Optional[dict]:
    """Load a tuned policy as ``{"boost": float, "veto_min_confidence": float, ...}``."""
    policy_file = Path(path) if path is not None else policy_path()
    if not policy_file.exists():
        return None
    try:
        payload = json.loads(policy_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    chosen = payload.get("chosen") or {}
    if "open_set_boost" not in chosen:
        return None
    return {
        "boost": float(chosen["open_set_boost"]),
        "veto_min_confidence": float(chosen["veto_min_confidence"]),
        "pseudo_open_country": payload.get("pseudo_open_country"),
        "path": str(policy_file),
    }


def resolve_open_set_policy(
    cli_boost: Optional[float] = None,
    cli_veto: Optional[float] = None,
    path: str | Path | None = None,
) -> tuple[float, float, str]:
    """Resolve ``(boost, veto)``: CLI -> tuned policy file -> config defaults."""
    tuned = load_open_set_policy(path)
    boost, veto, source = L10_OPEN_SET_TAU_BOOST, L10_OPEN_SET_VETO_MIN_CONFIDENCE, "config_default"
    if tuned is not None:
        boost, veto, source = tuned["boost"], tuned["veto_min_confidence"], "tuned_policy"
    if cli_boost is not None:
        boost, source = cli_boost, "cli"
    if cli_veto is not None:
        veto, source = cli_veto, "cli"
    return boost, veto, source
