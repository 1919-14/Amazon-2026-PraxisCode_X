"""L10b: decision v2 - per-country thresholds + entity-level expected-F0.5.

The v1 rule is a *global* cut: keep candidates within ``margin`` of the top once
the top clears ``tau_s``. Macro F0.5 is scored per Source-1 entity, so the right
question is per entity: which subset of my candidates maximises that entity's
expected F0.5? This module answers it directly from calibrated probabilities.

For one entity with candidate probabilities ``p1 >= p2 >= ...``:

  * keeping the top ``k`` gives ``E[precision] = mean(p[:k])``;
  * with the entity's true-match count estimated as ``T = sum(p) / recall_hat``,
    ``E[recall] = sum(p[:k]) / T``;
  * ``E[F0.5] = 1.25 * E[P] * E[R] / (0.25 * E[P] + E[R])``;
  * predicting empty (a singleton) scores ``prod(1 - p)``.

The layer picks the best of ``{empty, top-1, ..., top-n}``. ``recall_hat`` is
tuned globally and then per country, so seen countries (US/India) get their own
operating point and the open-set country (France) is tuned by the pseudo-open
tuner / fallback veto, exactly as in v1.

The expected-F0.5 rule is *applied* at both tuning and inference time from the
same function, so there is no train/serve reimplementation gap.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

try:  # pragma: no cover - path fallback when run outside the package
    from src.l1_validation.metrics import evaluate_with_thresholds
except ImportError:  # pragma: no cover
    from l1_validation.metrics import evaluate_with_thresholds

DEFAULT_RECALL_HAT: float = 0.80


def _normalize(cands: Iterable[tuple[str, float]]) -> list[tuple[str, float]]:
    """Deduplicate a reference's candidates (max score) and sort descending."""
    best: dict[str, float] = {}
    for cand, score in cands:
        value = float(score)
        if value > best.get(cand, float("-inf")):
            best[cand] = value
    return sorted(best.items(), key=lambda item: (-item[1], item[0]))


def entity_expected_f05(
    ranked: Sequence[tuple[str, float]],
    recall_hat: float = DEFAULT_RECALL_HAT,
    tau_floor: float = 0.0,
) -> list[str]:
    """Choose the candidate subset maximising this entity's expected F0.5.

    Args:
        ranked: ``(candidate_id, probability)`` pairs, best first.
        recall_hat: estimated fraction of an entity's true matches present in its
            candidate list (``sum(p) / recall_hat`` estimates the true count).
        tau_floor: candidates below this probability are never kept.

    Returns:
        The chosen candidate ids (possibly empty for a predicted singleton).
    """
    if not ranked:
        return []

    probs = [max(0.0, min(1.0, p)) for _, p in ranked]
    total = sum(probs)
    if total <= 0.0:
        return []

    recall_hat = min(max(recall_hat, 1e-3), 1.0)
    t_est = max(total / recall_hat, 1e-9)

    # Empty (singleton) score: probability all candidates are false.
    e_empty = 1.0
    for p in probs:
        e_empty *= 1.0 - p

    cumsum = 0.0
    best_k = 0
    best_score = e_empty

    for k, p in enumerate(probs, start=1):
        cumsum += p
        e_p = cumsum / k
        e_r = min(cumsum / t_est, 1.0)
        e_f = (1.25 * e_p * e_r) / (0.25 * e_p + e_r + 1e-12)
        if e_f > best_score:
            best_score = e_f
            best_k = k

    if best_k == 0:
        return []

    kept = [cand for cand, p in ranked[:best_k] if p >= tau_floor]
    # Never keep the k-th candidate while dropping an earlier one: filtering by
    # tau_floor can only shorten the prefix, which stays a valid set.
    return kept


def apply_decision_v2(
    s1_to_cands: Mapping[str, Iterable[tuple[str, float]]],
    reference_ids: Iterable[str],
    recall_hat: float = DEFAULT_RECALL_HAT,
    per_country: Optional[Mapping[str, float]] = None,
    country_by_ref: Optional[Mapping[str, str]] = None,
    tau_floor: float = 0.0,
    open_set_ids: Iterable[str] = (),
    open_set_recall_hat: Optional[float] = None,
    veto_min_confidence: float = 0.0,
) -> dict[str, list[str]]:
    """Entity-level expected-F0.5 decision for every reference.

    Args:
        s1_to_cands: candidate ``(id, prob)`` pairs per reference.
        reference_ids: every reference that must receive a prediction row.
        recall_hat: global fallback true-count estimator.
        per_country: optional ``country -> recall_hat`` overrides.
        country_by_ref: ``reference -> country`` (drives ``per_country``).
        tau_floor: probability floor below which a candidate is dropped.
        open_set_ids: references whose country is unseen in training.
        open_set_recall_hat: recall_hat override for open-set references.
        veto_min_confidence: open-set references below this top score are vetoed.

    Returns:
        ``{s1_id: [cand_id, ...]}``; references with nothing kept map to ``[]``.
    """
    per_country = per_country or {}
    country_by_ref = country_by_ref or {}
    open_set = set(open_set_ids)
    predictions: dict[str, list[str]] = {}

    for s1_id in reference_ids:
        cands = s1_to_cands.get(s1_id)
        if not cands:
            predictions[s1_id] = []
            continue

        ranked = _normalize(cands)
        top_score = ranked[0][1] if ranked else 0.0

        if s1_id in open_set:
            if top_score < veto_min_confidence:
                predictions[s1_id] = []
                continue
            rh = open_set_recall_hat if open_set_recall_hat is not None else recall_hat
        else:
            country = country_by_ref.get(s1_id)
            rh = float(per_country.get(country, recall_hat))

        predictions[s1_id] = entity_expected_f05(ranked, recall_hat=rh, tau_floor=tau_floor)

    return predictions


def tune_decision_v2(
    scores: Mapping[tuple[str, str], float],
    gt_map: Mapping[str, set[str]],
    reference_ids: Optional[Iterable[str]] = None,
    country_by_ref: Optional[Mapping[str, str]] = None,
    recall_hats: Sequence[float] = (0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.0),
    tau_floors: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.5),
    open_set_ids: Iterable[str] = (),
) -> dict:
    """Tune the global + per-country ``recall_hat`` on out-of-fold scores.

    Scored against the real ground truth with the official macro-F0.5 formula, so
    candidate-recall losses count as false negatives (the true competition
    metric). Returns the best parameter set plus the score it achieved.
    """
    from l1_validation.metrics import macro_f05

    refs = list(reference_ids) if reference_ids is not None else sorted(
        {s1 for s1, _ in scores}
    )
    grouped: dict[str, list[tuple[str, float]]] = {}
    for (s1_id, cand_id), prob in scores.items():
        grouped.setdefault(s1_id, []).append((cand_id, float(prob)))
    gt_only = {s1: set(gt_map.get(s1, set())) for s1 in refs}
    country_by_ref = country_by_ref or {}
    open_set = set(open_set_ids)

    def evaluate(recall_hat: float, per_country: dict[str, float], tau_floor: float) -> float:
        preds = apply_decision_v2(
            grouped,
            refs,
            recall_hat=recall_hat,
            per_country=per_country,
            country_by_ref=country_by_ref,
            tau_floor=tau_floor,
            open_set_ids=open_set,
            open_set_recall_hat=recall_hat,
        )
        return macro_f05(preds, gt_only)["macro_f05"]

    best = {"macro_f05": -1.0, "recall_hat": DEFAULT_RECALL_HAT, "tau_floor": 0.0, "per_country": {}}
    for tau_floor in tau_floors:
        for rh in recall_hats:
            score = evaluate(rh, {}, tau_floor)
            if score > best["macro_f05"]:
                best = {
                    "macro_f05": score,
                    "recall_hat": rh,
                    "tau_floor": tau_floor,
                    "per_country": {},
                }

    # Per-country refinement: hold the global recall_hat, tune each country's own.
    global_rh = best["recall_hat"]
    tau_floor = best["tau_floor"]
    per_country: dict[str, float] = {}
    country_scores: dict[str, float] = {}
    for country in sorted(set(country_by_ref.values())):
        if any(country_by_ref.get(r) == country for r in open_set):
            continue
        best_country = (global_rh, best["macro_f05"])
        for rh in recall_hats:
            trial = dict(per_country)
            trial[country] = rh
            score = evaluate(global_rh, trial, tau_floor)
            if score > best_country[1]:
                best_country = (rh, score)
        per_country[country] = best_country[0]
        country_scores[country] = best_country[1]

    final_score = evaluate(global_rh, per_country, tau_floor)
    return {
        "macro_f05": final_score,
        "recall_hat": global_rh,
        "tau_floor": tau_floor,
        "per_country": per_country,
        "country_macro": country_scores,
    }


def compare_with_v1(
    scores: Mapping[tuple[str, str], float],
    gt_map: Mapping[str, set[str]],
    tau_match_grid: Iterable[float] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    tau_s_grid: Iterable[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
    margin: float = 0.05,
) -> dict:
    """Best v1 (global threshold) score for the same score table, for comparison."""
    best = -1.0
    best_cfg = None
    for tau_match in tau_match_grid:
        for tau_s in tau_s_grid:
            result = evaluate_with_thresholds(scores, gt_map, tau_match, tau_s, margin)
            if result["macro_f05"] > best:
                best = result["macro_f05"]
                best_cfg = {"tau_match": tau_match, "tau_s": tau_s, "margin": margin}
    return {"macro_f05": best, "config": best_cfg}
