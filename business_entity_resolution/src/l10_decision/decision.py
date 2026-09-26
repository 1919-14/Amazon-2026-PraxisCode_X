"""L10: decision engine — thresholding with singleton protection.

The official scorer treats every Source-1 entity independently:

  * if the highest candidate score is below the **singleton threshold**
    ``tau_s``, predict the empty list (a singleton / no-match entity);
  * otherwise keep every candidate whose score is at least
    ``max(tau_match, top - margin)``.

This module is the production copy of that rule. The L1 scorer's
``evaluate_with_thresholds`` is the *evaluation* copy; having a single explicit
implementation here means the thresholds tuned during validation transfer to
inference without a reimplementation gap.

Open-set fallback veto
----------------------
Entities whose country never appears in the training ground truth (France is
test-only) must be handled more conservatively. For those references the
singleton threshold is raised by ``open_set_boost`` and any top score below
``veto_min_confidence`` is vetoed to empty. Macro F0.5 is precision-weighted, so
avoiding a spurious match on an unseen country is worth far more than the
occasional missed match.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping

from config import TAU_MATCH_GRID, TAU_S_GRID

try:  # pragma: no cover - path fallback when run outside package
    from src.l1_validation.metrics import evaluate_with_thresholds
except ImportError:  # pragma: no cover
    from l1_validation.metrics import evaluate_with_thresholds


def group_scores(scores: Mapping[tuple[str, str], float]) -> dict[str, list[tuple[str, float]]]:
    """Group ``(s1_id, cand_id) -> score`` into ``s1_id -> [(cand_id, score)]``.

    Duplicate candidate ids for a reference are collapsed to their maximum score
    so a reference never emits the same id twice (the scorer rejects dupes).
    """
    grouped: dict[str, dict[str, float]] = {}
    for (s1_id, cand_id), score in scores.items():
        bucket = grouped.setdefault(s1_id, {})
        value = float(score)
        if value > bucket.get(cand_id, float("-inf")):
            bucket[cand_id] = value
    return {s1_id: list(cands.items()) for s1_id, cands in grouped.items()}


def apply_decision_rule(
    s1_to_cands: Mapping[str, Iterable[tuple[str, float]]],
    reference_ids: Iterable[str],
    tau_match: float,
    tau_s: float,
    margin: float = 0.05,
    open_set_ids: Iterable[str] = (),
    open_set_boost: float = 0.0,
    veto_min_confidence: float = 0.0,
) -> dict[str, list[str]]:
    """Apply the tau_match/tau_s decision rule with singleton + open-set guards.

    Args:
        s1_to_cands: candidate ``(id, score)`` pairs per reference.
        reference_ids: every reference that must receive a prediction row.
        tau_match: minimum absolute score for a candidate to count as a match.
        tau_s: singleton threshold on the reference's top candidate score.
        margin: candidates within this distance of the top survive ``tau_match``.
        open_set_ids: references whose country is unseen in training.
        open_set_boost: amount added to ``tau_s`` for open-set references.
        veto_min_confidence: open-set references below this top score are vetoed.

    Returns:
        ``{s1_id: [cand_id, ...]}`` sorted by descending score (ties by id);
        references with no surviving candidate map to an empty list.
    """
    open_set = set(open_set_ids)
    predictions: dict[str, list[str]] = {}

    for s1_id in reference_ids:
        cands = s1_to_cands.get(s1_id)
        if not cands:
            predictions[s1_id] = []
            continue

        ranked = sorted(cands, key=lambda item: (-item[1], item[0]))
        top_score = ranked[0][1]
        is_open = s1_id in open_set

        # Open-set fallback veto: refuse low-confidence guesses on unseen countries.
        if is_open and top_score < veto_min_confidence:
            predictions[s1_id] = []
            continue

        effective_tau_s = tau_s + (open_set_boost if is_open else 0.0)
        if top_score < effective_tau_s:
            predictions[s1_id] = []
            continue

        effective_threshold = max(tau_match, top_score - margin)
        predictions[s1_id] = [cand for cand, score in ranked if score >= effective_threshold]

    return predictions


def tune_thresholds(
    scores: Mapping[tuple[str, str], float],
    gt_map: Mapping[str, set[str]],
    tau_match_grid: Iterable[float] = TAU_MATCH_GRID,
    tau_s_grid: Iterable[float] = TAU_S_GRID,
    margin: float = 0.05,
) -> dict:
    """Jointly grid search ``(tau_match, tau_s)`` for the best macro F0.5.

    Uses the official L1 scorer so the tuned pair is directly comparable to the
    leaderboard metric. Only references present in ``gt_map`` participate.
    """
    best: dict | None = None
    for tau_match in tau_match_grid:
        for tau_s in tau_s_grid:
            result = evaluate_with_thresholds(scores, gt_map, tau_match, tau_s, margin)
            if best is None or result["macro_f05"] > best["macro_f05"]:
                best = result
    if best is None:
        return {"macro_f05": 0.0, "tau_match": 0.0, "tau_s": 0.0, "margin": margin}
    return best


def build_reference_ids(
    score_ids: Iterable[str],
    candidate_ids: Iterable[str] | None = None,
) -> list[str]:
    """Union the references seen in scores and (optionally) the candidate set.

    The submission requires exactly one row per test Source-1 entity, including
    those with no candidates, so the candidate file is the authoritative source
    when available. Returns a sorted list for deterministic output.
    """
    ids = set(score_ids)
    if candidate_ids is not None:
        ids.update(candidate_ids)
    return sorted(ids)


def write_id_list_tsv(
    path: str | Path,
    rows: Mapping[str, Iterable[str]],
    reference_ids: Iterable[str],
    header: tuple[str, str] = ("source1_entity_id", "matched_entity_ids"),
) -> int:
    """Write a ``source1_entity_id <tab> comma-separated ids`` submission TSV.

    One row per reference id, in the given order, empty second column for
    singletons. Returns the number of rows written.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(list(header))
        for s1_id in reference_ids:
            ids = list(rows.get(s1_id, []))
            writer.writerow([s1_id, ",".join(ids)])
            written += 1
    return written
