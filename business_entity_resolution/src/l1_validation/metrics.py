"""Official per-reference macro F0.5 evaluation, precision/recall metrics, and threshold evaluation."""

from typing import Any


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Calculate F-beta score with precision weighting (beta=0.5 defaults to macro F0.5)."""
    denom = (beta**2 * precision) + recall
    if denom == 0.0:
        return 0.0
    return ((1.0 + beta**2) * precision * recall) / denom


def precision_recall(pred: set[str], gt: set[str]) -> tuple[float, float]:
    """Compute precision and recall between predicted and ground-truth ID sets."""
    if not pred and not gt:
        return 0.0, 0.0
    if not pred or not gt:
        return 0.0, 0.0
    tp = len(pred & gt)
    precision = tp / len(pred)
    recall = tp / len(gt)
    return precision, recall


def macro_f05(
    predictions: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
) -> dict[str, Any]:
    """Compute macro-averaged F0.5 score across all reference entities with explicit singleton scoring."""
    n_entities = len(ground_truth)
    if n_entities == 0:
        return {
            "macro_f05": 0.0,
            "per_entity": {},
            "n_entities": 0,
            "n_singletons": 0,
            "singleton_f05": 0.0,
            "match_f05": 0.0,
        }

    per_entity: dict[str, float] = {}
    singleton_scores: list[float] = []
    match_scores: list[float] = []

    for s1_id, gt in ground_truth.items():
        pred = set(predictions.get(s1_id, []))
        if not gt and not pred:
            score = 1.0
            singleton_scores.append(score)
        elif not gt and pred:
            score = 0.0
            singleton_scores.append(score)
        elif gt and not pred:
            score = 0.0
            match_scores.append(score)
        else:
            tp = len(pred & gt)
            prec = tp / len(pred)
            rec = tp / len(gt)
            score = f_beta(prec, rec, 0.5)
            match_scores.append(score)
        per_entity[s1_id] = score

    macro = sum(per_entity.values()) / n_entities
    n_singletons = len(singleton_scores)
    singleton_f05 = (sum(singleton_scores) / n_singletons) if n_singletons > 0 else 0.0
    n_matches = len(match_scores)
    match_f05 = (sum(match_scores) / n_matches) if n_matches > 0 else 0.0

    return {
        "macro_f05": float(macro),
        "per_entity": per_entity,
        "n_entities": n_entities,
        "n_singletons": n_singletons,
        "singleton_f05": float(singleton_f05),
        "match_f05": float(match_f05),
    }


def evaluate_with_thresholds(
    scores: dict[tuple[str, str], float],
    ground_truth: dict[str, set[str]],
    tau_match: float,
    tau_s: float,
    margin: float = 0.05,
) -> dict[str, Any]:
    """Evaluate candidate pair probabilities with singleton thresholding and adaptive top-margin rule."""
    # Group pairwise scores by S1 entity ID for fast lookup
    s1_to_cands: dict[str, list[tuple[str, float]]] = {}
    for (s1_id, cand_id), score in scores.items():
        s1_to_cands.setdefault(s1_id, []).append((cand_id, float(score)))

    predictions: dict[str, list[str]] = {}
    for s1_id in ground_truth:
        cand_scores = s1_to_cands.get(s1_id, [])
        if not cand_scores:
            predictions[s1_id] = []
            continue

        max_score = max(score for _, score in cand_scores)
        if max_score < tau_s:
            # Singleton path: maximum confidence is below singleton cutoff
            predictions[s1_id] = []
        else:
            # Match path: filter candidates above tau_match and within margin of top candidate
            effective_threshold = max(tau_match, max_score - margin)
            kept = [(cand, sc) for cand, sc in cand_scores if sc >= effective_threshold]
            kept.sort(key=lambda x: x[1], reverse=True)
            predictions[s1_id] = [cand for cand, _ in kept]

    eval_result = macro_f05(predictions, ground_truth)
    eval_result["tau_match"] = tau_match
    eval_result["tau_s"] = tau_s
    eval_result["margin"] = margin
    return eval_result
