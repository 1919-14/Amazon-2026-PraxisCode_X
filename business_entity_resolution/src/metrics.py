"""Official per-reference macro F0.5, including singleton credit."""

from collections.abc import Mapping, Set


def entity_f05(truth: Set[str], predicted: Set[str]) -> float:
    """Score one reference entity, with explicit empty-set behavior."""
    if not truth:
        return float(not predicted)
    true_positives = len(truth & predicted)
    return 5.0 * true_positives / (len(truth) + 4 * len(predicted))


def macro_f05(
    truth: Mapping[str, Set[str]], predicted: Mapping[str, Set[str]]
) -> float:
    """Require identical reference coverage and average entity-level scores."""
    if not truth:
        raise ValueError("Cannot score an empty evaluation set")
    if truth.keys() != predicted.keys():
        raise ValueError("Predictions must cover exactly the evaluation references")
    return sum(entity_f05(matches, predicted[entity_id])
               for entity_id, matches in truth.items()) / len(truth)