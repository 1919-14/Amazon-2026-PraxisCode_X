"""Layer 10: decision engine — thresholding with singleton and open-set protection."""

from l10_decision.decision import (
    apply_decision_rule,
    build_reference_ids,
    group_scores,
    tune_thresholds,
    write_id_list_tsv,
)

__all__ = [
    "apply_decision_rule",
    "build_reference_ids",
    "group_scores",
    "tune_thresholds",
    "write_id_list_tsv",
]
