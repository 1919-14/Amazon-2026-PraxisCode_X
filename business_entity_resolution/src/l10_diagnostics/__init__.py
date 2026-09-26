"""Layer 10.5: diagnostic error analysis and open-set (France) spot-check."""

from l10_diagnostics.diagnostics import (
    aggregate,
    classify_entity,
    error_focus,
    france_spot_check,
    group_metrics,
    length_bucket,
    load_predictions_tsv,
    load_reference_metadata,
    source_composition,
)

__all__ = [
    "aggregate",
    "classify_entity",
    "error_focus",
    "france_spot_check",
    "group_metrics",
    "length_bucket",
    "load_predictions_tsv",
    "load_reference_metadata",
    "source_composition",
]
