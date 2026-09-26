"""L9: conditional isotonic calibration and reliability metrics.

Isotonic regression is a **monotonic** map, so it preserves the ranking of the
OOF scores. That means the best macro-F0.5 achievable by tuning a threshold is
unchanged by calibration. Calibration therefore matters for *probability
reliability* (Brier score / expected calibration error), which feeds per-country
decision rules and any downstream stacking - not for raw ranking quality.

Functions:
  * ``ece`` / ``brier`` / ``reliability_curve`` - calibration diagnostics.
  * ``fit_isotonic`` / ``apply_isotonic`` - calibrator fit/apply.
  * ``calibrated_oof`` - grouped cross-validated calibration of OOF scores so the
    calibrated values we *report* are not fit on the same rows they score.
  * ``serialize_isotonic`` / ``deserialize_isotonic`` - persist the step function
    for inference (a list of knots is enough).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

try:
    from sklearn.isotonic import IsotonicRegression
except ImportError as exc:  # pragma: no cover - dependency pinned in requirements
    raise ImportError("scikit-learn is required for Layer 9") from exc

EPS = 1e-12


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier score (mean squared error of the probabilities). Lower is better."""
    return float(np.mean((np.asarray(probs) - np.asarray(labels)) ** 2))


def log_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    """Binary log-loss with clipping. Lower is better."""
    p = np.clip(np.asarray(probs), EPS, 1.0 - EPS)
    y = np.asarray(labels)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> list[tuple[float, float, int]]:
    """Return ``(mean_confidence, mean_accuracy, count)`` per probability bin."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs, edges[1:-1])
    curve: list[tuple[float, float, int]] = []
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            continue
        curve.append((float(probs[mask].mean()), float(labels[mask].mean()), count))
    return curve


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (count-weighted mean |confidence - accuracy|)."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    n = len(probs)
    if n == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs, edges[1:-1])
    total = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            continue
        total += (count / n) * abs(float(probs[mask].mean()) - float(labels[mask].mean()))
    return float(total)


def fit_isotonic(probs: Sequence[float], labels: Sequence[int]) -> IsotonicRegression:
    """Fit a monotonic calibrator mapping raw scores to calibrated probabilities."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(probs, dtype=np.float64), np.asarray(labels, dtype=np.int32))
    return iso


def apply_isotonic(iso: IsotonicRegression, probs: Sequence[float]) -> np.ndarray:
    """Apply a fitted calibrator to new scores."""
    return iso.predict(np.asarray(probs, dtype=np.float64))


def calibrate_grouped(
    probs: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Cross-validated calibration: fit on out-of-fold groups, never on own rows.

    Args:
        probs: raw OOF probabilities.
        labels: binary labels.
        groups: group id per row (``s1_id``) used to build folds.

    Returns:
        Calibrated probabilities aligned with the input rows.
    """
    from sklearn.model_selection import GroupKFold

    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    groups = np.asarray(groups)
    calibrated = np.zeros(len(probs), dtype=np.float64)

    n_groups = len(np.unique(groups))
    folds = max(2, min(n_folds, n_groups))
    if n_groups < 2:
        return probs.copy()

    for train_idx, val_idx in GroupKFold(n_splits=folds).split(probs, labels, groups):
        iso = fit_isotonic(probs[train_idx], labels[train_idx])
        calibrated[val_idx] = apply_isotonic(iso, probs[val_idx])
    return calibrated


def serialize_isotonic(iso: Optional[IsotonicRegression]) -> Optional[dict]:
    """Persist the calibrator's step-function knots (or ``None``)."""
    if iso is None:
        return None
    return {
        "x_thresholds": [float(v) for v in iso.X_thresholds_],
        "y_thresholds": [float(v) for v in iso.y_thresholds_],
    }


def apply_serialized(knots: dict, probs: Sequence[float]) -> np.ndarray:
    """Apply a serialized calibrator by linear interpolation between knots."""
    x = np.asarray(knots["x_thresholds"], dtype=np.float64)
    y = np.asarray(knots["y_thresholds"], dtype=np.float64)
    return np.interp(np.asarray(probs, dtype=np.float64), x, y, left=y[0], right=y[-1])
