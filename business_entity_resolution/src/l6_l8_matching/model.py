"""L8: LightGBM matcher training with grouped out-of-fold probabilities.

Pairs must be split by *reference entity* (``s1_id``): pairs sharing an S1
entity are highly correlated, so a plain row split would leak and inflate the
validation score. ``GroupKFold`` on ``s1_id`` prevents that.

Outputs from :func:`train_oof`:
* ``oof`` - out-of-fold positive-class probability per pair (leakage-free),
  suitable for calibrating the decision threshold without a separate holdout.
* the per-fold models and averaged feature importances.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - dependency pinned in requirements
    raise ImportError("lightgbm is required for Layer 8") from exc

from sklearn.model_selection import GroupKFold

DEFAULT_PARAMS: dict = {
    "objective": "binary",
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbose": -1,
}


def train_oof(
    df: pd.DataFrame,
    feature_names: Sequence[str],
    n_folds: int = 5,
    seed: int = 42,
    params: Optional[dict] = None,
) -> tuple[np.ndarray, list, dict[str, float]]:
    """Train grouped K-fold LightGBM and return out-of-fold probabilities.

    Args:
        df: must contain ``label``, ``s1_id`` and all ``feature_names``.
        feature_names: feature columns in model order.
        n_folds: number of grouped folds (capped by the number of groups).
        seed: random seed.
        params: optional LightGBM parameter overrides.

    Returns:
        ``(oof, models, importances)`` where ``oof`` is aligned with ``df`` rows.
    """
    merged_params = {**DEFAULT_PARAMS, **(params or {}), "random_state": seed}
    x = df[list(feature_names)].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)
    groups = df["s1_id"].to_numpy()

    n_groups = len(np.unique(groups))
    folds = max(2, min(n_folds, n_groups))
    oof = np.zeros(len(df), dtype=np.float64)
    models: list = []
    importances = np.zeros(len(feature_names), dtype=np.float64)
    covered = np.zeros(len(df), dtype=bool)

    splitter = GroupKFold(n_splits=folds)
    for train_idx, val_idx in splitter.split(x, y, groups):
        model = lgb.LGBMClassifier(**merged_params)
        model.fit(x[train_idx], y[train_idx])
        oof[val_idx] = model.predict_proba(x[val_idx])[:, 1]
        covered[val_idx] = True
        models.append(model)
        importances += model.feature_importances_

    # Guard: any row not covered (should not happen) gets its fold-model average.
    if not covered.all():
        fallback = np.mean([m.predict_proba(x[~covered])[:, 1] for m in models], axis=0)
        oof[~covered] = fallback

    importances /= max(1, len(models))
    importance_map = {name: float(value) for name, value in zip(feature_names, importances)}
    return oof, models, importance_map


def train_full(
    df: pd.DataFrame,
    feature_names: Sequence[str],
    seed: int = 42,
    params: Optional[dict] = None,
):
    """Fit a single LightGBM model on all rows (for inference / export)."""
    merged_params = {**DEFAULT_PARAMS, **(params or {}), "random_state": seed}
    x = df[list(feature_names)].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)
    model = lgb.LGBMClassifier(**merged_params)
    model.fit(x, y)
    return model


def top_importances(importance_map: dict[str, float], n: int = 15) -> list[tuple[str, float]]:
    """Return the ``n`` highest-importance features."""
    return sorted(importance_map.items(), key=lambda item: item[1], reverse=True)[:n]
