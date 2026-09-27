"""L8b: GBDT stack (LightGBM + CatBoost + XGBoost) with a meta-learner.

The single LightGBM matcher captures ~89% of the candidate-set ceiling. A stack
of three gradient-boosted trees with different inductive biases (leaf-wise
LightGBM, symmetric CatBoost, level-wise XGBoost) plus a logistic-regression
meta-learner over their out-of-fold probabilities closes part of that gap, which
matters because macro F0.5 is precision-weighted and rewards a sharper ranking.

Leakage control
---------------
Every split is grouped by ``s1_id`` (pairs sharing a reference entity are highly
correlated), both for the base models and for the meta-learner. The meta-learner
is trained on *out-of-fold* base probabilities only, so the OOF column it sees was
never produced by a model that saw those rows.

Inference
---------
Base models are kept per fold and averaged at scoring time (fold bagging), which
is more stable than a single full-data refit and needs no extra training pass.
:class:`StackScorer` loads the serialized stack and reproduces the OOF pipeline
exactly. Missing optional libraries (catboost/xgboost) are skipped gracefully, so
a LightGBM-only environment still trains and scores a one-model stack.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

# LightGBM before scikit-learn: Windows OpenMP import-order hazard.
import lightgbm as lgb  # noqa: F401  (import order is deliberate)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

DEFAULT_BASES: tuple[str, ...] = ("lgbm", "catboost", "xgboost")

LGBM_PARAMS: dict = {
    "objective": "binary",
    "n_estimators": 500,
    "learning_rate": 0.04,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbose": -1,
}

CATBOOST_PARAMS: dict = {
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "loss_function": "Logloss",
    "verbose": 100,
    "allow_writing_files": False,
}

XGBOOST_PARAMS: dict = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "eval_metric": "logloss",
    "n_jobs": -1,
}


def available_bases(requested: Sequence[str]) -> list[str]:
    """Filter the requested base learners to the ones importable here."""
    usable: list[str] = ["lgbm"]  # lightgbm is a hard dependency of this module
    if "catboost" in requested:
        try:
            import catboost  # noqa: F401
            usable.append("catboost")
        except ImportError:
            pass
    if "xgboost" in requested:
        try:
            import xgboost  # noqa: F401
            usable.append("xgboost")
        except ImportError:
            pass
    return usable


def _fit_predict_base(
    base: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    seed: int,
    fold: int = 1,
    total_folds: int = 5,
):
    """Fit one base learner; return its validation probabilities and model."""
    import time
    t0 = time.time()
    print(f"    [{base.upper()}] Fold {fold}/{total_folds} starting ({len(x_train):,} train, {len(x_val):,} val)...", flush=True)
    if base == "lgbm":
        model = lgb.LGBMClassifier(**{**LGBM_PARAMS, "random_state": seed})
        model.fit(x_train, y_train)
        print(f"    [{base.upper()}] Fold {fold}/{total_folds} complete in {time.time()-t0:.1f}s", flush=True)
        return model.predict_proba(x_val)[:, 1], model
    if base == "catboost":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(**{**CATBOOST_PARAMS, "random_seed": seed})
        model.fit(x_train, y_train, verbose=100)
        print(f"    [{base.upper()}] Fold {fold}/{total_folds} complete in {time.time()-t0:.1f}s", flush=True)
        return model.predict_proba(x_val)[:, 1], model
    if base == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(**{**XGBOOST_PARAMS, "random_state": seed})
        model.fit(x_train, y_train)
        print(f"    [{base.upper()}] Fold {fold}/{total_folds} complete in {time.time()-t0:.1f}s", flush=True)
        return model.predict_proba(x_val)[:, 1], model
    raise ValueError(f"unknown base learner: {base}")


@dataclass
class StackResult:
    """OOF probabilities plus the fitted meta-learner and per-fold base models."""

    feature_names: list[str]
    bases: list[str]
    oof_prob: np.ndarray  # final (meta) OOF probability, aligned with the input rows
    base_oof: np.ndarray  # (n_rows, n_bases) out-of-fold base probabilities
    meta_coef: np.ndarray
    meta_intercept: float
    meta_mean: np.ndarray
    meta_scale: np.ndarray
    fold_models: dict[str, list] = field(default_factory=dict)
    importances: dict[str, float] = field(default_factory=dict)

    def _base_matrix(self, x: np.ndarray) -> np.ndarray:
        """Average the per-fold base models into an ``(n, n_bases)`` matrix."""
        cols = []
        for base in self.bases:
            models = self.fold_models.get(base) or []
            if not models:
                cols.append(np.zeros(x.shape[0], dtype=np.float64))
                continue
            preds = [m.predict_proba(x)[:, 1] for m in models]
            cols.append(np.mean(preds, axis=0))
        return np.column_stack(cols)

    def _meta(self, base_matrix: np.ndarray) -> np.ndarray:
        z = (base_matrix - self.meta_mean) / self.meta_scale
        logit = z @ self.meta_coef + self.meta_intercept
        return 1.0 / (1.0 + np.exp(-logit))

    def predict_proba(self, df_or_x) -> np.ndarray:
        """Score a dataframe (using ``feature_names``) or a raw feature matrix."""
        if isinstance(df_or_x, pd.DataFrame):
            x = df_or_x[self.feature_names].to_numpy(dtype=np.float32)
        else:
            x = np.asarray(df_or_x, dtype=np.float32)
        return self._meta(self._base_matrix(x))

    def save(self, directory: str | Path) -> Path:
        """Serialize every fold model + the meta-learner; return the directory."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        for base in self.bases:
            for fold, model in enumerate(self.fold_models.get(base, [])):
                if base == "lgbm":
                    model.booster_.save_model(str(out / f"{base}_fold{fold}.txt"))
                elif base == "catboost":
                    model.save_model(str(out / f"{base}_fold{fold}.cbm"))
                elif base == "xgboost":
                    model.save_model(str(out / f"{base}_fold{fold}.json"))
        meta = {
            "feature_names": self.feature_names,
            "bases": self.bases,
            "meta_coef": self.meta_coef.tolist(),
            "meta_intercept": float(self.meta_intercept),
            "meta_mean": self.meta_mean.tolist(),
            "meta_scale": self.meta_scale.tolist(),
            "importances": self.importances,
        }
        (out / "stack_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return out


def train_stack(
    df: pd.DataFrame,
    feature_names: Sequence[str],
    n_folds: int = 5,
    seed: int = 42,
    bases: Sequence[str] = DEFAULT_BASES,
    meta_c: float = 1.0,
) -> StackResult:
    """Train the grouped OOF base models and the meta-learner.

    Args:
        df: must contain ``label``, ``s1_id`` and every ``feature_names`` column.
        feature_names: feature columns in model order.
        n_folds: grouped folds (shared by the base learners and the meta-learner).
        seed: random seed.
        bases: requested base learners (unavailable ones are dropped).
        meta_c: inverse regularisation strength of the logistic meta-learner.

    Returns:
        A :class:`StackResult` whose ``oof_prob`` is the leak-free meta OOF.
    """
    usable = available_bases(bases)
    x = df[list(feature_names)].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)
    groups = df["s1_id"].to_numpy()

    n_groups = len(np.unique(groups))
    folds = max(2, min(n_folds, n_groups))
    base_oof = np.zeros((len(df), len(usable)), dtype=np.float64)
    fold_models: dict[str, list] = {base: [] for base in usable}
    importances = np.zeros(len(feature_names), dtype=np.float64)

    splitter = GroupKFold(n_splits=folds)
    for fold_num, (train_idx, val_idx) in enumerate(splitter.split(x, y, groups), 1):
        print(f"\n  ── Fold {fold_num}/{folds} ──", flush=True)
        for col, base in enumerate(usable):
            probs, model = _fit_predict_base(
                base, x[train_idx], y[train_idx], x[val_idx], seed, fold=fold_num, total_folds=folds
            )
            base_oof[val_idx, col] = probs
            fold_models[base].append(model)
        # LightGBM importances are the ones we report (consistent across bases).
        if "lgbm" in usable:
            importances += fold_models["lgbm"][-1].feature_importances_

    if "lgbm" in usable and fold_models["lgbm"]:
        importances /= len(fold_models["lgbm"])

    # Meta-learner with its own grouped OOF so its contribution is leak-free too.
    mean = base_oof.mean(axis=0)
    scale = base_oof.std(axis=0)
    scale[scale == 0.0] = 1.0
    z = (base_oof - mean) / scale
    meta_oof = np.zeros(len(df), dtype=np.float64)
    meta_splitter = GroupKFold(n_splits=folds)
    for train_idx, val_idx in meta_splitter.split(z, y, groups):
        meta = LogisticRegression(C=meta_c, max_iter=2000, solver="lbfgs")
        meta.fit(z[train_idx], y[train_idx])
        meta_oof[val_idx] = meta.predict_proba(z[val_idx])[:, 1]

    final_meta = LogisticRegression(C=meta_c, max_iter=2000, solver="lbfgs")
    final_meta.fit(z, y)

    importance_map = {
        name: float(value) for name, value in zip(feature_names, importances)
    }
    return StackResult(
        feature_names=list(feature_names),
        bases=usable,
        oof_prob=meta_oof,
        base_oof=base_oof,
        meta_coef=final_meta.coef_.ravel(),
        meta_intercept=float(final_meta.intercept_[0]),
        meta_mean=mean,
        meta_scale=scale,
        fold_models=fold_models,
        importances=importance_map,
    )


def load_stack(directory: str | Path) -> StackResult:
    """Load a serialized stack for inference (mirrors the training pipeline)."""
    # LightGBM first (OpenMP).
    import lightgbm as lgb

    out = Path(directory)
    meta = json.loads((out / "stack_meta.json").read_text(encoding="utf-8"))
    bases = list(meta["bases"])
    fold_models: dict[str, list] = {}
    for base in bases:
        models = []
        if base == "lgbm":
            for path in sorted(out.glob("lgbm_fold*.txt")):
                models.append(lgb.Booster(model_file=str(path)))
        elif base == "catboost":
            from catboost import CatBoostClassifier

            for path in sorted(out.glob("catboost_fold*.cbm")):
                model = CatBoostClassifier()
                model.load_model(str(path))
                models.append(model)
        elif base == "xgboost":
            from xgboost import XGBClassifier

            for path in sorted(out.glob("xgboost_fold*.json")):
                model = XGBClassifier()
                model.load_model(str(path))
                models.append(model)
        fold_models[base] = models

    class _BoosterPredictAdapter:
        """Wrap an lgb.Booster so ``predict_proba`` matches the sklearn API."""

        def __init__(self, booster):
            self.booster = booster

        def predict_proba(self, x):
            prob = np.asarray(self.booster.predict(x), dtype=np.float64)
            return np.column_stack([1.0 - prob, prob])

    if "lgbm" in fold_models:
        fold_models["lgbm"] = [_BoosterPredictAdapter(b) for b in fold_models["lgbm"]]

    return StackResult(
        feature_names=list(meta["feature_names"]),
        bases=bases,
        oof_prob=np.zeros(0, dtype=np.float64),
        base_oof=np.zeros((0, len(bases)), dtype=np.float64),
        meta_coef=np.asarray(meta["meta_coef"], dtype=np.float64),
        meta_intercept=float(meta["meta_intercept"]),
        meta_mean=np.asarray(meta["meta_mean"], dtype=np.float64),
        meta_scale=np.asarray(meta["meta_scale"], dtype=np.float64),
        fold_models=fold_models,
        importances=dict(meta.get("importances") or {}),
    )


def top_importances(importance_map: dict[str, float], n: int = 15) -> list[tuple[str, float]]:
    """Return the ``n`` highest-importance features."""
    return sorted(importance_map.items(), key=lambda item: item[1], reverse=True)[:n]
