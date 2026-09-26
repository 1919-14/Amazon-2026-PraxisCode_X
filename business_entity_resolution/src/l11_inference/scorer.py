"""L11: load the trained matcher (and optional L9 calibrator) for inference.

The expensive part of inference is *feature computation*, not scoring — LightGBM
predicts millions of rows in seconds. This module keeps the model side small: a
loader for the exported booster and a loader for the serialized isotonic
calibrator, so :mod:`l11_inference.inference` can focus on streaming features.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

# LightGBM must be imported before scikit-learn: on Windows their OpenMP runtimes
# conflict and fitting/loading after an sklearn import can raise an access
# violation. Keep this import first, before any sklearn-backed import below.
import lightgbm as lgb  # noqa: F401  (import order is deliberate)

import numpy as np

try:  # pragma: no cover - path fallback when run outside the package
    from src.l6_l8_matching.calibration import apply_serialized
except ImportError:  # pragma: no cover
    from l6_l8_matching.calibration import apply_serialized


def load_booster(path: str | Path) -> lgb.Booster:
    """Load a native LightGBM booster exported by L8 (`lgbm_variant_<v>.txt`)."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"model file not found: {model_path}")
    return lgb.Booster(model_file=str(model_path))


def load_calibrator(path: str | Path | None) -> Optional[dict]:
    """Load the serialized isotonic calibrator chosen by L9 (or ``None``).

    The payload records which option was adopted. When L9 chose ``none`` there is
    no correction to apply, so we return ``None`` even though the file exists.
    """
    if path is None:
        return None
    cal_path = Path(path)
    if not cal_path.exists():
        return None
    payload = json.loads(cal_path.read_text(encoding="utf-8"))
    chosen = payload.get("chosen", "none")
    if chosen == "global" and payload.get("global"):
        return {"mode": "global", "knots": payload["global"]}
    if chosen == "per_country" and payload.get("per_country"):
        return {"mode": "per_country", "knots": payload["per_country"]}
    return None


def score_matrix(booster: lgb.Booster, features: np.ndarray) -> np.ndarray:
    """Predict positive-class probabilities for a ``(n, n_features)`` matrix."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"expected a 2-D feature matrix, got shape {features.shape}")
    if features.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(booster.predict(features), dtype=np.float64)


def calibrate_scores(
    probs: np.ndarray,
    calibrator: Optional[dict],
    countries: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Apply the L9 calibrator (global or per-country) to raw probabilities."""
    probs = np.asarray(probs, dtype=np.float64)
    if calibrator is None or len(probs) == 0:
        return probs

    if calibrator["mode"] == "global":
        return apply_serialized(calibrator["knots"], probs)

    # per-country: calibrate each country's rows with its own knot set, falling
    # back to the raw probability where no calibrator was fit.
    out = probs.copy()
    if countries is None:
        return out
    countries = list(countries)
    for country in set(countries):
        knots = calibrator["knots"].get(str(country))
        if not knots:
            continue
        mask = np.array([c == country for c in countries], dtype=bool)
        out[mask] = apply_serialized(knots, probs[mask])
    return out
