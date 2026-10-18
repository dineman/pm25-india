"""ML models for step 07: definitions, search spaces and block-wise CV tuning.

Every model is wrapped the same way:
- optional log1p target (``target="log"``), back-transformed with expm1;
- predictions clipped to [0, max PM2.5 seen in training], so no model can
  return negative or absurdly large values (step 06: a log-linear model
  predicted 55,041 ug m-3 without clipping).

Tuning is a seeded random search: each model gets the same number of
candidate settings, each scored by grouped cross-validation over the spatial
CV folds of step 05 (whole blocks held out), on raw-scale RMSE.
"""

import json
import logging
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pm25.metrics import scores

logger = logging.getLogger(__name__)

MODEL_NAMES = ["ridge", "rf", "lgbm", "mlp"]

# Candidate settings per model. "target" (raw / log1p) is searched for all.
SEARCH_SPACES: dict[str, dict[str, list]] = {
    "ridge": {
        "target": ["raw", "log"],
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    },
    "rf": {
        "target": ["raw", "log"],
        "n_estimators": [100, 200],
        "max_depth": [None, 20, 30],
        "min_samples_leaf": [5, 20, 50],
        "max_features": [0.33, 0.5, 1.0],
        "max_samples": [0.1, 0.25],
    },
    "lgbm": {
        "target": ["raw", "log"],
        "n_estimators": [300, 600, 1000],
        "learning_rate": [0.03, 0.05, 0.1],
        "num_leaves": [31, 63, 127, 255],
        "min_child_samples": [20, 100, 500],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0],
        "reg_lambda": [0.0, 1.0, 10.0],
    },
    "mlp": {
        "target": ["raw", "log"],
        "hidden_layer_sizes": [(64, 64), (128, 64), (128, 128, 64)],
        "alpha": [1e-5, 1e-4, 1e-3],
        "learning_rate_init": [1e-3, 3e-3],
    },
}


def build_estimator(name: str, params: dict[str, Any], seed: int) -> BaseEstimator:
    """The underlying scikit-learn/LightGBM estimator (without target handling)."""
    p = {k: v for k, v in params.items() if k != "target"}
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(**p))
    if name == "rf":
        return RandomForestRegressor(n_jobs=-1, random_state=seed, **p)
    if name == "lgbm":
        return LGBMRegressor(
            subsample_freq=1, n_jobs=-1, random_state=seed, verbose=-1, **p
        )
    if name == "mlp":
        mlp = MLPRegressor(
            batch_size=2048,
            max_iter=50,
            early_stopping=True,
            n_iter_no_change=5,
            random_state=seed,
            **p,
        )
        return make_pipeline(StandardScaler(), mlp)
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")


class ClippedTargetModel(RegressorMixin, BaseEstimator):
    """Fit on raw or log1p target; predict on the raw scale, clipped to [0, max]."""

    def __init__(self, estimator: BaseEstimator, target: str = "raw") -> None:
        self.estimator = estimator
        self.target = target

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ClippedTargetModel":
        """Fit the wrapped estimator; remember the training maximum for clipping."""
        self.y_max_ = float(np.max(y))
        y_fit = np.log1p(y) if self.target == "log" else y
        self.estimator_ = clone(self.estimator).fit(x, y_fit)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Raw-scale predictions clipped to [0, training max]."""
        pred = self.estimator_.predict(x)
        if self.target == "log":
            pred = np.expm1(np.clip(pred, None, np.log1p(self.y_max_)))
        return np.clip(pred, 0.0, self.y_max_)


def make_model(name: str, params: dict[str, Any], seed: int) -> ClippedTargetModel:
    """Model ``name`` with ``params`` (including ``target``), ready to fit."""
    return ClippedTargetModel(build_estimator(name, params, seed), params["target"])


def sample_params(
    space: dict[str, list], n: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    """``n`` distinct random settings from ``space`` (fewer if the space is smaller)."""
    keys = list(space)
    total = int(np.prod([len(space[k]) for k in keys]))
    chosen: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    while len(chosen) < min(n, total):
        idx = tuple(int(rng.integers(len(space[k]))) for k in keys)
        if idx in seen:
            continue
        seen.add(idx)
        chosen.append({k: space[k][i] for k, i in zip(keys, idx)})
    return chosen


def fold_indices(folds: np.ndarray) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """(fold, train rows, validation rows) for each fold id in ``folds``."""
    for k in np.unique(folds):
        yield int(k), np.flatnonzero(folds != k), np.flatnonzero(folds == k)


def cross_validate(
    name: str,
    params: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    seed: int,
) -> dict[str, float]:
    """Grouped CV: mean RMSE/MAE/R² over folds (each fold = held-out blocks)."""
    per_fold = []
    start = time.perf_counter()
    for _, tr, va in fold_indices(folds):
        model = make_model(name, params, seed).fit(x[tr], y[tr])
        per_fold.append(scores(y[va].astype("float64"), model.predict(x[va])))
    out = {
        f"cv_{m}": float(np.mean([s[m] for s in per_fold]))
        for m in ("rmse", "mae", "r2", "bias")
    }
    out["cv_rmse_std"] = float(np.std([s["rmse"] for s in per_fold]))
    out["seconds"] = time.perf_counter() - start
    return out


def best_settings(cv_table: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Setting with the lowest mean CV RMSE for each model.

    Args:
        cv_table: One row per tried setting with columns ``model``, ``params``
            (JSON text) and ``cv_rmse``. The row index is ignored, so tables
            concatenated from several models (with repeated indices) are fine.
    """
    best = {}
    for model, rows in cv_table.groupby("model", sort=False):
        row = rows.iloc[int(np.argmin(rows["cv_rmse"].to_numpy()))]
        best[str(model)] = json.loads(row["params"])
    return best
