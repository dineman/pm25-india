"""Regression kriging: ML prediction + interpolated ML residuals (step 08).

The ML model predicts PM2.5 from the predictors at a cell. Its errors are
spatially correlated (a region the model gets too low tends to be too low in
neighbouring cells too), so interpolating the errors seen at nearby training
cells and adding them to the ML prediction can correct it.

The residuals at training cells must be *out-of-fold*: predicted by a model
that did not see that cell. In-sample residuals are much smaller than the
errors at new places and would make the correction too weak.
"""

import logging
import time
from typing import Any

import numpy as np

from pm25.models import fold_indices, make_model

logger = logging.getLogger(__name__)


def out_of_fold_predictions(
    name: str,
    params: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Predict every row with a model fitted on the other CV folds.

    Args:
        name, params: Model and setting (as in :func:`pm25.models.make_model`).
        x, y: All training rows.
        folds: CV fold of each row; rows of one fold are predicted by a model
            fitted on all other folds.

    Returns:
        Predictions, same length as ``y``.
    """
    pred = np.empty(len(y), dtype="float32")
    for k, tr, va in fold_indices(folds):
        start = time.perf_counter()
        model = make_model(name, params, seed).fit(x[tr], y[tr])
        pred[va] = model.predict(x[va])
        logger.info(
            "  out-of-fold %s fold %d: fit on %d rows, predicted %d rows, %.0f s",
            name,
            k,
            tr.size,
            va.size,
            time.perf_counter() - start,
        )
    return pred


def add_residual_correction(
    ml_test: np.ndarray, residual_train: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """ML test prediction + interpolated training residuals, clipped at 0.

    Args:
        ml_test: (time, n_test) ML predictions at test cells.
        residual_train: (time, n_train) observed − out-of-fold prediction.
        weights: (n_test, n_train) interpolation weights (e.g. kriging).
    """
    return np.clip(ml_test + residual_train @ weights.T, 0.0, None)
