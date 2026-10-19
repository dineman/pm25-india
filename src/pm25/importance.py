"""Feature importance for the step 07 tree model (step 10).

Two complementary views:

- **SHAP values** (what the model *uses*): for every prediction, how much each
  feature pushed it up or down from the average prediction. LightGBM computes
  exact TreeSHAP values itself (``pred_contrib=True``). For a model fitted on
  log1p(PM2.5) they are in log units: +0.1 means about 10% higher PM2.5.
- **Permutation importance** (what the *accuracy* depends on): shuffle one
  feature (or a group of related features) across rows and measure how much the
  error on held-out cells grows. Shuffling a group together avoids the
  underestimate for correlated features (e.g. t2m, d2m and rh2m), where the
  model can partly recover a shuffled feature from its partners.
"""

import logging

import numpy as np
import pandas as pd

from pm25.metrics import scores
from pm25.models import ClippedTargetModel

logger = logging.getLogger(__name__)

# Physically related features, shuffled together and summed for SHAP.
FEATURE_GROUPS: dict[str, list[str]] = {
    "aerosol (aod550)": ["aod550"],
    "wind (u10, v10, speed)": ["u10", "v10", "wind_speed"],
    "temperature & humidity (t2m, d2m, rh2m)": ["t2m", "d2m", "rh2m"],
    "surface pressure (sp)": ["sp"],
    "location (lat, lon)": ["lat", "lon"],
    "hour of day (hour_sin, hour_cos)": ["hour_sin", "hour_cos"],
    "day of year (doy_sin, doy_cos)": ["doy_sin", "doy_cos"],
}


def shap_values(model: ClippedTargetModel, x: np.ndarray) -> np.ndarray:
    """Exact TreeSHAP values of a fitted LightGBM model, (rows, features).

    The last column LightGBM returns (the expected value) is dropped. Values are
    on the scale the model was fitted on (log1p for ``target="log"``).
    """
    contrib = model.estimator_.predict(x, pred_contrib=True)
    return np.asarray(contrib)[:, :-1]


def group_shap(shap: np.ndarray, features: list[str]) -> pd.DataFrame:
    """Sum SHAP values within each :data:`FEATURE_GROUPS` group, (rows, groups)."""
    cols = {
        name: shap[:, [features.index(f) for f in members]].sum(axis=1)
        for name, members in FEATURE_GROUPS.items()
        if all(f in features for f in members)
    }
    return pd.DataFrame(cols)


def mean_abs_by(values: pd.DataFrame, by: pd.Series | None = None) -> pd.DataFrame:
    """Mean |value| per column, overall (``by=None``) or per group of ``by``."""
    absolute = values.abs()
    if by is None:
        return absolute.mean().to_frame("mean_abs_shap").T
    return absolute.groupby(by.to_numpy(), observed=True).mean()


def permutation_importance(
    model: ClippedTargetModel,
    x: np.ndarray,
    y: np.ndarray,
    columns: dict[str, list[int]],
    n_repeats: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Increase in RMSE/MAE when the given columns are shuffled together.

    Args:
        columns: Name -> column indices to shuffle jointly (one feature or a
            group). Jointly means the same row permutation for all of them, so
            their relation to each other is kept but their link to PM2.5 is broken.

    Returns:
        One row per (name, repeat) with rmse, mae and their increase over the
        unshuffled score.
    """
    base = scores(y, model.predict(x))
    rows = []
    for name, idx in columns.items():
        for repeat in range(n_repeats):
            shuffled = x.copy()
            perm = rng.permutation(len(x))
            shuffled[:, idx] = x[perm][:, idx]
            s = scores(y, model.predict(shuffled))
            rows.append(
                {
                    "name": name,
                    "repeat": repeat,
                    "rmse": s["rmse"],
                    "mae": s["mae"],
                    "d_rmse": s["rmse"] - base["rmse"],
                    "d_mae": s["mae"] - base["mae"],
                }
            )
        logger.info("  permuted %s", name)
    return pd.DataFrame(rows)


def binned_dependence(
    value: np.ndarray, shap: np.ndarray, n_bins: int = 20
) -> pd.DataFrame:
    """Mean SHAP value per quantile bin of the feature value (dependence curve)."""
    edges = np.unique(np.quantile(value, np.linspace(0, 1, n_bins + 1)))
    bins = np.clip(np.digitize(value, edges[1:-1]), 0, len(edges) - 2)
    table = pd.DataFrame({"bin": bins, "value": value, "shap": shap})
    return (
        table.groupby("bin")
        .agg(value=("value", "median"), shap=("shap", "mean"), n=("shap", "size"))
        .reset_index(drop=True)
    )
