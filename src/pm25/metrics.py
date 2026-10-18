"""Evaluation metrics shared by all models: RMSE, MAE, R², bias.

Bias is mean(prediction - observation): positive means the model predicts
too high. R² is 1 - SSE / SST computed on the rows given, so a per-region R²
measures skill within that region (relative to that region's own mean).
"""

import numpy as np
import pandas as pd


def scores(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """RMSE, MAE, R², bias and row count, ignoring rows where either is NaN."""
    ok = ~(np.isnan(obs) | np.isnan(pred))
    obs, pred = obs[ok], pred[ok]
    if obs.size == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "bias": np.nan, "n": 0}
    err = pred - obs
    sst = np.sum((obs - obs.mean()) ** 2)
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1.0 - np.sum(err**2) / sst) if sst > 0 else np.nan,
        "bias": float(np.mean(err)),
        "n": int(obs.size),
    }


def scores_by_group(
    table: pd.DataFrame,
    pred_cols: list[str],
    by: list[str] | None = None,
    obs_col: str = "obs",
) -> pd.DataFrame:
    """Scores of each prediction column, overall (``by=None``) or per group.

    Returns:
        Long table with columns ``by`` + ``method``, rmse, mae, r2, bias, n.
    """
    rows = []
    groups = [((), table)] if not by else table.groupby(by, observed=True)
    for key, sub in groups:
        key = key if isinstance(key, tuple) else (key,)
        obs = sub[obs_col].to_numpy(dtype="float64")
        for col in pred_cols:
            pred = sub[col].to_numpy(dtype="float64")
            rows.append(
                {**dict(zip(by or [], key)), "method": col, **scores(obs, pred)}
            )
    return pd.DataFrame(rows)
