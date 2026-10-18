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


# Groupings reported for every model (CLAUDE.md: overall, by region and season).
GROUPINGS: dict[str, list[str] | None] = {
    "overall": None,
    "by_region": ["region"],
    "by_season": ["season"],
    "by_region_season": ["region", "season"],
    "by_distance": ["dist_bin"],
}
DIST_BINS = [0, 100, 200, 300, 1000]


def evaluation_table(
    times: np.ndarray,
    seasons: np.ndarray,
    regions: np.ndarray,
    dist_to_train_km: np.ndarray,
    obs: np.ndarray,
    preds: dict[str, np.ndarray],
) -> pd.DataFrame:
    """One row per (time, test cell) with grouping columns, ``obs`` and predictions.

    Args:
        times, seasons: Per time step, shape (time,).
        regions, dist_to_train_km: Per test cell, shape (cell,).
        obs: Observed PM2.5, shape (time, cell).
        preds: Method name -> predictions, each shape (time, cell).
    """
    n_time, n_cell = obs.shape
    return pd.DataFrame(
        {
            "time": np.repeat(times, n_cell),
            "season": pd.Categorical(np.repeat(seasons, n_cell)),
            "region": pd.Categorical(np.tile(regions, n_time)),
            "dist_bin": pd.cut(np.tile(dist_to_train_km, n_time), DIST_BINS),
            "obs": obs.ravel(),
            **{m: p.ravel() for m, p in preds.items()},
        }
    )


def all_scores(table: pd.DataFrame, methods: list[str]) -> dict[str, pd.DataFrame]:
    """Scores of ``methods`` for every grouping in :data:`GROUPINGS`."""
    return {name: scores_by_group(table, methods, by) for name, by in GROUPINGS.items()}
