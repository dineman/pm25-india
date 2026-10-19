"""Step 09 evaluation helpers: extra groupings, block bootstrap and CV-based checks.

The test set is only 6 spatial blocks, so step 09 adds two things to the
plain tables of steps 06-08:

- a **block bootstrap**: resample whole test blocks with replacement and
  recompute the scores, giving an uncertainty range for each score and for the
  difference between two methods;
- a **CV check**: the same comparison on the training cells, where each of the
  5 block folds is predicted only from the other 4 (27 blocks instead of 6).
"""

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from pm25.models import fold_indices, make_model

logger = logging.getLogger(__name__)

# India National AQI breakpoints for PM2.5 (µg m⁻³, defined on 24 h means;
# here applied to 3-hourly values, so only as a scale of "how polluted").
AQI_EDGES = [0, 30, 60, 90, 120, 250, np.inf]
AQI_LABELS = ["good", "satisfactory", "moderate", "poor", "very_poor", "severe"]
IST_OFFSET_MINUTES = 330


def ist_hour(times: np.ndarray) -> np.ndarray:
    """Local Indian time (UTC + 5:30) of each time step as "HH:MM" labels."""
    local = pd.DatetimeIndex(times) + pd.Timedelta(minutes=IST_OFFSET_MINUTES)
    return local.strftime("%H:%M").to_numpy()


def aqi_level(obs: np.ndarray) -> pd.Categorical:
    """AQI category of each observed PM2.5 value (lower edge inclusive)."""
    return pd.cut(obs, AQI_EDGES, labels=AQI_LABELS, right=False)


def add_time_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Add ``hour_ist``, ``year`` and ``aqi_level`` grouping columns in place."""
    times = table["time"].to_numpy()
    unique, inverse = np.unique(times, return_inverse=True)
    table["hour_ist"] = pd.Categorical(ist_hour(unique)[inverse])
    table["year"] = pd.DatetimeIndex(unique).year.to_numpy()[inverse]
    table["aqi_level"] = aqi_level(table["obs"].to_numpy())
    return table


# --- Block bootstrap ---------------------------------------------------------

STAT_COLS = ["n", "sum_err", "sum_abs", "sum_sq", "sum_obs", "sum_obs_sq"]


def block_sums(table: pd.DataFrame, methods: list[str], block_col: str) -> pd.DataFrame:
    """Per block and method, the sums from which RMSE, MAE, R² and bias follow.

    Rows where the observation or prediction is NaN are skipped.

    Returns:
        Table indexed by (block, method) with columns :data:`STAT_COLS`.
    """
    rows = []
    obs_all = table["obs"].to_numpy(dtype="float64")
    pred_all = {m: table[m].to_numpy(dtype="float64") for m in methods}
    for block, idx in table.groupby(block_col, observed=True).indices.items():
        obs_b = obs_all[idx]
        for m in methods:
            pred = pred_all[m][idx]
            ok = ~(np.isnan(obs_b) | np.isnan(pred))
            o, err = obs_b[ok], pred[ok] - obs_b[ok]
            rows.append(
                {
                    "block": block,
                    "method": m,
                    "n": o.size,
                    "sum_err": err.sum(),
                    "sum_abs": np.abs(err).sum(),
                    "sum_sq": (err**2).sum(),
                    "sum_obs": o.sum(),
                    "sum_obs_sq": (o**2).sum(),
                }
            )
    return pd.DataFrame(rows).set_index(["block", "method"])


def scores_from_sums(sums: np.ndarray) -> dict[str, np.ndarray]:
    """RMSE, MAE, R², bias from summed :data:`STAT_COLS` (last axis)."""
    n, s_err, s_abs, s_sq, s_obs, s_obs_sq = np.moveaxis(sums, -1, 0)
    sst = s_obs_sq - s_obs**2 / n
    return {
        "rmse": np.sqrt(s_sq / n),
        "mae": s_abs / n,
        "r2": 1.0 - s_sq / sst,
        "bias": s_err / n,
    }


def block_bootstrap(
    sums: pd.DataFrame,
    pairs: list[tuple[str, str]],
    n_boot: int,
    rng: np.random.Generator,
    level: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap scores by resampling whole blocks with replacement.

    Every method is scored on the same resampled blocks, so differences
    between methods are paired.

    Args:
        sums: Output of :func:`block_sums`.
        pairs: (a, b) method pairs; the difference reported is score(a) - score(b).
        n_boot: Number of bootstrap samples.
        level: Width of the percentile interval.

    Returns:
        (per-method scores, per-pair differences), each with the point
        estimate, interval bounds and, for differences, the share of bootstrap
        samples in which ``a`` beats ``b``.
    """
    blocks = sums.index.get_level_values("block").unique()
    methods = list(sums.index.get_level_values("method").unique())
    # cube[block, method, stat]
    cube = np.stack(
        [sums.xs(m, level="method").loc[blocks, STAT_COLS].to_numpy() for m in methods],
        axis=1,
    )
    draws = rng.integers(0, len(blocks), size=(n_boot, len(blocks)))
    counts = np.stack([np.bincount(d, minlength=len(blocks)) for d in draws])
    boot = scores_from_sums(np.einsum("rb,bms->rms", counts, cube))
    point = scores_from_sums(cube.sum(axis=0))
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2

    per_method = []
    for j, m in enumerate(methods):
        row: dict[str, Any] = {"method": m}
        for metric, values in boot.items():
            row[metric] = point[metric][j]
            row[f"{metric}_lo"], row[f"{metric}_hi"] = np.nanpercentile(
                values[:, j], [lo, hi]
            )
        per_method.append(row)

    per_pair = []
    for a, b in pairs:
        ia, ib = methods.index(a), methods.index(b)
        for metric, values in boot.items():
            diff = values[:, ia] - values[:, ib]
            if metric == "r2":
                better = diff > 0
            elif metric == "bias":
                better = np.abs(values[:, ia]) < np.abs(values[:, ib])
            else:
                better = diff < 0
            per_pair.append(
                {
                    "a": a,
                    "b": b,
                    "metric": metric,
                    "diff": point[metric][ia] - point[metric][ib],
                    "diff_lo": np.nanpercentile(diff, lo),
                    "diff_hi": np.nanpercentile(diff, hi),
                    "share_a_better": float(np.mean(better)),
                }
            )
    return pd.DataFrame(per_method), pd.DataFrame(per_pair)


# --- CV check on training cells ----------------------------------------------

WeightFn = Callable[
    [np.ndarray, np.ndarray, np.ndarray], list[tuple[np.ndarray, np.ndarray]]
]


def cv_interpolate(
    values: np.ndarray,
    dist: np.ndarray,
    folds: np.ndarray,
    make_weights: WeightFn,
) -> np.ndarray:
    """Predict each fold's cells by interpolating from the other folds' cells.

    Args:
        values: (time, cell) field at all training cells, or (n_folds, time, cell)
            with a separate field per predicted fold (e.g. nested residuals, in
            the order of ``np.unique(folds)``). Only source cells are read.
        dist: (cell, cell) distances.
        folds: Fold of each cell.
        make_weights: Called as ``make_weights(values_src, dist_src, dist_tgt_src)``
            and returns a list of (time mask, weights (n_tgt, n_src)) so that
            weights can differ by season (kriging).

    Returns:
        (time, cell) interpolated values.
    """
    out = np.full(values.shape[-2:], np.nan)
    for i, (_, src, tgt) in enumerate(fold_indices(folds)):
        field = values[i] if values.ndim == 3 else values
        src_values = field[:, src]
        for rows, w in make_weights(
            src_values, dist[np.ix_(src, src)], dist[np.ix_(tgt, src)]
        ):
            out[np.ix_(rows, tgt)] = src_values[rows] @ w.T
    return out


def nested_oof_residuals(
    name: str,
    params: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    cell_folds: np.ndarray,
    n_time: int,
    seed: int,
) -> np.ndarray:
    """Residuals for regression kriging inside the CV check, without leakage.

    For outer fold k, the residuals at the other folds' cells must come from
    models that never saw fold k. So for each inner fold j != k a model is fitted
    on folds other than j and k and predicts fold j.

    Args:
        x, y: All training rows, time-major (``time * cell``).
        cell_folds: Fold of each training cell.

    Returns:
        (n_folds, time, cell) residuals obs - pred; NaN at the outer fold's own cells.
    """
    fold_ids = np.unique(cell_folds)
    n_cell = cell_folds.size
    row_folds = np.tile(cell_folds, n_time)
    resid = np.full((fold_ids.size, n_time, n_cell), np.nan, dtype="float32")
    for i, k in enumerate(fold_ids):
        for j in fold_ids[fold_ids != k]:
            start = time.perf_counter()
            fit_rows = np.flatnonzero((row_folds != k) & (row_folds != j))
            pred_rows = np.flatnonzero(row_folds == j)
            model = make_model(name, params, seed).fit(x[fit_rows], y[fit_rows])
            pred = model.predict(x[pred_rows]).reshape(n_time, -1)
            cells = np.flatnonzero(cell_folds == j)
            resid[i][:, cells] = y[pred_rows].reshape(n_time, -1) - pred
            logger.info(
                "  nested %s outer %d inner %d: fit %d rows, %.0f s",
                name,
                k,
                j,
                fit_rows.size,
                time.perf_counter() - start,
            )
    return resid
