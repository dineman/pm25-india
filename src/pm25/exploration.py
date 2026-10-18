"""Exploratory statistics for step 04: summaries, correlations and variograms.

The variogram measures how fast PM2.5 values stop being similar as distance
between grid cells grows. Its range sets the block size for the spatial
train/test split (step 05): blocks should be at least as wide as the range so
test cells are not just near-copies of their training neighbours.
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit

EARTH_RADIUS_KM = 6371.0
KM_PER_DEG_LAT = np.pi * EARTH_RADIUS_KM / 180.0

# Predictors used for the diagnostic linear model and the correlation table.
PREDICTORS = [
    "aod550",
    "u10",
    "v10",
    "wind_speed",
    "t2m",
    "d2m",
    "rh2m",
    "sp",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]


def india_cells(ds: xr.Dataset) -> xr.Dataset:
    """Stack (lat, lon) into a ``cell`` dim and keep only cells inside India."""
    stacked = ds.stack(cell=("lat", "lon"))
    return stacked.isel(cell=stacked["in_india"].values)


def to_table(ds: xr.Dataset, variables: list[str]) -> pd.DataFrame:
    """Flatten India cells into one row per (time, cell) with region and season.

    Variables with fewer dims (e.g. ``hour_sin`` on (time, lon)) are broadcast.
    """
    cells = india_cells(ds)
    cols = {}
    for name in variables:
        arr = cells[name].broadcast_like(cells["pm2p5"]).transpose("time", "cell")
        cols[name] = arr.values.ravel()
    n_time, n_cell = cells.sizes["time"], cells.sizes["cell"]
    table = pd.DataFrame(cols)
    table["region"] = pd.Categorical(np.tile(cells["region"].values, n_time))
    table["season"] = pd.Categorical(np.repeat(cells["season"].values, n_cell))
    table["time"] = np.repeat(cells["time"].values, n_cell)
    return table


def summarize(
    table: pd.DataFrame, by: list[str], column: str = "pm2p5"
) -> pd.DataFrame:
    """Mean, median, 5th/95th percentile, std and count of ``column`` per group."""
    grouped = table.groupby(by, observed=True)[column]
    out = grouped.agg(["mean", "median", "std", "count"])
    out["p05"] = grouped.quantile(0.05)
    out["p95"] = grouped.quantile(0.95)
    return out.reset_index()


def spearman_with_target(
    table: pd.DataFrame, predictors: list[str], target: str = "pm2p5"
) -> pd.Series:
    """Spearman rank correlation of each predictor with the target."""
    ranks = table[predictors + [target]].rank()
    return ranks[predictors].corrwith(ranks[target])


def haversine_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Pairwise great-circle distances (km) between points given in degrees."""
    phi, lam = np.radians(lat), np.radians(lon)
    dphi = phi[:, None] - phi[None, :]
    dlam = lam[:, None] - lam[None, :]
    a = (
        np.sin(dphi / 2) ** 2
        + np.cos(phi[:, None]) * np.cos(phi[None, :]) * np.sin(dlam / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def standardize_rows(values: np.ndarray) -> np.ndarray:
    """Z-score each row (one time step) across cells.

    This puts every time step on the same scale, so the pooled variogram is
    not dominated by high-variance winter days, and its sill is close to 1.
    """
    mean = np.nanmean(values, axis=1, keepdims=True)
    std = np.nanstd(values, axis=1, keepdims=True)
    return (values - mean) / np.where(std > 0, std, np.nan)


def empirical_variogram(
    values: np.ndarray,
    dist_km: np.ndarray,
    bin_edges: np.ndarray,
    chunk: int = 32,
) -> pd.DataFrame:
    """Semivariance per distance bin, pooled over time steps.

    Args:
        values: Array (time, cell). NaNs are ignored pair by pair.
        dist_km: Pairwise distances (cell, cell).
        bin_edges: Distance bin edges in km.
        chunk: Time steps processed at once (bounds memory use).

    Returns:
        Table with columns ``lag_km`` (mean pair distance in the bin),
        ``gamma`` (semivariance, 0.5 * mean squared difference) and ``n_pairs``.
    """
    i, j = np.triu_indices(dist_km.shape[0], k=1)
    d = dist_km[i, j]
    bins = np.digitize(d, bin_edges) - 1
    keep = (bins >= 0) & (bins < len(bin_edges) - 1)
    i, j, d, bins = i[keep], j[keep], d[keep], bins[keep]
    n_bins = len(bin_edges) - 1

    sq_sum = np.zeros(n_bins)
    count = np.zeros(n_bins)
    for start in range(0, values.shape[0], chunk):
        block = values[start : start + chunk]
        sq = (block[:, i] - block[:, j]) ** 2
        valid = ~np.isnan(sq)
        sq_sum += np.bincount(
            bins, weights=np.where(valid, sq, 0.0).sum(0), minlength=n_bins
        )
        count += np.bincount(bins, weights=valid.sum(0), minlength=n_bins)

    lag = np.bincount(bins, weights=d, minlength=n_bins) / np.maximum(
        np.bincount(bins, minlength=n_bins), 1
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        gamma = 0.5 * sq_sum / count
    out = pd.DataFrame({"lag_km": lag, "gamma": gamma, "n_pairs": count})
    return out[out["n_pairs"] > 0].reset_index(drop=True)


def spherical(h: np.ndarray, nugget: float, psill: float, rng: float) -> np.ndarray:
    """Spherical variogram model; reaches the sill exactly at ``rng``."""
    r = np.minimum(h / rng, 1.0)
    return nugget + psill * (1.5 * r - 0.5 * r**3)


def exponential(h: np.ndarray, nugget: float, psill: float, rng: float) -> np.ndarray:
    """Exponential variogram model with *practical* range ``rng`` (95% of sill)."""
    return nugget + psill * (1.0 - np.exp(-3.0 * h / rng))


VARIOGRAM_MODELS = {"spherical": spherical, "exponential": exponential}


def fit_variogram(
    vgm: pd.DataFrame, model: str = "spherical", max_lag_km: float | None = None
) -> dict[str, float]:
    """Fit a nugget + sill + range model to an empirical variogram.

    Bins are weighted by their number of pairs. Only lags up to ``max_lag_km``
    are used: long lags have few pairs and reflect large-scale trends rather
    than local correlation.

    Returns:
        Dict with ``nugget``, ``psill``, ``sill``, ``range_km`` and ``rmse``.
    """
    func = VARIOGRAM_MODELS[model]
    data = vgm if max_lag_km is None else vgm[vgm["lag_km"] <= max_lag_km]
    h, g, n = data["lag_km"].to_numpy(), data["gamma"].to_numpy(), data["n_pairs"]
    p0 = [max(g.min(), 0.0), max(g.max() - g.min(), 1e-6), h.max() / 2]
    bounds = ([0.0, 0.0, 1.0], [np.inf, np.inf, 10 * h.max()])
    params, _ = curve_fit(
        func, h, g, p0=p0, bounds=bounds, sigma=1 / np.sqrt(n.to_numpy())
    )
    nugget, psill, rng = (float(p) for p in params)
    rmse = float(np.sqrt(np.mean((func(h, *params) - g) ** 2)))
    return {
        "model": model,
        "nugget": nugget,
        "psill": psill,
        "sill": nugget + psill,
        "range_km": rng,
        "rmse": rmse,
    }


def linear_residuals(values: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Residuals of an ordinary least-squares fit of ``values`` on ``features``.

    Args:
        values: Shape (n,).
        features: Shape (n, k); an intercept column is added.

    Rows with any NaN get a NaN residual.
    """
    ok = ~np.isnan(values) & ~np.isnan(features).any(axis=1)
    design = np.column_stack([np.ones(ok.sum()), features[ok]])
    coef, *_ = np.linalg.lstsq(design, values[ok], rcond=None)
    resid = np.full(values.shape, np.nan)
    resid[ok] = values[ok] - design @ coef
    return resid


def block_size_cells(
    range_km: float, cell_deg: float = 0.75, ref_lat: float = 25.0
) -> int:
    """Smallest square block (in grid cells) whose side is at least ``range_km``.

    Uses the east-west cell width at ``ref_lat`` (~76 km for 0.75 deg at 25 N),
    the narrower side of a cell over most of India, so the block is at least
    as wide as the range in both directions there.
    """
    cell_km = cell_deg * KM_PER_DEG_LAT * np.cos(np.radians(ref_lat))
    return max(1, int(np.ceil(range_km / cell_km)))
