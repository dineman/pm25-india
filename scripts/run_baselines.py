"""Step 06: baseline predictions of PM2.5 at held-out test cells.

Usage:
    conda run -n pm25 python scripts/run_baselines.py
    conda run -n pm25 python scripts/run_baselines.py \
        --data data/processed/eac4_india_3h_2020_2021.nc \
        --out-dir results/06_baselines_2020_2021

Interpolation baselines (nearest, IDW, Delaunay-linear, ordinary kriging,
training-cell mean) use only PM2.5 at training cells at the same time step.
Linear regression uses the predictors at the test cell itself. Writes
predictions, metrics (overall, by region, season, region x season and
distance to the nearest training cell), the kriging variograms, config and figures.
"""

import argparse
import json
import logging
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LinearRegression

from pm25.baselines import (
    apply_weights,
    delaunay_weights,
    idw_weights,
    nearest_weights,
    ordinary_kriging_weights,
    project_km,
)
from pm25.exploration import (
    PREDICTORS,
    VARIOGRAM_MODELS,
    empirical_variogram,
    fit_variogram,
    haversine_km,
    standardize_rows,
)
from pm25.logging_utils import log_run_info, setup_logging
from pm25.metrics import all_scores, evaluation_table

logger = logging.getLogger(__name__)

SEASON_ORDER = ["winter", "pre_monsoon", "monsoon", "post_monsoon"]
REGION_ORDER = ["igp", "northwest", "central", "south", "northeast", "himalaya"]
METHODS = [
    "train_mean",
    "nearest",
    "idw",
    "linear_delaunay",
    "kriging",
    "linreg",
    "linreg_log",
]
INK, MUTED, GRID = "#2b2b29", "#6b6a63", "#e4e3de"
BAR = "#2a78d6"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95"]
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data", type=Path, default=Path("data/processed/eac4_india_3h.nc")
    )
    parser.add_argument(
        "--split", type=Path, default=Path("data/processed/splits/split_block5.nc")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/06_baselines"))
    parser.add_argument("--idw-k", type=int, default=8)
    parser.add_argument("--idw-power", type=float, default=2.0)
    parser.add_argument(
        "--vgm-times",
        type=int,
        default=1000,
        help="random time steps per season for the kriging variogram",
    )
    parser.add_argument("--fit-max-lag-km", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stack_cells(ds: xr.Dataset, mask: np.ndarray) -> xr.Dataset:
    """Stack (lat, lon) to ``cell`` and keep cells where ``mask`` (lat, lon) is True."""
    stacked = ds.stack(cell=("lat", "lon"))
    return stacked.isel(cell=mask.ravel())


def feature_matrix(cells: xr.Dataset) -> np.ndarray:
    """Predictors plus lat/lon as (time * cell, k), time-major like ``.ravel()``."""
    cols = []
    ref = cells["pm2p5"].transpose("time", "cell")
    for name in PREDICTORS:
        cols.append(cells[name].broadcast_like(ref).transpose("time", "cell").values)
    for name in ("lat", "lon"):
        cols.append(np.broadcast_to(cells[name].values, ref.shape))
    return np.stack([c.ravel() for c in cols], axis=1).astype("float64")


def kriging_variograms(
    pm_train: np.ndarray,
    seasons: np.ndarray,
    dist_train: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[dict[str, dict], pd.DataFrame]:
    """Fit one exponential variogram per season on TRAINING cells only.

    Same recipe as step 04 (log1p, standardised per time step), so the fitted
    nugget/sill ratio and range describe the shape of spatial correlation.
    Kriging weights depend only on that shape, not on the sill's scale.
    """
    edges = np.arange(0.0, 1500.0 + 75.0, 75.0)
    fits, bins = {}, []
    for season in SEASON_ORDER:
        idx = np.flatnonzero(seasons == season)
        idx = rng.choice(idx, size=min(args.vgm_times, idx.size), replace=False)
        field = standardize_rows(np.log1p(pm_train[idx].astype("float64")))
        vgm = empirical_variogram(field, dist_train, edges)
        fits[season] = fit_variogram(vgm, "exponential", args.fit_max_lag_km)
        bins.append(vgm.assign(season=season))
        logger.info(
            "Kriging variogram %s: range %.0f km, nugget/sill %.2f",
            season,
            fits[season]["range_km"],
            fits[season]["nugget"] / fits[season]["sill"],
        )
    return fits, pd.concat(bins)


def plot_overall(overall: pd.DataFrame, path: Path) -> None:
    """Horizontal bars of overall test RMSE per method, labelled with RMSE and R²."""
    data = overall.sort_values("rmse", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(data["method"], data["rmse"], color=BAR, height=0.6)
    for y, (rmse, r2) in enumerate(zip(data["rmse"], data["r2"])):
        ax.text(
            rmse, y, f"  {rmse:.1f}  (R² {r2:.2f})", va="center", color=INK, fontsize=9
        )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.set_xlim(0, data["rmse"].max() * 1.35)
    ax.set_xlabel("RMSE on test cells (µg m⁻³)", color=INK)
    ax.set_title("Baselines: overall test error", color=INK, loc="left")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_region_table(by_region: pd.DataFrame, path: Path) -> None:
    """Heatmap of RMSE, method x region, with values written in each cell."""
    table = by_region.pivot(index="method", columns="region", values="rmse")
    table = table.loc[METHODS, REGION_ORDER]
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(table.values, cmap=SEQUENTIAL, aspect="auto")
    for (r, c), v in np.ndenumerate(table.values):
        dark = v > np.nanpercentile(table.values, 70)
        ax.text(
            c,
            r,
            f"{v:.0f}",
            ha="center",
            va="center",
            fontsize=9,
            color="white" if dark else INK,
        )
    ax.set_xticks(range(len(REGION_ORDER)), REGION_ORDER)
    ax.set_yticks(range(len(METHODS)), METHODS)
    ax.tick_params(length=0, labelcolor=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.8).set_label("RMSE (µg m⁻³)", color=INK)
    ax.set_title("Baselines: test RMSE by region", color=INK, loc="left")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", path)


def main() -> None:
    """Fit/apply all baselines, score them on test cells and save everything."""
    args = parse_args()
    setup_logging(log_file=Path("results/logs/run_baselines.log"))
    log_run_info(args)
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(args.data)
    split = xr.open_dataset(args.split)
    xr.testing.assert_allclose(split["lat"], ds["lat"])
    xr.testing.assert_allclose(split["lon"], ds["lon"])
    is_train = (split["set"] == "train").values
    is_test = (split["set"] == "test").values

    variables = ["pm2p5", *PREDICTORS]
    train = stack_cells(ds[variables], is_train).load()
    test = stack_cells(ds[variables], is_test).load()
    seasons = ds["season"].values
    logger.info(
        "Data %s: %d time steps; %d train cells, %d test cells (split %s, block %s)",
        args.data,
        ds.sizes["time"],
        train.sizes["cell"],
        test.sizes["cell"],
        args.split,
        split.attrs.get("block_size_cells"),
    )

    pm_train = train["pm2p5"].transpose("time", "cell").values
    pm_test = test["pm2p5"].transpose("time", "cell").values
    lat_tr, lon_tr = train["lat"].values, train["lon"].values
    lat_te, lon_te = test["lat"].values, test["lon"].values
    dist_train = haversine_km(lat_tr, lon_tr)
    dist_tt = haversine_km(
        np.concatenate([lat_te, lat_tr]), np.concatenate([lon_te, lon_tr])
    )[: lat_te.size, lat_te.size :]

    # --- Interpolation weights (computed once) -------------------------------
    ref_lat = float(np.mean(lat_tr))
    lin_w, outside = delaunay_weights(
        project_km(lat_tr, lon_tr, ref_lat),
        project_km(lat_te, lon_te, ref_lat),
        dist_tt,
    )
    logger.info(
        "Delaunay: %d of %d test cells outside the triangulation -> nearest",
        int(outside.sum()),
        outside.size,
    )
    weights = {
        "train_mean": np.full(dist_tt.shape, 1.0 / dist_tt.shape[1]),
        "nearest": nearest_weights(dist_tt),
        "idw": idw_weights(dist_tt, args.idw_k, args.idw_power),
        "linear_delaunay": lin_w,
    }
    preds = {name: apply_weights(w, pm_train) for name, w in weights.items()}

    vgm_fits, vgm_bins = kriging_variograms(pm_train, seasons, dist_train, args, rng)
    kriging = np.empty_like(pm_test, dtype="float64")
    for season, fit in vgm_fits.items():
        variogram = partial(
            VARIOGRAM_MODELS[fit["model"]],
            nugget=fit["nugget"],
            psill=fit["psill"],
            rng=fit["range_km"],
        )
        w = ordinary_kriging_weights(dist_train, dist_tt, variogram)
        rows = seasons == season
        kriging[rows] = apply_weights(w, pm_train[rows])
        logger.info(
            "Kriging %s: weights sum %.3f, %.1f%% negative weights",
            season,
            float(w.sum(axis=1).mean()),
            100 * float((w < 0).mean()),
        )
    preds["kriging"] = kriging

    # --- Linear regression on predictors + lat/lon ---------------------------
    x_train, x_test = feature_matrix(train), feature_matrix(test)
    y_train = pm_train.ravel().astype("float64")
    shape = pm_test.shape
    raw = LinearRegression().fit(x_train, y_train)
    preds["linreg"] = raw.predict(x_test).reshape(shape)
    log = LinearRegression().fit(x_train, np.log1p(y_train))
    preds["linreg_log"] = np.expm1(log.predict(x_test)).reshape(shape)
    coef = pd.DataFrame(
        {
            "feature": [*PREDICTORS, "lat", "lon"],
            "coef_raw": raw.coef_,
            "coef_log": log.coef_,
        }
    )
    coef.loc[len(coef)] = ["intercept", raw.intercept_, log.intercept_]
    coef.round(5).to_csv(args.out_dir / "linreg_coefficients.csv", index=False)
    logger.info("Linear regression fitted on %d rows, %d features", *x_train.shape)

    # --- Table of test predictions and metrics -------------------------------
    n_time, n_test = shape
    region = split["region"].values[is_test]
    dist_to_train = split["dist_to_train_km"].values[is_test]
    table = evaluation_table(
        ds["time"].values, seasons, region, dist_to_train, pm_test, preds
    )
    results = all_scores(table, METHODS)
    for name, result in results.items():
        result.round(4).to_csv(args.out_dir / f"metrics_{name}.csv", index=False)
    logger.info(
        "Overall test metrics:\n%s",
        results["overall"].round(3).to_string(index=False),
    )
    rmse_region = results["by_region"].pivot(
        index="method", columns="region", values="rmse"
    )
    logger.info(
        "Test RMSE by region:\n%s", rmse_region.loc[METHODS].round(1).to_string()
    )
    rmse_season = results["by_season"].pivot(
        index="method", columns="season", values="rmse"
    )
    logger.info(
        "Test RMSE by season:\n%s",
        rmse_season.loc[METHODS, SEASON_ORDER].round(1).to_string(),
    )
    rmse_dist = results["by_distance"].pivot(
        index="method", columns="dist_bin", values="rmse"
    )
    logger.info(
        "Test RMSE by distance to train (km):\n%s",
        rmse_dist.loc[METHODS].round(1).to_string(),
    )

    # --- Save predictions, variograms, config, figures -----------------------
    out = xr.Dataset(
        {m: (("time", "cell"), preds[m].astype("float32")) for m in METHODS}
        | {"obs": (("time", "cell"), pm_test.astype("float32"))},
        coords={
            "time": ds["time"].values,
            "lat": ("cell", lat_te),
            "lon": ("cell", lon_te),
            "region": ("cell", region),
            "dist_to_train_km": ("cell", dist_to_train),
        },
    )
    out.to_netcdf(args.out_dir / "predictions_test.nc")
    vgm_bins.round(4).to_csv(args.out_dir / "kriging_variogram_bins.csv", index=False)
    config = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "n_train_cells": int(train.sizes["cell"]),
        "n_test_cells": int(n_test),
        "n_time_steps": int(n_time),
        "delaunay_outside_hull": int(outside.sum()),
        "kriging_variograms": {
            s: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items()}
            for s, f in vgm_fits.items()
        },
        "linreg_features": [*PREDICTORS, "lat", "lon"],
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2))
    plot_overall(results["overall"], args.out_dir / "rmse_overall.png")
    plot_region_table(results["by_region"], args.out_dir / "rmse_by_region.png")
    logger.info("Done; outputs in %s", args.out_dir)


if __name__ == "__main__":
    main()
