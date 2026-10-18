"""Step 08: regression kriging, ML prediction + kriged ML residuals.

Usage:
    conda run -n pm25 python scripts/run_hybrid.py

1. Out-of-fold ML predictions at all training cells (model fitted on the other
   CV folds), giving honest residuals = observed - predicted (cached).
2. Variogram of those residuals per season, on training cells only.
3. At each time step, krige (and, as a check, IDW-interpolate) the training
   residuals to the test cells and add them to the step 07 ML test prediction.
4. Score ML only, ML + residual interpolation, and interpolation only on the
   test cells with the shared metrics.
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

from pm25.baselines import idw_weights, ordinary_kriging_weights
from pm25.exploration import (
    VARIOGRAM_MODELS,
    empirical_variogram,
    fit_variogram,
    haversine_km,
    standardize_rows,
)
from pm25.features import feature_matrix, stack_cells, target_vector
from pm25.hybrid import add_residual_correction, out_of_fold_predictions
from pm25.logging_utils import log_run_info, setup_logging
from pm25.metrics import all_scores, evaluation_table, scores

logger = logging.getLogger(__name__)

SEASON_ORDER = ["winter", "pre_monsoon", "monsoon", "post_monsoon"]
REGION_ORDER = ["igp", "northwest", "central", "south", "northeast", "himalaya"]
BASELINE_METHODS = ["nearest", "idw", "kriging"]
INK, MUTED, GRID = "#2b2b29", "#6b6a63", "#e4e3de"
# Categorical slots 1-3 (validated for all pairs): interpolation, ML, hybrid.
COLORS = {"kriging": "#2a78d6", "lgbm": "#eb6834", "lgbm_rk": "#1baf7a"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data", type=Path, default=Path("data/processed/eac4_india_3h.nc")
    )
    parser.add_argument(
        "--split", type=Path, default=Path("data/processed/splits/split_block5.nc")
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("results/07_models/all"),
        help="step 07 run: best_params.json, config.json, predictions_test.nc",
    )
    parser.add_argument("--model", default="lgbm")
    parser.add_argument(
        "--baselines",
        type=Path,
        default=Path("results/06_baselines/predictions_test.nc"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/08_hybrid"))
    parser.add_argument(
        "--vgm-times", type=int, default=1000, help="time steps per season (variogram)"
    )
    parser.add_argument("--fit-max-lag-km", type=float, default=1000.0)
    parser.add_argument("--idw-k", type=int, default=8)
    parser.add_argument("--idw-power", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def residual_variograms(
    resid: np.ndarray,
    seasons: np.ndarray,
    dist_train: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[dict[str, dict], pd.DataFrame]:
    """Exponential variogram of standardised residual fields, one per season."""
    edges = np.arange(0.0, 1500.0 + 75.0, 75.0)
    fits, bins = {}, []
    for season in SEASON_ORDER:
        idx = np.flatnonzero(seasons == season)
        idx = rng.choice(idx, size=min(args.vgm_times, idx.size), replace=False)
        vgm = empirical_variogram(
            standardize_rows(resid[idx].astype("float64")), dist_train, edges
        )
        fits[season] = fit_variogram(vgm, "exponential", args.fit_max_lag_km)
        bins.append(vgm.assign(season=season))
        logger.info(
            "Residual variogram %s: range %.0f km, nugget/sill %.2f%s",
            season,
            fits[season]["range_km"],
            fits[season]["nugget"] / fits[season]["sill"],
            "  (range beyond fitted lags)"
            if fits[season]["range_km"] > args.fit_max_lag_km
            else "",
        )
    return fits, pd.concat(bins)


def plot_by_region(by_region: pd.DataFrame, path: Path) -> None:
    """Dot plot of test RMSE per region: interpolation vs ML vs hybrid."""
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(REGION_ORDER))
    offsets = {"kriging": -0.2, "lgbm": 0.0, "lgbm_rk": 0.2}
    labels = {
        "kriging": "kriging (interpolation only)",
        "lgbm": "LightGBM (ML only)",
        "lgbm_rk": "LightGBM + residual kriging",
    }
    for method, color in COLORS.items():
        sub = by_region[by_region["method"] == method].set_index("region")
        ax.plot(
            sub.loc[REGION_ORDER, "rmse"],
            y + offsets[method],
            "o",
            color=color,
            ms=8,
            label=labels[method],
            markeredgecolor="white",
            markeredgewidth=1.5,
        )
    ax.set_yticks(y, REGION_ORDER)
    ax.invert_yaxis()
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.set_xlim(left=0)
    ax.set_xlabel("RMSE on test cells (µg m⁻³)", color=INK)
    ax.set_title("Test RMSE by region", color=INK, loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", path)


def main() -> None:
    """Build out-of-fold residuals, krige them and score the hybrid."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=Path("results/logs/run_hybrid.log"))
    log_run_info(args)
    rng = np.random.default_rng(args.seed)

    model_config = json.loads((args.models_dir / "config.json").read_text())
    features = model_config["features"]
    params = json.loads((args.models_dir / "best_params.json").read_text())[args.model]
    logger.info("Model %s, setting %s", args.model, params)
    logger.info("Features: %s", ", ".join(features))

    ds = xr.open_dataset(args.data)
    split = xr.open_dataset(args.split)
    is_train = (split["set"] == "train").values
    is_test = (split["set"] == "test").values
    seasons = ds["season"].values
    n_time = ds.sizes["time"]
    lat_tr = split["lat"].values[np.nonzero(is_train)[0]]
    lon_tr = split["lon"].values[np.nonzero(is_train)[1]]
    lat_te = split["lat"].values[np.nonzero(is_test)[0]]
    lon_te = split["lon"].values[np.nonzero(is_test)[1]]

    # --- 1. Out-of-fold residuals at training cells (cached) ----------------
    oof_path = args.out_dir / f"oof_train_{args.model}.nc"
    if oof_path.exists():
        oof = xr.open_dataset(oof_path)
        obs_train, pred_train = oof["obs"].values, oof["oof_pred"].values
        logger.info("Loaded cached out-of-fold predictions from %s", oof_path)
    else:
        variables = ["pm2p5"] + [f for f in features if f not in ("lat", "lon")]
        train = stack_cells(ds[variables], is_train).load()
        x_train, y_train = feature_matrix(train, features), target_vector(train)
        del train
        fold_rows = np.tile(split["fold"].values[is_train], n_time)
        logger.info("Out-of-fold predictions on %d training rows", len(y_train))
        oof_flat = out_of_fold_predictions(
            args.model, params, x_train, y_train, fold_rows, args.seed
        )
        obs_train = y_train.reshape(n_time, -1)
        pred_train = oof_flat.reshape(n_time, -1)
        del x_train
        xr.Dataset(
            {
                "obs": (("time", "cell"), obs_train),
                "oof_pred": (("time", "cell"), pred_train),
            },
            coords={
                "time": ds["time"].values,
                "lat": ("cell", lat_tr),
                "lon": ("cell", lon_tr),
            },
        ).to_netcdf(oof_path)
        logger.info("Saved %s", oof_path)
    resid_train = (obs_train - pred_train).astype("float64")
    oof_scores = scores(obs_train.ravel().astype("float64"), pred_train.ravel())
    logger.info(
        "Out-of-fold %s on training cells: RMSE %.2f, MAE %.2f, R² %.3f, bias %.2f",
        args.model,
        oof_scores["rmse"],
        oof_scores["mae"],
        oof_scores["r2"],
        oof_scores["bias"],
    )

    # --- 2. Residual variograms and interpolation weights -------------------
    dist_train = haversine_km(lat_tr, lon_tr)
    dist_tt = haversine_km(
        np.concatenate([lat_te, lat_tr]), np.concatenate([lon_te, lon_tr])
    )[: lat_te.size, lat_te.size :]
    fits, vgm_bins = residual_variograms(resid_train, seasons, dist_train, args, rng)

    # --- 3. ML + interpolated residuals at test cells ------------------------
    ml_run = xr.open_dataset(args.models_dir / "predictions_test.nc")
    ml_test = ml_run[args.model].values.astype("float64")
    obs_test = ml_run["obs"].values
    rk = np.empty_like(ml_test)
    for season, fit in fits.items():
        variogram = partial(
            VARIOGRAM_MODELS[fit["model"]],
            nugget=fit["nugget"],
            psill=fit["psill"],
            rng=fit["range_km"],
        )
        w = ordinary_kriging_weights(dist_train, dist_tt, variogram)
        rows = seasons == season
        rk[rows] = add_residual_correction(ml_test[rows], resid_train[rows], w)
    idw = add_residual_correction(
        ml_test, resid_train, idw_weights(dist_tt, args.idw_k, args.idw_power)
    )
    name = args.model
    preds = {name: ml_test, f"{name}_rk": rk, f"{name}_idw": idw}

    # --- 4. Score next to interpolation-only baselines -----------------------
    base = xr.open_dataset(args.baselines)
    assert np.allclose(base["obs"].values, obs_test), "baseline test set differs"
    for m in BASELINE_METHODS:
        preds[m] = base[m].values
    methods = list(preds)
    table = evaluation_table(
        ds["time"].values,
        seasons,
        split["region"].values[is_test],
        split["dist_to_train_km"].values[is_test],
        obs_test,
        preds,
    )
    results = all_scores(table, methods)
    for grouping, result in results.items():
        result.round(4).to_csv(args.out_dir / f"metrics_{grouping}.csv", index=False)
    logger.info(
        "Test metrics (overall):\n%s",
        results["overall"].round(3).to_string(index=False),
    )
    for grouping, col, order in [
        ("by_region", "region", REGION_ORDER),
        ("by_season", "season", SEASON_ORDER),
    ]:
        pivot = results[grouping].pivot(index="method", columns=col, values="rmse")
        logger.info(
            "Test RMSE %s:\n%s",
            grouping,
            pivot.loc[methods, order].round(1).to_string(),
        )
    dist = results["by_distance"].pivot(
        index="method", columns="dist_bin", values="rmse"
    )
    logger.info(
        "Test RMSE by distance (km):\n%s", dist.loc[methods].round(1).to_string()
    )

    xr.Dataset(
        {m: (("time", "cell"), preds[m].astype("float32")) for m in methods[:3]}
        | {"obs": (("time", "cell"), obs_test)},
        coords={
            "time": ds["time"].values,
            "lat": ("cell", lat_te),
            "lon": ("cell", lon_te),
            "region": ("cell", split["region"].values[is_test]),
            "dist_to_train_km": ("cell", split["dist_to_train_km"].values[is_test]),
        },
    ).to_netcdf(args.out_dir / "predictions_test.nc")
    vgm_bins.round(4).to_csv(args.out_dir / "residual_variogram_bins.csv", index=False)
    config = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "model": args.model,
        "params": params,
        "features": features,
        "oof_train_scores": {k: round(v, 4) for k, v in oof_scores.items()},
        "residual_variograms": {
            s: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items()}
            for s, f in fits.items()
        },
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2))
    plot_by_region(results["by_region"], args.out_dir / "rmse_by_region.png")
    logger.info("Done; outputs in %s", args.out_dir)


if __name__ == "__main__":
    main()
