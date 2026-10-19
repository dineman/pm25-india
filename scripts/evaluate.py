"""Step 09: evaluation of all methods on test cells, with uncertainty and a CV check.

Usage:
    conda run -n pm25 python scripts/evaluate.py
    conda run -n pm25 python scripts/evaluate.py --skip-cv     # test part only

1. Test cells: predictions of steps 06 (interpolation, linear regression),
   07 (ML models) and 08 (hybrids) in one table, scored overall and by region,
   season, region x season, distance, hour (IST), AQI level and year.
2. Block bootstrap over the 6 test blocks: uncertainty range of every score and
   of the difference between method pairs.
3. CV check on the 399 training cells: each of the 5 block folds is predicted
   only from the other 4 folds (interpolation of PM2.5, out-of-fold LightGBM,
   LightGBM + nested out-of-fold residuals interpolated), then scored and
   bootstrapped the same way over 27 blocks.
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

from pm25.baselines import idw_weights, nearest_weights, ordinary_kriging_weights
from pm25.evaluation import (
    AQI_LABELS,
    add_time_columns,
    block_bootstrap,
    block_sums,
    cv_interpolate,
    nested_oof_residuals,
)
from pm25.exploration import (
    VARIOGRAM_MODELS,
    empirical_variogram,
    fit_variogram,
    haversine_km,
    standardize_rows,
)
from pm25.features import feature_matrix, stack_cells, target_vector
from pm25.logging_utils import log_run_info, setup_logging
from pm25.metrics import DIST_BINS, GROUPINGS, scores_by_group

logger = logging.getLogger(__name__)

SEASON_ORDER = ["winter", "pre_monsoon", "monsoon", "post_monsoon"]
REGION_ORDER = ["igp", "northwest", "central", "south", "northeast", "himalaya"]
EXTRA_GROUPINGS = {
    "by_hour": ["hour_ist"],
    "by_aqi_level": ["aqi_level"],
    "by_year": ["year"],
}
# Order used in tables and figures: interpolation, regression, hybrid.
TEST_METHODS = [
    "nearest",
    "idw",
    "linear_delaunay",
    "kriging",
    "linreg",
    "ridge",
    "rf",
    "mlp",
    "lgbm",
    "lgbm_idw",
    "lgbm_rk",
]
CV_METHODS = ["nearest", "idw", "kriging", "lgbm", "lgbm_idw", "lgbm_rk"]
PAIRS = [
    ("lgbm_rk", "lgbm"),
    ("lgbm_rk", "kriging"),
    ("lgbm_rk", "nearest"),
    ("lgbm_rk", "lgbm_idw"),
    ("lgbm", "kriging"),
    ("lgbm", "nearest"),
    ("lgbm", "mlp"),
    ("lgbm", "rf"),
]
LABELS = {
    "nearest": "nearest cell",
    "idw": "IDW",
    "linear_delaunay": "linear (Delaunay)",
    "kriging": "kriging",
    "linreg": "linear regression",
    "ridge": "ridge",
    "rf": "random forest",
    "mlp": "MLP",
    "lgbm": "LightGBM",
    "lgbm_idw": "LightGBM + IDW residuals",
    "lgbm_rk": "LightGBM + kriged residuals",
}
INK, MUTED, GRID = "#2b2b29", "#6b6a63", "#e4e3de"
# Categorical slots 1-3 as in step 08: interpolation, ML, hybrid; others muted.
COLORS = {"kriging": "#2a78d6", "lgbm": "#eb6834", "lgbm_rk": "#1baf7a"}
OTHER = "#a8a79f"


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
        "--baselines",
        type=Path,
        default=Path("results/06_baselines/predictions_test.nc"),
    )
    parser.add_argument(
        "--models-dir", type=Path, default=Path("results/07_models/all")
    )
    parser.add_argument("--hybrid-dir", type=Path, default=Path("results/08_hybrid"))
    parser.add_argument("--model", default="lgbm")
    parser.add_argument("--out-dir", type=Path, default=Path("results/09_evaluation"))
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument(
        "--vgm-times", type=int, default=1000, help="time steps per season (variogram)"
    )
    parser.add_argument("--fit-max-lag-km", type=float, default=1000.0)
    parser.add_argument("--idw-k", type=int, default=8)
    parser.add_argument("--idw-power", type=float, default=2.0)
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# --- Scoring -------------------------------------------------------------------


def score_everything(
    table: pd.DataFrame,
    methods: list[str],
    label: str,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    """All groupings, block bootstrap and per-block scores; saved as CSV."""
    results = {
        name: scores_by_group(table, methods, by)
        for name, by in (GROUPINGS | EXTRA_GROUPINGS).items()
    }
    results["by_block"] = scores_by_group(table, methods, ["block"])
    sums = block_sums(table, methods, "block")
    pairs = [(a, b) for a, b in PAIRS if a in methods and b in methods]
    results["bootstrap"], results["bootstrap_pairs"] = block_bootstrap(
        sums, pairs, args.n_boot, rng
    )
    for name, result in results.items():
        result.round(4).to_csv(args.out_dir / f"{label}_{name}.csv", index=False)
    n_blocks = table["block"].nunique()
    logger.info(
        "%s: %d rows, %d blocks. Scores with 95%% block-bootstrap ranges:\n%s",
        label,
        len(table),
        n_blocks,
        results["bootstrap"][
            ["method", "rmse", "rmse_lo", "rmse_hi", "r2", "r2_lo", "r2_hi", "bias"]
        ]
        .round(3)
        .to_string(index=False),
    )
    logger.info(
        "%s: paired differences (a - b) with 95%% ranges:\n%s",
        label,
        results["bootstrap_pairs"].round(3).to_string(index=False),
    )
    for grouping, col, order in [
        ("by_region", "region", REGION_ORDER),
        ("by_season", "season", SEASON_ORDER),
        ("by_aqi_level", "aqi_level", AQI_LABELS),
    ]:
        for metric in ("rmse", "bias"):
            pivot = results[grouping].pivot(index="method", columns=col, values=metric)
            logger.info(
                "%s %s %s:\n%s",
                label,
                metric.upper(),
                grouping,
                pivot.loc[methods, [c for c in order if c in pivot.columns]]
                .round(1)
                .to_string(),
            )
    return results


def build_table(
    times: np.ndarray,
    seasons: np.ndarray,
    regions: np.ndarray,
    dist_km: np.ndarray,
    blocks: np.ndarray,
    obs: np.ndarray,
    preds: dict[str, np.ndarray],
) -> pd.DataFrame:
    """One row per (time, cell): grouping columns, block, obs and predictions."""
    n_time, n_cell = obs.shape
    table = pd.DataFrame(
        {
            "time": np.repeat(times, n_cell),
            "season": pd.Categorical(np.repeat(seasons, n_cell)),
            "region": pd.Categorical(np.tile(regions, n_time)),
            "dist_bin": pd.cut(np.tile(dist_km, n_time), DIST_BINS),
            "block": np.tile(blocks, n_time),
            "obs": obs.ravel(),
            **{m: p.ravel() for m, p in preds.items()},
        }
    )
    return add_time_columns(table)


# --- CV check --------------------------------------------------------------------


def seasonal_kriging_weights(
    values_src: np.ndarray,
    dist_src: np.ndarray,
    dist_tgt_src: np.ndarray,
    seasons: np.ndarray,
    transform: str,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Ordinary kriging weights per season, variogram fitted on source cells only.

    ``transform="log1p"`` follows step 06 (PM2.5 fields), ``"none"`` step 08
    (residual fields); both standardise each time step first.
    """
    edges = np.arange(0.0, 1500.0 + 75.0, 75.0)
    out = []
    for season in SEASON_ORDER:
        rows = seasons == season
        idx = rng.choice(
            np.flatnonzero(rows), size=min(args.vgm_times, rows.sum()), replace=False
        )
        field = values_src[idx].astype("float64")
        if transform == "log1p":
            field = np.log1p(field)
        vgm = empirical_variogram(standardize_rows(field), dist_src, edges)
        fit = fit_variogram(vgm, "exponential", args.fit_max_lag_km)
        variogram = partial(
            VARIOGRAM_MODELS[fit["model"]],
            nugget=fit["nugget"],
            psill=fit["psill"],
            rng=fit["range_km"],
        )
        out.append((rows, ordinary_kriging_weights(dist_src, dist_tgt_src, variogram)))
    return out


def run_cv_check(
    ds: xr.Dataset,
    split: xr.Dataset,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Predict every training fold from the other folds; return the CV table."""
    is_train = (split["set"] == "train").values
    seasons = ds["season"].values
    n_time = ds.sizes["time"]
    lat = split["lat"].values[np.nonzero(is_train)[0]]
    lon = split["lon"].values[np.nonzero(is_train)[1]]
    cell_folds = split["fold"].values[is_train]
    dist = haversine_km(lat, lon)

    oof = xr.open_dataset(args.hybrid_dir / f"oof_train_{args.model}.nc")
    assert np.allclose(oof["lat"].values, lat) and np.allclose(oof["lon"].values, lon)
    obs = oof["obs"].values.astype("float64")
    ml = oof["oof_pred"].values.astype("float64")

    # Nested out-of-fold residuals (cached, ~20 model fits).
    nested_path = args.out_dir / f"nested_resid_{args.model}.nc"
    if nested_path.exists():
        resid = xr.open_dataset(nested_path)["resid"].values
        logger.info("Loaded cached nested residuals from %s", nested_path)
    else:
        model_config = json.loads((args.models_dir / "config.json").read_text())
        features = model_config["features"]
        params = json.loads((args.models_dir / "best_params.json").read_text())[
            args.model
        ]
        variables = ["pm2p5"] + [f for f in features if f not in ("lat", "lon")]
        train = stack_cells(ds[variables], is_train).load()
        x, y = feature_matrix(train, features), target_vector(train)
        del train
        assert np.allclose(y.reshape(n_time, -1), obs), "training rows out of order"
        logger.info("Nested out-of-fold residuals: %s, setting %s", args.model, params)
        resid = nested_oof_residuals(
            args.model, params, x, y, cell_folds, n_time, args.seed
        )
        del x, y
        xr.Dataset(
            {"resid": (("outer_fold", "time", "cell"), resid)},
            coords={
                "outer_fold": np.unique(cell_folds),
                "time": ds["time"].values,
                "lat": ("cell", lat),
                "lon": ("cell", lon),
            },
        ).to_netcdf(nested_path)
        logger.info("Saved %s", nested_path)
    resid = resid.astype("float64")

    def fixed(make):
        return lambda v, d_src, d_ts: [(np.ones(n_time, bool), make(d_ts))]

    nearest = fixed(nearest_weights)
    idw = fixed(partial(idw_weights, k=args.idw_k, power=args.idw_power))
    krige_pm = partial(
        seasonal_kriging_weights, seasons=seasons, transform="log1p", args=args, rng=rng
    )
    krige_resid = partial(
        seasonal_kriging_weights, seasons=seasons, transform="none", args=args, rng=rng
    )
    preds = {
        "nearest": cv_interpolate(obs, dist, cell_folds, nearest),
        "idw": cv_interpolate(obs, dist, cell_folds, idw),
        "kriging": cv_interpolate(obs, dist, cell_folds, krige_pm),
        "lgbm": ml,
        "lgbm_idw": np.clip(ml + cv_interpolate(resid, dist, cell_folds, idw), 0, None),
        "lgbm_rk": np.clip(
            ml + cv_interpolate(resid, dist, cell_folds, krige_resid), 0, None
        ),
    }
    for m, p in preds.items():
        assert not np.isnan(p).any(), f"CV predictions for {m} contain NaN"

    other_fold = cell_folds[:, None] != cell_folds[None, :]
    dist_to_other = np.where(other_fold, dist, np.inf).min(axis=1)
    return build_table(
        ds["time"].values,
        seasons,
        split["region"].values[is_train],
        dist_to_other,
        split["block"].values[is_train],
        obs,
        preds,
    )


# --- Figures -------------------------------------------------------------------


def style(ax: plt.Axes) -> None:
    """Recessive axes and grid, shared by all figures."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)


def plot_overall(boot: dict[str, pd.DataFrame], path: Path) -> None:
    """RMSE with 95% block-bootstrap range per method: test vs CV check."""
    panels = [(k, v) for k, v in boot.items() if v is not None]
    fig, axes = plt.subplots(
        1, len(panels), figsize=(5 * len(panels), 4.6), sharex=True, squeeze=False
    )
    for ax, (title, table) in zip(axes[0], panels):
        table = table.set_index("method")
        methods = [m for m in TEST_METHODS if m in table.index]
        y = np.arange(len(methods))
        for i, m in enumerate(methods):
            color = COLORS.get(m, OTHER)
            row = table.loc[m]
            ax.plot([row["rmse_lo"], row["rmse_hi"]], [i, i], color=color, lw=2)
            ax.plot(
                row["rmse"], i, "o", color=color, ms=8, mec="white", mew=1.5, zorder=3
            )
        ax.set_yticks(y, [LABELS[m] for m in methods])
        ax.invert_yaxis()
        style(ax)
        ax.set_title(title, color=INK, loc="left", fontsize=10)
        ax.set_xlabel("RMSE (µg m⁻³), dot = estimate, line = 95% range", color=INK)
    fig.suptitle(
        "Overall error with block-bootstrap uncertainty", color=INK, x=0.01, ha="left"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_lines(
    tables: dict[str, pd.DataFrame],
    col: str,
    order: list[str],
    metric: str,
    xlabel: str,
    title: str,
    path: Path,
) -> None:
    """One panel per evaluation (test / CV): ``metric`` along ``col`` for 3 methods."""
    panels = [(k, v) for k, v in tables.items() if v is not None]
    fig, axes = plt.subplots(
        1, len(panels), figsize=(5.5 * len(panels), 3.8), sharey=True, squeeze=False
    )
    for ax, (panel_title, table) in zip(axes[0], panels):
        x = np.arange(len(order))
        for m, color in COLORS.items():
            sub = table[table["method"] == m].set_index(col)
            sub.index = sub.index.astype(str)
            ax.plot(
                x,
                sub.loc[order, metric],
                "-o",
                color=color,
                lw=2,
                ms=6,
                mec="white",
                mew=1.2,
                label=LABELS[m],
            )
        if metric == "bias":
            ax.axhline(0, color=MUTED, lw=1)
        ax.set_xticks(x, [o.replace("_", " ") for o in order], rotation=30, ha="right")
        style(ax)
        ax.grid(axis="y", color=GRID)
        ax.set_title(panel_title, color=INK, loc="left", fontsize=10)
        ax.set_xlabel(xlabel, color=INK)
    axes[0][0].set_ylabel(f"{metric.upper()} (µg m⁻³)", color=INK)
    axes[0][-1].legend(frameon=False, fontsize=8)
    fig.suptitle(title, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", path)


# --- Main ------------------------------------------------------------------------


def main() -> None:
    """Score all methods on test cells and in the CV check; save tables and figures."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=Path("results/logs/evaluate.log"))
    log_run_info(args)
    rng = np.random.default_rng(args.seed)

    ds = xr.open_dataset(args.data)
    split = xr.open_dataset(args.split)
    is_test = (split["set"] == "test").values
    seasons = ds["season"].values

    # --- 1-2. Test cells ---------------------------------------------------------
    sources = [
        xr.open_dataset(args.baselines),
        xr.open_dataset(args.models_dir / "predictions_test.nc"),
        xr.open_dataset(args.hybrid_dir / "predictions_test.nc"),
    ]
    obs = sources[0]["obs"].values
    preds = {}
    for src in sources:
        assert np.allclose(src["obs"].values, obs), "test sets differ between steps"
        for m in TEST_METHODS:
            if m in src and m not in preds:
                preds[m] = src[m].values
    missing = [m for m in TEST_METHODS if m not in preds]
    assert not missing, f"missing test predictions: {missing}"
    test_table = build_table(
        ds["time"].values,
        seasons,
        split["region"].values[is_test],
        split["dist_to_train_km"].values[is_test],
        split["block"].values[is_test],
        obs,
        preds,
    )
    test = score_everything(test_table, TEST_METHODS, "test", args, rng)
    del test_table

    # --- 3. CV check on training cells -------------------------------------------
    cv = None
    if not args.skip_cv:
        cv_table = run_cv_check(ds, split, args, rng)
        cv = score_everything(cv_table, CV_METHODS, "cv", args, rng)
        del cv_table

    # --- Figures and config ------------------------------------------------------
    test_title = "Test cells (6 blocks)"
    cv_title = "CV check, training cells (27 blocks)"
    plot_overall(
        {test_title: test["bootstrap"], cv_title: cv and cv["bootstrap"]},
        args.out_dir / "rmse_overall_ci.png",
    )
    for grouping, col, order, metric, xlabel, title in [
        ("by_region", "region", REGION_ORDER, "rmse", "region", "RMSE by region"),
        ("by_season", "season", SEASON_ORDER, "rmse", "season", "RMSE by season"),
        (
            "by_hour",
            "hour_ist",
            sorted(test["by_hour"]["hour_ist"].astype(str).unique()),
            "rmse",
            "hour of day (IST)",
            "RMSE by hour of day",
        ),
        (
            "by_aqi_level",
            "aqi_level",
            AQI_LABELS,
            "bias",
            "observed PM2.5 level (India AQI category)",
            "Bias by pollution level (negative = predicts too low)",
        ),
    ]:
        plot_lines(
            {test_title: test[grouping], cv_title: cv and cv[grouping]},
            col,
            order,
            metric,
            xlabel,
            title,
            args.out_dir / f"{metric}_{grouping}.png",
        )
    config = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "test_methods": TEST_METHODS,
        "cv_methods": CV_METHODS if cv else [],
        "pairs": PAIRS,
        "n_test_blocks": int(np.unique(split["block"].values[is_test]).size),
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2))
    logger.info("Done; outputs in %s", args.out_dir)


if __name__ == "__main__":
    main()
