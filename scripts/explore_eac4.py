"""Step 04: explore the preprocessed EAC4 data and fit PM2.5 variograms.

Usage:
    conda run -n pm25 python scripts/explore_eac4.py
    conda run -n pm25 python scripts/explore_eac4.py \
        --data data/processed/eac4_india_3h_2020_2021.nc --out-dir results/04_exploration

Writes summary tables (CSV), figures (PNG) and summary.json with the fitted
variogram ranges and the block size they imply for the spatial split (step 05).
Only cells inside India are used.
"""

import argparse
import json
import logging
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from pm25.exploration import (
    PREDICTORS,
    VARIOGRAM_MODELS,
    block_size_cells,
    empirical_variogram,
    fit_variogram,
    haversine_km,
    india_cells,
    linear_residuals,
    spearman_with_target,
    standardize_rows,
    summarize,
    to_table,
)
from pm25.logging_utils import log_run_info, setup_logging

logger = logging.getLogger(__name__)

REGION_ORDER = ["igp", "northwest", "central", "south", "northeast", "himalaya"]
SEASON_ORDER = ["winter", "pre_monsoon", "monsoon", "post_monsoon"]
# Categorical slots 1-6 of the project palette (validated for adjacent pairs).
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
REGION_COLORS = dict(zip(REGION_ORDER, CATEGORICAL))
SEASON_COLORS = dict(zip(SEASON_ORDER, CATEGORICAL))
INK, MUTED, GRID = "#2b2b29", "#6b6a63", "#e4e3de"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list(
    "orange_gray_blue", ["#c4501f", "#eb6834", "#e8e7e3", "#2a78d6", "#184f95"]
)
IST_OFFSET = pd.Timedelta(hours=5, minutes=30)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data", type=Path, default=Path("data/processed/eac4_india_3h.nc")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/04_exploration"))
    parser.add_argument(
        "--n-times",
        type=int,
        default=2000,
        help="random time steps used for variograms and the residual model",
    )
    parser.add_argument(
        "--bin-km", type=float, default=75.0, help="variogram bin width"
    )
    parser.add_argument(
        "--max-lag-km", type=float, default=1500.0, help="largest lag computed"
    )
    parser.add_argument(
        "--fit-max-lag-km", type=float, default=1000.0, help="largest lag used in fits"
    )
    parser.add_argument(
        "--corr-rows", type=int, default=300_000, help="rows sampled per correlation"
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def style_axes(ax: plt.Axes) -> None:
    """Recessive axes: no top/right spines, light horizontal grid."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def label_line_ends(ax: plt.Axes, x: float, ys: dict[str, float]) -> None:
    """Direct-label lines at their right end, nudging labels apart vertically."""
    lo, hi = ax.get_ylim()
    min_gap = 0.045 * (hi - lo)
    placed: list[float] = []
    for name, y in sorted(ys.items(), key=lambda kv: kv[1]):
        if placed and y - placed[-1] < min_gap:
            y = placed[-1] + min_gap
        placed.append(y)
        ax.annotate(
            name,
            (x, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK,
            annotation_clip=False,
        )


def save(fig: plt.Figure, path: Path) -> None:
    """Save a figure at report resolution and close it."""
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_seasonal_maps(ds: xr.Dataset, path: Path) -> None:
    """Mean PM2.5 per IMD season on the grid, India cells only, log colour scale."""
    pm = ds["pm2p5"].where(ds["in_india"])
    means = pm.groupby("season").mean("time").compute()
    norm = LogNorm(vmin=10, vmax=float(means.max()))
    fig, axes = plt.subplots(
        2, 2, figsize=(10, 10), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    for ax, season in zip(axes.ravel(), SEASON_ORDER):
        mesh = ax.pcolormesh(
            means["lon"],
            means["lat"],
            means.sel(season=season),
            cmap=SEQUENTIAL,
            norm=norm,
            shading="nearest",
            transform=ccrs.PlateCarree(),
        )
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor=MUTED)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor=MUTED)
        ax.set_extent([67.5, 98.0, 5.5, 38.0], crs=ccrs.PlateCarree())
        ax.set_title(season.replace("_", "-"), color=INK, fontsize=11)
    cbar = fig.colorbar(mesh, ax=axes, shrink=0.6, pad=0.03)
    cbar.set_label("Mean PM2.5 (µg m⁻³, log scale)", color=INK)
    fig.suptitle("CAMS EAC4 mean PM2.5 by IMD season", color=INK, fontsize=13)
    save(fig, path)


def plot_monthly_cycle(table: pd.DataFrame, path: Path) -> None:
    """Mean PM2.5 by calendar month for each region."""
    month = table["time"].dt.month
    means = table.groupby([month, "region"], observed=True)["pm2p5"].mean().unstack()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for region in REGION_ORDER:
        ax.plot(
            means.index, means[region], color=REGION_COLORS[region], lw=2, label=region
        )
    style_axes(ax)
    ax.set_xticks(
        range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    )
    ax.set_xlim(1, 12)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Mean PM2.5 (µg m⁻³)", color=INK)
    ax.set_title("Seasonal cycle of PM2.5 by region", color=INK, loc="left")
    label_line_ends(ax, 12, {r: means.loc[12, r] for r in REGION_ORDER})
    ax.legend(frameon=False, ncols=6, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    save(fig, path)


def plot_diurnal_cycle(table: pd.DataFrame, path: Path) -> None:
    """Mean PM2.5 by IST hour for each region, one panel per season."""
    hour = (table["time"] + IST_OFFSET).dt.hour + 0.5
    means = table.groupby(["season", hour, "region"], observed=True)["pm2p5"].mean()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, season in zip(axes.ravel(), SEASON_ORDER):
        sub = means.loc[season].unstack()
        for region in REGION_ORDER:
            ax.plot(
                sub.index,
                sub[region],
                color=REGION_COLORS[region],
                lw=2,
                marker="o",
                ms=4,
                label=region,
            )
        style_axes(ax)
        ax.set_title(season.replace("_", "-"), color=INK, loc="left", fontsize=10)
        ax.set_xticks(sub.index, [f"{h:04.1f}".replace(".5", ":30") for h in sub.index])
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean PM2.5 (µg m⁻³)", color=INK)
        ax.set_ylim(bottom=0)
    for ax in axes[1]:
        ax.set_xlabel("Hour (IST)", color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncols=6,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("Daily cycle of PM2.5 by region and season", color=INK, fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, path)


def plot_correlations(corr: pd.DataFrame, path: Path) -> None:
    """Heatmap of Spearman correlation with PM2.5, predictors x season."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(corr.shape[1]), [c.replace("_", "-") for c in corr.columns])
    ax.set_yticks(range(corr.shape[0]), corr.index)
    for (r, c), value in np.ndenumerate(corr.values):
        ax.text(
            c,
            r,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if abs(value) > 0.5 else INK,
        )
    ax.tick_params(length=0, labelcolor=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Spearman correlation with PM2.5", color=INK)
    ax.set_title("Rank correlation of each predictor with PM2.5", color=INK, loc="left")
    save(fig, path)


def plot_variograms(
    vgms: dict[str, dict[str, pd.DataFrame]],
    fits: dict[str, dict[str, float]],
    path: Path,
) -> None:
    """One panel per field: seasonal variograms + pooled fit."""
    titles = {
        "log_pm25": "log(1 + PM2.5)",
        "residual": "Residual: predictors",
        "residual_trend": "Residual: predictors + lat/lon trend",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, field in zip(axes, titles):
        for season in SEASON_ORDER:
            v = vgms[field][season]
            ax.plot(
                v["lag_km"],
                v["gamma"],
                color=SEASON_COLORS[season],
                lw=2,
                label=season.replace("_", "-"),
            )
        v = vgms[field]["all"]
        ax.plot(v["lag_km"], v["gamma"], "o", color=INK, ms=5, label="all seasons")
        fit = fits[field]
        h = np.linspace(0, v["lag_km"].max(), 200)
        params = (fit["nugget"], fit["psill"], fit["range_km"])
        ax.plot(
            h,
            VARIOGRAM_MODELS[fit["model"]](h, *params),
            "--",
            color=INK,
            lw=1.2,
            label=f"{fit['model']} fit",
        )
        if fit["range_km"] <= v["lag_km"].max():
            ax.axvline(fit["range_km"], color=MUTED, lw=1, ls=":")
            ax.annotate(
                f"range ≈ {fit['range_km']:.0f} km",
                (fit["range_km"], 1.3),
                xytext=(4, 0),
                textcoords="offset points",
                color=INK,
                fontsize=9,
            )
        else:
            ax.text(
                0.03,
                0.93,
                f"no sill: fitted range {fit['range_km']:.0f} km is off the plot",
                transform=ax.transAxes,
                color=INK,
                fontsize=9,
            )
        style_axes(ax)
        ax.set_title(titles[field], color=INK, loc="left", fontsize=10)
        ax.set_xlabel("Distance between cells (km)", color=INK)
        ax.set_xlim(0, v["lag_km"].max() + 40)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Semivariance (standardised)", color=INK)
    axes[-1].legend(frameon=False, loc="lower right", fontsize=9)
    fig.suptitle(
        "Empirical variograms over India cells (each field standardised per time step)",
        color=INK,
        fontsize=13,
    )
    fig.tight_layout()
    save(fig, path)


def season_correlations(
    table: pd.DataFrame, n_rows: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Spearman correlation of each predictor with PM2.5, overall and per season."""
    groups = {"all": table} | {s: table[table["season"] == s] for s in SEASON_ORDER}
    out = {}
    for name, sub in groups.items():
        idx = rng.choice(len(sub), size=min(n_rows, len(sub)), replace=False)
        out[name] = spearman_with_target(sub.iloc[idx], PREDICTORS)
    return pd.DataFrame(out)


def variogram_fields(cells: xr.Dataset, time_idx: np.ndarray) -> dict[str, np.ndarray]:
    """The (time, cell) fields whose variograms are computed.

    ``log_pm25``: log1p(PM2.5); the log tames the extreme values that would
    otherwise dominate squared differences. ``residual``: what is left after a
    linear model on the predictors (fitted on the same time steps), which is
    the spatial structure a predictor-based model cannot explain.
    ``residual_trend``: as ``residual`` but the model also gets a quadratic
    lat/lon trend, removing the country-wide north-south gradient so only
    local correlation remains. This is the field that sets the block size.
    """
    sub = cells.isel(time=time_idx)
    log_pm = np.log1p(sub["pm2p5"].transpose("time", "cell").values.astype("float64"))
    feats = []
    for name in PREDICTORS:
        arr = sub[name].broadcast_like(sub["pm2p5"]).transpose("time", "cell").values
        feats.append(np.log1p(arr) if name == "aod550" else arr)
    features = np.stack([f.ravel() for f in feats], axis=1).astype("float64")
    resid = linear_residuals(log_pm.ravel(), features).reshape(log_pm.shape)

    lat = sub["lat"].values - sub["lat"].values.mean()
    lon = sub["lon"].values - sub["lon"].values.mean()
    trend = np.stack([lat, lon, lat**2, lon**2, lat * lon], axis=1)
    trend = np.tile(trend, (sub.sizes["time"], 1))
    resid_trend = linear_residuals(log_pm.ravel(), np.hstack([features, trend]))
    return {
        "log_pm25": standardize_rows(log_pm),
        "residual": standardize_rows(resid),
        "residual_trend": standardize_rows(resid_trend.reshape(log_pm.shape)),
    }


def main() -> None:
    """Run all exploration outputs for step 04."""
    args = parse_args()
    setup_logging(log_file=Path("results/logs/explore_eac4.log"))
    log_run_info(args)
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(args.data)
    variables = ["pm2p5", *PREDICTORS]
    cells = india_cells(ds[variables]).load()
    logger.info(
        "Loaded %s: %d time steps (%s to %s), %d India cells",
        args.data,
        cells.sizes["time"],
        str(cells["time"].values[0])[:10],
        str(cells["time"].values[-1])[:10],
        cells.sizes["cell"],
    )

    # --- Summaries -------------------------------------------------------
    table = to_table(ds[variables], variables)
    table["ist_hour"] = (table["time"] + IST_OFFSET).dt.hour
    for name, by in {
        "summary_region_season": ["region", "season"],
        "summary_region": ["region"],
        "summary_season": ["season"],
        "summary_region_season_hour": ["region", "season", "ist_hour"],
    }.items():
        summarize(table, by).round(2).to_csv(args.out_dir / f"{name}.csv", index=False)
    overall = summarize(table.assign(all="india"), ["all"]).iloc[0]
    logger.info(
        "PM2.5 over India cells: mean %.1f, median %.1f, p95 %.1f ug m-3 (%d rows)",
        overall["mean"],
        overall["median"],
        overall["p95"],
        overall["count"],
    )
    by_rs = summarize(table, ["region", "season"]).set_index(["region", "season"])
    logger.info(
        "Mean PM2.5 by region x season:\n%s",
        by_rs["mean"].unstack()[SEASON_ORDER].round(1).to_string(),
    )

    corr = season_correlations(table, args.corr_rows, rng)
    corr.round(3).to_csv(args.out_dir / "spearman_pm25_by_season.csv")
    logger.info("Spearman correlation with PM2.5:\n%s", corr.round(2).to_string())

    plot_seasonal_maps(ds, args.out_dir / "seasonal_mean_maps.png")
    plot_monthly_cycle(table, args.out_dir / "monthly_cycle_by_region.png")
    plot_diurnal_cycle(table, args.out_dir / "diurnal_cycle_by_region.png")
    plot_correlations(corr, args.out_dir / "spearman_heatmap.png")
    del table

    # --- Variograms ------------------------------------------------------
    n_times = min(args.n_times, cells.sizes["time"])
    time_idx = np.sort(rng.choice(cells.sizes["time"], size=n_times, replace=False))
    dist = haversine_km(cells["lat"].values, cells["lon"].values)
    edges = np.arange(0.0, args.max_lag_km + args.bin_km, args.bin_km)
    fields = variogram_fields(cells, time_idx)
    seasons = cells["season"].values[time_idx]

    vgms: dict[str, dict[str, pd.DataFrame]] = {}
    fits: dict[str, dict[str, float]] = {}
    all_fits = []
    for field, values in fields.items():
        vgms[field] = {"all": empirical_variogram(values, dist, edges)}
        for season in SEASON_ORDER:
            vgms[field][season] = empirical_variogram(
                values[seasons == season], dist, edges
            )
        for group, vgm in vgms[field].items():
            for model in VARIOGRAM_MODELS:
                fit = fit_variogram(vgm, model, max_lag_km=args.fit_max_lag_km)
                all_fits.append({"field": field, "group": group, **fit})
        pooled = [f for f in all_fits if f["field"] == field and f["group"] == "all"]
        fits[field] = min(pooled, key=lambda f: f["rmse"])
        logger.info(
            "Variogram %s (all seasons): best %s, range %.0f km, nugget/sill %.2f",
            field,
            fits[field]["model"],
            fits[field]["range_km"],
            fits[field]["nugget"] / fits[field]["sill"],
        )

    pd.concat(
        [v.assign(field=f, group=g) for f, d in vgms.items() for g, v in d.items()]
    ).round(4).to_csv(args.out_dir / "variogram_bins.csv", index=False)
    fits_table = pd.DataFrame(all_fits).round(4)
    fits_table.to_csv(args.out_dir / "variogram_fits.csv", index=False)
    logger.info("All variogram fits:\n%s", fits_table.to_string(index=False))
    plot_variograms(vgms, fits, args.out_dir / "variograms.png")

    summary = {
        "data": str(args.data),
        "n_time_steps": int(cells.sizes["time"]),
        "n_india_cells": int(cells.sizes["cell"]),
        "variogram_time_steps": int(n_times),
        "seed": args.seed,
        "variogram": {
            field: {
                "model": fit["model"],
                "range_km": round(fit["range_km"], 1),
                "nugget_to_sill": round(fit["nugget"] / fit["sill"], 3),
                "block_size_cells": block_size_cells(fit["range_km"]),
                # A range past the fitted lags is extrapolated: no sill was seen.
                "range_beyond_fitted_lags": fit["range_km"] > args.fit_max_lag_km,
            }
            for field, fit in fits.items()
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Summary: %s", json.dumps(summary["variogram"]))


if __name__ == "__main__":
    main()
