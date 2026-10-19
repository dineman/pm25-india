"""Step 11: spatially enhanced PM2.5 maps over India at 0.25° (CAMS: 0.75°).

Usage:
    conda run -n pm25 python scripts/make_maps.py

Methods mapped at every 0.25° point inside India's official boundary:
- bilinear: CAMS PM2.5 regridded bilinearly (standard interpolation);
- kriging:  ordinary kriging of PM2.5 from the 494 India cells (step 06 variograms);
- lgbm:     the step 07 LightGBM applied to bilinearly interpolated predictors;
- hybrid:   lgbm + ML residuals of the 494 India cells, kriged (step 08 variograms).
The residuals are CAMS minus the *same* LightGBM's prediction at each cell, so
the hybrid reproduces CAMS at the cell centres. Out-of-fold residuals (step 08)
would double-count at training cells, which the final model has already seen.

Outputs: seasonal mean maps, three example time steps, a north India zoom, and a
consistency check (fine maps averaged back to 0.75° vs CAMS).
"""

import argparse
import json
import logging
from functools import partial
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

from pm25.baselines import ordinary_kriging_weights
from pm25.exploration import VARIOGRAM_MODELS, haversine_km
from pm25.features import feature_matrix, stack_cells, target_vector
from pm25.logging_utils import log_run_info, setup_logging
from pm25.maps import (
    aggregate_to_coarse,
    coarse_membership,
    fine_axis,
    load_india_official,
    points_inside,
    predictors_at_points,
)
from pm25.metrics import scores

logger = logging.getLogger(__name__)

SEASON_ORDER = ["winter", "pre_monsoon", "monsoon", "post_monsoon"]
METHODS = ["bilinear", "kriging", "lgbm", "hybrid"]
LABELS = {
    "cams": "CAMS (0.75°)",
    "bilinear": "Bilinear (0.25°)",
    "kriging": "Kriging (0.25°)",
    "lgbm": "LightGBM (0.25°)",
    "hybrid": "LightGBM + kriged residuals (0.25°)",
}
INK, MUTED = "#2b2b29", "#6b6a63"
PM_BINS = [0, 15, 30, 45, 60, 90, 120, 180, 250, 400]
PM_CMAP = LinearSegmentedColormap.from_list(
    "pm", ["#fdf3e9", "#f6b37e", "#eb6834", "#a8390f", "#4a1503"]
)
DIFF_BINS = [-60, -30, -15, -5, 5, 15, 30, 60]
DIFF_CMAP = LinearSegmentedColormap.from_list(
    "diff", ["#174a8b", "#2a78d6", "#f0efec", "#eb6834", "#8a2f0b"]
)
ZOOM = {"lon": (72.0, 82.0), "lat": (26.0, 37.5)}


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
        "--models-dir", type=Path, default=Path("results/07_models/all")
    )
    parser.add_argument("--hybrid-dir", type=Path, default=Path("results/08_hybrid"))
    parser.add_argument(
        "--baselines-config",
        type=Path,
        default=Path("results/06_baselines/config.json"),
    )
    parser.add_argument("--model", default="lgbm")
    parser.add_argument("--factor", type=int, default=3, help="0.75° / factor")
    parser.add_argument("--out-dir", type=Path, default=Path("results/11_maps"))
    parser.add_argument("--year", type=int, default=2024, help="year of the examples")
    parser.add_argument(
        "--max-months", type=int, default=None, help="limit months (smoke test)"
    )
    return parser.parse_args()


# --- Setup ---------------------------------------------------------------------


def all_cell_residuals(
    ds: xr.Dataset, split: xr.Dataset, model, features: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lat, lon, region, CAMS PM2.5 and residual of ``model`` at all India cells.

    Residual = CAMS − prediction of the same fitted model that makes the fine
    map, so kriging it back reproduces CAMS at the cells (exactly where the
    variogram has no nugget).
    """
    is_india = np.isin(split["set"].values, ["train", "test"])
    variables = ["pm2p5"] + [f for f in features if f not in ("lat", "lon")]
    cells = stack_cells(ds[variables], is_india).load()
    n_time = cells.sizes["time"]
    obs = target_vector(cells).reshape(n_time, -1).astype("float64")
    pred = model.predict(feature_matrix(cells, features)).reshape(n_time, -1)
    region = stack_cells(split[["region"]], is_india)["region"].values
    return cells["lat"].values, cells["lon"].values, region, obs, obs - pred


def kriging_weights_by_season(
    fits: dict[str, dict], dist_cc: np.ndarray, dist_fc: np.ndarray
) -> dict[str, np.ndarray]:
    """Ordinary kriging weights (fine point x cell) for each season's variogram."""
    out = {}
    for season, fit in fits.items():
        variogram = partial(
            VARIOGRAM_MODELS[fit["model"]],
            nugget=fit["nugget"],
            psill=fit["psill"],
            rng=fit["range_km"],
        )
        out[season] = ordinary_kriging_weights(dist_cc, dist_fc, variogram)
    return out


def pick_examples(
    ds: xr.Dataset, obs: np.ndarray, region: np.ndarray, year: int
) -> dict[str, int]:
    """Indices of three example time steps in ``year``, chosen from CAMS itself."""
    times = pd.DatetimeIndex(ds["time"].values)
    in_year = times.year == year
    igp = obs[:, region == "igp"].mean(axis=1)
    ne = obs[:, region == "northeast"].mean(axis=1)
    india = obs.mean(axis=1)
    smog = np.flatnonzero(in_year)[np.argmax(igp[in_year])]
    fire_rows = in_year & times.month.isin([3, 4, 5])
    fire = np.flatnonzero(fire_rows)[np.argmax(ne[fire_rows])]
    # Monsoon afternoon: July, 09 UTC (14:30 IST), the day with the median India mean.
    mon_rows = np.flatnonzero(in_year & (times.month == 7) & (times.hour == 9))
    monsoon = mon_rows[np.argsort(india[mon_rows])[len(mon_rows) // 2]]
    return {
        "smog_igp": int(smog),
        "fire_northeast": int(fire),
        "monsoon_afternoon": int(monsoon),
    }


# --- Figures -------------------------------------------------------------------


def to_grid(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Values at the masked points -> full (lat, lon) grid with NaN elsewhere."""
    grid = np.full(mask.shape, np.nan)
    grid[mask] = values
    return grid


def edges(axis: np.ndarray) -> np.ndarray:
    """Cell edges for pcolormesh from regularly spaced centres."""
    step = axis[1] - axis[0]
    return np.concatenate([axis - step / 2, [axis[-1] + step / 2]])


def draw_outlines(ax: plt.Axes, countries: list, india: shapely.Geometry) -> None:
    """Neighbouring countries in grey, India (official boundary) in dark ink."""
    for geom, color, lw in [(g, "#a8a79f", 0.5) for g in countries] + [
        (india, INK, 0.9)
    ]:
        for poly in getattr(geom, "geoms", [geom]):
            x, y = poly.exterior.xy
            ax.plot(x, y, color=color, lw=lw)


def map_axes(ax: plt.Axes, extent: dict | None = None) -> None:
    """Equal-area-ish lon/lat axes without frame clutter."""
    extent = extent or {"lon": (67.5, 98.0), "lat": (5.5, 38.0)}
    ax.set_xlim(*extent["lon"])
    ax.set_ylim(*extent["lat"])
    ax.set_aspect(1 / np.cos(np.radians(np.mean(extent["lat"]))))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_panels(
    panels: list[list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]],
    row_titles: list[str],
    outlines: tuple[list, shapely.Geometry],
    path: Path,
    title: str,
    extent: dict | None = None,
    diff_last: bool = False,
) -> None:
    """Grid of maps. Each panel: (title, lat edges, lon edges, values)."""
    n_row, n_col = len(panels), len(panels[0])
    fig, axes = plt.subplots(
        n_row, n_col, figsize=(3.6 * n_col, 3.9 * n_row), squeeze=False
    )
    pm_norm = BoundaryNorm(PM_BINS, PM_CMAP.N, extend="max")
    diff_norm = BoundaryNorm(DIFF_BINS, DIFF_CMAP.N, extend="both")
    pm_im = diff_im = None
    for r, row in enumerate(panels):
        for c, (panel_title, lat_e, lon_e, values) in enumerate(row):
            ax = axes[r][c]
            is_diff = diff_last and c == n_col - 1
            im = ax.pcolormesh(
                lon_e,
                lat_e,
                values,
                cmap=DIFF_CMAP if is_diff else PM_CMAP,
                norm=diff_norm if is_diff else pm_norm,
            )
            if is_diff:
                diff_im = im
            else:
                pm_im = im
            draw_outlines(ax, *outlines)
            map_axes(ax, extent)
            if r == 0:
                ax.set_title(panel_title, color=INK, fontsize=9, loc="left")
            if c == 0:
                ax.text(
                    -0.04,
                    0.5,
                    row_titles[r],
                    transform=ax.transAxes,
                    rotation=90,
                    ha="right",
                    va="center",
                    color=INK,
                    fontsize=10,
                )
    fig.suptitle(title, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.965))
    cax = fig.add_axes((0.1, 0.035, 0.45 if diff_last else 0.8, 0.015))
    fig.colorbar(pm_im, cax=cax, orientation="horizontal").set_label(
        "PM2.5 (µg m⁻³)", color=INK
    )
    if diff_last:
        dax = fig.add_axes((0.62, 0.035, 0.3, 0.015))
        fig.colorbar(diff_im, cax=dax, orientation="horizontal").set_label(
            "hybrid − bilinear (µg m⁻³)", color=INK
        )
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", path)


# --- Main ----------------------------------------------------------------------


def main() -> None:
    """Predict every month on the fine grid, accumulate means, save maps and checks."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=Path("results/logs/make_maps.log"))
    log_run_info(args)

    ds = xr.open_dataset(args.data)
    split = xr.open_dataset(args.split)
    seasons = ds["season"].values
    times = pd.DatetimeIndex(ds["time"].values)
    features = json.loads((args.models_dir / "config.json").read_text())["features"]
    model = joblib.load(args.models_dir / f"model_{args.model}.joblib")

    # Fine points inside India's official boundary.
    india = load_india_official()
    fine_lat = fine_axis(ds["lat"].values, args.factor)
    fine_lon = fine_axis(ds["lon"].values, args.factor)
    mask = points_inside(fine_lat, fine_lon, india)
    lon2, lat2 = np.meshgrid(fine_lon, fine_lat)
    pt_lat, pt_lon = lat2[mask], lon2[mask]
    logger.info(
        "Fine grid %.2f°: %d x %d, %d points inside India (official boundary)",
        fine_lat[1] - fine_lat[0],
        fine_lat.size,
        fine_lon.size,
        mask.sum(),
    )

    # CAMS cells, residuals and interpolation weights.
    c_lat, c_lon, c_region, c_obs, c_resid = all_cell_residuals(
        ds, split, model, features
    )
    dist = haversine_km(
        np.concatenate([pt_lat, c_lat]), np.concatenate([pt_lon, c_lon])
    )
    n_pt = pt_lat.size
    dist_fc, dist_cc = dist[:n_pt, n_pt:], dist[n_pt:, n_pt:]
    del dist
    resid_fits = json.loads((args.hybrid_dir / "config.json").read_text())[
        "residual_variograms"
    ]
    pm_fits = json.loads(args.baselines_config.read_text())["kriging_variograms"]
    w_resid = kriging_weights_by_season(resid_fits, dist_cc, dist_fc)
    w_pm = kriging_weights_by_season(pm_fits, dist_cc, dist_fc)
    logger.info("Kriging weights for %d cells -> %d fine points", c_lat.size, n_pt)
    members = coarse_membership(pt_lat, pt_lon, c_lat, c_lon, half_width=0.375 + 1e-9)
    full_cells = sum(m.size == args.factor**2 for m in members)
    logger.info(
        "%d of %d India cells fully covered by fine points", full_cells, c_lat.size
    )

    examples = pick_examples(ds, c_obs, c_region, args.year)
    for name, i in examples.items():
        logger.info("Example %s: %s UTC (%s)", name, times[i], seasons[i])

    # --- Monthly loop: predict, accumulate seasonal sums -------------------------
    sums = {m: {s: np.zeros(n_pt) for s in SEASON_ORDER} for m in METHODS}
    counts = {s: 0 for s in SEASON_ORDER}
    abs_diff = {s: np.zeros(n_pt) for s in SEASON_ORDER}
    snapshots = {m: {} for m in METHODS}
    agg = {m: {"obs": [], "pred": []} for m in METHODS}
    months = pd.period_range(times[0], times[-1], freq="M")
    if args.max_months:
        months = months[: args.max_months]
    for k, month in enumerate(months):
        rows = np.flatnonzero((times.year == month.year) & (times.month == month.month))
        chunk = ds.isel(time=rows)
        fine = predictors_at_points(chunk, pt_lat, pt_lon).load()
        n_t = rows.size
        ml = model.predict(feature_matrix(fine, features)).reshape(n_t, n_pt)
        bilinear = fine["pm2p5"].transpose("time", "cell").values.astype("float64")
        season_rows = seasons[rows]
        hybrid = np.empty_like(ml)
        kriging = np.empty_like(ml)
        for s in np.unique(season_rows):
            r = season_rows == s
            hybrid[r] = np.clip(ml[r] + c_resid[rows][r] @ w_resid[s].T, 0, None)
            kriging[r] = c_obs[rows][r] @ w_pm[s].T
        preds = {"bilinear": bilinear, "kriging": kriging, "lgbm": ml, "hybrid": hybrid}
        for s in np.unique(season_rows):
            r = season_rows == s
            counts[s] += int(r.sum())
            abs_diff[s] += np.abs(hybrid[r] - bilinear[r]).sum(axis=0)
            for m, p in preds.items():
                sums[m][s] += p[r].sum(axis=0)
        for m, p in preds.items():
            agg[m]["pred"].append(aggregate_to_coarse(p, members, args.factor**2))
            agg[m]["obs"].append(c_obs[rows])
            for name, i in examples.items():
                if i in rows:
                    snapshots[m][name] = p[np.flatnonzero(rows == i)[0]]
        logger.info("Month %s done (%d/%d)", month, k + 1, len(months))

    # --- Consistency check: fine maps averaged back to 0.75° vs CAMS ------------
    check = []
    for m in METHODS:
        pred = np.concatenate(agg[m]["pred"])
        obs = np.concatenate(agg[m]["obs"])
        check.append({"method": m, **scores(obs.ravel(), pred.ravel())})
    check = pd.DataFrame(check)
    check.round(4).to_csv(args.out_dir / "consistency_with_cams.csv", index=False)
    logger.info(
        "Fine maps averaged to 0.75° vs CAMS (%d fully covered cells):\n%s",
        full_cells,
        check.round(3).to_string(index=False),
    )

    # --- Save seasonal means and examples on the fine grid ----------------------
    season_means = {
        m: np.stack([sums[m][s] / counts[s] for s in SEASON_ORDER]) for m in METHODS
    }
    mean_abs_diff = np.stack([abs_diff[s] / counts[s] for s in SEASON_ORDER])
    grid = {
        f"{m}_season_mean": (
            ("season", "lat", "lon"),
            np.stack([to_grid(v, mask) for v in season_means[m]]).astype("float32"),
        )
        for m in METHODS
    }
    grid["mean_abs_hybrid_minus_bilinear"] = (
        ("season", "lat", "lon"),
        np.stack([to_grid(v, mask) for v in mean_abs_diff]).astype("float32"),
    )
    names = [n for n in examples if all(n in snapshots[m] for m in METHODS)]
    for m in METHODS:
        grid[f"{m}_example"] = (
            ("example", "lat", "lon"),
            np.stack([to_grid(snapshots[m][n], mask) for n in names]).astype("float32"),
        )
    xr.Dataset(
        grid,
        coords={
            "season": SEASON_ORDER,
            "example": names,
            "example_time": ("example", times[[examples[n] for n in names]]),
            "lat": fine_lat,
            "lon": fine_lon,
        },
    ).to_netcdf(args.out_dir / "maps_0p25.nc")

    summary = []
    for i, s in enumerate(SEASON_ORDER):
        row = {"season": s, "time_steps": counts[s]}
        for m in METHODS:
            row[f"{m}_mean"] = float(np.mean(season_means[m][i]))
        row["mean_abs_hybrid_minus_bilinear"] = float(np.mean(mean_abs_diff[i]))
        row["p95_abs_hybrid_minus_bilinear"] = float(
            np.percentile(mean_abs_diff[i], 95)
        )
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.round(3).to_csv(args.out_dir / "season_summary.csv", index=False)
    logger.info("Seasonal India means (fine points):\n%s", summary.round(1).to_string())

    # --- Figures -----------------------------------------------------------------
    import cartopy.io.shapereader as shpreader

    path = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries_ind"
    )
    frame = shapely.box(60, 0, 105, 45)
    countries = [
        r.geometry
        for r in shpreader.Reader(path).records()
        if r.attributes["ADM0_A3"] != "IND" and r.geometry.intersects(frame)
    ]
    outlines = (countries, india)
    lat_e, lon_e = edges(fine_lat), edges(fine_lon)
    cams_lat, cams_lon = ds["lat"].values, ds["lon"].values
    cams_mask = points_inside(cams_lat, cams_lon, india)
    clat_e, clon_e = edges(cams_lat), edges(cams_lon)
    cams_season = ds["pm2p5"].groupby("season").mean("time")

    def cams(values: np.ndarray) -> np.ndarray:
        return np.where(cams_mask, values, np.nan)

    seasonal = []
    for i, s in enumerate(SEASON_ORDER):
        seasonal.append(
            [
                (
                    LABELS["cams"],
                    clat_e,
                    clon_e,
                    cams(cams_season.sel(season=s).values),
                ),
                (
                    LABELS["bilinear"],
                    lat_e,
                    lon_e,
                    to_grid(season_means["bilinear"][i], mask),
                ),
                (
                    LABELS["hybrid"],
                    lat_e,
                    lon_e,
                    to_grid(season_means["hybrid"][i], mask),
                ),
                (
                    "Hybrid − bilinear",
                    lat_e,
                    lon_e,
                    to_grid(
                        season_means["hybrid"][i] - season_means["bilinear"][i], mask
                    ),
                ),
            ]
        )
    plot_panels(
        seasonal,
        [s.replace("_", "-") for s in SEASON_ORDER],
        outlines,
        args.out_dir / "season_means.png",
        "Mean PM2.5 by season, 2020–2024",
        diff_last=True,
    )

    example_rows, example_titles = [], []
    for j, n in enumerate(names):
        i = examples[n]
        ist = times[i] + pd.Timedelta(minutes=330)
        example_titles.append(f"{n.replace('_', ' ')}\n{ist:%d %b %Y %H:%M} IST")
        example_rows.append(
            [(LABELS["cams"], clat_e, clon_e, cams(ds["pm2p5"].values[i]))]
            + [
                (LABELS[m], lat_e, lon_e, to_grid(snapshots[m][n], mask))
                for m in ("bilinear", "lgbm", "hybrid")
            ]
        )
    plot_panels(
        example_rows,
        example_titles,
        outlines,
        args.out_dir / "examples.png",
        f"Three example time steps ({args.year})",
    )

    w = SEASON_ORDER.index("winter")
    plot_panels(
        [
            [
                (
                    LABELS["cams"],
                    clat_e,
                    clon_e,
                    cams(cams_season.sel(season="winter").values),
                )
            ]
            + [
                (LABELS[m], lat_e, lon_e, to_grid(season_means[m][w], mask))
                for m in ("bilinear", "lgbm", "hybrid")
            ]
        ],
        ["winter"],
        outlines,
        args.out_dir / "zoom_north_winter.png",
        "North India, winter mean PM2.5 (2020–2024)",
        extent=ZOOM,
    )
    config = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "n_fine_points": int(n_pt),
        "n_cells": int(c_lat.size),
        "n_full_cells": int(full_cells),
        "examples": {n: str(times[i]) for n, i in examples.items()},
        "boundary": "Natural Earth 10m admin_0_countries_ind (India point of view)",
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2))
    logger.info("Done; outputs in %s", args.out_dir)


if __name__ == "__main__":
    main()
