"""Step 12: final figures for the report, rebuilt from saved results (no model runs).

Usage:
    conda run -n pm25 python scripts/report_figures.py

- SHAP heatmaps by region and season (step 10 CSVs) with text colour chosen by
  position on the colour scale (fixes low-contrast labels).
- Seasonal mean maps and the north India zoom (step 11 ``maps_0p25.nc``) with the
  area outside the model domain hatched: PoK, Aksai Chin and the coastal cells
  excluded by the 50% rule are filled by the methods but were never trained or
  evaluated on.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Scripts in this folder are importable when this file is run as a script.
from feature_importance import plot_heatmap
from make_maps import (
    LABELS,
    SEASON_ORDER,
    ZOOM,
    edges,
    neighbour_outlines,
    plot_panels,
)

from pm25.logging_utils import log_run_info, setup_logging
from pm25.maps import load_india_official, points_inside

logger = logging.getLogger(__name__)

REGION_ORDER = ["igp", "northwest", "central", "south", "northeast", "himalaya"]


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
        "--importance-dir", type=Path, default=Path("results/10_importance")
    )
    parser.add_argument("--maps-dir", type=Path, default=Path("results/11_maps"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/12_report"))
    return parser.parse_args()


def outside_domain(
    fine_lat: np.ndarray, fine_lon: np.ndarray, split: xr.Dataset
) -> np.ndarray:
    """(lat, lon) mask of fine points whose 0.75° cell is not a model-domain cell."""
    in_domain = np.isin(split["set"].values, ["train", "test"])
    c_lat, c_lon = split["lat"].values, split["lon"].values
    step = c_lat[1] - c_lat[0]
    i = np.clip(np.rint((fine_lat - c_lat[0]) / step).astype(int), 0, c_lat.size - 1)
    j = np.clip(np.rint((fine_lon - c_lon[0]) / step).astype(int), 0, c_lon.size - 1)
    return ~in_domain[np.ix_(i, j)]


def main() -> None:
    """Rebuild the report figures."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=Path("results/logs/report_figures.log"))
    log_run_info(args)

    # --- SHAP heatmaps -----------------------------------------------------------
    overall = pd.read_csv(args.importance_dir / "shap_groups_overall.csv", index_col=0)
    order = overall.iloc[:, 0].sort_values(ascending=False).index
    for by, cols in [("region", REGION_ORDER), ("season", SEASON_ORDER)]:
        table = pd.read_csv(
            args.importance_dir / f"shap_groups_by_{by}.csv", index_col=0
        )
        plot_heatmap(
            table.loc[order, cols],
            args.out_dir / f"shap_groups_by_{by}.png",
            f"Mean |SHAP| by {by}",
        )

    # --- Maps with the out-of-domain area hatched -------------------------------
    maps = xr.open_dataset(args.maps_dir / "maps_0p25.nc")
    ds = xr.open_dataset(args.data)
    split = xr.open_dataset(args.split)
    india = load_india_official()
    outlines = (neighbour_outlines(), india)
    fine_lat, fine_lon = maps["lat"].values, maps["lon"].values
    fine_in_india = points_inside(fine_lat, fine_lon, india)
    hatch_mask = outside_domain(fine_lat, fine_lon, split) & fine_in_india
    logger.info(
        "Fine points in India (official) outside the model domain: %d of %d",
        int(hatch_mask.sum()),
        int(fine_in_india.sum()),
    )
    hatch = (fine_lat, fine_lon, hatch_mask)
    lat_e, lon_e = edges(fine_lat), edges(fine_lon)
    c_lat, c_lon = ds["lat"].values, ds["lon"].values
    clat_e, clon_e = edges(c_lat), edges(c_lon)
    cams_mask = points_inside(c_lat, c_lon, india)
    cams_season = ds["pm2p5"].groupby("season").mean("time")

    def cams(season: str) -> np.ndarray:
        return np.where(cams_mask, cams_season.sel(season=season).values, np.nan)

    def fine(method: str, season: str) -> np.ndarray:
        return maps[f"{method}_season_mean"].sel(season=season).values

    rows = [
        [
            (LABELS["cams"], clat_e, clon_e, cams(s)),
            (LABELS["bilinear"], lat_e, lon_e, fine("bilinear", s)),
            (LABELS["hybrid"], lat_e, lon_e, fine("hybrid", s)),
            (
                "Hybrid − bilinear",
                lat_e,
                lon_e,
                fine("hybrid", s) - fine("bilinear", s),
            ),
        ]
        for s in SEASON_ORDER
    ]
    plot_panels(
        rows,
        [s.replace("_", "-") for s in SEASON_ORDER],
        outlines,
        args.out_dir / "season_means.png",
        "Mean PM2.5 by season, 2020–2024 (hatched: outside the model domain)",
        diff_last=True,
        hatch=hatch,
    )
    plot_panels(
        [
            [(LABELS["cams"], clat_e, clon_e, cams("winter"))]
            + [
                (LABELS[m], lat_e, lon_e, fine(m, "winter"))
                for m in ("bilinear", "lgbm", "hybrid")
            ]
        ],
        ["winter"],
        outlines,
        args.out_dir / "zoom_north_winter.png",
        "North India, winter mean PM2.5 (hatched: outside the model domain)",
        extent=ZOOM,
        hatch=hatch,
    )
    logger.info("Done; outputs in %s", args.out_dir)


if __name__ == "__main__":
    main()
