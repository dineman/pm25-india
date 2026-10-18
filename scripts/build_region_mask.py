"""Build the India land mask and region labels on the EAC4 grid.

Usage:
    conda run -n pm25 python scripts/build_region_mask.py

Reads the grid from one raw EAC4 file and writes data/processed/india_regions.nc
plus a check map to results/figures/region_mask.png.
"""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pm25.logging_utils import setup_logging
from pm25.preprocess import standardize
from pm25.regions import build_region_mask, load_india_states

logger = logging.getLogger(__name__)

REGION_ORDER = ["igp", "northwest", "central", "south", "northeast", "himalaya"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--grid-file",
        type=Path,
        default=Path("data/raw/eac4/eac4_india_2020-01.nc"),
        help="raw EAC4 file whose lat/lon grid is used",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/processed/india_regions.nc")
    )
    parser.add_argument(
        "--figure", type=Path, default=Path("results/figures/region_mask.png")
    )
    parser.add_argument("--min-fraction", type=float, default=0.5)
    return parser.parse_args()


def plot_regions(mask: xr.Dataset, path: Path) -> None:
    """Save a map of region labels with coastlines and country borders."""
    codes = np.full(mask["region"].shape, np.nan)
    for k, name in enumerate(REGION_ORDER):
        codes[mask["region"].values == name] = k

    fig = plt.figure(figsize=(7, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    cmap = plt.get_cmap("tab10", len(REGION_ORDER))
    mesh = ax.pcolormesh(
        mask["lon"],
        mask["lat"],
        codes,
        cmap=cmap,
        vmin=-0.5,
        vmax=len(REGION_ORDER) - 0.5,
        transform=ccrs.PlateCarree(),
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6)
    cbar = fig.colorbar(mesh, ax=ax, ticks=range(len(REGION_ORDER)), shrink=0.7)
    cbar.ax.set_yticklabels(REGION_ORDER)
    ax.set_title("EAC4 0.75° cells assigned to India regions")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Build, save and plot the region mask."""
    args = parse_args()
    setup_logging()

    grid = standardize(xr.open_dataset(args.grid_file))
    mask = build_region_mask(
        grid["lat"].values, grid["lon"].values, load_india_states(), args.min_fraction
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mask.to_netcdf(args.out)
    logger.info("Saved %s", args.out)

    counts = mask["region"].to_series().value_counts()
    logger.info("India cells: %d", int(mask["in_india"].sum()))
    for name in REGION_ORDER:
        logger.info("  %-10s %d cells", name, counts.get(name, 0))

    plot_regions(mask, args.figure)
    logger.info("Saved %s", args.figure)


if __name__ == "__main__":
    main()
