"""Step 05: build spatial block train/test splits and CV folds for India cells.

Usage:
    conda run -n pm25 python scripts/make_splits.py
    conda run -n pm25 python scripts/make_splits.py --block-sizes 5 8 \
        --data data/processed/eac4_india_3h_2020_2021.nc

For each block size writes data/processed/splits/split_block<N>.nc (grid of
block id, set, fold and distance to the nearest training cell) and, in
results/05_splits/, the per-cell table, per-region summary, a PM2.5 balance
check, a map and summary.json. Re-run if the India mask (step 02) changes.
"""

import argparse
import json
import logging
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import ListedColormap

from pm25.logging_utils import log_run_info, setup_logging
from pm25.splits import TEST, TRAIN, make_split, split_summary

logger = logging.getLogger(__name__)

INK, MUTED = "#2b2b29", "#6b6a63"
TRAIN_COLOR, TEST_COLOR = "#9ec5f4", "#eb6834"
FOLD_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mask", type=Path, default=Path("data/processed/india_regions.nc")
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/eac4_india_3h.nc"),
        help="preprocessed data, used only to check PM2.5 balance (skipped if missing)",
    )
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[5, 8])
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-dir", type=Path, default=Path("data/processed/splits"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/05_splits"))
    return parser.parse_args()


def india_cell_table(mask: xr.Dataset) -> pd.DataFrame:
    """One row per India cell: grid indices, coordinates and region."""
    lat_idx, lon_idx = np.nonzero(mask["in_india"].values)
    return pd.DataFrame(
        {
            "lat_idx": lat_idx,
            "lon_idx": lon_idx,
            "lat": mask["lat"].values[lat_idx],
            "lon": mask["lon"].values[lon_idx],
            "region": mask["region"].values[lat_idx, lon_idx],
        }
    )


def to_grid(split: pd.DataFrame, mask: xr.Dataset, block_size: int) -> xr.Dataset:
    """Put the per-cell split back on the (lat, lon) grid; -1 / "" outside India."""
    shape = mask["in_india"].shape
    idx = (split["lat_idx"].to_numpy(), split["lon_idx"].to_numpy())

    def grid(column: str, fill, dtype) -> np.ndarray:
        arr = np.full(shape, fill, dtype=dtype)
        arr[idx] = split[column].to_numpy()
        return arr

    dims = ("lat", "lon")
    ds = xr.Dataset(
        {
            "block": (dims, grid("block", -1, "int32")),
            "set": (dims, grid("set", "", "<U5")),
            "fold": (dims, grid("fold", -1, "int8")),
            "dist_to_train_km": (dims, grid("dist_to_train_km", np.nan, "float32")),
        },
        coords={"lat": mask["lat"], "lon": mask["lon"], "region": mask["region"]},
    )
    ds.attrs["block_size_cells"] = block_size
    ds["set"].attrs["description"] = "train / test; empty outside India"
    ds["fold"].attrs["description"] = (
        "CV fold of training cells; -1 for test and non-India"
    )
    return ds


def pm25_balance(split: pd.DataFrame, data_path: Path) -> pd.DataFrame | None:
    """Time-mean PM2.5 of train vs test cells, per region and overall."""
    if not data_path.exists():
        logger.warning("No data at %s: skipping PM2.5 balance check", data_path)
        return None
    with xr.open_dataset(data_path) as ds:
        mean_pm = ds["pm2p5"].mean("time").values
    cells = split.assign(
        pm25_mean=mean_pm[split["lat_idx"].to_numpy(), split["lon_idx"].to_numpy()]
    )
    by_region = cells.pivot_table(
        index="region", columns="set", values="pm25_mean", aggfunc="mean"
    )
    overall = cells.groupby("set")["pm25_mean"].mean().rename("all")
    out = pd.concat([by_region, overall.to_frame().T])
    out["test_minus_train"] = out[TEST] - out[TRAIN]
    return out.round(1)


def draw_block_edges(
    ax: plt.Axes, block: np.ndarray, lat: np.ndarray, lon: np.ndarray
) -> None:
    """Draw straight cell-edge segments wherever two neighbouring cells differ in
    block id (India cells only; -1 marks cells outside India)."""
    half = (lat[1] - lat[0]) / 2
    padded = np.pad(block, 1, constant_values=-1)
    segments = []
    for i in range(block.shape[0]):
        for j in range(block.shape[1]):
            b = padded[i + 1, j + 1]
            if b < 0:
                continue
            y, x = lat[i], lon[j]
            if padded[i + 1, j + 2] != b:  # east edge
                segments.append(([x + half, x + half], [y - half, y + half]))
            if padded[i + 1, j] != b:  # west edge
                segments.append(([x - half, x - half], [y - half, y + half]))
            if padded[i + 2, j + 1] != b:  # north edge
                segments.append(([x - half, x + half], [y + half, y + half]))
            if padded[i, j + 1] != b:  # south edge
                segments.append(([x - half, x + half], [y - half, y - half]))
    for xs, ys in segments:
        ax.plot(xs, ys, color=INK, lw=0.6, transform=ccrs.PlateCarree())


def plot_split(split: pd.DataFrame, grid: xr.Dataset, path: Path) -> None:
    """Two maps: train/test cells with block outlines, and CV folds of train blocks."""
    lat, lon = grid["lat"].values, grid["lon"].values
    set_code = np.where(
        grid["set"] == TRAIN, 0, np.where(grid["set"] == TEST, 1, np.nan)
    )
    fold = np.where(grid["fold"] >= 0, grid["fold"], np.nan)
    n_folds = int(split["fold"].max()) + 1

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 7), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    panels = [
        (set_code, ListedColormap([TRAIN_COLOR, TEST_COLOR]), 2, "Train / test cells"),
        (
            fold,
            ListedColormap(FOLD_COLORS[:n_folds]),
            n_folds,
            "CV folds (training cells)",
        ),
    ]
    for ax, (values, cmap, n, title) in zip(axes, panels):
        ax.pcolormesh(
            lon,
            lat,
            values,
            cmap=cmap,
            vmin=-0.5,
            vmax=n - 0.5,
            shading="nearest",
            transform=ccrs.PlateCarree(),
        )
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor=MUTED)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor=MUTED)
        ax.set_extent([67.5, 98.0, 5.5, 38.0], crs=ccrs.PlateCarree())
        ax.set_title(title, color=INK, loc="left")
        draw_block_edges(ax, grid["block"].values, lat, lon)
    # Fold number at the centre of each training block (identity is not colour-only).
    train = split[split["set"] == TRAIN]
    for _, b in train.groupby("block"):
        axes[1].text(
            b["lon"].mean(),
            b["lat"].mean(),
            str(b["fold"].iloc[0] + 1),
            ha="center",
            va="center",
            fontsize=8,
            color=INK,
            transform=ccrs.PlateCarree(),
        )
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (TRAIN_COLOR, TEST_COLOR)]
    axes[0].legend(handles, ["train", "test"], frameon=False, loc="lower left")
    fold_handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FOLD_COLORS[:n_folds]]
    axes[1].legend(
        fold_handles,
        [f"fold {k + 1}" for k in range(n_folds)],
        frameon=False,
        loc="lower left",
    )
    block_size = grid.attrs["block_size_cells"]
    fig.suptitle(
        f"Spatial split, {block_size} × {block_size}-cell blocks",
        color=INK,
        fontsize=13,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", path)


def main() -> None:
    """Build, save and check one split per block size."""
    args = parse_args()
    setup_logging(log_file=Path("results/logs/make_splits.log"))
    log_run_info(args)
    args.split_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mask = xr.open_dataset(args.mask)
    cells = india_cell_table(mask)
    logger.info("%d India cells from %s", len(cells), args.mask)

    summary = {"config": {k: str(v) for k, v in vars(args).items()}, "splits": {}}
    for block_size in args.block_sizes:
        name = f"block{block_size}"
        split = make_split(
            cells, block_size, args.test_fraction, args.n_folds, args.seed
        )
        grid = to_grid(split, mask, block_size)
        grid.to_netcdf(args.split_dir / f"split_{name}.nc")
        split.to_csv(args.out_dir / f"cells_{name}.csv", index=False)

        by_region = split_summary(split)
        by_region.round(3).to_csv(args.out_dir / f"summary_{name}.csv", index=False)
        test = split[split["set"] == TEST]
        fold_cells = split[split["set"] == TRAIN].groupby("fold").size()
        dist = test["dist_to_train_km"]
        logger.info(
            "%s: %d blocks, test %d cells (%.1f%%) in %d blocks; folds %s cells",
            name,
            split["block"].nunique(),
            len(test),
            100 * len(test) / len(split),
            test["block"].nunique(),
            fold_cells.tolist(),
        )
        logger.info(
            "%s per region:\n%s", name, by_region.round(3).to_string(index=False)
        )
        logger.info(
            "%s test-cell distance to nearest training cell: median %.0f km, "
            "max %.0f km",
            name,
            dist.median(),
            dist.max(),
        )

        balance = pm25_balance(split, args.data)
        if balance is not None:
            balance.to_csv(args.out_dir / f"pm25_balance_{name}.csv")
            logger.info("%s mean PM2.5 train vs test:\n%s", name, balance.to_string())

        plot_split(split, grid, args.out_dir / f"split_{name}.png")
        summary["splits"][name] = {
            "block_size_cells": block_size,
            "n_blocks": int(split["block"].nunique()),
            "n_test_blocks": int(test["block"].nunique()),
            "n_train_cells": int((split["set"] == TRAIN).sum()),
            "n_test_cells": len(test),
            "fold_cells": fold_cells.astype(int).tolist(),
            "test_dist_km": {
                "median": round(float(dist.median()), 1),
                "max": round(float(dist.max()), 1),
            },
            "file": str(args.split_dir / f"split_{name}.nc"),
        }

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s", args.out_dir / "summary.json")


if __name__ == "__main__":
    main()
