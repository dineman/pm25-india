"""Combine raw monthly EAC4 files into one preprocessed dataset.

Usage:
    conda run -n pm25 python scripts/preprocess_eac4.py

Applies pm25.preprocess (renaming, units, derived features), attaches the
India mask and region labels from scripts/build_region_mask.py, and writes
data/processed/eac4_india_3h.nc. The full grid is kept (cells outside India
included) so interpolation baselines and maps can use it; filter on
``in_india`` for modelling.
"""

import argparse
import logging
from pathlib import Path

import xarray as xr

from pm25.logging_utils import setup_logging
from pm25.preprocess import preprocess

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/eac4"))
    parser.add_argument(
        "--mask", type=Path, default=Path("data/processed/india_regions.nc")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/processed/eac4_india_3h.nc")
    )
    return parser.parse_args()


def main() -> None:
    """Preprocess all raw months and write a single compressed netCDF file."""
    args = parse_args()
    setup_logging(log_file=Path("results/logs/preprocess_eac4.log"))

    files = sorted(args.raw_dir.glob("eac4_india_*.nc"))
    if not files:
        raise SystemExit(f"No raw files found in {args.raw_dir}")
    logger.info(
        "Found %d raw files (%s to %s)", len(files), files[0].name, files[-1].name
    )

    raw = xr.open_mfdataset(files, combine="by_coords", chunks={"valid_time": 248})
    ds = preprocess(raw)

    mask = xr.open_dataset(args.mask)
    xr.testing.assert_allclose(mask["lat"], ds["lat"])
    xr.testing.assert_allclose(mask["lon"], ds["lon"])
    ds = ds.assign_coords(
        in_india=mask["in_india"], region=mask["region"], state=mask["state"]
    )

    encoding = {
        name: {"zlib": True, "complevel": 4}
        for name, var in ds.data_vars.items()
        if var.dtype.kind == "f"
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".part")
    ds.to_netcdf(tmp, encoding=encoding)
    tmp.rename(args.out)
    logger.info(
        "Saved %s: %d times x %d lat x %d lon, %d India cells",
        args.out,
        ds.sizes["time"],
        ds.sizes["lat"],
        ds.sizes["lon"],
        int(mask["in_india"].sum()),
    )


if __name__ == "__main__":
    main()
