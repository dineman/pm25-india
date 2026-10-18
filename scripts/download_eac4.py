"""Download CAMS EAC4 3-hourly fields for India, one file per month.

Usage:
    conda run -n pm25 python scripts/download_eac4.py --start 2020-01 --end 2024-12

Requires ADS credentials in ~/.cdsapirc. Re-running skips finished months.
"""

import argparse
import logging
from pathlib import Path

import cdsapi

from pm25.download import download_month, iter_months
from pm25.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="2020-01", help="first month, YYYY-MM")
    parser.add_argument("--end", default="2024-12", help="last month, YYYY-MM")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/raw/eac4"), help="output directory"
    )
    return parser.parse_args()


def main() -> None:
    """Download every month in the requested range, logging failures."""
    args = parse_args()
    setup_logging(log_file=Path("results/logs/download_eac4.log"))
    # The ADS client adds its own console handler to a logger that has none,
    # which prints every status message twice; a NullHandler prevents that.
    logging.getLogger("ecmwf.datastores.legacy_client").addHandler(
        logging.NullHandler()
    )
    client = cdsapi.Client()

    failed = []
    for year, month in iter_months(args.start, args.end):
        try:
            download_month(client, args.out_dir, year, month)
        except Exception:
            logger.exception("Download failed for %d-%02d", year, month)
            failed.append(f"{year}-{month:02d}")

    if failed:
        logger.error("Failed months (re-run to retry): %s", ", ".join(failed))
        raise SystemExit(1)
    logger.info("All months downloaded")


if __name__ == "__main__":
    main()
