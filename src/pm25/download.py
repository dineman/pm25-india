"""Build and run CAMS EAC4 download requests for the India domain.

One request per calendar month keeps each ADS job small and lets interrupted
runs resume: months whose file already exists are skipped.
"""

import calendar
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

DATASET = "cams-global-reanalysis-eac4"

# Target first, then predictors (ADS variable names).
VARIABLES = [
    "particulate_matter_2.5um",
    "total_aerosol_optical_depth_550nm",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
]

# Native EAC4 3-hourly analysis times (UTC).
TIMES = [f"{hour:02d}:00" for hour in range(0, 24, 3)]

# India bounding box as [north, west, south, east]; land masking happens later.
AREA = [38.0, 68.0, 6.0, 98.0]


class Client(Protocol):
    """The part of ``cdsapi.Client`` this module uses."""

    def retrieve(self, name: str, request: dict[str, Any], target: str) -> Any: ...


def iter_months(start: str, end: str) -> Iterator[tuple[int, int]]:
    """Yield (year, month) pairs from ``start`` to ``end`` inclusive.

    Args:
        start: First month as ``"YYYY-MM"``.
        end: Last month as ``"YYYY-MM"``.
    """
    year, month = (int(part) for part in start.split("-"))
    end_year, end_month = (int(part) for part in end.split("-"))
    if (year, month) > (end_year, end_month):
        raise ValueError(f"start {start} is after end {end}")
    while (year, month) <= (end_year, end_month):
        yield year, month
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


def build_request(year: int, month: int) -> dict[str, Any]:
    """Return the ADS request for one month of 3-hourly EAC4 fields."""
    last_day = calendar.monthrange(year, month)[1]
    return {
        "variable": VARIABLES,
        "date": f"{year}-{month:02d}-01/{year}-{month:02d}-{last_day:02d}",
        "time": TIMES,
        "area": AREA,
        "data_format": "netcdf",
    }


def month_path(out_dir: Path, year: int, month: int) -> Path:
    """Return the raw file path for one month, e.g. ``eac4_india_2020-01.nc``."""
    return out_dir / f"eac4_india_{year}-{month:02d}.nc"


def download_month(client: Client, out_dir: Path, year: int, month: int) -> Path:
    """Download one month unless its file already exists.

    The file is written under a ``.part`` name and renamed when complete, so a
    crash never leaves a truncated file that a later run would skip.
    """
    target = month_path(out_dir, year, month)
    if target.exists():
        logger.info("Skipping %s (already downloaded)", target.name)
        return target

    out_dir.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    logger.info("Requesting %d-%02d", year, month)
    client.retrieve(DATASET, build_request(year, month), str(partial))
    partial.rename(target)
    logger.info("Saved %s", target.name)
    return target
