"""India land mask and region labels on the EAC4 grid.

Boundaries come from Natural Earth admin-1 (states), which follows de facto
lines of control in Kashmir. A cell belongs to India if at least
``min_fraction`` of its area lies inside Indian states; its state (and so its
region) is the one covering the largest share of the cell.
"""

from collections.abc import Iterable

import numpy as np
import xarray as xr
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

GRID_RES = 0.75

NO_REGION = ""

REGION_BY_STATE = {
    # Indo-Gangetic Plain
    "Punjab": "igp",
    "Haryana": "igp",
    "Delhi": "igp",
    "Chandigarh": "igp",
    "Uttar Pradesh": "igp",
    "Bihar": "igp",
    "West Bengal": "igp",
    # Northwest
    "Rajasthan": "northwest",
    "Gujarat": "northwest",
    "Dadra and Nagar Haveli and Daman and Diu": "northwest",
    # Central
    "Madhya Pradesh": "central",
    "Chhattisgarh": "central",
    "Jharkhand": "central",
    "Odisha": "central",
    "Maharashtra": "central",
    # South (peninsula and islands)
    "Andhra Pradesh": "south",
    "Telangana": "south",
    "Karnataka": "south",
    "Goa": "south",
    "Kerala": "south",
    "Tamil Nadu": "south",
    "Puducherry": "south",
    "Lakshadweep": "south",
    "Andaman and Nicobar": "south",
    # Northeast
    "Assam": "northeast",
    "Arunachal Pradesh": "northeast",
    "Meghalaya": "northeast",
    "Manipur": "northeast",
    "Mizoram": "northeast",
    "Nagaland": "northeast",
    "Tripura": "northeast",
    "Sikkim": "northeast",
    # Himalaya
    "Jammu and Kashmir": "himalaya",
    "Ladakh": "himalaya",
    "Himachal Pradesh": "himalaya",
    "Uttarakhand": "himalaya",
}


def load_india_states() -> dict[str, BaseGeometry]:
    """Return Natural Earth 10 m admin-1 geometries for India, keyed by state name.

    The shapefile is downloaded and cached by cartopy on first use.
    """
    import cartopy.io.shapereader as shpreader

    path = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_1_states_provinces"
    )
    return {
        rec.attributes["name"]: rec.geometry
        for rec in shpreader.Reader(path).records()
        if rec.attributes["admin"] == "India"
    }


def check_state_names(names: Iterable[str]) -> None:
    """Raise if any state lacks a region, so a naming mismatch can't pass silently."""
    missing = sorted(set(names) - REGION_BY_STATE.keys())
    if missing:
        raise KeyError(f"States without a region mapping: {missing}")


def cell_box(lat: float, lon: float, res: float = GRID_RES) -> BaseGeometry:
    """Square cell of side ``res`` degrees centred on (lat, lon)."""
    half = res / 2
    return box(lon - half, lat - half, lon + half, lat + half)


def build_region_mask(
    lat: np.ndarray,
    lon: np.ndarray,
    states: dict[str, BaseGeometry],
    min_fraction: float = 0.5,
) -> xr.Dataset:
    """Compute India coverage, state and region for every grid cell.

    Areas are computed in degree space, which is accurate enough for the
    small, near-square cells of a 0.75 degree grid.

    Returns:
        Dataset on (lat, lon) with ``india_fraction`` (0-1), ``in_india``
        (bool), ``state`` and ``region`` (empty string outside India).
    """
    check_state_names(states)
    names = list(states)
    tree = STRtree([states[n] for n in names])

    shape = (lat.size, lon.size)
    fraction = np.zeros(shape, dtype="float32")
    state = np.full(shape, NO_REGION, dtype=object)

    for i, la in enumerate(lat):
        for j, lo in enumerate(lon):
            cell = cell_box(float(la), float(lo))
            overlaps = {
                names[k]: cell.intersection(states[names[k]]).area / cell.area
                for k in tree.query(cell)
            }
            overlaps = {n: f for n, f in overlaps.items() if f > 0}
            if overlaps:
                fraction[i, j] = min(sum(overlaps.values()), 1.0)
                state[i, j] = max(overlaps, key=overlaps.__getitem__)

    in_india = fraction >= min_fraction
    state = np.where(in_india, state, NO_REGION)
    region = np.vectorize(lambda s: REGION_BY_STATE.get(s, NO_REGION))(state)

    dims = ("lat", "lon")
    return xr.Dataset(
        {
            "india_fraction": (dims, fraction),
            "in_india": (dims, in_india),
            "state": (dims, state.astype(str)),
            "region": (dims, region.astype(str)),
        },
        coords={"lat": lat, "lon": lon},
        attrs={"source": "Natural Earth 10m admin-1", "min_fraction": min_fraction},
    )
