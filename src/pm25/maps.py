"""Spatially enhanced PM2.5 maps on a finer grid (step 11).

CAMS gives PM2.5 and the predictors on a 0.75° grid. To map PM2.5 at 0.25°:

1. interpolate the *predictors* bilinearly to the fine points and derive wind
   speed, humidity and time features exactly as in step 03;
2. apply the step 07 ML model at every fine point;
3. add the ML residuals of the CAMS cells, kriged to the fine points (the
   step 08 hybrid).

Below 0.75° there is no reference data, so the maps can't be validated at
their own resolution; the only possible check is that averaging the fine
values back to 0.75° reproduces CAMS.
"""

import numpy as np
import shapely
import xarray as xr
from shapely.geometry.base import BaseGeometry

from pm25.preprocess import add_met_features, add_time_features

BASE_PREDICTORS = ["aod550", "u10", "v10", "t2m", "d2m", "sp"]


def load_india_official() -> BaseGeometry:
    """India's outline as officially claimed (Natural Earth 10 m, India point of view).

    Includes all of Jammu & Kashmir and Ladakh (PoK, Gilgit-Baltistan, Aksai
    Chin), unlike the de facto boundaries used for the region mask (step 02).
    """
    import cartopy.io.shapereader as shpreader

    path = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries_ind"
    )
    geoms = [
        rec.geometry
        for rec in shpreader.Reader(path).records()
        if rec.attributes["ADM0_A3"] == "IND"
    ]
    return shapely.union_all(geoms)


def fine_axis(coarse: np.ndarray, factor: int) -> np.ndarray:
    """Fine coordinate axis spanning the coarse axis, ``factor`` points per coarse step."""
    step = (coarse[1] - coarse[0]) / factor
    n = (len(coarse) - 1) * factor + 1
    return coarse[0] + step * np.arange(n)


def points_inside(
    lat: np.ndarray, lon: np.ndarray, geometry: BaseGeometry
) -> np.ndarray:
    """Boolean (lat, lon) mask of grid points inside ``geometry``."""
    lon2, lat2 = np.meshgrid(lon, lat)
    return shapely.contains_xy(geometry, lon2, lat2)


def predictors_at_points(
    ds: xr.Dataset, lat: np.ndarray, lon: np.ndarray
) -> xr.Dataset:
    """Bilinearly interpolated PM2.5 and base predictors at points, plus derived features.

    The interpolated ``pm2p5`` is the bilinear-regridding baseline; it is also
    the shape reference :func:`pm25.features.feature_matrix` needs.

    Args:
        ds: Gridded dataset (time, lat, lon) with ``pm2p5`` and
            :data:`BASE_PREDICTORS` in step 03 units.
        lat, lon: Point coordinates, 1-D, same length.

    Returns:
        Dataset (time, cell) with PM2.5, base and derived features and lat/lon
        coordinates along ``cell``, ready for :func:`pm25.features.feature_matrix`.
    """
    points = {
        "lat": xr.DataArray(lat, dims="cell"),
        "lon": xr.DataArray(lon, dims="cell"),
    }
    out = ds[["pm2p5", *BASE_PREDICTORS]].interp(points, method="linear")
    out = add_met_features(out)
    return add_time_features(out)


def coarse_membership(
    fine_lat: np.ndarray,
    fine_lon: np.ndarray,
    coarse_lat: np.ndarray,
    coarse_lon: np.ndarray,
    half_width: float,
) -> list[np.ndarray]:
    """For each coarse cell, the indices of fine points inside its box.

    A fine point exactly on a box edge belongs to both neighbouring cells; with
    0.25° points and 0.75° cells this never happens (edges fall at ±0.375°).
    """
    return [
        np.flatnonzero(
            (np.abs(fine_lat - la) < half_width) & (np.abs(fine_lon - lo) < half_width)
        )
        for la, lo in zip(coarse_lat, coarse_lon)
    ]


def aggregate_to_coarse(
    fine: np.ndarray, members: list[np.ndarray], min_points: int
) -> np.ndarray:
    """Mean of fine values (time, fine point) per coarse cell, (time, coarse cell).

    Cells with fewer than ``min_points`` fine points (coast, border) are NaN.
    """
    out = np.full((fine.shape[0], len(members)), np.nan)
    for j, idx in enumerate(members):
        if idx.size >= min_points:
            out[:, j] = fine[:, idx].mean(axis=1)
    return out
