"""Turn the gridded dataset into (rows x features) tables for modelling.

Rows are ordered time-major: all cells of time step 0, then time step 1, ...
so ``array.ravel()`` of a (time, cell) field lines up with the rows.
"""

import numpy as np
import xarray as xr

# The predictors named in the project brief, plus location.
BRIEF = ["aod550", "u10", "v10", "t2m", "d2m", "sp", "lat", "lon"]
# Derived in step 03: wind speed, relative humidity and cyclic time.
DERIVED = ["wind_speed", "rh2m", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]

FEATURE_SETS = {
    "brief": BRIEF,
    "all": BRIEF + DERIVED,
    "all_no_latlon": [f for f in BRIEF + DERIVED if f not in ("lat", "lon")],
}

TARGET = "pm2p5"


def stack_cells(ds: xr.Dataset, mask: np.ndarray) -> xr.Dataset:
    """Stack (lat, lon) to ``cell`` and keep cells where ``mask`` (lat, lon) is True."""
    stacked = ds.stack(cell=("lat", "lon"))
    return stacked.isel(cell=np.asarray(mask).ravel())


def feature_matrix(cells: xr.Dataset, features: list[str]) -> np.ndarray:
    """Features as a float32 array (time * cell, n_features), time-major.

    ``lat``/``lon`` come from the cell coordinates; variables with fewer dims
    (e.g. ``hour_sin`` on (time, lon), ``doy_sin`` on (time,)) are broadcast.
    """
    ref = cells[TARGET].transpose("time", "cell")
    cols = []
    for name in features:
        if name in ("lat", "lon"):
            cols.append(np.broadcast_to(cells[name].values, ref.shape))
        else:
            arr = cells[name].broadcast_like(ref).transpose("time", "cell").values
            cols.append(arr)
    return np.stack([c.ravel() for c in cols], axis=1).astype("float32")


def target_vector(cells: xr.Dataset) -> np.ndarray:
    """PM2.5 as float32, time-major, matching :func:`feature_matrix` rows."""
    return cells[TARGET].transpose("time", "cell").values.ravel().astype("float32")
