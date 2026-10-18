"""Turn raw EAC4 downloads into a tidy dataset with derived features.

Steps: rename to the project's (time, lat, lon) convention, convert units,
and add derived meteorological and calendar features. Land masking and
region labels are handled separately.
"""

import numpy as np
import xarray as xr

RENAME = {"valid_time": "time", "latitude": "lat", "longitude": "lon"}

KG_M3_TO_UG_M3 = 1e9
KELVIN_OFFSET = 273.15

# IMD seasons by calendar month.
SEASON_BY_MONTH = {
    1: "winter",
    2: "winter",
    3: "pre_monsoon",
    4: "pre_monsoon",
    5: "pre_monsoon",
    6: "monsoon",
    7: "monsoon",
    8: "monsoon",
    9: "monsoon",
    10: "post_monsoon",
    11: "post_monsoon",
    12: "post_monsoon",
}


def standardize(ds: xr.Dataset) -> xr.Dataset:
    """Rename dims to (time, lat, lon), sort lat ascending, drop extra coords."""
    ds = ds.rename({k: v for k, v in RENAME.items() if k in ds.dims})
    ds = ds.drop_vars([c for c in ds.coords if c not in ("time", "lat", "lon")])
    return ds.sortby("lat")


def convert_units(ds: xr.Dataset) -> xr.Dataset:
    """Convert PM2.5 to ug m-3, temperatures to degC and pressure to hPa."""
    out = ds.copy()
    out["pm2p5"] = ds["pm2p5"] * KG_M3_TO_UG_M3
    out["pm2p5"].attrs = {"units": "ug m-3", "long_name": "PM2.5 mass concentration"}
    for name in ("t2m", "d2m"):
        out[name] = ds[name] - KELVIN_OFFSET
        out[name].attrs = {**ds[name].attrs, "units": "degC"}
    out["sp"] = ds["sp"] / 100.0
    out["sp"].attrs = {**ds["sp"].attrs, "units": "hPa"}
    return out


def relative_humidity(t2m_c: xr.DataArray, d2m_c: xr.DataArray) -> xr.DataArray:
    """Relative humidity (%) from temperature and dewpoint in degC (Magnus formula)."""

    def vapour_pressure(temp: xr.DataArray) -> xr.DataArray:
        return np.exp(17.625 * temp / (243.04 + temp))

    rh = 100.0 * vapour_pressure(d2m_c) / vapour_pressure(t2m_c)
    return rh.clip(max=100.0)


def add_met_features(ds: xr.Dataset) -> xr.Dataset:
    """Add wind speed and relative humidity (expects converted units)."""
    out = ds.copy()
    out["wind_speed"] = np.hypot(ds["u10"], ds["v10"])
    out["wind_speed"].attrs = {"units": "m s-1", "long_name": "10 m wind speed"}
    out["rh2m"] = relative_humidity(ds["t2m"], ds["d2m"])
    out["rh2m"].attrs = {"units": "%", "long_name": "2 m relative humidity"}
    return out


def add_time_features(ds: xr.Dataset) -> xr.Dataset:
    """Add cyclic local-solar-hour and day-of-year features plus an IMD season label.

    Local solar hour is UTC hour + lon / 15, so it varies across the domain
    (about 2 h from west to east) and tracks the boundary-layer cycle better
    than a single time zone.
    """
    out = ds.copy()
    utc_hour = ds["time"].dt.hour + ds["time"].dt.minute / 60.0
    solar_hour = (utc_hour + ds["lon"] / 15.0) % 24.0
    out["hour_sin"] = np.sin(2 * np.pi * solar_hour / 24.0).astype("float32")
    out["hour_cos"] = np.cos(2 * np.pi * solar_hour / 24.0).astype("float32")

    doy = ds["time"].dt.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25).astype("float32")
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25).astype("float32")

    months = ds["time"].dt.month.values
    out = out.assign_coords(
        season=("time", np.array([SEASON_BY_MONTH[m] for m in months]))
    )
    return out


def preprocess(ds: xr.Dataset) -> xr.Dataset:
    """Run all preprocessing steps on a raw EAC4 dataset."""
    ds = standardize(ds)
    ds = convert_units(ds)
    ds = add_met_features(ds)
    ds = add_time_features(ds)
    return ds
