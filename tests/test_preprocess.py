import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pm25.preprocess import preprocess, relative_humidity


def make_raw(times: list[str]) -> xr.Dataset:
    """A tiny dataset shaped like an ADS EAC4 download (lat descending)."""
    time = pd.to_datetime(times)
    lat = np.array([30.0, 29.25])
    lon = np.array([75.0, 75.75, 76.5])
    shape = (len(time), lat.size, lon.size)
    coords = {"valid_time": time, "latitude": lat, "longitude": lon}
    dims = ("valid_time", "latitude", "longitude")
    values = {
        "pm2p5": 5e-8,
        "aod550": 0.5,
        "u10": 3.0,
        "v10": 4.0,
        "t2m": 293.15,
        "d2m": 293.15,
        "sp": 100000.0,
    }
    data = {k: (dims, np.full(shape, v, dtype="float32")) for k, v in values.items()}
    return xr.Dataset(data, coords=coords)


def test_preprocess_renames_and_sorts() -> None:
    ds = preprocess(make_raw(["2020-01-01T00:00"]))
    assert ds["pm2p5"].dims == ("time", "lat", "lon")
    assert np.all(np.diff(ds["lat"].values) > 0)


def test_preprocess_converts_units() -> None:
    ds = preprocess(make_raw(["2020-01-01T00:00"]))
    assert float(ds["pm2p5"].max()) == pytest.approx(50.0, rel=1e-5)
    assert float(ds["t2m"].max()) == pytest.approx(20.0, abs=1e-3)
    assert float(ds["sp"].max()) == pytest.approx(1000.0)


def test_met_features() -> None:
    ds = preprocess(make_raw(["2020-01-01T00:00"]))
    assert float(ds["wind_speed"].max()) == pytest.approx(5.0)
    # Dewpoint equal to temperature means saturated air.
    assert float(ds["rh2m"].max()) == pytest.approx(100.0)


def test_relative_humidity_is_lower_for_drier_air() -> None:
    t = xr.DataArray([30.0])
    assert float(relative_humidity(t, xr.DataArray([10.0]))[0]) == pytest.approx(
        28.9, abs=0.5
    )


def test_solar_hour_shifts_with_longitude() -> None:
    # 07 UTC at 75 E (UTC + 5 h) is solar noon: hour_cos == -1 (midnight: +1).
    ds = preprocess(make_raw(["2020-01-01T07:00"]))
    assert float(ds["hour_cos"].sel(lon=75.0).isel(time=0)) == pytest.approx(
        -1.0, abs=1e-5
    )
    assert not np.allclose(
        ds["hour_cos"].sel(lon=75.0).values, ds["hour_cos"].sel(lon=76.5).values
    )


def test_imd_seasons() -> None:
    ds = preprocess(make_raw(["2020-02-15", "2020-04-15", "2020-07-15", "2020-11-15"]))
    assert list(ds["season"].values) == [
        "winter",
        "pre_monsoon",
        "monsoon",
        "post_monsoon",
    ]
