import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box

from pm25.maps import (
    aggregate_to_coarse,
    coarse_membership,
    fine_axis,
    points_inside,
    predictors_at_points,
)


def test_fine_axis_contains_coarse_points() -> None:
    coarse = np.array([6.0, 6.75, 7.5])
    fine = fine_axis(coarse, 3)
    assert np.allclose(fine, [6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5])


def test_points_inside_box() -> None:
    lat, lon = np.array([0.0, 1.0, 2.0]), np.array([10.0, 11.0])
    mask = points_inside(lat, lon, box(9.5, 0.5, 11.5, 2.5))
    assert mask.tolist() == [[False, False], [True, True], [True, True]]


def test_predictors_at_points_interpolates_and_derives() -> None:
    times = pd.date_range("2020-01-01", periods=2, freq="3h")
    lat, lon = np.array([10.0, 11.0]), np.array([70.0, 71.0])
    shape = (2, 2, 2)
    ds = xr.Dataset(
        {
            "pm2p5": (
                ("time", "lat", "lon"),
                np.broadcast_to([[10.0, 30.0], [10.0, 30.0]], shape).copy(),
            ),
            "aod550": (("time", "lat", "lon"), np.zeros(shape)),
            "u10": (("time", "lat", "lon"), np.full(shape, 3.0)),
            "v10": (("time", "lat", "lon"), np.full(shape, 4.0)),
            "t2m": (("time", "lat", "lon"), np.full(shape, 20.0)),
            "d2m": (("time", "lat", "lon"), np.full(shape, 20.0)),
            "sp": (
                ("time", "lat", "lon"),
                np.broadcast_to([[1000.0, 1000.0], [900.0, 900.0]], shape).copy(),
            ),
        },
        coords={"time": times, "lat": lat, "lon": lon},
    )
    out = predictors_at_points(ds, np.array([10.5]), np.array([70.5]))
    assert out["sp"].values.ravel().tolist() == [950.0, 950.0]  # halfway in lat
    assert out["pm2p5"].values.ravel().tolist() == [20.0, 20.0]  # halfway in lon
    assert np.allclose(out["wind_speed"], 5.0)
    assert np.allclose(out["rh2m"], 100.0)
    # Local solar hour at 70.5° E and 00 UTC = 4.7 h.
    assert np.isclose(out["hour_sin"].values[0, 0], np.sin(2 * np.pi * 4.7 / 24))


def test_aggregate_to_coarse_means_and_drops_sparse_cells() -> None:
    fine_lat = np.array([0.0, 0.25, -0.25, 5.0])
    fine_lon = np.array([0.0, 0.0, 0.0, 5.0])
    members = coarse_membership(
        fine_lat, fine_lon, np.array([0.0, 5.0]), np.array([0.0, 5.0]), 0.375
    )
    assert [m.tolist() for m in members] == [[0, 1, 2], [3]]
    fine = np.array([[1.0, 2.0, 3.0, 9.0]])
    out = aggregate_to_coarse(fine, members, min_points=2)
    assert out[0, 0] == 2.0 and np.isnan(out[0, 1])
