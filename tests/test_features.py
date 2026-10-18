import numpy as np
import pandas as pd
import xarray as xr

from pm25.features import (
    FEATURE_SETS,
    feature_matrix,
    stack_cells,
    target_vector,
)


def make_ds() -> xr.Dataset:
    time = pd.date_range("2020-01-01", periods=2, freq="3h")
    lat, lon = [20.0, 20.75], [78.0, 78.75]
    pm = np.arange(8.0).reshape(2, 2, 2)
    return xr.Dataset(
        {
            "pm2p5": (("time", "lat", "lon"), pm),
            "aod550": (("time", "lat", "lon"), pm / 10),
            "hour_sin": (("time", "lon"), np.array([[0.1, 0.2], [0.3, 0.4]])),
            "doy_sin": (("time",), np.array([0.5, 0.6])),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )


def test_rows_are_time_major_and_aligned_with_target() -> None:
    mask = np.array([[True, False], [True, True]])
    cells = stack_cells(make_ds(), mask)
    x = feature_matrix(cells, ["aod550", "hour_sin", "doy_sin", "lat", "lon"])
    y = target_vector(cells)
    assert x.shape == (6, 5) and x.dtype == np.float32
    assert y.tolist() == [0.0, 2.0, 3.0, 4.0, 6.0, 7.0]
    assert np.allclose(x[:, 0], y / 10)
    assert np.allclose(x[:, 1], [0.1, 0.1, 0.2, 0.3, 0.3, 0.4])
    assert np.allclose(x[:, 2], [0.5, 0.5, 0.5, 0.6, 0.6, 0.6])
    assert np.allclose(x[:, 3], [20.0, 20.75, 20.75] * 2)
    assert np.allclose(x[:, 4], [78.0, 78.0, 78.75] * 2)


def test_feature_sets() -> None:
    assert "lat" not in FEATURE_SETS["all_no_latlon"]
    assert set(FEATURE_SETS["brief"]) < set(FEATURE_SETS["all"])
    assert len(FEATURE_SETS["all"]) == 14
