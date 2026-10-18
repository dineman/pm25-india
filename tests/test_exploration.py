import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pm25.exploration import (
    block_size_cells,
    empirical_variogram,
    exponential,
    fit_variogram,
    haversine_km,
    linear_residuals,
    spearman_with_target,
    standardize_rows,
    to_table,
)


def test_haversine_one_degree_latitude() -> None:
    d = haversine_km(np.array([20.0, 21.0]), np.array([78.0, 78.0]))
    assert d[0, 1] == pytest.approx(111.2, abs=0.2)
    assert d[0, 0] == 0.0


def test_variogram_of_constant_field_is_zero() -> None:
    lat, lon = np.meshgrid(np.arange(5.0), np.arange(5.0), indexing="ij")
    dist = haversine_km(lat.ravel(), lon.ravel())
    values = np.ones((3, 25))
    vgm = empirical_variogram(values, dist, np.arange(0, 800, 100))
    assert np.allclose(vgm["gamma"], 0.0)
    # 25 cells -> 300 pairs per time step, all within 800 km.
    assert vgm["n_pairs"].sum() == 3 * 300


def test_variogram_ignores_nan_pairs() -> None:
    dist = np.array([[0.0, 50.0], [50.0, 0.0]])
    values = np.array([[0.0, 2.0], [np.nan, 1.0]])
    vgm = empirical_variogram(values, dist, np.array([0.0, 100.0]))
    assert vgm["n_pairs"].tolist() == [1]
    assert vgm["gamma"].iloc[0] == pytest.approx(0.5 * 4.0)


def test_fit_variogram_recovers_known_range() -> None:
    lag = np.linspace(40, 1500, 30)
    vgm = pd.DataFrame(
        {"lag_km": lag, "gamma": exponential(lag, 0.1, 0.9, 600.0), "n_pairs": 100}
    )
    fit = fit_variogram(vgm, model="exponential")
    assert fit["range_km"] == pytest.approx(600.0, rel=0.01)
    assert fit["nugget"] == pytest.approx(0.1, abs=0.01)
    assert fit["sill"] == pytest.approx(1.0, rel=0.01)


def test_standardize_rows_gives_zero_mean_unit_std() -> None:
    out = standardize_rows(np.array([[1.0, 2.0, 3.0], [10.0, 10.0, 40.0]]))
    assert np.allclose(out.mean(axis=1), 0.0)
    assert np.allclose(out.std(axis=1), 1.0)


def test_linear_residuals_removes_linear_signal() -> None:
    x = np.arange(10.0)[:, None]
    y = 3.0 + 2.0 * x[:, 0]
    y[4] = np.nan
    resid = linear_residuals(y, x)
    assert np.isnan(resid[4])
    assert np.allclose(np.delete(resid, 4), 0.0)


def test_block_size_rounds_up() -> None:
    assert block_size_cells(1.0) == 1
    # One cell is ~75.6 km wide at 25 N, so 300 km needs 4 cells.
    assert block_size_cells(300.0) == 4


def test_spearman_detects_monotonic_relation() -> None:
    table = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1], "pm2p5": [1, 4, 9, 16]})
    corr = spearman_with_target(table, ["a", "b"])
    assert corr["a"] == pytest.approx(1.0)
    assert corr["b"] == pytest.approx(-1.0)


def test_to_table_keeps_only_india_cells_and_broadcasts() -> None:
    time = pd.date_range("2020-01-01", periods=2, freq="3h")
    lat, lon = [20.0, 20.75], [78.0, 78.75]
    ds = xr.Dataset(
        {
            "pm2p5": (("time", "lat", "lon"), np.arange(8.0).reshape(2, 2, 2)),
            "hour_sin": (("time", "lon"), np.array([[0.1, 0.2], [0.3, 0.4]])),
        },
        coords={
            "time": time,
            "lat": lat,
            "lon": lon,
            "season": ("time", ["winter", "winter"]),
            "in_india": (("lat", "lon"), np.array([[True, False], [True, True]])),
            "region": (("lat", "lon"), np.array([["south", ""], ["south", "igp"]])),
        },
    )
    table = to_table(ds, ["pm2p5", "hour_sin"])
    assert len(table) == 2 * 3
    assert table["pm2p5"].tolist() == [0.0, 2.0, 3.0, 4.0, 6.0, 7.0]
    assert table["hour_sin"].tolist() == [0.1, 0.1, 0.2, 0.3, 0.3, 0.4]
    assert set(table["region"]) == {"south", "igp"}
