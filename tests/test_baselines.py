import numpy as np
import pytest
from pykrige.ok import OrdinaryKriging
from scipy.spatial.distance import cdist

from pm25.baselines import (
    apply_weights,
    delaunay_weights,
    idw_weights,
    nearest_weights,
    ordinary_kriging_weights,
)
from pm25.exploration import exponential


@pytest.fixture
def points() -> tuple[np.ndarray, np.ndarray]:
    """A 6 x 6 grid of training points and a few test points inside it."""
    gx, gy = np.meshgrid(np.arange(6.0), np.arange(6.0))
    train = np.column_stack([gx.ravel(), gy.ravel()]) * 100.0
    test = np.array([[150.0, 220.0], [310.0, 90.0], [420.0, 430.0]])
    return train, test


def test_nearest_picks_closest(points) -> None:
    train, test = points
    w = nearest_weights(cdist(test, train))
    assert np.allclose(w.sum(axis=1), 1.0)
    assert np.allclose(w @ train, [[100.0, 200.0], [300.0, 100.0], [400.0, 400.0]])


def test_idw_uses_k_neighbours_and_sums_to_one(points) -> None:
    train, test = points
    w = idw_weights(cdist(test, train), k=4)
    assert np.allclose(w.sum(axis=1), 1.0)
    assert ((w > 0).sum(axis=1) == 4).all()


def test_delaunay_reproduces_linear_field_inside_hull(points) -> None:
    train, test = points
    field = lambda xy: 2.0 * xy[:, 0] - 0.5 * xy[:, 1] + 7.0
    w, outside = delaunay_weights(train, test, cdist(test, train))
    assert not outside.any()
    assert np.allclose(w @ field(train), field(test))


def test_delaunay_falls_back_to_nearest_outside_hull(points) -> None:
    train, _ = points
    test = np.array([[-50.0, 250.0]])
    w, outside = delaunay_weights(train, test, cdist(test, train))
    assert outside.all()
    assert np.allclose(w @ train, [[0.0, 200.0]]) or np.allclose(
        w @ train, [[0.0, 300.0]]
    )


def test_kriging_weights_sum_to_one_and_match_pykrige(points) -> None:
    train, test = points
    rng = np.random.default_rng(0)
    values = rng.normal(size=len(train))
    nugget, psill, rng_km = 0.1, 0.9, 300.0

    w = ordinary_kriging_weights(
        cdist(train, train),
        cdist(test, train),
        lambda h: exponential(h, nugget, psill, rng_km),
    )
    assert np.allclose(w.sum(axis=1), 1.0)

    ok = OrdinaryKriging(
        train[:, 0],
        train[:, 1],
        values,
        variogram_model="exponential",
        variogram_parameters={
            "sill": nugget + psill,
            "range": rng_km,
            "nugget": nugget,
        },
        exact_values=True,
    )
    expected, _ = ok.execute("points", test[:, 0], test[:, 1])
    assert np.allclose(w @ values, expected, atol=1e-8)


def test_apply_weights_shapes() -> None:
    w = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
    values = np.array([[1.0, 3.0, 5.0], [2.0, 2.0, 2.0]])
    out = apply_weights(w, values)
    assert out.shape == (2, 2)
    assert np.allclose(out, [[2.0, 5.0], [2.0, 2.0]])


def test_idw_ties_go_to_lower_index() -> None:
    # Four training cells at the same distance, k = 2: the first two are used.
    dist = np.full((1, 4), 100.0)
    w = idw_weights(dist, k=2)
    assert np.allclose(w, [[0.5, 0.5, 0.0, 0.0]])
