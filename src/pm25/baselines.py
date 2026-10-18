"""Spatial interpolation baselines as fixed weight matrices.

Training and test cells stay in the same places at every time step, so each
interpolation method reduces to a matrix W (n_test x n_train) computed once:
prediction(t) = W @ values_train(t). This makes even ordinary kriging over
thousands of 3-hourly steps a single matrix multiplication.
"""

from collections.abc import Callable

import numpy as np
from scipy.spatial import Delaunay

from pm25.exploration import KM_PER_DEG_LAT


def project_km(lat: np.ndarray, lon: np.ndarray, ref_lat: float) -> np.ndarray:
    """Equirectangular projection to (x, y) km around ``ref_lat``.

    Accurate to a few percent over India; used only to build triangles.
    """
    x = lon * KM_PER_DEG_LAT * np.cos(np.radians(ref_lat))
    y = lat * KM_PER_DEG_LAT
    return np.column_stack([x, y])


def nearest_weights(dist: np.ndarray) -> np.ndarray:
    """Weight 1 on the closest training cell. ``dist`` is (n_test, n_train)."""
    w = np.zeros_like(dist)
    w[np.arange(dist.shape[0]), dist.argmin(axis=1)] = 1.0
    return w


def idw_weights(dist: np.ndarray, k: int = 8, power: float = 2.0) -> np.ndarray:
    """Inverse-distance weights over the ``k`` nearest training cells.

    Weight of training cell j for test cell i is d_ij^-power, normalised to
    sum to 1 over the k nearest cells; all other weights are 0.
    """
    k = min(k, dist.shape[1])
    nearest = np.argsort(dist, axis=1)[:, :k]
    rows = np.arange(dist.shape[0])[:, None]
    inv = 1.0 / np.maximum(dist[rows, nearest], 1e-6) ** power
    w = np.zeros_like(dist)
    w[rows, nearest] = inv / inv.sum(axis=1, keepdims=True)
    return w


def delaunay_weights(
    train_xy: np.ndarray, test_xy: np.ndarray, dist: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear interpolation on a Delaunay triangulation of training cells.

    This is the scattered-point equivalent of bilinear interpolation (training
    cells no longer form a full grid once test blocks are removed). Each test
    cell gets barycentric weights on the 3 corners of the triangle containing
    it. Test cells outside all triangles fall back to the nearest training cell.

    Returns:
        Weights (n_test, n_train) and a boolean array marking fallback cells.
    """
    tri = Delaunay(train_xy)
    simplex = tri.find_simplex(test_xy)
    outside = simplex < 0
    w = nearest_weights(dist)
    inside = np.flatnonzero(~outside)
    transform = tri.transform[simplex[inside]]
    bary2 = np.einsum("nij,nj->ni", transform[:, :2], test_xy[inside] - transform[:, 2])
    bary = np.column_stack([bary2, 1.0 - bary2.sum(axis=1)])
    w[inside] = 0.0
    corners = tri.simplices[simplex[inside]]
    w[inside[:, None], corners] = bary
    return w, outside


def ordinary_kriging_weights(
    dist_train: np.ndarray,
    dist_test_train: np.ndarray,
    variogram: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Ordinary kriging weights from a variogram model.

    Solves, for all test cells at once,
        [Γ  1] [w]   [γ0]
        [1ᵀ 0] [μ] = [1 ]
    where Γ_jk = γ(d_jk) between training cells (γ(0) = 0 on the diagonal),
    γ0_j = γ(d_ij) to the test cell, and μ is the Lagrange multiplier that
    makes the weights sum to 1 (unbiased for an unknown constant mean).

    Args:
        dist_train: (n_train, n_train) distances.
        dist_test_train: (n_test, n_train) distances.
        variogram: Function of distance returning semivariance.
    """
    n = dist_train.shape[0]
    a = np.ones((n + 1, n + 1))
    a[:n, :n] = variogram(dist_train)
    np.fill_diagonal(a[:n, :n], 0.0)
    a[n, n] = 0.0
    b = np.ones((n + 1, dist_test_train.shape[0]))
    b[:n] = variogram(dist_test_train).T
    solution = np.linalg.solve(a, b)
    return solution[:n].T


def apply_weights(weights: np.ndarray, values_train: np.ndarray) -> np.ndarray:
    """Predict test values: (time, n_train) values -> (time, n_test)."""
    return values_train @ weights.T
