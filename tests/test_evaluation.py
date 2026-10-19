import numpy as np
import pandas as pd

from pm25.evaluation import (
    aqi_level,
    block_bootstrap,
    block_sums,
    cv_interpolate,
    ist_hour,
    nested_oof_residuals,
    scores_from_sums,
)
from pm25.metrics import scores


def test_ist_hour_is_utc_plus_five_thirty() -> None:
    times = np.array(["2020-01-01T00:00", "2020-01-01T21:00"], dtype="datetime64[ns]")
    assert list(ist_hour(times)) == ["05:30", "02:30"]


def test_aqi_level_lower_edge_inclusive() -> None:
    levels = aqi_level(np.array([0.0, 29.9, 30.0, 250.0, 5000.0]))
    assert list(levels) == ["good", "good", "satisfactory", "severe", "severe"]


def _table(rng: np.random.Generator) -> pd.DataFrame:
    obs = rng.gamma(2.0, 30.0, size=400)
    return pd.DataFrame(
        {
            "block": np.repeat(np.arange(8), 50),
            "obs": obs,
            "good": obs + rng.normal(0, 5, size=400),
            "bad": obs + rng.normal(10, 30, size=400),
        }
    )


def test_block_sums_reproduce_plain_scores() -> None:
    table = _table(np.random.default_rng(0))
    sums = block_sums(table, ["good", "bad"], "block")
    for m in ["good", "bad"]:
        pooled = scores_from_sums(sums.xs(m, level="method").sum().to_numpy())
        plain = scores(table["obs"].to_numpy(), table[m].to_numpy())
        for metric in ["rmse", "mae", "r2", "bias"]:
            assert np.isclose(pooled[metric], plain[metric])


def test_block_bootstrap_interval_contains_point_and_ranks_methods() -> None:
    table = _table(np.random.default_rng(1))
    sums = block_sums(table, ["good", "bad"], "block")
    per_method, per_pair = block_bootstrap(
        sums, [("good", "bad")], n_boot=300, rng=np.random.default_rng(2)
    )
    for _, row in per_method.iterrows():
        assert row["rmse_lo"] <= row["rmse"] <= row["rmse_hi"]
    rmse = per_pair[per_pair["metric"] == "rmse"].iloc[0]
    assert rmse["diff"] < 0 and rmse["diff_hi"] < 0
    assert rmse["share_a_better"] == 1.0


def test_cv_interpolate_uses_only_other_folds() -> None:
    # Two folds of two cells; each fold's cells are predicted as the mean of the
    # other fold, so a fold's own values never leak into its prediction.
    values = np.array([[1.0, 3.0, 10.0, 30.0]])
    folds = np.array([0, 0, 1, 1])
    dist = np.ones((4, 4))

    def mean_weights(src, dist_src, dist_tgt_src):
        n_tgt, n_src = dist_tgt_src.shape
        return [(np.ones(src.shape[0], bool), np.full((n_tgt, n_src), 1.0 / n_src))]

    out = cv_interpolate(values, dist, folds, mean_weights)
    assert np.allclose(out, [[20.0, 20.0, 2.0, 2.0]])


def test_nested_residuals_never_use_the_outer_fold() -> None:
    # 3 folds, one cell each, constant target per fold. For outer fold k, the
    # residual at inner fold j comes from a model fitted only on the remaining
    # fold, so it equals y_j - y_remaining and doesn't depend on y_k.
    n_time = 5
    cell_folds = np.array([0, 1, 2])
    y = np.tile([10.0, 20.0, 60.0], n_time).astype("float32")
    x = np.zeros((y.size, 1), dtype="float32")
    resid = nested_oof_residuals(
        "ridge", {"target": "raw", "alpha": 1.0}, x, y, cell_folds, n_time, seed=0
    )
    assert resid.shape == (3, n_time, 3)
    assert np.isnan(resid[0][:, 0]).all()
    assert np.allclose(resid[0][:, 1], 20.0 - 60.0)  # outer 0, inner 1: fit on fold 2
    assert np.allclose(resid[0][:, 2], 60.0 - 20.0)  # outer 0, inner 2: fit on fold 1
    assert np.allclose(resid[2][:, 0], 10.0 - 20.0)  # outer 2, inner 0: fit on fold 1


def test_cv_interpolate_per_fold_fields() -> None:
    # With one field per predicted fold, fold 0 reads only field 0, fold 1 only field 1.
    folds = np.array([0, 0, 1, 1])
    fields = np.array([[[np.nan, np.nan, 4.0, 4.0]], [[8.0, 8.0, np.nan, np.nan]]])

    def mean_weights(src, dist_src, dist_tgt_src):
        n_tgt, n_src = dist_tgt_src.shape
        return [(np.ones(src.shape[0], bool), np.full((n_tgt, n_src), 1.0 / n_src))]

    out = cv_interpolate(fields, np.ones((4, 4)), folds, mean_weights)
    assert np.allclose(out, [[4.0, 4.0, 8.0, 8.0]])
