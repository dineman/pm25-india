import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from pm25.models import (
    MODEL_NAMES,
    SEARCH_SPACES,
    ClippedTargetModel,
    best_settings,
    build_estimator,
    cross_validate,
    fold_indices,
    make_model,
    sample_params,
)


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 3)).astype("float32")
    y = np.exp(1.0 + 0.5 * x[:, 0] - 0.3 * x[:, 1]).astype("float32")
    folds = np.repeat(np.arange(3), 200)
    return x, y, folds


def test_clipping_keeps_predictions_in_training_range() -> None:
    x = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 10.0, 20.0])
    model = ClippedTargetModel(LinearRegression(), "raw").fit(x, y)
    pred = model.predict(np.array([[-5.0], [100.0]]))
    assert pred.tolist() == [0.0, 20.0]


def test_log_target_back_transforms_and_clips() -> None:
    x = np.array([[0.0], [1.0], [2.0]])
    y = np.expm1(np.array([1.0, 2.0, 3.0]))
    model = ClippedTargetModel(LinearRegression(), "log").fit(x, y)
    assert model.predict(np.array([[1.0]]))[0] == pytest.approx(np.expm1(2.0))
    assert model.predict(np.array([[50.0]]))[0] == pytest.approx(y.max())


def test_sample_params_distinct_and_seeded() -> None:
    space = {"a": [1, 2], "b": ["x", "y", "z"]}
    first = sample_params(space, 4, np.random.default_rng(1))
    again = sample_params(space, 4, np.random.default_rng(1))
    assert first == again
    assert len({tuple(p.items()) for p in first}) == 4
    assert len(sample_params(space, 100, np.random.default_rng(1))) == 6


def test_fold_indices_cover_rows_once() -> None:
    folds = np.array([0, 1, 0, 2, 1])
    val = np.concatenate([va for _, _, va in fold_indices(folds)])
    assert sorted(val.tolist()) == list(range(5))
    for k, tr, va in fold_indices(folds):
        assert set(folds[va]) == {k} and k not in set(folds[tr])


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_fits_and_predicts(name: str, data) -> None:
    x, y, _ = data
    params = sample_params(SEARCH_SPACES[name], 1, np.random.default_rng(0))[0]
    if name == "rf":
        params = {**params, "n_estimators": 10}
    if name == "lgbm":
        params = {**params, "n_estimators": 20}
    model = make_model(name, params, seed=0).fit(x, y)
    pred = model.predict(x)
    assert pred.shape == y.shape
    assert (pred >= 0).all() and (pred <= y.max()).all()


def test_cross_validate_reports_fold_means(data) -> None:
    x, y, folds = data
    out = cross_validate("ridge", {"target": "log", "alpha": 1.0}, x, y, folds, seed=0)
    assert {"cv_rmse", "cv_mae", "cv_r2", "cv_bias", "cv_rmse_std"} <= out.keys()
    assert out["cv_r2"] > 0.5


def test_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_estimator("xgb", {"target": "raw"}, seed=0)


def test_best_settings_ignores_repeated_index() -> None:
    # Regression test: tables of several models concatenated without
    # ignore_index have repeated row labels 0..n-1; the first version of
    # train_models.py then picked settings from the wrong rows.
    a = pd.DataFrame(
        {"model": "a", "params": ['{"k": 1}', '{"k": 2}'], "cv_rmse": [5.0, 3.0]}
    )
    b = pd.DataFrame(
        {"model": "b", "params": ['{"k": 3}', '{"k": 4}'], "cv_rmse": [1.0, 9.0]}
    )
    best = best_settings(pd.concat([a, b]))
    assert best == {"a": {"k": 2}, "b": {"k": 3}}
