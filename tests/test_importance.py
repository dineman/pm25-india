import numpy as np
import pandas as pd

from pm25.importance import (
    binned_dependence,
    group_shap,
    mean_abs_by,
    permutation_importance,
    shap_values,
)
from pm25.models import make_model

LGBM = {
    "target": "log",
    "n_estimators": 50,
    "learning_rate": 0.1,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 0.0,
}


def _data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    # PM2.5 depends strongly on column 0, weakly on column 1, not at all on column 2.
    x = rng.normal(size=(3000, 3)).astype("float32")
    y = np.exp(3.0 + 0.8 * x[:, 0] + 0.1 * x[:, 1]).astype("float32")
    return x, y


def test_shap_values_add_up_to_log_prediction() -> None:
    x, y = _data(np.random.default_rng(0))
    model = make_model("lgbm", LGBM, seed=0).fit(x, y)
    shap = shap_values(model, x[:100])
    assert shap.shape == (100, 3)
    full = model.estimator_.predict(x[:100], pred_contrib=True)
    raw = model.estimator_.predict(x[:100])
    assert np.allclose(shap.sum(axis=1) + full[:, -1], raw, atol=1e-5)
    importance = np.abs(shap).mean(axis=0)
    assert importance[0] > importance[1] > importance[2]


def test_group_shap_sums_members() -> None:
    features = ["u10", "v10", "wind_speed", "aod550"]
    shap = np.array([[1.0, 2.0, 3.0, 4.0]])
    groups = group_shap(shap, features)
    assert list(groups.columns) == ["aerosol (aod550)", "wind (u10, v10, speed)"]
    assert groups.iloc[0].tolist() == [4.0, 6.0]


def test_mean_abs_by_group() -> None:
    values = pd.DataFrame({"a": [1.0, -3.0, 2.0]})
    by = pd.Series(["x", "x", "y"])
    out = mean_abs_by(values, by)
    assert out.loc["x", "a"] == 2.0 and out.loc["y", "a"] == 2.0


def test_permutation_ranks_informative_feature_first() -> None:
    x, y = _data(np.random.default_rng(1))
    model = make_model("lgbm", LGBM, seed=0).fit(x, y)
    table = permutation_importance(
        model, x, y, {"c0": [0], "c1": [1], "c2": [2]}, 2, np.random.default_rng(2)
    )
    d = table.groupby("name")["d_rmse"].mean()
    assert d["c0"] > d["c1"] > 0
    assert abs(d["c2"]) < d["c1"]


def test_binned_dependence_is_monotone_for_linear_effect() -> None:
    value = np.linspace(0, 1, 1000)
    out = binned_dependence(value, 2 * value, n_bins=10)
    assert len(out) == 10 and out["n"].sum() == 1000
    assert np.all(np.diff(out["shap"]) > 0)
