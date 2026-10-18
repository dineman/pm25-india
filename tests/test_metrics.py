import numpy as np
import pandas as pd
import pytest

from pm25.metrics import scores, scores_by_group


def test_scores_known_values() -> None:
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([2.0, 2.0, 4.0, 4.0])
    s = scores(obs, pred)
    assert s["rmse"] == pytest.approx(np.sqrt(0.5))
    assert s["mae"] == pytest.approx(0.5)
    assert s["bias"] == pytest.approx(0.5)
    assert s["r2"] == pytest.approx(1 - 2.0 / 5.0)
    assert s["n"] == 4


def test_scores_ignore_nan_and_perfect_prediction() -> None:
    obs = np.array([1.0, np.nan, 3.0])
    s = scores(obs, np.array([1.0, 5.0, 3.0]))
    assert s["rmse"] == 0.0 and s["r2"] == 1.0 and s["n"] == 2


def test_scores_by_group_long_format() -> None:
    table = pd.DataFrame(
        {
            "obs": [1.0, 2.0, 3.0, 4.0],
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [2.0, 3.0, 4.0, 5.0],
            "region": ["x", "x", "y", "y"],
        }
    )
    out = scores_by_group(table, ["a", "b"], by=["region"])
    assert len(out) == 4
    b_y = out[(out["region"] == "y") & (out["method"] == "b")].iloc[0]
    assert b_y["bias"] == pytest.approx(1.0)
    overall = scores_by_group(table, ["a"])
    assert overall["rmse"].iloc[0] == 0.0
