import numpy as np

from pm25.hybrid import add_residual_correction, out_of_fold_predictions


def test_out_of_fold_rows_are_predicted_without_their_fold() -> None:
    # Each fold has its own constant target; a ridge model fitted on the other
    # folds can't know it, so out-of-fold predictions differ from the fold's own
    # value, while an in-sample model would reproduce it.
    x = np.zeros((90, 1), dtype="float32")
    folds = np.repeat(np.arange(3), 30)
    y = np.repeat([10.0, 20.0, 60.0], 30).astype("float32")
    pred = out_of_fold_predictions(
        "ridge", {"target": "raw", "alpha": 1.0}, x, y, folds, seed=0
    )
    assert pred.shape == y.shape
    assert np.allclose(pred[folds == 0], 40.0)  # mean of folds 1 and 2
    assert np.allclose(pred[folds == 2], 15.0)  # mean of folds 0 and 1


def test_residual_correction_adds_weighted_residuals_and_clips() -> None:
    ml = np.array([[10.0, 5.0]])
    resid = np.array([[4.0, -20.0]])
    w = np.array([[0.5, 0.5], [0.0, 1.0]])
    out = add_residual_correction(ml, resid, w)
    assert np.allclose(out, [[2.0, 0.0]])  # 10 + (-8) = 2; 5 - 20 -> clipped to 0
