"""Step 07: tune, train and test ML models that predict PM2.5 from predictors.

Usage:
    conda run -n pm25 python scripts/train_models.py                 # all 4 models
    conda run -n pm25 python scripts/train_models.py --models lgbm ridge \
        --feature-set brief --params-from results/07_models/all/best_params.json

For each model: random search over its settings (same number for every model),
each scored by block-wise CV on training cells only (step 05 folds), on a
random subset of time steps. The best setting is refitted on the training
cells and scored once on the held-out test cells. Test cells are never used
for any choice. Writes CV results, best settings, test metrics (overall,
region, season, region x season, distance), test predictions, fitted models
(joblib, not in git) and a comparison with the step 06 baselines.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xarray as xr

from pm25.features import FEATURE_SETS, feature_matrix, stack_cells, target_vector
from pm25.logging_utils import log_run_info, setup_logging
from pm25.metrics import all_scores, evaluation_table
from pm25.models import (
    MODEL_NAMES,
    SEARCH_SPACES,
    best_settings,
    cross_validate,
    make_model,
    sample_params,
)

logger = logging.getLogger(__name__)

SEASON_ORDER = ["winter", "pre_monsoon", "monsoon", "post_monsoon"]
REGION_ORDER = ["igp", "northwest", "central", "south", "northeast", "himalaya"]
BASELINE_METHODS = ["nearest", "idw", "kriging", "linreg"]
# Models too slow / memory-hungry to fit on all ~5.8 M training rows.
SUBSAMPLED_MODELS = {"rf", "mlp"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data", type=Path, default=Path("data/processed/eac4_india_3h.nc")
    )
    parser.add_argument(
        "--split", type=Path, default=Path("data/processed/splits/split_block5.nc")
    )
    parser.add_argument("--feature-set", choices=list(FEATURE_SETS), default="all")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    parser.add_argument(
        "--n-iter", type=int, default=8, help="settings tried per model (same budget)"
    )
    parser.add_argument(
        "--cv-time-frac",
        type=float,
        default=0.1,
        help="fraction of time steps (random) used for CV tuning",
    )
    parser.add_argument(
        "--final-max-rows",
        type=int,
        default=2_000_000,
        help="row cap for the final fit of rf and mlp (random rows)",
    )
    parser.add_argument(
        "--params-from",
        type=Path,
        default=None,
        help="best_params.json of an earlier run: skip tuning, reuse those settings",
    )
    parser.add_argument(
        "--cv-from",
        type=Path,
        default=None,
        help="cv_results.csv of an earlier run: skip tuning, pick the best rows from it",
    )
    parser.add_argument(
        "--baselines",
        type=Path,
        default=Path("results/06_baselines/predictions_test.nc"),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def tune(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Score ``args.n_iter`` random settings of model ``name`` by block-wise CV."""
    rows = []
    for i, params in enumerate(sample_params(SEARCH_SPACES[name], args.n_iter, rng)):
        result = cross_validate(name, params, x, y, folds, args.seed)
        rows.append(
            {"model": name, "candidate": i, "params": json.dumps(params), **result}
        )
        logger.info(
            "  %s %d/%d: CV RMSE %.2f (± %.2f), R² %.3f, %.0f s  %s",
            name,
            i + 1,
            args.n_iter,
            result["cv_rmse"],
            result["cv_rmse_std"],
            result["cv_r2"],
            result["seconds"],
            params,
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Tune (or load settings), fit, test and save every requested model."""
    args = parse_args()
    out_dir = args.out_dir or Path("results/07_models") / args.feature_set
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=Path("results/logs/train_models.log"))
    log_run_info(args)
    rng = np.random.default_rng(args.seed)
    features = FEATURE_SETS[args.feature_set]

    # --- Data ----------------------------------------------------------------
    ds = xr.open_dataset(args.data)
    split = xr.open_dataset(args.split)
    xr.testing.assert_allclose(split["lat"], ds["lat"])
    xr.testing.assert_allclose(split["lon"], ds["lon"])
    is_train = (split["set"] == "train").values
    is_test = (split["set"] == "test").values
    variables = ["pm2p5"] + [f for f in features if f not in ("lat", "lon")]
    train = stack_cells(ds[variables], is_train).load()
    test = stack_cells(ds[variables], is_test).load()
    n_time = ds.sizes["time"]

    x_train, y_train = feature_matrix(train, features), target_vector(train)
    x_test, y_test = feature_matrix(test, features), target_vector(test)
    n_train_cells, n_test_cells = train.sizes["cell"], test.sizes["cell"]
    test_lat, test_lon = test["lat"].values, test["lon"].values
    del train, test  # keep only the numpy tables: memory is tight on this machine
    fold_rows = np.tile(split["fold"].values[is_train], n_time)
    time_rows = np.repeat(np.arange(n_time), n_train_cells)
    logger.info("Features (%s): %s", args.feature_set, ", ".join(features))
    logger.info(
        "Train %d rows (%d cells x %d times), test %d rows (%d cells); %.0f MB",
        len(y_train),
        n_train_cells,
        n_time,
        len(y_test),
        n_test_cells,
        (x_train.nbytes + x_test.nbytes) / 1e6,
    )

    # CV subset: random time steps, all training cells at those times.
    cv_times = rng.choice(n_time, size=int(args.cv_time_frac * n_time), replace=False)
    cv_rows = np.flatnonzero(np.isin(time_rows, cv_times))
    logger.info(
        "CV subset: %d time steps -> %d rows; folds %s",
        cv_times.size,
        cv_rows.size,
        np.bincount(fold_rows[cv_rows]).tolist(),
    )

    # --- Tune (or reuse settings) ---------------------------------------------
    if args.params_from:
        best_params = json.loads(args.params_from.read_text())
        best_params = {m: best_params[m] for m in args.models}
        logger.info("Reusing settings from %s: %s", args.params_from, best_params)
    elif args.cv_from:
        cv_all = pd.read_csv(args.cv_from)
        cv_all = cv_all[cv_all["model"].isin(args.models)]
        best_params = best_settings(cv_all)
        if args.cv_from.resolve() != (out_dir / "cv_results.csv").resolve():
            cv_all.to_csv(out_dir / "cv_results.csv", index=False)
        logger.info("Reusing tuning results from %s", args.cv_from)
    else:
        cv_tables = []
        for name in args.models:
            logger.info("Tuning %s (%d settings, %d folds)", name, args.n_iter, 5)
            table = tune(
                name, x_train[cv_rows], y_train[cv_rows], fold_rows[cv_rows], args, rng
            )
            cv_tables.append(table)
            pd.concat(cv_tables, ignore_index=True).round(4).to_csv(
                out_dir / "cv_results.csv", index=False
            )
        cv_all = pd.concat(cv_tables, ignore_index=True)
        best_params = best_settings(cv_all)
    if not args.params_from:
        best_rows = cv_all.loc[cv_all.groupby("model")["cv_rmse"].idxmin()]
        logger.info(
            "Best CV setting per model:\n%s",
            best_rows[["model", "cv_rmse", "cv_rmse_std", "cv_mae", "cv_r2", "params"]]
            .round(3)
            .to_string(index=False),
        )
    (out_dir / "best_params.json").write_text(json.dumps(best_params, indent=2))

    # --- Final fit on training cells, one score on test cells ------------------
    preds, fit_info = {}, {}
    final_rng = np.random.default_rng(args.seed + 1)
    for name in args.models:
        rows = np.arange(len(y_train))
        if name in SUBSAMPLED_MODELS and len(rows) > args.final_max_rows:
            rows = np.sort(
                final_rng.choice(rows, size=args.final_max_rows, replace=False)
            )
        start = time.perf_counter()
        model = make_model(name, best_params[name], args.seed).fit(
            x_train[rows], y_train[rows]
        )
        fit_s = time.perf_counter() - start
        start = time.perf_counter()
        preds[name] = model.predict(x_test).reshape(n_time, -1)
        pred_s = time.perf_counter() - start
        fit_info[name] = {
            "params": best_params[name],
            "train_rows": int(rows.size),
            "fit_seconds": round(fit_s, 1),
            "predict_seconds": round(pred_s, 1),
        }
        joblib.dump(model, out_dir / f"model_{name}.joblib")
        logger.info(
            "Final %s: fitted on %d rows in %.0f s, predicted test in %.1f s",
            name,
            rows.size,
            fit_s,
            pred_s,
        )

    # --- Test metrics, with step 06 baselines alongside ------------------------
    methods = list(args.models)
    obs = y_test.reshape(n_time, -1)
    if args.baselines.exists():
        base = xr.open_dataset(args.baselines)
        assert np.allclose(base["obs"].values, obs), "baseline test set differs"
        for m in BASELINE_METHODS:
            preds[m] = base[m].values
        methods += BASELINE_METHODS
    table = evaluation_table(
        ds["time"].values,
        ds["season"].values,
        split["region"].values[is_test],
        split["dist_to_train_km"].values[is_test],
        obs,
        {m: preds[m] for m in methods},
    )
    results = all_scores(table, methods)
    for name, result in results.items():
        result.round(4).to_csv(out_dir / f"metrics_{name}.csv", index=False)
    logger.info(
        "Test metrics (overall):\n%s",
        results["overall"].round(3).to_string(index=False),
    )
    for grouping, col, order in [
        ("by_region", "region", REGION_ORDER),
        ("by_season", "season", SEASON_ORDER),
    ]:
        pivot = results[grouping].pivot(index="method", columns=col, values="rmse")
        logger.info(
            "Test RMSE %s:\n%s",
            grouping,
            pivot.loc[methods, order].round(1).to_string(),
        )
    dist = results["by_distance"].pivot(
        index="method", columns="dist_bin", values="rmse"
    )
    logger.info(
        "Test RMSE by distance (km):\n%s", dist.loc[methods].round(1).to_string()
    )

    out = xr.Dataset(
        {m: (("time", "cell"), preds[m].astype("float32")) for m in args.models}
        | {"obs": (("time", "cell"), obs)},
        coords={
            "time": ds["time"].values,
            "lat": ("cell", test_lat),
            "lon": ("cell", test_lon),
            "region": ("cell", split["region"].values[is_test]),
            "dist_to_train_km": ("cell", split["dist_to_train_km"].values[is_test]),
        },
    )
    out.to_netcdf(out_dir / "predictions_test.nc")
    config = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "features": features,
        "n_train_rows": len(y_train),
        "n_test_rows": len(y_test),
        "cv_time_steps": int(cv_times.size),
        "cv_rows": int(cv_rows.size),
        "models": fit_info,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    logger.info("Done; outputs in %s", out_dir)


if __name__ == "__main__":
    main()
