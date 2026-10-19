"""Step 10: feature importance of the LightGBM model (SHAP + permutation).

Usage:
    conda run -n pm25 python scripts/feature_importance.py

1. SHAP values (exact TreeSHAP from LightGBM) on a random sample of rows from
   all India cells (training and test): mean |SHAP| per feature and per feature
   group, overall, by region and by season; dependence curves by season.
2. Permutation importance on a random sample of test-cell rows: increase in
   RMSE and MAE when one feature, or a group of related features, is shuffled.
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap

from pm25.features import feature_matrix, stack_cells, target_vector
from pm25.importance import (
    FEATURE_GROUPS,
    binned_dependence,
    group_shap,
    mean_abs_by,
    permutation_importance,
    shap_values,
)
from pm25.logging_utils import log_run_info, setup_logging

logger = logging.getLogger(__name__)

SEASON_ORDER = ["winter", "pre_monsoon", "monsoon", "post_monsoon"]
REGION_ORDER = ["igp", "northwest", "central", "south", "northeast", "himalaya"]
INK, MUTED, GRID = "#2b2b29", "#6b6a63", "#e4e3de"
BAR = "#2a78d6"
# Categorical slots 1-4 (validated for adjacent lines): one per season.
SEASON_COLORS = dict(zip(SEASON_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blues", ["#eef4fc", "#2a78d6", "#0d3a73"]
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data", type=Path, default=Path("data/processed/eac4_india_3h.nc")
    )
    parser.add_argument(
        "--split", type=Path, default=Path("data/processed/splits/split_block5.nc")
    )
    parser.add_argument(
        "--models-dir", type=Path, default=Path("results/07_models/all")
    )
    parser.add_argument("--model", default="lgbm")
    parser.add_argument("--out-dir", type=Path, default=Path("results/10_importance"))
    parser.add_argument("--shap-rows", type=int, default=200_000)
    parser.add_argument("--perm-rows", type=int, default=300_000)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--n-dependence", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sample_rows(
    ds: xr.Dataset,
    mask: np.ndarray,
    features: list[str],
    regions: np.ndarray,
    n_rows: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Random (time, cell) rows of the cells in ``mask``: features, PM2.5, labels."""
    variables = ["pm2p5", "season"] + [f for f in features if f not in ("lat", "lon")]
    cells = stack_cells(ds[variables], mask).load()
    x, y = feature_matrix(cells, features), target_vector(cells)
    n_cell = cells.sizes["cell"]
    rows = np.sort(rng.choice(len(y), size=min(n_rows, len(y)), replace=False))
    labels = pd.DataFrame(
        {
            "season": ds["season"].values[rows // n_cell],
            "region": regions[rows % n_cell],
        }
    )
    return x[rows], y[rows], labels


def style(ax: plt.Axes, grid_axis: str = "x") -> None:
    """Recessive axes and grid."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis=grid_axis, color=GRID)
    ax.set_axisbelow(True)


def plot_overview(
    shap_mean: pd.Series, perm: pd.DataFrame, path: Path, title: str
) -> None:
    """Two panels: mean |SHAP| (what the model uses) and permutation ΔRMSE (accuracy)."""
    order = shap_mean.sort_values().index
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.45 * len(order) + 1.5), sharey=True)
    y = np.arange(len(order))
    axes[0].barh(y, shap_mean.loc[order], color=BAR, height=0.6)
    axes[0].set_xlabel("mean |SHAP| (log units; 0.1 ≈ ±10% PM2.5)", color=INK)
    axes[0].set_title("What the model uses", color=INK, loc="left", fontsize=10)
    stats = perm.groupby("name")["d_rmse"].agg(["mean", "min", "max"]).loc[order]
    axes[1].barh(y, stats["mean"], color=BAR, height=0.6)
    axes[1].errorbar(
        stats["mean"],
        y,
        xerr=[stats["mean"] - stats["min"], stats["max"] - stats["mean"]],
        fmt="none",
        ecolor=INK,
        elinewidth=1,
    )
    axes[1].set_xlabel("RMSE increase when shuffled (µg m⁻³)", color=INK)
    axes[1].set_title(
        "What the accuracy depends on (test cells)", color=INK, loc="left", fontsize=10
    )
    axes[0].set_yticks(y, order)
    for ax in axes:
        style(ax)
    fig.suptitle(title, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_heatmap(table: pd.DataFrame, path: Path, title: str) -> None:
    """Mean |SHAP| of feature groups (rows) by region or season (columns)."""
    fig, ax = plt.subplots(
        figsize=(1.3 * table.shape[1] + 4, 0.5 * table.shape[0] + 1.5)
    )
    im = ax.imshow(table.values, cmap=SEQUENTIAL, aspect="auto")
    # White text only on the dark part of the ramp (by position on the colour
    # scale, not by percentile, so mid-blue cells keep dark text).
    lo, hi = np.nanmin(table.values), np.nanmax(table.values)
    threshold = lo + 0.55 * (hi - lo)
    for (r, c), v in np.ndenumerate(table.values):
        ax.text(
            c,
            r,
            f"{v:.2f}",
            ha="center",
            va="center",
            fontsize=9,
            color="white" if v > threshold else INK,
        )
    ax.set_xticks(range(table.shape[1]), [c.replace("_", " ") for c in table.columns])
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.tick_params(length=0, labelcolor=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.8).set_label("mean |SHAP| (log units)", color=INK)
    ax.set_title(title, color=INK, loc="left")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_dependence(curves: pd.DataFrame, features: list[str], path: Path) -> None:
    """Small multiples: mean SHAP vs feature value, one line per season."""
    n_col = 3
    n_row = int(np.ceil(len(features) / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(13, 3.4 * n_row), squeeze=False)
    for ax, feature in zip(axes.ravel(), features):
        for season, color in SEASON_COLORS.items():
            sub = curves[(curves["feature"] == feature) & (curves["season"] == season)]
            ax.plot(
                sub["value"],
                sub["shap"],
                "-",
                color=color,
                lw=2,
                label=season.replace("_", " "),
            )
        ax.axhline(0, color=MUTED, lw=1)
        style(ax, grid_axis="both")
        ax.set_title(feature, color=INK, loc="left", fontsize=10)
        ax.set_ylabel("SHAP (log units)", color=INK, fontsize=8)
    for ax in axes.ravel()[len(features) :]:
        ax.set_visible(False)
    axes[0][-1].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Effect of each feature on predicted PM2.5, by season "
        "(above 0 = raises the prediction)",
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", path)


def main() -> None:
    """Compute SHAP and permutation importance; save tables and figures."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=Path("results/logs/feature_importance.log"))
    log_run_info(args)
    rng = np.random.default_rng(args.seed)

    features = json.loads((args.models_dir / "config.json").read_text())["features"]
    model = joblib.load(args.models_dir / f"model_{args.model}.joblib")
    logger.info(
        "Model %s (target %s), features: %s", args.model, model.target, features
    )

    ds = xr.open_dataset(args.data)
    split = xr.open_dataset(args.split)
    is_india = np.isin(split["set"].values, ["train", "test"])
    is_test = (split["set"] == "test").values

    # --- 1. SHAP on a sample of all India cells --------------------------------
    x, _, labels = sample_rows(
        ds, is_india, features, split["region"].values[is_india], args.shap_rows, rng
    )
    shap = shap_values(model, x)
    shap_df = pd.DataFrame(shap, columns=features)
    groups = group_shap(shap, features)
    logger.info("SHAP on %d rows from %d India cells", len(x), int(is_india.sum()))

    feat_overall = mean_abs_by(shap_df).T["mean_abs_shap"].sort_values(ascending=False)
    group_overall = mean_abs_by(groups).T["mean_abs_shap"].sort_values(ascending=False)
    feat_overall.round(4).to_csv(args.out_dir / "shap_features_overall.csv")
    group_overall.round(4).to_csv(args.out_dir / "shap_groups_overall.csv")
    logger.info("Mean |SHAP| per feature:\n%s", feat_overall.round(3).to_string())
    logger.info("Mean |SHAP| per group:\n%s", group_overall.round(3).to_string())
    tables = {}
    for by, order in [("region", REGION_ORDER), ("season", SEASON_ORDER)]:
        for name, values in [("features", shap_df), ("groups", groups)]:
            table = mean_abs_by(values, labels[by]).loc[order].T
            table.round(4).to_csv(args.out_dir / f"shap_{name}_by_{by}.csv")
            tables[(name, by)] = table
        logger.info(
            "Mean |SHAP| per group by %s:\n%s",
            by,
            tables[("groups", by)].round(3).to_string(),
        )

    # Dependence curves for the top features, per season.
    top = list(feat_overall.index[: args.n_dependence])
    curves = []
    for feature in top:
        j = features.index(feature)
        for season in SEASON_ORDER:
            rows = (labels["season"] == season).to_numpy()
            curve = binned_dependence(x[rows, j], shap[rows, j])
            curves.append(curve.assign(feature=feature, season=season))
    curves = pd.concat(curves, ignore_index=True)
    curves.round(4).to_csv(args.out_dir / "shap_dependence.csv", index=False)
    del x, shap

    # --- 2. Permutation importance on test cells ------------------------------
    x_te, y_te, _ = sample_rows(
        ds, is_test, features, split["region"].values[is_test], args.perm_rows, rng
    )
    single = {f: [i] for i, f in enumerate(features)}
    grouped = {
        g: [features.index(f) for f in members] for g, members in FEATURE_GROUPS.items()
    }
    logger.info(
        "Permutation importance on %d test rows, %d repeats", len(y_te), args.n_repeats
    )
    perm_feat = permutation_importance(model, x_te, y_te, single, args.n_repeats, rng)
    perm_group = permutation_importance(model, x_te, y_te, grouped, args.n_repeats, rng)
    for name, table in [("features", perm_feat), ("groups", perm_group)]:
        table.round(4).to_csv(args.out_dir / f"permutation_{name}.csv", index=False)
        summary = (
            table.groupby("name")[["d_rmse", "d_mae"]]
            .agg(["mean", "std"])
            .sort_values(("d_rmse", "mean"), ascending=False)
        )
        logger.info(
            "Permutation (%s), increase over unshuffled:\n%s", name, summary.round(2)
        )

    # --- Figures and config ----------------------------------------------------
    plot_overview(
        feat_overall,
        perm_feat,
        args.out_dir / "importance_features.png",
        "Feature importance, LightGBM (single features)",
    )
    plot_overview(
        group_overall,
        perm_group,
        args.out_dir / "importance_groups.png",
        "Feature importance, LightGBM (groups of related features)",
    )
    plot_heatmap(
        tables[("groups", "region")].loc[group_overall.index],
        args.out_dir / "shap_groups_by_region.png",
        "Mean |SHAP| by region",
    )
    plot_heatmap(
        tables[("groups", "season")].loc[group_overall.index],
        args.out_dir / "shap_groups_by_season.png",
        "Mean |SHAP| by season",
    )
    plot_dependence(curves, top, args.out_dir / "shap_dependence.png")
    config = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "features": features,
        "groups": FEATURE_GROUPS,
        "model_target": model.target,
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2))
    logger.info("Done; outputs in %s", args.out_dir)


if __name__ == "__main__":
    main()
