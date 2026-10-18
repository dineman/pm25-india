"""Spatial block train/test splits and block-wise cross-validation folds.

Cells are grouped into square blocks of ``block_size`` x ``block_size`` grid
cells. Whole blocks go to the test set, so a test cell's nearest training
cell is usually several cells away. Test blocks are chosen so every region
has close to the target share of its cells in test. Training blocks are then
dealt into CV folds, again by whole blocks, for hyperparameter tuning (step 07).
"""

import numpy as np
import pandas as pd

from pm25.exploration import haversine_km

TRAIN, TEST = "train", "test"


def assign_blocks(
    lat_idx: np.ndarray, lon_idx: np.ndarray, block_size: int
) -> np.ndarray:
    """Block id for each cell from its grid indices.

    Blocks are aligned to the grid origin (south-west corner of the domain), so
    the same ``block_size`` always gives the same blocks.
    """
    n_lon_blocks = int(lon_idx.max()) // block_size + 1
    return (lat_idx // block_size) * n_lon_blocks + lon_idx // block_size


def region_counts(cells: pd.DataFrame) -> pd.DataFrame:
    """Number of cells of each region in each block (blocks x regions).

    Args:
        cells: One row per India cell with columns ``block`` and ``region``.
    """
    return pd.crosstab(cells["block"], cells["region"])


def pick_test_blocks(
    counts: pd.DataFrame,
    test_fraction: float,
    rng: np.random.Generator,
    n_candidates: int = 5000,
) -> set[int]:
    """Choose whole test blocks so that every region gets ~``test_fraction``.

    Blocks often span two regions, so regions are balanced on cell counts, not
    on a block label. Each candidate shuffles the blocks and takes the first k,
    with k chosen so the total test share is closest to ``test_fraction``.
    The candidate with the lowest score is kept:
    mean over regions of |region test share - test_fraction| + |total share
    - test_fraction|, plus 1 if any region has no test cells.

    Args:
        counts: Blocks x regions cell counts from :func:`region_counts`.
        n_candidates: Random candidates tried (deterministic given ``rng``).
    """
    cells = counts.to_numpy()
    per_block = cells.sum(axis=1)
    region_total = cells.sum(axis=0)
    total = per_block.sum()

    best_score, best = np.inf, np.array([], dtype=int)
    for _ in range(n_candidates):
        order = rng.permutation(len(cells))
        cumulative = np.cumsum(per_block[order])
        k = int(np.argmin(np.abs(cumulative / total - test_fraction))) + 1
        chosen = order[:k]
        share = cells[chosen].sum(axis=0) / region_total
        score = np.abs(share - test_fraction).mean()
        score += abs(cumulative[k - 1] / total - test_fraction)
        score += float((share == 0).any())
        if score < best_score:
            best_score, best = score, chosen
    return set(counts.index[best].tolist())


def assign_folds(
    block_sizes: pd.Series, n_folds: int, rng: np.random.Generator
) -> dict[int, int]:
    """Deal training blocks into ``n_folds`` CV folds with balanced cell counts.

    Blocks are taken largest first (ties in random order) and each goes to the
    fold with the fewest cells so far.

    Args:
        block_sizes: Number of cells per training block, indexed by block id.
    """
    shuffled = block_sizes.iloc[rng.permutation(len(block_sizes))]
    ordered = shuffled.sort_values(ascending=False, kind="stable")
    fold_cells = np.zeros(n_folds)
    folds: dict[int, int] = {}
    for block, n in ordered.items():
        fold = int(np.argmin(fold_cells))
        folds[int(block)] = fold
        fold_cells[fold] += n
    return folds


def make_split(
    cells: pd.DataFrame,
    block_size: int,
    test_fraction: float = 0.2,
    n_folds: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign every India cell to a block, train/test set and CV fold.

    Args:
        cells: One row per India cell with columns ``lat_idx``, ``lon_idx``,
            ``lat``, ``lon`` and ``region``.

    Returns:
        ``cells`` with added columns ``block``, ``set`` (train/test), ``fold``
        (0..n_folds-1 for training cells, -1 for test cells) and
        ``dist_to_train_km`` (distance to the nearest training cell; 0 for
        training cells).
    """
    rng = np.random.default_rng(seed)
    out = cells.copy()
    out["block"] = assign_blocks(
        out["lat_idx"].to_numpy(), out["lon_idx"].to_numpy(), block_size
    )
    test_blocks = pick_test_blocks(region_counts(out), test_fraction, rng)
    out["set"] = np.where(out["block"].isin(test_blocks), TEST, TRAIN)

    train_sizes = out.loc[out["set"] == TRAIN].groupby("block").size()
    folds = assign_folds(train_sizes, n_folds, rng)
    out["fold"] = out["block"].map(folds).fillna(-1).astype(int)

    dist = haversine_km(out["lat"].to_numpy(), out["lon"].to_numpy())
    is_train = (out["set"] == TRAIN).to_numpy()
    out["dist_to_train_km"] = dist[:, is_train].min(axis=1)
    return out


def split_summary(split: pd.DataFrame) -> pd.DataFrame:
    """Cells and blocks per region and set, with the test share of cells."""
    table = (
        split.groupby(["region", "set"])
        .agg(cells=("block", "size"), blocks=("block", "nunique"))
        .unstack("set", fill_value=0)
    )
    table.columns = [f"{a}_{b}" for a, b in table.columns]
    table["test_share"] = table["cells_test"] / (
        table["cells_test"] + table["cells_train"]
    )
    return table.reset_index()
