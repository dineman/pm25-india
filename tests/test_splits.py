import numpy as np
import pandas as pd
import pytest

from pm25.splits import TEST, TRAIN, assign_blocks, make_split, split_summary


def make_cells(n_lat: int = 12, n_lon: int = 12) -> pd.DataFrame:
    """A full rectangular grid of 0.75 deg cells; west half 'a', east half 'b'."""
    lat_idx, lon_idx = np.meshgrid(np.arange(n_lat), np.arange(n_lon), indexing="ij")
    lat_idx, lon_idx = lat_idx.ravel(), lon_idx.ravel()
    return pd.DataFrame(
        {
            "lat_idx": lat_idx,
            "lon_idx": lon_idx,
            "lat": 10.0 + 0.75 * lat_idx,
            "lon": 70.0 + 0.75 * lon_idx,
            "region": np.where(lon_idx < n_lon // 2, "a", "b"),
        }
    )


def test_assign_blocks_groups_neighbouring_cells() -> None:
    blocks = assign_blocks(np.array([0, 0, 1, 2, 2]), np.array([0, 1, 1, 0, 3]), 2)
    # Cells (0,0), (0,1), (1,1) share a block; (2,0) and (2,3) are in other rows.
    assert blocks[0] == blocks[1] == blocks[2]
    assert len({blocks[0], blocks[3], blocks[4]}) == 3


def test_blocks_are_never_split_between_sets_or_folds() -> None:
    split = make_split(make_cells(), block_size=2, seed=0)
    assert (split.groupby("block")["set"].nunique() == 1).all()
    assert (split.groupby("block")["fold"].nunique() == 1).all()


def test_every_region_has_test_cells_near_target_share() -> None:
    split = make_split(make_cells(), block_size=2, test_fraction=0.2, seed=1)
    summary = split_summary(split).set_index("region")
    assert (summary["cells_test"] > 0).all()
    assert summary["test_share"].between(0.1, 0.3).all()


def test_folds_only_for_training_cells_and_balanced() -> None:
    split = make_split(make_cells(), block_size=2, n_folds=4, seed=2)
    assert (split.loc[split["set"] == TEST, "fold"] == -1).all()
    train_folds = split.loc[split["set"] == TRAIN, "fold"]
    assert set(train_folds) == {0, 1, 2, 3}
    sizes = train_folds.value_counts()
    assert sizes.max() - sizes.min() <= 4  # at most one 2x2 block apart


def test_distance_to_train_zero_for_train_positive_for_test() -> None:
    split = make_split(make_cells(), block_size=3, seed=3)
    assert (split.loc[split["set"] == TRAIN, "dist_to_train_km"] == 0).all()
    test_dist = split.loc[split["set"] == TEST, "dist_to_train_km"]
    # Nearest training cell is at least one cell (~74-83 km) away.
    assert test_dist.min() == pytest.approx(
        0.75 * 111.2 * np.cos(np.radians(18.25)), rel=0.2
    )


def test_split_is_reproducible_with_seed() -> None:
    a = make_split(make_cells(), block_size=2, seed=7)
    b = make_split(make_cells(), block_size=2, seed=7)
    c = make_split(make_cells(), block_size=2, seed=8)
    pd.testing.assert_frame_equal(a, b)
    assert not a["set"].equals(c["set"])
