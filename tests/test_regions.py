import numpy as np
import pytest
from shapely.geometry import box

from pm25.regions import build_region_mask, check_state_names


def test_check_state_names_rejects_unknown_state() -> None:
    with pytest.raises(KeyError, match="Atlantis"):
        check_state_names(["Bihar", "Atlantis"])


def test_build_region_mask_uses_coverage_threshold_and_largest_state() -> None:
    # Bihar covers lon 75-76.125; Rajasthan covers lon 76.125-77.
    states = {
        "Bihar": box(75.0, 29.0, 76.125, 31.0),
        "Rajasthan": box(76.125, 29.0, 77.0, 31.0),
    }
    lat = np.array([30.0])
    lon = np.array([74.625, 75.375, 76.125, 77.25])
    mask = build_region_mask(lat, lon, states)

    # Cell 74.625 spans 74.25-75.0: no overlap.
    # Cell 75.375 spans 75.0-75.75: fully Bihar.
    # Cell 76.125 spans 75.75-76.5: half Bihar, half Rajasthan -> in India.
    # Cell 77.25 spans 76.875-77.625: one sixth Rajasthan -> excluded.
    np.testing.assert_allclose(
        mask["india_fraction"].values[0], [0.0, 1.0, 1.0, 1 / 6], atol=1e-6
    )
    assert list(mask["in_india"].values[0]) == [False, True, True, False]
    assert mask["region"].values[0, 1] == "igp"
    assert mask["state"].values[0, 3] == ""
