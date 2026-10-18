from pathlib import Path
from typing import Any

import pytest

from pm25.download import DATASET, build_request, download_month, iter_months


class FakeClient:
    """Records requests and writes a placeholder file instead of calling ADS."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], str]] = []

    def retrieve(self, name: str, request: dict[str, Any], target: str) -> None:
        self.calls.append((name, request, target))
        Path(target).write_bytes(b"fake")


def test_iter_months_crosses_year_boundary() -> None:
    assert list(iter_months("2020-11", "2021-02")) == [
        (2020, 11),
        (2020, 12),
        (2021, 1),
        (2021, 2),
    ]


def test_iter_months_rejects_reversed_range() -> None:
    with pytest.raises(ValueError):
        list(iter_months("2021-01", "2020-12"))


def test_build_request_covers_whole_month_including_leap_day() -> None:
    request = build_request(2020, 2)
    assert request["date"] == "2020-02-01/2020-02-29"
    assert len(request["time"]) == 8
    assert request["variable"][0] == "particulate_matter_2.5um"


def test_download_month_writes_final_file(tmp_path: Path) -> None:
    client = FakeClient()
    path = download_month(client, tmp_path, 2020, 1)

    assert path == tmp_path / "eac4_india_2020-01.nc"
    assert path.read_bytes() == b"fake"
    assert not list(tmp_path.glob("*.part"))
    assert client.calls[0][0] == DATASET


def test_download_month_skips_existing_file(tmp_path: Path) -> None:
    existing = tmp_path / "eac4_india_2020-01.nc"
    existing.write_bytes(b"done")
    client = FakeClient()

    download_month(client, tmp_path, 2020, 1)

    assert client.calls == []
    assert existing.read_bytes() == b"done"
