import argparse
import logging
from pathlib import Path

from pm25.logging_utils import log_run_info, setup_logging


def test_log_run_info_records_arguments(tmp_path: Path) -> None:
    log_file = tmp_path / "run.log"
    setup_logging(log_file=log_file)

    log_run_info(argparse.Namespace(start="2020-01", end="2024-12"))
    logging.shutdown()

    content = log_file.read_text()
    assert "Run started:" in content
    assert "Git version:" in content
    assert "start = 2020-01" in content


def test_setup_logging_writes_errors_to_file(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "run.log"
    setup_logging(log_file=log_file)

    logging.getLogger("pm25.test").error("Missing input file: %s", "jan.nc")
    logging.shutdown()

    content = log_file.read_text()
    assert "ERROR pm25.test: Missing input file: jan.nc" in content


def test_setup_logging_respects_level(tmp_path: Path) -> None:
    log_file = tmp_path / "run.log"
    setup_logging(level="WARNING", log_file=log_file)

    logging.getLogger("pm25.test").info("should be hidden")
    logging.shutdown()

    assert "should be hidden" not in log_file.read_text()
