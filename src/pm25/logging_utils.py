"""Logging setup shared by entry-point scripts.

Library modules should only call ``logging.getLogger(__name__)``; scripts call
``setup_logging()`` once at startup to decide where messages go.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

logger = logging.getLogger(__name__)


def setup_logging(
    level: int | str = logging.INFO, log_file: str | Path | None = None
) -> None:
    """Configure the root logger to write to stderr and, optionally, a file.

    Args:
        level: Minimum level to emit, e.g. ``logging.DEBUG`` or ``"WARNING"``.
        log_file: If given, also append log messages to this file. Parent
            directories are created as needed.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path))

    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=handlers, force=True)


def git_version() -> str:
    """Return the current git commit (short hash), marked ``-dirty`` if uncommitted
    changes exist, or ``"unknown"`` outside a git checkout."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


def log_run_info(args: argparse.Namespace | None = None) -> None:
    """Log a header identifying this run: command, git version and arguments.

    Call right after ``setup_logging()`` so every run in a step's log file
    records exactly which code and settings produced its outputs.
    """
    logger.info("=" * 60)
    logger.info("Run started: %s", " ".join(sys.argv))
    logger.info("Git version: %s | Python %s", git_version(), sys.version.split()[0])
    for name, value in sorted(vars(args or argparse.Namespace()).items()):
        logger.info("  %s = %s", name, value)
