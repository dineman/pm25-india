"""Logging setup shared by entry-point scripts.

Library modules should only call ``logging.getLogger(__name__)``; scripts call
``setup_logging()`` once at startup to decide where messages go.
"""

import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


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
