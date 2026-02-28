"""Logging setup for the summarizer CLI.

Call ``setup_logging`` once from ``cli.main()`` to configure the ``"summarizer"``
package logger with timestamps and optional file output.  All other modules
obtain a child logger via ``logging.getLogger(__name__)`` and let records
propagate here.
"""

import logging
import sys
from pathlib import Path

_FMT = "%(asctime)s  %(levelname)-7s [%(threadName)s] %(message)s"
_DATE = "%H:%M:%S"


def setup_logging(verbose: bool = False, log_file: Path | None = None) -> None:
    """Configure the ``summarizer`` logger for a CLI session.

    Args:
        verbose:  If True, set level to DEBUG (shows prompt-size diagnostics
                  and other fine-grained detail).  Default level is INFO.
        log_file: If provided, attach a ``FileHandler`` that writes to this
                  path in addition to stderr.  Parent directories are created
                  automatically.

    Calling this function a second time (e.g., in tests) is safe: existing
    handlers are cleared before new ones are added.
    """
    logger = logging.getLogger("summarizer")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(_FMT, datefmt=_DATE)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
