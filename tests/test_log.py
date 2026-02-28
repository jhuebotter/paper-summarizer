"""Tests for summarizer/log.py — logging setup."""

import logging
import pytest

from summarizer.log import setup_logging


# ---------------------------------------------------------------------------
# setup_logging  (logger state reset handled by conftest._reset_summarizer_logger)
# ---------------------------------------------------------------------------


def test_setup_logging_adds_stderr_handler():
    """setup_logging attaches a StreamHandler (stderr) to the summarizer logger."""
    setup_logging()
    logger = logging.getLogger("summarizer")
    stream_handlers = [h for h in logger.handlers if type(h) is logging.StreamHandler]
    assert len(stream_handlers) == 1


def test_setup_logging_default_level_is_info():
    """Without verbose=True the summarizer logger is set to INFO."""
    setup_logging()
    assert logging.getLogger("summarizer").level == logging.INFO


def test_setup_logging_verbose_sets_debug_level():
    """verbose=True raises the level to DEBUG."""
    setup_logging(verbose=True)
    assert logging.getLogger("summarizer").level == logging.DEBUG


def test_setup_logging_file_handler_created(tmp_path):
    """Passing log_file attaches a FileHandler and creates the file."""
    log_file = tmp_path / "logs" / "run.log"
    setup_logging(log_file=log_file)
    logger = logging.getLogger("summarizer")
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1
    assert log_file.exists()


def test_setup_logging_file_handler_parent_dir_created(tmp_path):
    """setup_logging creates missing parent directories for the log file."""
    log_file = tmp_path / "deep" / "nested" / "run.log"
    setup_logging(log_file=log_file)
    assert log_file.parent.exists()


def test_setup_logging_no_file_handler_by_default():
    """Without log_file, only a StreamHandler is added (no FileHandler)."""
    setup_logging()
    logger = logging.getLogger("summarizer")
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 0


def test_setup_logging_is_idempotent():
    """Calling setup_logging twice leaves exactly one StreamHandler."""
    setup_logging()
    setup_logging()
    logger = logging.getLogger("summarizer")
    stream_handlers = [h for h in logger.handlers if type(h) is logging.StreamHandler]
    assert len(stream_handlers) == 1


def test_setup_logging_output_contains_timestamp(tmp_path, capsys):
    """Log output includes a HH:MM:SS timestamp prefix."""
    setup_logging()
    logging.getLogger("summarizer").info("sentinel-message")
    err = capsys.readouterr().err
    # Format: "HH:MM:SS  INFO    sentinel-message"
    import re

    assert re.search(r"\d{2}:\d{2}:\d{2}", err), f"No timestamp found in: {err!r}"
    assert "sentinel-message" in err
    assert "MainThread" in err
