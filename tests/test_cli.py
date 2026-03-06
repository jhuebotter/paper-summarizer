"""Tests for summarizer/cli.py — argument parsing and high-level CLI behaviour."""

import sys
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from summarizer.cli import _build_parser, _check_lm_studio, main
from summarizer.models import Config, PaperSummary


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def test_parser_requires_source_or_file():
    """--source or --file is required; neither raises SystemExit."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_source_and_file_mutually_exclusive():
    """--source and --file cannot be used together."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--source", "/tmp", "--file", "paper.pdf"])


def test_parser_defaults():
    """Default values match Config defaults (with LLM_MODEL unset)."""
    import os

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("LLM_MODEL", None)
        parser = _build_parser()
        args = parser.parse_args(["--source", "/tmp"])
    assert args.model == "openai/gpt-oss-120b:free"
    assert args.base_url == "https://openrouter.ai/api/v1"
    assert args.max_chars == 200_000
    assert args.skill_data_dir == "skill_data/references"
    assert args.output_dir == "output_summaries"
    assert args.force_summary is False
    assert args.reparse is False
    assert args.dry_run is False
    assert args.verbose is True
    assert args.timeout == 120
    assert args.workers == 3
    assert args.extractor == "auto"


def test_parser_model_env_var_used_as_default():
    """LLM_MODEL env var is used when --model is not passed."""
    import os

    with patch.dict(os.environ, {"LLM_MODEL": "my-env-model"}):
        parser = _build_parser()
        args = parser.parse_args(["--source", "/tmp"])
    assert args.model == "my-env-model"


def test_parser_model_cli_overrides_env():
    """--model CLI flag takes precedence over LLM_MODEL env var."""
    import os

    with patch.dict(os.environ, {"LLM_MODEL": "my-env-model"}):
        parser = _build_parser()
        args = parser.parse_args(["--source", "/tmp", "--model", "my-cli-model"])
    assert args.model == "my-cli-model"


@pytest.mark.parametrize("cli_flag,cli_value,config_attr,expected", [
    ("--timeout", "300", "timeout_s", 300),
    ("--workers", "6", "workers", 6),
    ("--extractor", "pypdf", "extractor", "pypdf"),
])
def test_main_cli_flag_propagates_to_config(tmp_path, cli_flag, cli_value, config_attr, expected):
    """CLI flags are forwarded as the corresponding Config fields."""
    (tmp_path / "paper.pdf").write_bytes(b"%PDF")
    with (
        patch(
            "sys.argv",
            ["summarize-papers", "--source", str(tmp_path), "--dry-run", cli_flag, cli_value],
        ),
        patch("summarizer.cli.run_batch") as mock_run_batch,
    ):
        mock_run_batch.return_value = MagicMock(
            processed=0, skipped=1, failed=0, failed_papers=[], total_cost=0.0
        )
        main()
    passed_config = mock_run_batch.call_args[0][1]
    assert getattr(passed_config, config_attr) == expected


def test_parser_custom_flags():
    """Custom flag values are parsed correctly."""
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--source",
            "/tmp",
            "--force-summary",
            "--reparse",
            "--dry-run",
            "--output-dir",
            "/custom/output",
            "--model",
            "my-model",
            "--base-url",
            "http://localhost:8080/v1",
            "--max-chars",
            "50000",
            "--skill-data-dir",
            "/custom/refs",
            "--verbose",
            "--workers",
            "5",
            "--extractor",
            "pypdf",
        ]
    )
    assert args.force_summary is True
    assert args.reparse is True
    assert args.dry_run is True
    assert args.output_dir == "/custom/output"
    assert args.model == "my-model"
    assert args.base_url == "http://localhost:8080/v1"
    assert args.max_chars == 50_000
    assert args.skill_data_dir == "/custom/refs"
    assert args.verbose is True
    assert args.workers == 5
    assert args.extractor == "pypdf"


# ---------------------------------------------------------------------------
# LM Studio health check
# ---------------------------------------------------------------------------


def test_check_lm_studio_succeeds_when_reachable():
    """_check_lm_studio does not exit when the server responds."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        _check_lm_studio("http://localhost:1234/v1")  # should not raise
    # Must hit the root host, not /v1 or /api
    called_url = mock_urlopen.call_args[0][0]
    assert called_url == "http://localhost:1234"


def test_check_backend_strips_to_root_for_openrouter():
    """_check_lm_studio strips to scheme://netloc for OpenRouter URLs."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        _check_lm_studio("https://openrouter.ai/api/v1")
    called_url = mock_urlopen.call_args[0][0]
    assert called_url == "https://openrouter.ai"


def test_check_backend_http_error_is_treated_as_reachable():
    """A 403/404 HTTP response means the server is up (cloud backends need auth)."""
    http_err = urllib.error.HTTPError(
        url=None, code=403, msg="Forbidden", hdrs=None, fp=None
    )
    with patch("urllib.request.urlopen", side_effect=http_err):
        _check_lm_studio("https://openrouter.ai/api/v1")  # should not raise


def test_check_lm_studio_exits_when_unreachable():
    """_check_lm_studio calls sys.exit(1) when the server is not reachable."""
    with (
        patch("urllib.request.urlopen", side_effect=OSError("connection refused")),
        pytest.raises(SystemExit) as exc_info,
    ):
        _check_lm_studio("http://localhost:9999/v1")
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# main() — dry-run batch mode (no LLM calls)
# ---------------------------------------------------------------------------


def test_main_dry_run_batch(tmp_path, capsys):
    """main() with --dry-run --source does not call process_pdf."""
    (tmp_path / "paper.pdf").write_bytes(b"%PDF")

    with (
        patch("sys.argv", ["summarize-papers", "--source", str(tmp_path), "--dry-run"]),
        patch("summarizer.cli.run_batch") as mock_run_batch,
    ):
        mock_run_batch.return_value = MagicMock(
            processed=0, skipped=1, failed=0, failed_papers=[], total_cost=0.0
        )
        main()

    mock_run_batch.assert_called_once()
    _, call_kwargs = mock_run_batch.call_args
    # dry_run flag should be set in the Config passed to run_batch
    passed_config = mock_run_batch.call_args[0][1]
    assert passed_config.dry_run is True


def test_run_single_force_summary_creates_versioned_file(tmp_path):
    """--force-summary on a single file creates _v2.md instead of overwriting."""
    from unittest.mock import MagicMock
    from summarizer.cli import _run_single

    output_dir = tmp_path / "output_summaries"
    existing = output_dir / "synthesis" / "dewolf2021spiking_summary.md"
    existing.parent.mkdir(parents=True)
    existing.write_text("# original", encoding="utf-8")

    abs_pdf = str((tmp_path / "paper.pdf").resolve())
    (output_dir / "processed.txt").write_text(
        f"{abs_pdf}, {existing}\n", encoding="utf-8"
    )

    config = Config(
        base_url="http://localhost:1234/v1",
        model="test-model",
        output_dir=output_dir,
        force_summary=True,
    )
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")

    mock_summary = MagicMock()
    mock_summary.metadata.paper_type = "synthesis"
    mock_summary.metadata.citation_key = "dewolf2021spiking"

    with (
        patch("summarizer.cli.process_pdf", return_value=mock_summary),
        patch("summarizer.cli.render_summary", return_value="# new"),
    ):
        _run_single(pdf, config)

    versioned = output_dir / "synthesis" / "dewolf2021spiking_summary_v2.md"
    assert existing.read_text(encoding="utf-8") == "# original", "original must not be overwritten"
    assert versioned.exists(), "_v2.md must be created"
    assert versioned.read_text(encoding="utf-8") == "# new"


def test_main_single_file_skips_when_in_processed_index(tmp_path, capsys):
    """main() --file skips and exits 0 when the PDF is already in processed.txt."""
    pdf = tmp_path / "huebotter2025spiking.pdf"
    pdf.write_bytes(b"%PDF")

    # Create output_dir and populate processed.txt
    output_dir = tmp_path / "output_summaries"
    output_dir.mkdir()
    (output_dir / "processed.txt").write_text(
        str(pdf.resolve()) + "\n", encoding="utf-8"
    )

    with (
        patch(
            "sys.argv",
            [
                "summarize-papers",
                "--file",
                str(pdf),
                "--output-dir",
                str(output_dir),
            ],
        ),
        patch("summarizer.cli._check_lm_studio"),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# --log-file flag
# ---------------------------------------------------------------------------


def test_parser_log_file_flag(tmp_path):
    """--log-file sets args.log_file."""
    log_file = tmp_path / "output.log"
    parser = _build_parser()
    args = parser.parse_args(["--source", "/tmp", "--log-file", str(log_file)])
    assert args.log_file == str(log_file)


def test_parser_log_file_default_is_none():
    """--log-file defaults to None when not provided."""
    parser = _build_parser()
    args = parser.parse_args(["--source", "/tmp"])
    assert args.log_file is None


# ---------------------------------------------------------------------------
# Phase 6: Done summary includes cost
# ---------------------------------------------------------------------------


def test_run_batch_done_log_includes_cost(tmp_path, caplog):
    """_run_batch logs 'cost=' in the Done summary line."""
    import logging
    from summarizer.cli import _run_batch
    from summarizer.models import Config, BatchReport

    config = Config(
        base_url="http://localhost:1234/v1",
        model="test-model",
        output_dir=tmp_path / "output_summaries",
    )
    report = BatchReport(processed=3, skipped=1, failed=0, failed_papers=[], total_cost=0.0345)

    with (
        caplog.at_level(logging.INFO, logger="summarizer.cli"),
        patch("summarizer.cli.run_batch", return_value=report),
    ):
        _run_batch(tmp_path, config)

    done_lines = [r.message for r in caplog.records if "Done" in r.message]
    assert done_lines, "Expected a 'Done' log line"
    assert "cost=" in done_lines[0]
    assert "0.0345" in done_lines[0]
