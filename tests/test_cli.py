"""Tests for summarizer/cli.py — argument parsing and high-level CLI behaviour."""

import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from summarizer.cli import _build_parser, _check_backend, main
from summarizer.models import DEFAULT_MODEL, DEFAULT_SKILL_DATA_DIR, Config

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
    assert args.model == DEFAULT_MODEL
    assert args.base_url == "https://openrouter.ai/api/v1"
    assert args.max_chars == 200_000
    assert Path(args.skill_data_dir) == DEFAULT_SKILL_DATA_DIR
    assert Path(args.skill_data_dir).is_dir()
    assert args.output_dir == "output_summaries"
    assert args.force_summary is False
    assert args.reparse is False
    assert args.dry_run is False
    assert args.verbose is False
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


@pytest.mark.parametrize(
    "cli_flag,cli_value,config_attr,expected",
    [
        ("--timeout", "300", "timeout_s", 300),
        ("--workers", "6", "workers", 6),
        ("--extractor", "pypdf", "extractor", "pypdf"),
    ],
)
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
# Backend health check
# ---------------------------------------------------------------------------


def test_check_backend_succeeds_when_reachable():
    """_check_backend does not exit when the server responds."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        _check_backend("http://localhost:1234/v1")  # should not raise
    # Must hit the root host, not /v1 or /api
    called_url = mock_urlopen.call_args[0][0]
    assert called_url == "http://localhost:1234"


def test_check_backend_strips_to_root_for_openrouter():
    """_check_backend strips to scheme://netloc for OpenRouter URLs."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        _check_backend("https://openrouter.ai/api/v1")
    called_url = mock_urlopen.call_args[0][0]
    assert called_url == "https://openrouter.ai"


def test_check_backend_http_error_is_treated_as_reachable():
    """A 403/404 HTTP response means the server is up (cloud backends need auth)."""
    http_err = urllib.error.HTTPError(url=None, code=403, msg="Forbidden", hdrs=None, fp=None)
    with patch("urllib.request.urlopen", side_effect=http_err):
        _check_backend("https://openrouter.ai/api/v1")  # should not raise


def test_check_backend_exits_when_unreachable():
    """_check_backend calls sys.exit(1) when the server is not reachable."""
    with (
        patch("urllib.request.urlopen", side_effect=OSError("connection refused")),
        pytest.raises(SystemExit) as exc_info,
    ):
        _check_backend("http://localhost:9999/v1")
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
    (output_dir / "processed.txt").write_text(f"{abs_pdf}, {existing}\n", encoding="utf-8")

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
        patch("summarizer.batch.create_client"),
        patch("summarizer.batch.process_pdf", return_value=mock_summary),
        patch("summarizer.batch.render_summary", return_value="# new"),
    ):
        _run_single(pdf, config)

    versioned = output_dir / "synthesis" / "dewolf2021spiking_summary_v2.md"
    assert existing.read_text(encoding="utf-8") == "# original", "original must not be overwritten"
    assert versioned.exists(), "_v2.md must be created"
    assert versioned.read_text(encoding="utf-8") == "# new"


def test_main_single_file_skips_when_in_processed_index(tmp_path, capsys):
    """main() --file skips and exits 0 when the PDF is in the (legacy) processed.txt."""
    pdf = tmp_path / "huebotter2025spiking.pdf"
    pdf.write_bytes(b"%PDF")

    # Create output_dir and populate processed.txt
    output_dir = tmp_path / "output_summaries"
    output_dir.mkdir()
    (output_dir / "processed.txt").write_text(str(pdf.resolve()) + "\n", encoding="utf-8")

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
        patch("summarizer.cli._check_backend"),
        patch("summarizer.cli._check_openrouter_config"),
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
    from summarizer.models import BatchReport, Config

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


# ---------------------------------------------------------------------------
# OpenRouter preflight and single-file path
# ---------------------------------------------------------------------------


def _or_config(**kwargs):
    return Config(base_url="https://openrouter.ai/api/v1", model="meta/some-model", **kwargs)


def test_openrouter_preflight_exits_without_api_key(monkeypatch):
    from summarizer.cli import _check_openrouter_config

    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        _check_openrouter_config(_or_config())
    assert exc_info.value.code == 1


def test_openrouter_preflight_exits_when_model_not_listed(monkeypatch):
    """Regression: a retired default model id made every paper fail one by one."""
    from summarizer.cli import _check_openrouter_config

    monkeypatch.setenv("LLM_API_KEY", "k")
    with (
        patch("summarizer.cli.fetch_openrouter_model_ids", return_value={"other/model"}),
        pytest.raises(SystemExit) as exc_info,
    ):
        _check_openrouter_config(_or_config())
    assert exc_info.value.code == 1


def test_openrouter_preflight_passes_for_listed_model(monkeypatch):
    from summarizer.cli import _check_openrouter_config

    monkeypatch.setenv("LLM_API_KEY", "k")
    with patch("summarizer.cli.fetch_openrouter_model_ids", return_value={"meta/some-model"}):
        _check_openrouter_config(_or_config())  # no exit


def test_openrouter_preflight_tolerates_unreachable_model_list(monkeypatch):
    from summarizer.cli import _check_openrouter_config

    monkeypatch.setenv("LLM_API_KEY", "k")
    with patch("summarizer.cli.fetch_openrouter_model_ids", return_value=None):
        _check_openrouter_config(_or_config())  # no exit


def test_preflight_is_noop_for_local_backends(monkeypatch):
    from summarizer.cli import _check_openrouter_config

    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with patch("summarizer.cli.fetch_openrouter_model_ids") as mock_ids:
        _check_openrouter_config(Config(base_url="http://localhost:1234/v1"))
    mock_ids.assert_not_called()


def test_run_single_reports_cost_and_exits_1_on_failure(tmp_path, caplog):
    """Single-file mode now shares the batch path: cost is logged, failures exit 1."""
    import logging

    from summarizer.cli import _run_single
    from summarizer.models import PipelineError

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    config = Config(base_url="http://localhost:1234/v1", model="m", output_dir=tmp_path / "out")
    with (
        caplog.at_level(logging.INFO, logger="summarizer.cli"),
        patch("summarizer.batch.create_client"),
        patch("summarizer.batch.process_pdf", side_effect=PipelineError(pdf, Exception("boom"))),
        pytest.raises(SystemExit) as exc_info,
    ):
        _run_single(pdf, config)
    assert exc_info.value.code == 1
    assert any("cost=" in r.message for r in caplog.records if "Done" in r.message)


def test_run_single_missing_file_exits_1(tmp_path):
    from summarizer.cli import _run_single

    with pytest.raises(SystemExit) as exc_info:
        _run_single(tmp_path / "nope.pdf", Config(output_dir=tmp_path))
    assert exc_info.value.code == 1


@pytest.mark.parametrize(
    "model,listed",
    [
        ("meta/some-model", True),
        ("meta/some-model:nitro", True),
        ("meta/some-model:floor", True),
        ("meta/some-model:online", True),
        ("meta/some-model:exacto", True),
        ("meta/some-model:free", False),  # retired :free variants must be caught
        ("meta/other", False),
        ("@preset/my-preset", True),  # presets aren't in the models list
    ],
)
def test_openrouter_model_listed(model, listed):
    from summarizer.cli import _openrouter_model_listed

    assert _openrouter_model_listed(model, {"meta/some-model"}) is listed


def test_main_works_from_another_directory(tmp_path, monkeypatch):
    """Regression: the default references path was relative to the CWD."""
    (tmp_path / "paper.pdf").write_bytes(b"%PDF")
    monkeypatch.chdir(tmp_path)
    with (
        patch("sys.argv", ["summarize-papers", "--file", "paper.pdf"]),
        patch("summarizer.cli._check_backend"),
        patch("summarizer.cli._check_openrouter_config"),
        patch("summarizer.batch.create_client"),
        patch("summarizer.pipeline.parse_pdf", return_value="text"),
        patch("summarizer.pipeline.call_llm", side_effect=RuntimeError("stop after prompt")),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()
    # Fails at the (mocked) LLM step, not with "References directory not found".
    assert exc_info.value.code == 1
    log = next((tmp_path / "logs").glob("run_*.log")).read_text()
    assert "References directory not found" not in log
    assert "stop after prompt" in log


def test_source_must_be_a_directory(tmp_path):
    from summarizer.cli import _run_batch

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    with pytest.raises(SystemExit) as exc_info:
        _run_batch(pdf, Config(output_dir=tmp_path / "out"))
    assert exc_info.value.code == 1


def test_keyboard_interrupt_exits_130(tmp_path):
    (tmp_path / "paper.pdf").write_bytes(b"%PDF")
    with (
        patch("sys.argv", ["summarize-papers", "--source", str(tmp_path)]),
        patch("summarizer.cli._check_backend"),
        patch("summarizer.cli._check_openrouter_config"),
        patch("summarizer.cli.run_batch", side_effect=KeyboardInterrupt),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()
    assert exc_info.value.code == 130
