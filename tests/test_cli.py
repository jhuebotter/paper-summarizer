"""Tests for summarizer/cli.py — argument parsing and high-level CLI behaviour."""

import itertools
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from summarizer.cli import _build_parser, _check_backend, main
from summarizer.models import DEFAULT_MODEL, DEFAULT_SKILL_DATA_DIR, Config
from summarizer.parser import ParsedText

_PDF_COUNTER = itertools.count()


def _pdf_bytes() -> bytes:
    """Distinct content per test PDF (identical PDFs are de-duplicated by sha256)."""
    return f"%PDF {next(_PDF_COUNTER)}".encode()


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
    assert args.zotero is True
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
        ("--max-cost", "0.25", "max_cost", 0.25),
    ],
)
def test_main_cli_flag_propagates_to_config(tmp_path, cli_flag, cli_value, config_attr, expected):
    """CLI flags are forwarded as the corresponding Config fields."""
    (tmp_path / "paper.pdf").write_bytes(_pdf_bytes())
    with (
        patch(
            "sys.argv",
            ["summarize-papers", "--source", str(tmp_path), "--dry-run", cli_flag, cli_value],
        ),
        patch("summarizer.cli.run_batch") as mock_run_batch,
    ):
        mock_run_batch.return_value = MagicMock(
            processed=0, skipped=1, failed=0, failed_papers=[], total_cost=0.0, stopped_reason=None
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
    (tmp_path / "paper.pdf").write_bytes(_pdf_bytes())

    with (
        patch("sys.argv", ["summarize-papers", "--source", str(tmp_path), "--dry-run"]),
        patch("summarizer.cli.run_batch") as mock_run_batch,
    ):
        mock_run_batch.return_value = MagicMock(
            processed=0, skipped=1, failed=0, failed_papers=[], total_cost=0.0, stopped_reason=None
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
    pdf.write_bytes(_pdf_bytes())

    mock_summary = MagicMock()
    mock_summary.metadata.paper_type = "synthesis"
    mock_summary.metadata.citation_key = "dewolf2021spiking"
    mock_summary.model_dump_json.return_value = "{}"

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


def test_main_single_file_skips_when_in_processed_index(tmp_path):
    """main() --file skips a PDF that is in the (legacy) processed.txt."""
    pdf = tmp_path / "huebotter2025spiking.pdf"
    pdf.write_bytes(_pdf_bytes())
    output_dir = tmp_path / "output_summaries"
    output_dir.mkdir()
    (output_dir / "processed.txt").write_text(str(pdf.resolve()) + "\n", encoding="utf-8")

    with (
        patch(
            "sys.argv", ["summarize-papers", "--file", str(pdf), "--output-dir", str(output_dir)]
        ),
        patch("summarizer.cli._check_backend"),
        patch("summarizer.cli._check_openrouter_config"),
        patch("summarizer.cli._log_key_info"),
        patch("summarizer.batch.process_pdf") as process,
    ):
        main()
    process.assert_not_called()


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
        patch("summarizer.cli.fetch_openrouter_models", return_value=[{"id": "other/model"}]),
        pytest.raises(SystemExit) as exc_info,
    ):
        _check_openrouter_config(_or_config())
    assert exc_info.value.code == 1


def test_openrouter_preflight_passes_for_listed_model(monkeypatch):
    from summarizer.cli import _check_openrouter_config

    monkeypatch.setenv("LLM_API_KEY", "k")
    with patch("summarizer.cli.fetch_openrouter_models", return_value=[{"id": "meta/some-model"}]):
        _check_openrouter_config(_or_config())  # no exit


def test_openrouter_preflight_tolerates_unreachable_model_list(monkeypatch):
    from summarizer.cli import _check_openrouter_config

    monkeypatch.setenv("LLM_API_KEY", "k")
    with patch("summarizer.cli.fetch_openrouter_models", return_value=None):
        _check_openrouter_config(_or_config())  # no exit


def test_preflight_is_noop_for_local_backends(monkeypatch):
    from summarizer.cli import _check_openrouter_config

    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with patch("summarizer.cli.fetch_openrouter_models") as mock_ids:
        _check_openrouter_config(Config(base_url="http://localhost:1234/v1"))
    mock_ids.assert_not_called()


def test_run_single_reports_cost_and_exits_1_on_failure(tmp_path, caplog):
    """Single-file mode now shares the batch path: cost is logged, failures exit 1."""
    import logging

    from summarizer.cli import _run_single
    from summarizer.models import PipelineError

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(_pdf_bytes())
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
    (tmp_path / "paper.pdf").write_bytes(_pdf_bytes())
    monkeypatch.chdir(tmp_path)
    with (
        patch("sys.argv", ["summarize-papers", "--file", "paper.pdf"]),
        patch("summarizer.cli._check_backend"),
        patch("summarizer.cli._check_openrouter_config"),
        patch("summarizer.batch.create_client"),
        patch("summarizer.pipeline.load_text", return_value=ParsedText("text", "pypdf", "0" * 64)),
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
    pdf.write_bytes(_pdf_bytes())
    with pytest.raises(SystemExit) as exc_info:
        _run_batch(pdf, Config(output_dir=tmp_path / "out"))
    assert exc_info.value.code == 1


def test_keyboard_interrupt_exits_130(tmp_path):
    (tmp_path / "paper.pdf").write_bytes(_pdf_bytes())
    with (
        patch("sys.argv", ["summarize-papers", "--source", str(tmp_path)]),
        patch("summarizer.cli._check_backend"),
        patch("summarizer.cli._check_openrouter_config"),
        patch("summarizer.cli.run_batch", side_effect=KeyboardInterrupt),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()
    assert exc_info.value.code == 130


def test_preflight_rejects_structured_output_for_unsupported_model(monkeypatch):
    from summarizer.cli import _check_openrouter_config

    monkeypatch.setenv("LLM_API_KEY", "k")
    models = [{"id": "meta/some-model", "supported_parameters": ["max_tokens"]}]
    with (
        patch("summarizer.cli.fetch_openrouter_models", return_value=models),
        pytest.raises(SystemExit) as exc_info,
    ):
        _check_openrouter_config(_or_config(structured_output=True))
    assert exc_info.value.code == 1

    models[0]["supported_parameters"].append("structured_outputs")
    with patch("summarizer.cli.fetch_openrouter_models", return_value=models):
        _check_openrouter_config(_or_config(structured_output=True))  # no exit


@pytest.mark.parametrize("model,warned", [("meta/m:free", True), ("meta/m", False)])
def test_key_info_warns_about_free_tier_daily_cap(monkeypatch, caplog, model, warned):
    import logging

    from summarizer.cli import _log_key_info

    monkeypatch.setenv("LLM_API_KEY", "k")
    info = {"usage": 0.0, "limit": None, "limit_remaining": None, "is_free_tier": True}
    with (
        patch("summarizer.cli.fetch_openrouter_key_info", return_value=info),
        caplog.at_level(logging.INFO, logger="summarizer.cli"),
    ):
        _log_key_info(Config(base_url="https://openrouter.ai/api/v1", model=model))
    assert any("50 requests/day" in r.message for r in caplog.records) is warned


def test_structured_output_preflight_with_routing_suffix(monkeypatch):
    from summarizer.cli import _check_openrouter_config

    monkeypatch.setenv("LLM_API_KEY", "k")
    models = [{"id": "a/b", "supported_parameters": ["structured_outputs"]}]
    config = Config(
        base_url="https://openrouter.ai/api/v1", model="a/b:nitro", structured_output=True
    )
    with patch("summarizer.cli.fetch_openrouter_models", return_value=models):
        _check_openrouter_config(config)  # no exit


def test_eval_flags_propagate(tmp_path):
    (tmp_path / "p.pdf").write_bytes(_pdf_bytes())
    with (
        patch("summarizer.cli._check_backend"),
        patch("summarizer.cli._check_openrouter_config"),
        patch("summarizer.cli.run_eval", return_value=([], None)) as run,
    ):
        main(
            [
                "eval",
                "--source",
                str(tmp_path),
                "--extractors",
                "pypdf",
                "--no-strip-references",
                "--max-cost",
                "0.5",
                "--out",
                str(tmp_path / "run"),
            ]
        )
    config = run.call_args.args[1]
    assert (config.strip_references, config.max_cost) == (False, 0.5)


def test_stopped_eval_exits_1(tmp_path):
    (tmp_path / "p.pdf").write_bytes(_pdf_bytes())
    with (
        patch("summarizer.cli._check_backend"),
        patch("summarizer.cli._check_openrouter_config"),
        patch("summarizer.cli.run_eval", return_value=([], "daily cap")),
        pytest.raises(SystemExit) as exc_info,
    ):
        main(
            [
                "eval",
                "--source",
                str(tmp_path),
                "--extractors",
                "pypdf",
                "--out",
                str(tmp_path / "r"),
            ]
        )
    assert exc_info.value.code == 1


def test_stopped_run_exits_1(caplog):
    from summarizer.cli import _report_and_exit
    from summarizer.models import BatchReport

    report = BatchReport(
        processed=1, skipped=4, failed=0, failed_papers=[], stopped_reason="daily cap"
    )
    with pytest.raises(SystemExit) as exc_info:
        _report_and_exit(report)
    assert exc_info.value.code == 1


@pytest.mark.parametrize(
    "flag,attr,expected",
    [
        ("--no-strip-references", "strip_references", False),
        ("--structured-output", "structured_output", True),
    ],
)
def test_boolean_flags_propagate_to_config(tmp_path, flag, attr, expected):
    (tmp_path / "paper.pdf").write_bytes(_pdf_bytes())
    with (
        patch("sys.argv", ["summarize-papers", "--source", str(tmp_path), "--dry-run", flag]),
        patch("summarizer.cli.run_batch") as mock_run_batch,
    ):
        mock_run_batch.return_value = MagicMock(
            processed=0, skipped=1, failed=0, failed_papers=[], total_cost=0.0, stopped_reason=None
        )
        main()
    assert getattr(mock_run_batch.call_args[0][1], attr) is expected


def test_new_flag_defaults():
    args = _build_parser().parse_args(["--source", "/tmp"])
    assert (args.strip_references, args.structured_output, args.max_cost) == (True, False, None)


def test_render_subcommand(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    with patch("summarizer.cli.render_all", return_value=(3, 0)) as render:
        main(["render", "--output-dir", str(out)])
    render.assert_called_once_with(out)


def test_render_reports_a_locked_output_dir(tmp_path):
    from summarizer.batch import OutputDirLocked

    out = tmp_path / "out"
    out.mkdir()
    with (
        patch("summarizer.cli.render_all", side_effect=OutputDirLocked("busy")),
        pytest.raises(SystemExit) as exc_info,
    ):
        main(["render", "--output-dir", str(out)])
    assert exc_info.value.code == 1


def test_locked_output_dir_exits_1(tmp_path):
    from summarizer.batch import OutputDirLocked

    (tmp_path / "p.pdf").write_bytes(_pdf_bytes())
    with (
        patch("summarizer.cli._check_backend"),
        patch("summarizer.cli._check_openrouter_config"),
        patch("summarizer.cli._log_key_info"),
        patch("summarizer.cli.run_batch", side_effect=OutputDirLocked("busy")),
        pytest.raises(SystemExit) as exc_info,
    ):
        main(["--source", str(tmp_path)])
    assert exc_info.value.code == 1


def test_render_exits_1_when_sidecars_fail(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    with (
        patch("summarizer.cli.render_all", return_value=(2, 1)),
        pytest.raises(SystemExit) as exc_info,
    ):
        main(["render", "--output-dir", str(out)])
    assert exc_info.value.code == 1


def test_render_command_end_to_end(tmp_path, mock_part1_dict, mock_part2_dict):
    from summarizer.models import PaperSummary

    meta_keys = {"citation_key", "title", "authors", "year", "venue", "paper_type", "tags"}
    metadata = {k: mock_part1_dict[k] for k in meta_keys} | {
        "is_research_paper": True,
        "rejection_reason": None,
    }
    part1 = {k: v for k, v in mock_part1_dict.items() if k not in meta_keys - {"paper_type"}}
    summary = PaperSummary(metadata=metadata, part1=part1, part2=mock_part2_dict)
    out = tmp_path / "out" / "primary"
    out.mkdir(parents=True)
    (out / "a2020x_summary.json").write_text(summary.model_dump_json())
    main(["render", "--output-dir", str(tmp_path / "out")])
    assert "### Classification" in (out / "a2020x_summary.md").read_text()


def test_render_rejects_a_missing_output_dir(tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        main(["render", "--output-dir", str(tmp_path / "missing")])
    assert exc_info.value.code == 1


def test_no_zotero_flag():
    args = _build_parser().parse_args(["--source", ".", "--no-zotero"])
    assert args.zotero is False
