"""Tests for summarizer/batch.py — directory scanning, processed index, and batch execution."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from summarizer.batch import (
    find_pdfs,
    get_output_path,
    load_processed_index,
    run_batch,
    save_processed_index,
    should_skip,
)
from summarizer.models import BatchReport, Config, PipelineError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_path):
    """Config with output_dir pointing at a tmp directory."""
    references_dir = Path(__file__).parent.parent / "skill_data" / "references"
    return Config(
        base_url="http://localhost:1234/v1",
        model="test-model",
        skill_data_dir=references_dir,
        output_dir=tmp_path / "output_summaries",
    )


@pytest.fixture
def pdf_dir(tmp_path):
    """Directory with three PDFs."""
    (tmp_path / "paper_a.pdf").write_bytes(b"%PDF")
    (tmp_path / "paper_b.pdf").write_bytes(b"%PDF")
    (tmp_path / "paper_c.pdf").write_bytes(b"%PDF")
    return tmp_path


# ---------------------------------------------------------------------------
# find_pdfs (unchanged from v1.0)
# ---------------------------------------------------------------------------


def test_find_pdfs_returns_all_pdfs(pdf_dir):
    pdfs = find_pdfs(pdf_dir)
    assert len(pdfs) == 3
    assert all(p.suffix == ".pdf" for p in pdfs)


def test_find_pdfs_recursive(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "root.pdf").write_bytes(b"%PDF")
    (sub / "nested.pdf").write_bytes(b"%PDF")
    pdfs = find_pdfs(tmp_path)
    assert len(pdfs) == 2


def test_find_pdfs_empty_dir(tmp_path):
    assert find_pdfs(tmp_path) == []


# ---------------------------------------------------------------------------
# load_processed_index
# ---------------------------------------------------------------------------


def test_load_processed_index_returns_empty_set_when_file_missing(tmp_path):
    output_dir = tmp_path / "output_summaries"
    output_dir.mkdir()
    result = load_processed_index(output_dir)
    assert result == set()


def test_load_processed_index_reads_paths(tmp_path):
    output_dir = tmp_path / "output_summaries"
    output_dir.mkdir()
    index = output_dir / "processed.txt"
    index.write_text("/path/a.pdf\n/path/b.pdf\n", encoding="utf-8")
    result = load_processed_index(output_dir)
    assert "/path/a.pdf" in result
    assert "/path/b.pdf" in result
    assert len(result) == 2


def test_load_processed_index_ignores_blank_lines(tmp_path):
    output_dir = tmp_path / "output_summaries"
    output_dir.mkdir()
    (output_dir / "processed.txt").write_text("/a.pdf\n\n/b.pdf\n\n", encoding="utf-8")
    result = load_processed_index(output_dir)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# save_processed_index
# ---------------------------------------------------------------------------


def test_save_processed_index_writes_file(tmp_path):
    output_dir = tmp_path / "output_summaries"
    output_dir.mkdir()
    paths = {"/path/a.pdf", "/path/b.pdf"}
    save_processed_index(output_dir, paths)
    content = (output_dir / "processed.txt").read_text(encoding="utf-8")
    assert "/path/a.pdf" in content
    assert "/path/b.pdf" in content


def test_save_processed_index_creates_output_dir(tmp_path):
    output_dir = tmp_path / "new_output"
    save_processed_index(output_dir, {"/some/path.pdf"})
    assert output_dir.exists()
    assert (output_dir / "processed.txt").exists()


def test_load_save_roundtrip(tmp_path):
    output_dir = tmp_path / "output_summaries"
    output_dir.mkdir()
    paths = {"/a.pdf", "/b.pdf", "/c.pdf"}
    save_processed_index(output_dir, paths)
    loaded = load_processed_index(output_dir)
    assert loaded == paths


# ---------------------------------------------------------------------------
# should_skip
# ---------------------------------------------------------------------------


def test_should_skip_returns_true_when_in_processed(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    processed = {str(pdf.resolve())}
    assert should_skip(pdf, processed, force_summary=False) is True


def test_should_skip_returns_false_when_not_in_processed(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    assert should_skip(pdf, set(), force_summary=False) is False


def test_should_skip_force_summary_overrides(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    processed = {str(pdf.resolve())}
    assert should_skip(pdf, processed, force_summary=True) is False


# ---------------------------------------------------------------------------
# get_output_path
# ---------------------------------------------------------------------------


def test_get_output_path_returns_correct_path(tmp_path):
    output_dir = tmp_path / "output_summaries"
    path = get_output_path(output_dir, "primary", "smith2020foo")
    assert path == output_dir / "primary" / "smith2020foo_summary.md"


def test_get_output_path_creates_subdir(tmp_path):
    output_dir = tmp_path / "output_summaries"
    path = get_output_path(output_dir, "survey", "jones2021bar")
    assert path.parent.exists()


def test_get_output_path_non_research_category(tmp_path):
    output_dir = tmp_path / "output_summaries"
    path = get_output_path(output_dir, "non_research", "x2020y")
    assert path == output_dir / "non_research" / "x2020y_summary.md"


# ---------------------------------------------------------------------------
# run_batch
# ---------------------------------------------------------------------------


def _make_summary(citation_key: str, paper_type: str = "primary") -> MagicMock:
    s = MagicMock()
    s.metadata.citation_key = citation_key
    s.metadata.paper_type = paper_type
    return s


def test_run_batch_processes_all_pdfs(tmp_path, config):
    """run_batch calls process_pdf for each PDF and writes output files."""
    (tmp_path / "paper_a.pdf").write_bytes(b"%PDF")
    (tmp_path / "paper_b.pdf").write_bytes(b"%PDF")

    summaries = [_make_summary("smith2020foo"), _make_summary("jones2021bar")]

    with (
        patch("summarizer.batch.process_pdf", side_effect=summaries) as mock_process,
        patch("summarizer.batch.render_summary", return_value="# Markdown"),
    ):
        report = run_batch(tmp_path, config)

    assert report.processed == 2
    assert report.skipped == 0
    assert report.failed == 0
    assert mock_process.call_count == 2


def test_run_batch_uses_configured_worker_count(tmp_path, config):
    (tmp_path / "paper_a.pdf").write_bytes(b"%PDF")
    config.workers = 4

    summary = _make_summary("smith2020foo")
    with (
        patch("summarizer.batch.ThreadPoolExecutor") as mock_executor_cls,
        patch("summarizer.batch.process_pdf", return_value=summary),
        patch("summarizer.batch.render_summary", return_value="# Markdown"),
    ):
        mock_executor = MagicMock()
        mock_executor.__enter__.return_value = mock_executor
        mock_executor_cls.return_value = mock_executor

        fake_future = MagicMock()
        fake_future.result.return_value = {
            "pdf_path": tmp_path / "paper_a.pdf",
            "summary": summary,
            "markdown": "# Markdown",
        }
        mock_executor.submit.return_value = fake_future

        with patch("summarizer.batch.as_completed", return_value=[fake_future]):
            report = run_batch(tmp_path, config)

    mock_executor_cls.assert_called_once_with(
        max_workers=4, thread_name_prefix="worker"
    )
    assert report.processed == 1


def test_run_batch_uses_version_suffix_when_output_exists(tmp_path, config):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")

    existing = config.output_dir / "primary" / "smith2020foo_summary.md"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("# old", encoding="utf-8")

    summary = _make_summary("smith2020foo", paper_type="primary")
    with (
        patch("summarizer.batch.process_pdf", return_value=summary),
        patch("summarizer.batch.render_summary", return_value="# new"),
    ):
        run_batch(tmp_path, config)

    versioned = config.output_dir / "primary" / "smith2020foo_summary_v2.md"
    assert existing.read_text(encoding="utf-8") == "# old"
    assert versioned.exists()
    assert versioned.read_text(encoding="utf-8") == "# new"


def test_run_batch_skips_processed_pdfs(tmp_path, config):
    """run_batch skips PDFs already listed in processed.txt."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")

    config.output_dir.mkdir(parents=True)
    (config.output_dir / "processed.txt").write_text(
        str(pdf.resolve()) + "\n", encoding="utf-8"
    )

    with patch("summarizer.batch.process_pdf") as mock_process:
        report = run_batch(tmp_path, config)

    mock_process.assert_not_called()
    assert report.skipped == 1
    assert report.processed == 0


def test_run_batch_force_summary_reprocesses(tmp_path, config):
    """run_batch with force_summary=True re-processes processed files."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")

    config.output_dir.mkdir(parents=True)
    (config.output_dir / "processed.txt").write_text(
        str(pdf.resolve()) + "\n", encoding="utf-8"
    )
    config.force_summary = True

    summary = _make_summary("smith2020foo")
    with (
        patch("summarizer.batch.process_pdf", return_value=summary) as mock_process,
        patch("summarizer.batch.render_summary", return_value="# Markdown"),
    ):
        report = run_batch(tmp_path, config)

    assert report.processed == 1
    assert mock_process.call_count == 1


def test_run_batch_records_failures(tmp_path, config):
    """run_batch continues after a PipelineError and records the failure."""
    (tmp_path / "bad_paper.pdf").write_bytes(b"%PDF")

    err = PipelineError(tmp_path / "bad_paper.pdf", Exception("boom"))
    with (
        patch("summarizer.batch.process_pdf", side_effect=err),
        patch("summarizer.batch.render_summary", return_value="# Markdown"),
    ):
        report = run_batch(tmp_path, config)

    assert report.failed == 1
    assert report.processed == 0
    assert "bad_paper.pdf" in report.failed_papers[0].pdf_path


def test_run_batch_dry_run_makes_no_llm_calls(tmp_path, config):
    """run_batch with dry_run=True does not call process_pdf."""
    (tmp_path / "paper_a.pdf").write_bytes(b"%PDF")
    config.dry_run = True

    with patch("summarizer.batch.process_pdf") as mock_process:
        report = run_batch(tmp_path, config)

    mock_process.assert_not_called()
    assert report.skipped == 1
    assert report.processed == 0


def test_run_batch_logs_selection_summary(tmp_path, config, caplog):
    """Batch logs include discovered/selected/skipped counts before processing."""
    (tmp_path / "paper_a.pdf").write_bytes(b"%PDF")
    (tmp_path / "paper_b.pdf").write_bytes(b"%PDF")

    processed_pdf = tmp_path / "paper_b.pdf"
    config.output_dir.mkdir(parents=True)
    (config.output_dir / "processed.txt").write_text(
        str(processed_pdf.resolve()) + "\n", encoding="utf-8"
    )

    summary = _make_summary("smith2020foo")
    with (
        caplog.at_level("INFO", logger="summarizer.batch"),
        patch("summarizer.batch.process_pdf", return_value=summary),
        patch("summarizer.batch.render_summary", return_value="# md"),
    ):
        run_batch(tmp_path, config)

    messages = [r.message for r in caplog.records]
    assert any("Discovered PDFs:" in m for m in messages)
    assert any("Selected for processing:" in m for m in messages)
    assert any("Skipped by processed index:" in m for m in messages)


def test_run_batch_writes_to_centralized_output(tmp_path, config):
    """run_batch writes summary to output_summaries/{paper_type}/{citekey}_summary.md."""
    (tmp_path / "paper.pdf").write_bytes(b"%PDF")

    summary = _make_summary("smith2020foo", paper_type="primary")
    with (
        patch("summarizer.batch.process_pdf", return_value=summary),
        patch("summarizer.batch.render_summary", return_value="# Markdown content"),
    ):
        run_batch(tmp_path, config)

    expected = config.output_dir / "primary" / "smith2020foo_summary.md"
    assert expected.exists()
    assert expected.read_text(encoding="utf-8") == "# Markdown content"


def test_run_batch_appends_to_processed_txt(tmp_path, config):
    """After a successful run, the PDF path is appended to processed.txt."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")

    summary = _make_summary("smith2020foo")
    with (
        patch("summarizer.batch.process_pdf", return_value=summary),
        patch("summarizer.batch.render_summary", return_value="# md"),
    ):
        run_batch(tmp_path, config)

    processed = load_processed_index(config.output_dir)
    assert str(pdf.resolve()) in processed


def test_run_batch_failed_paper_not_in_processed(tmp_path, config):
    """Failed papers are NOT added to processed.txt."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")

    err = PipelineError(pdf, Exception("boom"))
    with patch("summarizer.batch.process_pdf", side_effect=err):
        run_batch(tmp_path, config)

    processed = load_processed_index(config.output_dir)
    assert str(pdf.resolve()) not in processed
