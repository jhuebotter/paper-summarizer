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
from summarizer.models import Config, PipelineError

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
# find_pdfs
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


def test_find_pdfs_matches_uppercase_extension_and_skips_appledouble(tmp_path):
    (tmp_path / "upper.PDF").write_bytes(b"%PDF")
    (tmp_path / "._upper.pdf").write_bytes(b"junk")
    (tmp_path / "notes.md").write_text("x")
    assert [p.name for p in find_pdfs(tmp_path)] == ["upper.PDF"]


# ---------------------------------------------------------------------------
# load_processed_index (legacy processed.txt is still read)
# ---------------------------------------------------------------------------


def test_load_processed_index_returns_empty_dict_when_file_missing(tmp_path):
    output_dir = tmp_path / "output_summaries"
    output_dir.mkdir()
    result = load_processed_index(output_dir)
    assert result == {}


def test_load_processed_index_reads_paths(tmp_path):
    output_dir = tmp_path / "output_summaries"
    output_dir.mkdir()
    index = output_dir / "processed.txt"
    index.write_text("/path/a.pdf\n/path/b.pdf\n", encoding="utf-8")
    result = load_processed_index(output_dir)
    assert "/path/a.pdf" in result
    assert "/path/b.pdf" in result
    assert len(result) == 2


def test_load_processed_index_reads_summary_paths(tmp_path):
    output_dir = tmp_path / "output_summaries"
    output_dir.mkdir()
    (output_dir / "processed.txt").write_text(
        "/path/a.pdf, /out/a_summary.md, /out/a_summary_v2.md\n/path/b.pdf, /out/b_summary.md\n",
        encoding="utf-8",
    )
    result = load_processed_index(output_dir)
    assert result["/path/a.pdf"] == ["/out/a_summary.md", "/out/a_summary_v2.md"]
    assert result["/path/b.pdf"] == ["/out/b_summary.md"]


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
    index = {"/path/a.pdf": ["/out/a_summary.md"], "/path/b.pdf": []}
    save_processed_index(output_dir, index)
    content = (output_dir / "processed.jsonl").read_text(encoding="utf-8")
    assert "/path/a.pdf" in content
    assert "/path/b.pdf" in content
    assert "/out/a_summary.md" in content


def test_save_processed_index_creates_output_dir(tmp_path):
    output_dir = tmp_path / "new_output"
    save_processed_index(output_dir, {"/some/path.pdf": []})
    assert output_dir.exists()
    assert (output_dir / "processed.jsonl").exists()


def test_load_save_roundtrip(tmp_path):
    output_dir = tmp_path / "output_summaries"
    output_dir.mkdir()
    index = {
        "/a.pdf": ["/out/a_summary.md"],
        "/b.pdf": ["/out/b_summary.md", "/out/b_summary_v2.md"],
        "/c.pdf": [],
    }
    save_processed_index(output_dir, index)
    loaded = load_processed_index(output_dir)
    assert loaded == index


def test_roundtrip_preserves_paths_with_commas(tmp_path):
    """Regression: the old comma-separated format truncated such paths."""
    pdf = tmp_path / "Smith et al. - 2024 - Spikes, Robots, and Control.pdf"
    pdf.write_bytes(b"%PDF")
    index = {str(pdf.resolve()): [str(tmp_path / "a, b_summary.md")]}
    save_processed_index(tmp_path, index)
    loaded = load_processed_index(tmp_path)
    assert loaded == index
    assert should_skip(pdf, loaded, force_summary=False) is True


def test_legacy_index_with_commas_in_paths_is_parsed(tmp_path):
    (tmp_path / "processed.txt").write_text(
        "/lib/Smith - 2024 - Spikes, Robots.pdf, /out/smith2024spikes_summary.md, "
        "/out/smith2024spikes_summary_v2.md\n",
        encoding="utf-8",
    )
    assert load_processed_index(tmp_path) == {
        "/lib/Smith - 2024 - Spikes, Robots.pdf": [
            "/out/smith2024spikes_summary.md",
            "/out/smith2024spikes_summary_v2.md",
        ]
    }


def test_jsonl_index_takes_precedence_over_legacy(tmp_path):
    (tmp_path / "processed.txt").write_text("/old.pdf\n", encoding="utf-8")
    save_processed_index(tmp_path, {"/new.pdf": []})
    assert load_processed_index(tmp_path) == {"/new.pdf": []}


def test_malformed_jsonl_line_is_ignored(tmp_path):
    (tmp_path / "processed.jsonl").write_text(
        '{"pdf_path": "/a.pdf", "outputs": []}\nnot json\n', encoding="utf-8"
    )
    assert load_processed_index(tmp_path) == {"/a.pdf": []}


def test_save_processed_index_leaves_no_temp_files(tmp_path):
    save_processed_index(tmp_path, {"/a.pdf": []})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["processed.jsonl"]


# ---------------------------------------------------------------------------
# should_skip
# ---------------------------------------------------------------------------


def test_should_skip_returns_true_when_in_processed(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    processed = {str(pdf.resolve()): ["/out/summary.md"]}
    assert should_skip(pdf, processed, force_summary=False) is True


def test_should_skip_returns_false_when_not_in_processed(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    assert should_skip(pdf, {}, force_summary=False) is False


def test_should_skip_force_summary_overrides(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    processed = {str(pdf.resolve()): ["/out/summary.md"]}
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
        str(pdf.resolve()) + ", /out/summary.md\n", encoding="utf-8"
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
        str(pdf.resolve()) + ", /out/summary.md\n", encoding="utf-8"
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
    """After a successful run, the PDF path and summary path are recorded in processed.txt."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")

    summary = _make_summary("smith2020foo")
    with (
        patch("summarizer.batch.process_pdf", return_value=summary),
        patch("summarizer.batch.render_summary", return_value="# md"),
    ):
        run_batch(tmp_path, config)

    processed = load_processed_index(config.output_dir)
    abs_path = str(pdf.resolve())
    assert abs_path in processed
    assert len(processed[abs_path]) == 1
    assert processed[abs_path][0].endswith("smith2020foo_summary.md")


def test_run_batch_force_summary_appends_new_summary_path(tmp_path, config):
    """Re-processing with force_summary appends a new summary path to the existing entry."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")

    abs_path = str(pdf.resolve())
    existing_summary = str(config.output_dir / "primary" / "smith2020foo_summary.md")
    config.output_dir.mkdir(parents=True)
    (config.output_dir / "processed.txt").write_text(
        f"{abs_path}, {existing_summary}\n", encoding="utf-8"
    )
    # pre-create the existing summary so versioning kicks in
    (config.output_dir / "primary").mkdir(parents=True, exist_ok=True)
    (config.output_dir / "primary" / "smith2020foo_summary.md").write_text(
        "# old", encoding="utf-8"
    )
    config.force_summary = True

    summary = _make_summary("smith2020foo", paper_type="primary")
    with (
        patch("summarizer.batch.process_pdf", return_value=summary),
        patch("summarizer.batch.render_summary", return_value="# new"),
    ):
        run_batch(tmp_path, config)

    processed = load_processed_index(config.output_dir)
    assert abs_path in processed
    assert len(processed[abs_path]) == 2
    assert processed[abs_path][0] == existing_summary
    assert processed[abs_path][1].endswith("smith2020foo_summary_v2.md")


def test_run_batch_failed_paper_not_in_processed(tmp_path, config):
    """Failed papers are NOT added to processed.txt."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")

    err = PipelineError(pdf, Exception("boom"))
    with patch("summarizer.batch.process_pdf", side_effect=err):
        run_batch(tmp_path, config)

    processed = load_processed_index(config.output_dir)
    assert str(pdf.resolve()) not in processed


# ---------------------------------------------------------------------------
# Phase 5: shared client, accumulator, BatchReport.total_cost
# ---------------------------------------------------------------------------


def test_run_batch_creates_client_once(tmp_path, config):
    """run_batch creates the LLM client exactly once before the thread pool."""
    (tmp_path / "paper_a.pdf").write_bytes(b"%PDF")
    (tmp_path / "paper_b.pdf").write_bytes(b"%PDF")

    summary_a = _make_summary("smith2020foo")
    summary_b = _make_summary("jones2021bar")
    with (
        patch("summarizer.batch.create_client") as mock_create,
        patch("summarizer.batch.process_pdf", side_effect=[summary_a, summary_b]),
        patch("summarizer.batch.render_summary", return_value="# md"),
    ):
        run_batch(tmp_path, config)

    mock_create.assert_called_once_with(config)


def test_run_batch_report_has_total_cost(tmp_path, config):
    """BatchReport returned by run_batch has a total_cost field."""
    (tmp_path / "paper.pdf").write_bytes(b"%PDF")

    summary = _make_summary("smith2020foo")
    with (
        patch("summarizer.batch.create_client"),
        patch("summarizer.batch.process_pdf", return_value=summary),
        patch("summarizer.batch.render_summary", return_value="# md"),
    ):
        report = run_batch(tmp_path, config)

    assert hasattr(report, "total_cost")
    assert isinstance(report.total_cost, float)


def test_run_batch_loads_references_once_and_passes_them(tmp_path, config):
    (tmp_path / "paper_a.pdf").write_bytes(b"%PDF")
    (tmp_path / "paper_b.pdf").write_bytes(b"%PDF")

    with (
        patch("summarizer.batch.create_client"),
        patch("summarizer.batch.load_references", return_value="REFS") as mock_refs,
        patch(
            "summarizer.batch.process_pdf",
            side_effect=[_make_summary("a2020x"), _make_summary("b2020y")],
        ) as mock_process,
        patch("summarizer.batch.render_summary", return_value="# md"),
    ):
        run_batch(tmp_path, config)

    mock_refs.assert_called_once()
    assert all(c.kwargs["references"] == "REFS" for c in mock_process.call_args_list)


def test_run_batch_nothing_to_do_creates_no_client(tmp_path, config):
    with patch("summarizer.batch.create_client") as mock_create:
        report = run_batch(tmp_path, config)
    mock_create.assert_not_called()
    assert report.processed == 0


def test_keyboard_interrupt_cancels_queued_papers(tmp_path, config):
    """Regression: Ctrl-C used to wait for (and pay for) every queued paper."""
    import threading

    for i in range(6):
        (tmp_path / f"p{i}.pdf").write_bytes(b"%PDF")
    config.workers = 1
    release = threading.Event()
    calls = []

    def fake_process(pdf_path, *args, **kwargs):
        calls.append(pdf_path.name)
        if len(calls) == 1:
            raise KeyboardInterrupt
        release.wait(5)
        return _make_summary(pdf_path.stem)

    with (
        patch("summarizer.batch.create_client"),
        patch("summarizer.batch.process_pdf", side_effect=fake_process),
        patch("summarizer.batch.render_summary", return_value="# md"),
        pytest.raises(KeyboardInterrupt),
    ):
        run_batch(tmp_path, config)
    release.set()
    assert len(calls) <= 2  # the interrupted paper plus at most one already in flight


def test_index_line_without_pdf_path_is_ignored(tmp_path):
    (tmp_path / "processed.jsonl").write_text(
        '{"outputs": []}\n[1, 2]\n{"pdf_path": "/a.pdf", "outputs": null}\n', encoding="utf-8"
    )
    assert load_processed_index(tmp_path) == {"/a.pdf": []}


def test_failed_index_write_keeps_previous_index(tmp_path):
    save_processed_index(tmp_path, {"/a.pdf": []})
    with (
        patch("summarizer.batch.os.replace", side_effect=OSError("disk full")),
        pytest.raises(OSError),
    ):
        save_processed_index(tmp_path, {"/b.pdf": []})
    assert load_processed_index(tmp_path) == {"/a.pdf": []}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["processed.jsonl"]


def test_legacy_index_is_migrated_on_save(tmp_path, config):
    pdf = tmp_path / "Doe, J - 2024 - Spikes, Robots.pdf"
    pdf.write_bytes(b"%PDF")
    config.output_dir.mkdir(parents=True)
    legacy = config.output_dir / "processed.txt"
    legacy.write_text(f"{pdf.resolve()}, /out/doe2024spikes_summary.md\n", encoding="utf-8")
    (tmp_path / "new.pdf").write_bytes(b"%PDF")

    with (
        patch("summarizer.batch.create_client"),
        patch("summarizer.batch.process_pdf", return_value=_make_summary("new2024x")) as proc,
        patch("summarizer.batch.render_summary", return_value="# md"),
    ):
        run_batch(tmp_path, config)

    assert proc.call_count == 1  # the legacy entry (comma path) was skipped
    migrated = load_processed_index(config.output_dir)
    assert migrated[str(pdf.resolve())] == ["/out/doe2024spikes_summary.md"]
    assert (config.output_dir / "processed.jsonl").exists()
    assert legacy.exists()


@pytest.mark.parametrize(
    "line,expected",
    [
        ("/no/pdf/here", ("/no/pdf/here", [])),
        (
            "/a.pdf, /out/a_summary.md, /out/trailing",
            ("/a.pdf", ["/out/a_summary.md", "/out/trailing"]),
        ),
    ],
)
def test_parse_legacy_line_edge_cases(line, expected):
    from summarizer.batch import _parse_legacy_line

    assert _parse_legacy_line(line) == expected


def test_run_batch_report_includes_token_totals(tmp_path, config):
    (tmp_path / "paper.pdf").write_bytes(b"%PDF")

    def fake_process(pdf_path, config, client, accumulator, references):
        from summarizer.llm import UsageStats

        accumulator.add(UsageStats(input_tokens=1000, output_tokens=200), 0.0)
        return _make_summary("a2020x")

    with (
        patch("summarizer.batch.create_client"),
        patch("summarizer.batch.process_pdf", side_effect=fake_process),
        patch("summarizer.batch.render_summary", return_value="# md"),
    ):
        report = run_batch(tmp_path, config)
    assert (report.input_tokens, report.output_tokens) == (1000, 200)
