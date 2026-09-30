"""Tests for summarizer/batch.py — directory scanning, processed index, and batch execution."""

import itertools
import json
import sys
import threading
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

_PDF_COUNTER = itertools.count()


def _pdf_bytes() -> bytes:
    """Distinct content per test PDF (identical PDFs are de-duplicated by sha256)."""
    return f"%PDF {next(_PDF_COUNTER)}".encode()


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
    (tmp_path / "paper_a.pdf").write_bytes(_pdf_bytes())
    (tmp_path / "paper_b.pdf").write_bytes(_pdf_bytes())
    (tmp_path / "paper_c.pdf").write_bytes(_pdf_bytes())
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
    (tmp_path / "root.pdf").write_bytes(_pdf_bytes())
    (sub / "nested.pdf").write_bytes(_pdf_bytes())
    pdfs = find_pdfs(tmp_path)
    assert len(pdfs) == 2


def test_find_pdfs_empty_dir(tmp_path):
    assert find_pdfs(tmp_path) == []


def test_find_pdfs_matches_uppercase_extension_and_skips_appledouble(tmp_path):
    (tmp_path / "upper.PDF").write_bytes(_pdf_bytes())
    (tmp_path / "._upper.pdf").write_bytes(b"junk")
    (tmp_path / "notes.md").write_text("x")
    assert [p.name for p in find_pdfs(tmp_path)] == ["upper.PDF"]


# ---------------------------------------------------------------------------
# load_processed_index (legacy processed.txt is still read)
# ---------------------------------------------------------------------------


def _outputs(index: dict) -> dict[str, list[str]]:
    """Index as ``pdf_path -> outputs`` for compact assertions."""
    return {record["pdf_path"]: record["outputs"] for record in index.values()}


def test_load_processed_index_returns_empty_dict_when_file_missing(tmp_path):
    assert load_processed_index(tmp_path) == {}


def test_legacy_index_paths_and_summaries_are_read(tmp_path):
    (tmp_path / "processed.txt").write_text(
        "/path/a.pdf, /out/a_summary.md, /out/a_summary_v2.md\n\n/path/b.pdf\n",
        encoding="utf-8",
    )
    assert _outputs(load_processed_index(tmp_path)) == {
        "/path/a.pdf": ["/out/a_summary.md", "/out/a_summary_v2.md"],
        "/path/b.pdf": [],
    }


def test_index_roundtrip_keeps_sha_and_path_keys(tmp_path):
    index = {
        "abc": {"pdf_path": "/a.pdf", "outputs": ["/out/a_summary.md"], "sha256": "abc"},
        "/legacy.pdf": {"pdf_path": "/legacy.pdf", "outputs": [], "sha256": None},
    }
    save_processed_index(tmp_path, index)
    assert load_processed_index(tmp_path) == index
    lines = (tmp_path / "processed.jsonl").read_text().splitlines()
    assert json.loads(lines[1]) == {"pdf_path": "/legacy.pdf", "outputs": []}  # no null sha


def test_roundtrip_preserves_paths_with_commas(tmp_path):
    """Regression: the old comma-separated format truncated such paths."""
    path = str(tmp_path / "Smith et al. - 2024 - Spikes, Robots, and Control.pdf")
    index = {"s": {"pdf_path": path, "outputs": [str(tmp_path / "a, b_summary.md")], "sha256": "s"}}
    save_processed_index(tmp_path, index)
    assert load_processed_index(tmp_path) == index


def test_legacy_index_with_commas_in_paths_is_parsed(tmp_path):
    (tmp_path / "processed.txt").write_text(
        "/lib/Smith - 2024 - Spikes, Robots.pdf, /out/smith2024spikes_summary.md, "
        "/out/smith2024spikes_summary_v2.md\n",
        encoding="utf-8",
    )
    assert _outputs(load_processed_index(tmp_path)) == {
        "/lib/Smith - 2024 - Spikes, Robots.pdf": [
            "/out/smith2024spikes_summary.md",
            "/out/smith2024spikes_summary_v2.md",
        ]
    }


def test_jsonl_index_takes_precedence_over_legacy(tmp_path):
    (tmp_path / "processed.txt").write_text("/old.pdf\n", encoding="utf-8")
    save_processed_index(tmp_path, {"n": {"pdf_path": "/new.pdf", "outputs": [], "sha256": "n"}})
    assert _outputs(load_processed_index(tmp_path)) == {"/new.pdf": []}


def test_malformed_jsonl_line_is_ignored(tmp_path):
    (tmp_path / "processed.jsonl").write_text(
        '{"pdf_path": "/a.pdf", "outputs": []}\nnot json\n', encoding="utf-8"
    )
    assert _outputs(load_processed_index(tmp_path)) == {"/a.pdf": []}


def test_save_processed_index_leaves_no_temp_files(tmp_path):
    save_processed_index(tmp_path, {"a": {"pdf_path": "/a.pdf", "outputs": [], "sha256": "a"}})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["processed.jsonl"]


# ---------------------------------------------------------------------------
# should_skip
# ---------------------------------------------------------------------------


def test_should_skip_by_sha():
    index = {"abc": {"pdf_path": "/elsewhere.pdf", "outputs": [], "sha256": "abc"}}
    assert should_skip("abc", index, force_summary=False) is True
    assert should_skip("other", index, force_summary=False) is False
    assert should_skip("abc", index, force_summary=True) is False


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
    s.model_dump_json.return_value = json.dumps({"citation_key": citation_key})
    return s


def test_run_batch_processes_all_pdfs(tmp_path, config):
    """run_batch calls process_pdf for each PDF and writes output files."""
    (tmp_path / "paper_a.pdf").write_bytes(_pdf_bytes())
    (tmp_path / "paper_b.pdf").write_bytes(_pdf_bytes())

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
    pdf.write_bytes(_pdf_bytes())

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
    pdf.write_bytes(_pdf_bytes())

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
    pdf.write_bytes(_pdf_bytes())

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
    (tmp_path / "bad_paper.pdf").write_bytes(_pdf_bytes())

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
    (tmp_path / "paper_a.pdf").write_bytes(_pdf_bytes())
    config.dry_run = True

    with patch("summarizer.batch.process_pdf") as mock_process:
        report = run_batch(tmp_path, config)

    mock_process.assert_not_called()
    assert report.skipped == 1
    assert report.processed == 0


def test_run_batch_logs_selection_summary(tmp_path, config, caplog):
    """Batch logs include discovered/selected/skipped counts before processing."""
    (tmp_path / "paper_a.pdf").write_bytes(_pdf_bytes())
    (tmp_path / "paper_b.pdf").write_bytes(_pdf_bytes())

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
    (tmp_path / "paper.pdf").write_bytes(_pdf_bytes())

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
    pdf.write_bytes(_pdf_bytes())

    summary = _make_summary("smith2020foo")
    with (
        patch("summarizer.batch.process_pdf", return_value=summary),
        patch("summarizer.batch.render_summary", return_value="# md"),
    ):
        run_batch(tmp_path, config)

    from summarizer.parser import sha256_file

    processed = load_processed_index(config.output_dir)
    record = processed[sha256_file(pdf)]  # keyed by content
    assert record["pdf_path"] == str(pdf.resolve())
    assert len(record["outputs"]) == 1 and record["outputs"][0].endswith("smith2020foo_summary.md")
    sidecar = Path(record["outputs"][0]).with_suffix(".json")
    assert json.loads(sidecar.read_text()) == {"citation_key": "smith2020foo"}


def test_run_batch_force_summary_appends_new_summary_path(tmp_path, config):
    """Re-processing with force_summary appends a new summary path to the existing entry."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(_pdf_bytes())

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

    outputs = _outputs(load_processed_index(config.output_dir))[abs_path]
    assert len(outputs) == 2
    assert outputs[0] == existing_summary
    assert outputs[1].endswith("smith2020foo_summary_v2.md")


def test_run_batch_failed_paper_not_in_processed(tmp_path, config):
    """Failed papers are NOT added to processed.txt."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(_pdf_bytes())

    err = PipelineError(pdf, Exception("boom"))
    with patch("summarizer.batch.process_pdf", side_effect=err):
        run_batch(tmp_path, config)

    assert load_processed_index(config.output_dir) == {}


# ---------------------------------------------------------------------------
# Phase 5: shared client, accumulator, BatchReport.total_cost
# ---------------------------------------------------------------------------


def test_run_batch_creates_client_once(tmp_path, config):
    """run_batch creates the LLM client exactly once before the thread pool."""
    (tmp_path / "paper_a.pdf").write_bytes(_pdf_bytes())
    (tmp_path / "paper_b.pdf").write_bytes(_pdf_bytes())

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
    (tmp_path / "paper.pdf").write_bytes(_pdf_bytes())

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
    (tmp_path / "paper_a.pdf").write_bytes(_pdf_bytes())
    (tmp_path / "paper_b.pdf").write_bytes(_pdf_bytes())

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

    for i in range(6):
        (tmp_path / f"p{i}.pdf").write_bytes(_pdf_bytes())
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
    for thread in threading.enumerate():  # let the in-flight worker finish inside this test
        if thread.name.startswith("worker"):
            thread.join(5)
    assert len(calls) <= 2  # the interrupted paper plus at most one already in flight


def test_index_line_without_pdf_path_is_ignored(tmp_path):
    (tmp_path / "processed.jsonl").write_text(
        '{"outputs": []}\n[1, 2]\n{"pdf_path": "/a.pdf", "outputs": null}\n', encoding="utf-8"
    )
    assert _outputs(load_processed_index(tmp_path)) == {"/a.pdf": []}


def test_failed_index_write_keeps_previous_index(tmp_path):
    save_processed_index(tmp_path, {"a": {"pdf_path": "/a.pdf", "outputs": [], "sha256": "a"}})
    with (
        patch("summarizer.batch.os.replace", side_effect=OSError("disk full")),
        pytest.raises(OSError),
    ):
        save_processed_index(tmp_path, {"b": {"pdf_path": "/b.pdf", "outputs": [], "sha256": "b"}})
    assert _outputs(load_processed_index(tmp_path)) == {"/a.pdf": []}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["processed.jsonl"]


def test_legacy_index_is_migrated_on_save(tmp_path, config):
    pdf = tmp_path / "Doe, J - 2024 - Spikes, Robots.pdf"
    pdf.write_bytes(_pdf_bytes())
    config.output_dir.mkdir(parents=True)
    legacy = config.output_dir / "processed.txt"
    legacy.write_text(f"{pdf.resolve()}, /out/doe2024spikes_summary.md\n", encoding="utf-8")
    (tmp_path / "new.pdf").write_bytes(_pdf_bytes())

    with (
        patch("summarizer.batch.create_client"),
        patch("summarizer.batch.process_pdf", return_value=_make_summary("new2024x")) as proc,
        patch("summarizer.batch.render_summary", return_value="# md"),
    ):
        run_batch(tmp_path, config)

    from summarizer.parser import sha256_file

    assert proc.call_count == 1  # the legacy entry (comma path) was skipped
    migrated = load_processed_index(config.output_dir)
    assert migrated[sha256_file(pdf)]["outputs"] == ["/out/doe2024spikes_summary.md"]
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
    (tmp_path / "paper.pdf").write_bytes(_pdf_bytes())

    def fake_process(pdf_path, config, client, accumulator, references, zotero=None):
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


def _quota_error(pdf_path):
    from summarizer.llm import QuotaExhausted

    return PipelineError(pdf_path, QuotaExhausted("free-models-per-day"))


def test_quota_exhaustion_stops_the_batch_without_failing_papers(tmp_path, config):
    for i in range(4):
        (tmp_path / f"p{i}.pdf").write_bytes(_pdf_bytes())
    config.workers = 1
    calls = []

    def fake_process(pdf_path, *args, **kwargs):
        calls.append(pdf_path.name)
        if len(calls) == 1:
            return _make_summary("first2020x")
        raise _quota_error(pdf_path)

    with (
        patch("summarizer.batch.create_client"),
        patch("summarizer.batch.process_pdf", side_effect=fake_process),
        patch("summarizer.batch.render_summary", return_value="# md"),
    ):
        report = run_batch(tmp_path, config)

    assert report.stopped_reason == "free-models-per-day"
    assert (report.processed, report.failed) == (1, 0)
    assert report.skipped == 3
    assert len(calls) == 2  # the rest never started
    assert len(load_processed_index(config.output_dir)) == 1


def test_max_cost_stops_starting_new_papers(tmp_path, config):
    from summarizer.llm import UsageStats

    for i in range(4):
        (tmp_path / f"p{i}.pdf").write_bytes(_pdf_bytes())
    config.workers = 1
    config.max_cost = 0.01

    def fake_process(pdf_path, config, client, accumulator, references, zotero=None):
        accumulator.add(UsageStats(), 0.005)
        return _make_summary(pdf_path.stem)

    with (
        patch("summarizer.batch.create_client"),
        patch("summarizer.batch.process_pdf", side_effect=fake_process),
        patch("summarizer.batch.render_summary", return_value="# md"),
    ):
        report = run_batch(tmp_path, config)

    assert report.stopped_reason == "--max-cost $0.01 reached"
    assert (report.processed, report.skipped) == (2, 2)  # 0.005 + 0.005 reaches 0.01


def test_stop_signal_keeps_first_reason_and_quota_race_is_safe(tmp_path, config):
    """Regression: a _Stopped seen before the quota error crashed on max_cost=None."""
    from summarizer.batch import StopSignal

    stop = StopSignal()
    stop.trip("daily cap")
    stop.trip("--max-cost $1 reached")
    assert stop.reason == "daily cap"

    for i in range(3):
        (tmp_path / f"p{i}.pdf").write_bytes(_pdf_bytes())
    config.workers = 2
    with (
        patch("summarizer.batch.create_client"),
        patch(
            "summarizer.batch.process_pdf",
            side_effect=lambda p, *a, **k: (_ for _ in ()).throw(_quota_error(p)),
        ),
    ):
        report = run_batch(tmp_path, config)
    assert report.stopped_reason == "free-models-per-day"
    assert (report.processed, report.failed, report.skipped) == (0, 0, 3)


def test_moved_pdf_is_recognized_by_content(tmp_path, config):
    """A PDF moved after being summarized is not summarized again."""
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    (old / "paper.pdf").write_bytes(_pdf_bytes())
    with (
        patch("summarizer.batch.create_client"),
        patch("summarizer.batch.process_pdf", return_value=_make_summary("a2020x")) as proc,
        patch("summarizer.batch.render_summary", return_value="# md"),
    ):
        run_batch(old, config)
        old.rename(new)
        report = run_batch(new, config)
    assert proc.call_count == 1 and report.skipped == 1


def test_duplicate_pdfs_in_one_batch_are_processed_once(tmp_path, config):
    content = _pdf_bytes()
    (tmp_path / "a.pdf").write_bytes(content)
    (tmp_path / "a_copy.pdf").write_bytes(content)
    with (
        patch("summarizer.batch.create_client"),
        patch("summarizer.batch.process_pdf", return_value=_make_summary("a2020x")) as proc,
        patch("summarizer.batch.render_summary", return_value="# md"),
    ):
        report = run_batch(tmp_path, config)
    assert (proc.call_count, report.processed, report.skipped) == (1, 1, 1)


@pytest.mark.skipif(sys.platform == "win32", reason="no advisory locks on Windows")
def test_output_dir_lock_blocks_a_second_run(tmp_path, config):
    from summarizer.batch import OutputDirLocked, output_dir_lock, render_all

    (tmp_path / "paper.pdf").write_bytes(_pdf_bytes())
    with (
        patch("summarizer.batch.create_client"),
        patch("summarizer.batch.process_pdf") as process,
        output_dir_lock(config.output_dir),
    ):
        with pytest.raises(OutputDirLocked):
            run_batch(tmp_path, config)
        with pytest.raises(OutputDirLocked):
            render_all(config.output_dir)
        config.dry_run = True
        run_batch(tmp_path, config)  # dry runs don't need the lock
    process.assert_not_called()


def test_render_all_rebuilds_markdown_from_sidecars(tmp_path, mock_part1_dict, mock_part2_dict):
    from summarizer.batch import render_all
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
    (out / "huebotter2025spiking_summary.json").write_text(summary.model_dump_json())
    (out / "huebotter2025spiking_summary.md").write_text("# stale")
    assert render_all(tmp_path / "out") == (1, 0)
    assert (out / "huebotter2025spiking_summary.md").read_text().startswith("# Spiking Neural")


def test_changed_pdf_at_an_old_path_is_summarized_again(tmp_path, config):
    """An old path-only entry is not carried over to different content."""
    import os

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(_pdf_bytes())
    old_output = tmp_path / "old_summary.md"
    old_output.write_text("# old")
    os.utime(old_output, (1_000_000, 1_000_000))  # summarized long before the file changed
    config.output_dir.mkdir(parents=True)
    (config.output_dir / "processed.txt").write_text(f"{pdf.resolve()}, {old_output}\n")
    with (
        patch("summarizer.batch.create_client"),
        patch("summarizer.batch.process_pdf", return_value=_make_summary("new2024x")) as proc,
        patch("summarizer.batch.render_summary", return_value="# md"),
    ):
        run_batch(tmp_path, config)
    proc.assert_called_once()


def test_unreadable_pdf_fails_only_itself(tmp_path, config):
    from summarizer.parser import sha256_file as real_sha

    (tmp_path / "ok.pdf").write_bytes(_pdf_bytes())
    (tmp_path / "locked.pdf").write_bytes(_pdf_bytes())

    def sha(path):
        if path.name == "locked.pdf":
            raise PermissionError("denied")
        return real_sha(path)

    with (
        patch("summarizer.batch.sha256_file", side_effect=sha),
        patch("summarizer.batch.create_client"),
        patch("summarizer.batch.process_pdf", return_value=_make_summary("ok2024x")),
        patch("summarizer.batch.render_summary", return_value="# md"),
    ):
        report = run_batch(tmp_path, config)
    assert (report.processed, report.failed) == (1, 1)
    assert "denied" in report.failed_papers[0].error


def test_versioned_path_respects_an_orphan_sidecar(tmp_path):
    from summarizer.batch import get_versioned_output_path

    md = tmp_path / "x2020y_summary.md"
    md.with_suffix(".json").write_text("{}")
    assert get_versioned_output_path(md).name == "x2020y_summary_v2.md"


def test_render_all_skips_invalid_sidecars(tmp_path):
    from summarizer.batch import render_all

    (tmp_path / "bad_summary.json").write_text("{}")
    assert render_all(tmp_path) == (0, 1)


def test_pdf_edited_in_place_is_summarized_again(tmp_path, config):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(_pdf_bytes())
    with (
        patch("summarizer.batch.create_client"),
        patch("summarizer.batch.process_pdf", return_value=_make_summary("a2020x")) as proc,
        patch("summarizer.batch.render_summary", return_value="# md"),
    ):
        run_batch(tmp_path, config)
        pdf.write_bytes(_pdf_bytes())  # new content, same path
        run_batch(tmp_path, config)
    assert proc.call_count == 2
    assert len(load_processed_index(config.output_dir)) == 2


def test_dry_run_writes_nothing(tmp_path, config):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(_pdf_bytes())
    config.output_dir.mkdir(parents=True)
    (config.output_dir / "processed.txt").write_text(f"{pdf.resolve()}\n")  # would migrate
    config.dry_run = True
    run_batch(tmp_path, config)
    assert sorted(p.name for p in config.output_dir.iterdir()) == ["processed.txt"]


def test_forced_rerun_after_a_move_updates_the_path(tmp_path, config):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    (old / "paper.pdf").write_bytes(_pdf_bytes())
    with (
        patch("summarizer.batch.create_client"),
        patch("summarizer.batch.process_pdf", return_value=_make_summary("a2020x")),
        patch("summarizer.batch.render_summary", return_value="# md"),
    ):
        run_batch(old, config)
        old.rename(new)
        config.force_summary = True
        run_batch(new, config)
    (record,) = load_processed_index(config.output_dir).values()
    assert record["pdf_path"] == str((new / "paper.pdf").resolve())
    assert len(record["outputs"]) == 2


def test_sidecar_is_written_before_the_markdown(tmp_path, config):
    from summarizer.batch import atomic_write_text as real_write

    (tmp_path / "paper.pdf").write_bytes(_pdf_bytes())

    def write(path, text):
        if path.suffix == ".md":
            raise OSError("disk full")
        real_write(path, text)

    with (
        patch("summarizer.batch.create_client"),
        patch("summarizer.batch.process_pdf", return_value=_make_summary("a2020x")),
        patch("summarizer.batch.render_summary", return_value="# md"),
        patch("summarizer.batch.atomic_write_text", side_effect=write),
    ):
        report = run_batch(tmp_path, config)
    assert report.failed == 1
    assert (config.output_dir / "primary" / "a2020x_summary.json").exists()
    assert load_processed_index(config.output_dir) == {}  # not indexed, so it will be retried


def test_render_all_includes_versioned_sidecars(tmp_path, mock_part1_dict, mock_part2_dict):
    from summarizer.batch import render_all
    from summarizer.models import PaperSummary

    meta_keys = {"citation_key", "title", "authors", "year", "venue", "paper_type", "tags"}
    metadata = {k: mock_part1_dict[k] for k in meta_keys} | {
        "is_research_paper": True,
        "rejection_reason": None,
    }
    part1 = {k: v for k, v in mock_part1_dict.items() if k not in meta_keys - {"paper_type"}}
    summary = PaperSummary(metadata=metadata, part1=part1, part2=mock_part2_dict)
    for name in ("x2020y_summary.json", "x2020y_summary_v2.json"):
        (tmp_path / name).write_text(summary.model_dump_json())
    assert render_all(tmp_path) == (2, 0)
    assert (tmp_path / "x2020y_summary_v2.md").exists()


# ---------------------------------------------------------------------------
# Zotero lookup
# ---------------------------------------------------------------------------


def test_batch_passes_each_pdf_its_zotero_record(tmp_path, config):
    pdf = tmp_path / "AAAAAAAA__paper.pdf"
    pdf.write_bytes(_pdf_bytes())
    record = object()
    seen = {}

    def fake_process(pdf_path, config, client, accumulator, references, zotero=None):
        seen[pdf_path.name] = zotero
        return _make_summary("a2020x")

    with (
        patch("summarizer.batch.create_client"),
        patch("summarizer.batch.lookup_all", return_value={pdf: record}) as lookup,
        patch("summarizer.batch.process_pdf", side_effect=fake_process),
        patch("summarizer.batch.render_summary", return_value="# md"),
    ):
        run_batch(tmp_path, config)
    lookup.assert_called_once_with([pdf])
    assert seen == {"AAAAAAAA__paper.pdf": record}


def test_no_zotero_and_dry_run_skip_the_lookup(tmp_path, config):
    (tmp_path / "AAAAAAAA__paper.pdf").write_bytes(_pdf_bytes())
    with (
        patch("summarizer.batch.create_client"),
        patch("summarizer.batch.lookup_all") as lookup,
        patch("summarizer.batch.process_pdf", return_value=_make_summary("a2020x")),
        patch("summarizer.batch.render_summary", return_value="# md"),
    ):
        config.zotero = False
        run_batch(tmp_path, config)
        config.zotero, config.dry_run = True, True
        run_batch(tmp_path, config)
    lookup.assert_not_called()
