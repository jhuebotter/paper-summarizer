"""Tests for summarizer/parser.py — docling PDF-to-markdown wrapper with cache."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import summarizer.parser as parser_mod
from summarizer.models import ParseError
from summarizer.parser import load_text, truncate_text


def parse_pdf(pdf_path, max_chars=200_000, reparse=False, extractor="auto", strip_references=False):
    """``load_text`` + ``truncate_text``, as the pipeline combines them."""
    parsed = load_text(pdf_path, extractor, reparse=reparse, strip_references=strip_references)
    return truncate_text(parsed.text, max_chars, pdf_path.name)


PROJECT_ROOT = Path(__file__).parent.parent


def _cache(tmp_path: Path, pdf: Path, extractor: str) -> Path:
    """Where the extraction cache for ``pdf`` lives (conftest sets XDG_CACHE_HOME)."""
    from summarizer.parser import sha256_file

    return tmp_path / "xdg-cache" / "paper-summarizer" / f"{sha256_file(pdf)}.{extractor}.md"


@pytest.fixture(autouse=True)
def _fresh_converter_cache():
    """The converter cache is process-wide; don't let one test reuse another's."""
    parser_mod._CONVERTERS.clear()
    yield
    parser_mod._CONVERTERS.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mocked_parse(
    tmp_path: Path,
    text: str,
    max_chars: int,
    reparse: bool = False,
    extractor: str = "auto",
) -> str:
    """Call parse_pdf with a mocked DocumentConverter."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake content")

    mock_result = MagicMock()
    mock_result.document.export_to_markdown.return_value = text

    with patch("summarizer.parser.DocumentConverter") as MockConverter:
        MockConverter.return_value.convert.return_value = mock_result
        return parse_pdf(
            pdf,
            max_chars=max_chars,
            reparse=reparse,
            extractor=extractor,
        )


# ---------------------------------------------------------------------------
# Truncation (unit — docling mocked)
# ---------------------------------------------------------------------------


def test_parse_pdf_returns_full_text_when_short(tmp_path):
    result = _mocked_parse(tmp_path, text="hello world", max_chars=100)
    assert result == "hello world"


def test_parse_pdf_truncates_long_text(tmp_path):
    result = _mocked_parse(tmp_path, text="x" * 1000, max_chars=100)
    assert len(result) == 100
    assert result == "x" * 100


def test_parse_pdf_exact_boundary_not_truncated(tmp_path):
    result = _mocked_parse(tmp_path, text="a" * 50, max_chars=50)
    assert len(result) == 50


def test_parse_pdf_raises_parse_error_on_docling_failure(tmp_path):
    pdf = tmp_path / "bad.pdf"
    pdf.write_bytes(b"not a real pdf")

    with patch("summarizer.parser.DocumentConverter") as MockConverter:
        MockConverter.return_value.convert.side_effect = Exception("docling internal error")
        with pytest.raises(ParseError, match="bad.pdf"):
            parse_pdf(pdf, max_chars=40_000)


def test_parse_pdf_parse_error_wraps_original_exception(tmp_path):
    pdf = tmp_path / "bad.pdf"
    pdf.write_bytes(b"not a real pdf")
    original = RuntimeError("deep failure")

    with patch("summarizer.parser.DocumentConverter") as MockConverter:
        MockConverter.return_value.convert.side_effect = original
        with pytest.raises(ParseError) as exc_info:
            parse_pdf(pdf, max_chars=40_000)
    assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# Cache logic
# ---------------------------------------------------------------------------


def test_parse_pdf_writes_cache_on_fresh_parse(tmp_path):
    """After a fresh docling parse, {pdf_stem}.docling.md is created next to the PDF."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake content")
    cache = _cache(tmp_path, pdf, "docling")

    mock_result = MagicMock()
    mock_result.document.export_to_markdown.return_value = "parsed content"

    with patch("summarizer.parser.DocumentConverter") as MockConverter:
        MockConverter.return_value.convert.return_value = mock_result
        parse_pdf(pdf, max_chars=10_000)

    assert cache.exists()
    assert cache.read_text(encoding="utf-8") == "parsed content"


def test_parse_pdf_reads_cache_when_present(tmp_path):
    """When an old {pdf_stem}.docling.md exists and is non-empty, docling is NOT called."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake content")
    cache = tmp_path / "paper.docling.md"
    cache.write_text("cached content", encoding="utf-8")

    with patch("summarizer.parser.DocumentConverter") as MockConverter:
        result = parse_pdf(pdf, max_chars=10_000)
        MockConverter.assert_not_called()

    assert result == "cached content"


def test_parse_pdf_truncates_cached_content(tmp_path):
    """Cached content is still subject to max_chars truncation."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake content")
    cache = tmp_path / "paper.docling.md"
    cache.write_text("x" * 1000, encoding="utf-8")

    with patch("summarizer.parser.DocumentConverter") as MockConverter:
        result = parse_pdf(pdf, max_chars=100)
        MockConverter.assert_not_called()

    assert len(result) == 100


def test_parse_pdf_zero_byte_cache_treated_as_miss(tmp_path):
    """An empty (zero-byte) cache file causes a fresh docling run."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake content")
    cache = tmp_path / "paper.docling.md"
    cache.write_text("", encoding="utf-8")  # zero bytes

    mock_result = MagicMock()
    mock_result.document.export_to_markdown.return_value = "fresh content"

    with patch("summarizer.parser.DocumentConverter") as MockConverter:
        MockConverter.return_value.convert.return_value = mock_result
        result = parse_pdf(pdf, max_chars=10_000)
        MockConverter.assert_called_once()

    assert result == "fresh content"


def test_parse_pdf_reparse_ignores_cache(tmp_path):
    """When reparse=True, docling runs even if a cache file exists."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake content")
    cache = tmp_path / "paper.docling.md"
    cache.write_text("stale cached content", encoding="utf-8")

    mock_result = MagicMock()
    mock_result.document.export_to_markdown.return_value = "fresh content"

    with patch("summarizer.parser.DocumentConverter") as MockConverter:
        MockConverter.return_value.convert.return_value = mock_result
        result = parse_pdf(pdf, max_chars=10_000, reparse=True)
        MockConverter.assert_called_once()

    assert result == "fresh content"


def test_parse_pdf_pypdf_extractor_skips_docling(tmp_path):
    """extractor='pypdf' bypasses docling and uses pypdf directly."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake content")

    with (
        patch("summarizer.parser.DocumentConverter") as MockConverter,
        patch("summarizer.parser._extract_text_with_pypdf", return_value="pypdf text"),
    ):
        result = parse_pdf(pdf, max_chars=10_000, extractor="pypdf")

    MockConverter.assert_not_called()
    assert result == "pypdf text"


def test_parse_pdf_docling_extractor_no_fallback(tmp_path):
    """extractor='docling' should not fall back to pypdf on failures."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake content")

    with (
        patch("summarizer.parser.DocumentConverter") as MockConverter,
        patch("summarizer.parser._extract_text_with_pypdf") as mock_pypdf,
    ):
        MockConverter.return_value.convert.side_effect = Exception("docling failed")
        with pytest.raises(ParseError):
            parse_pdf(pdf, max_chars=10_000, extractor="docling")

    mock_pypdf.assert_not_called()


def test_parse_pdf_failure_does_not_write_cache(tmp_path):
    """A parse failure must not create or overwrite the cache file."""
    pdf = tmp_path / "bad.pdf"
    pdf.write_bytes(b"not a real pdf")
    cache = tmp_path / "bad.md"

    with patch("summarizer.parser.DocumentConverter") as MockConverter:
        MockConverter.return_value.convert.side_effect = Exception("boom")
        with pytest.raises(ParseError):
            parse_pdf(pdf, max_chars=10_000)

    assert not cache.exists()


def test_parse_pdf_falls_back_to_pypdf_on_docling_failure(tmp_path):
    """If docling fails, parser falls back to pypdf extraction."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake content")

    with (
        patch("summarizer.parser.DocumentConverter") as MockConverter,
        patch("summarizer.parser._extract_text_with_pypdf", return_value="fallback text"),
    ):
        MockConverter.return_value.convert.side_effect = Exception("PdfHyperlink url_parsing")
        result = parse_pdf(pdf, max_chars=10_000)

    assert result == "fallback text"
    assert _cache(tmp_path, pdf, "pypdf").read_text(encoding="utf-8") == "fallback text"
    assert not _cache(tmp_path, pdf, "docling").exists()


def test_parse_pdf_raises_parse_error_if_docling_and_fallback_fail(tmp_path):
    """If both docling and fallback fail, ParseError is raised."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake content")

    with (
        patch("summarizer.parser.DocumentConverter") as MockConverter,
        patch(
            "summarizer.parser._extract_text_with_pypdf",
            side_effect=ParseError("fallback failed"),
        ),
    ):
        MockConverter.return_value.convert.side_effect = Exception("docling failed")
        with pytest.raises(ParseError, match="paper.pdf"):
            parse_pdf(pdf, max_chars=10_000)


# ---------------------------------------------------------------------------
# Integration — real docling parse (skippable in CI)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Logging (caplog)
# ---------------------------------------------------------------------------


def test_parse_pdf_logs_extraction_progress(tmp_path, caplog):
    """A fresh parse emits INFO messages for start and completion of extraction."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    mock_result = MagicMock()
    mock_result.document.export_to_markdown.return_value = "fresh content"

    with (
        caplog.at_level(logging.INFO, logger="summarizer.parser"),
        patch("summarizer.parser.DocumentConverter") as MockConverter,
    ):
        MockConverter.return_value.convert.return_value = mock_result
        parse_pdf(pdf, max_chars=10_000)

    messages = [r.message for r in caplog.records]
    assert any("Running auto extraction" in m for m in messages)
    assert any("Extraction complete" in m for m in messages)


@pytest.mark.integration
def test_parse_real_pdf_returns_nonempty_string(sample_pdf_path):
    result = parse_pdf(sample_pdf_path, max_chars=40_000)
    assert isinstance(result, str)
    assert len(result) > 100


@pytest.mark.integration
def test_parse_real_pdf_truncation_is_applied(sample_pdf_path):
    small_limit = 500
    result = parse_pdf(sample_pdf_path, max_chars=small_limit)
    assert len(result) <= small_limit


# ---------------------------------------------------------------------------
# Extractor-keyed cache, lazy docling, truncation warning
# ---------------------------------------------------------------------------


def _fake_pdf(tmp_path: Path) -> Path:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake content")
    return pdf


def test_pypdf_extractor_ignores_docling_cache(tmp_path):
    """Regression: switching --extractor must not reuse another backend's text."""
    pdf = _fake_pdf(tmp_path)
    (tmp_path / "paper.docling.md").write_text("docling text", encoding="utf-8")
    with patch("summarizer.parser._extract_text_with_pypdf", return_value="pypdf text"):
        assert parse_pdf(pdf, extractor="pypdf") == "pypdf text"
    assert _cache(tmp_path, pdf, "pypdf").read_text(encoding="utf-8") == "pypdf text"


def test_docling_extractor_ignores_legacy_and_pypdf_cache(tmp_path):
    (tmp_path / "paper.md").write_text("legacy", encoding="utf-8")
    (tmp_path / "paper.pypdf.md").write_text("pypdf", encoding="utf-8")
    assert _mocked_parse(tmp_path, "docling text", 10_000, extractor="docling") == "docling text"


def test_auto_prefers_docling_cache_over_pypdf_cache(tmp_path):
    pdf = _fake_pdf(tmp_path)
    (tmp_path / "paper.docling.md").write_text("docling", encoding="utf-8")
    (tmp_path / "paper.pypdf.md").write_text("pypdf", encoding="utf-8")
    with patch("summarizer.parser.DocumentConverter") as MockConverter:
        assert parse_pdf(pdf) == "docling"
        MockConverter.assert_not_called()


def test_unknown_extractor_raises(tmp_path):
    with pytest.raises(ValueError, match="extractor"):
        parse_pdf(_fake_pdf(tmp_path), extractor="ocr")


def test_converter_is_reused_across_pdfs(tmp_path):
    """Building a DocumentConverter loads models; it must happen once per process."""
    a = tmp_path / "a.pdf"
    b = tmp_path / "b.pdf"
    a.write_bytes(b"%PDF")
    b.write_bytes(b"%PDF")
    with patch("summarizer.parser.DocumentConverter") as MockConverter:
        MockConverter.return_value.convert.return_value.document.export_to_markdown.return_value = (
            "t"
        )
        parse_pdf(a, extractor="docling")
        parse_pdf(b, extractor="docling")
    MockConverter.assert_called_once()


def test_auto_falls_back_to_pypdf_when_docling_not_installed(tmp_path, caplog):
    import summarizer.parser as parser_mod

    pdf = _fake_pdf(tmp_path)
    with (
        patch.object(parser_mod, "DocumentConverter", None),
        patch.dict("sys.modules", {"docling": None, "docling.document_converter": None}),
        patch("summarizer.parser._extract_text_with_pypdf", return_value="pypdf text"),
        caplog.at_level(logging.WARNING, logger="summarizer.parser"),
    ):
        assert parse_pdf(pdf) == "pypdf text"
    assert any("docling is not installed" in r.message for r in caplog.records)


def test_docling_extractor_errors_clearly_when_not_installed(tmp_path):
    import summarizer.parser as parser_mod

    with (
        patch.object(parser_mod, "DocumentConverter", None),
        patch.dict("sys.modules", {"docling": None, "docling.document_converter": None}),
        pytest.raises(ParseError, match="not installed"),
    ):
        parse_pdf(_fake_pdf(tmp_path), extractor="docling")


def test_importing_cli_does_not_import_docling(tmp_path):
    """A fake ``docling`` that fails on import proves the import is lazy."""
    import os
    import subprocess
    import sys

    fake = tmp_path / "fake" / "docling"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text("raise RuntimeError('docling imported eagerly')\n")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(fake.parent), str(PROJECT_ROOT)])}
    result = subprocess.run(
        [sys.executable, "-c", "import summarizer.cli"], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr


def test_auto_falls_back_to_pypdf_when_docling_fails_to_start(tmp_path):
    """Regression: a converter constructor failure bypassed the pypdf fallback."""
    a = tmp_path / "a.pdf"
    b = tmp_path / "b.pdf"
    a.write_bytes(b"%PDF")
    b.write_bytes(b"%PDF")
    with (
        patch("summarizer.parser.DocumentConverter", side_effect=OSError("model download")) as cls,
        patch("summarizer.parser._extract_text_with_pypdf", return_value="pypdf text"),
    ):
        assert parse_pdf(a) == "pypdf text"
        assert parse_pdf(b) == "pypdf text"
    cls.assert_called_once()  # the failure is remembered, not retried per paper


def test_broken_docling_import_counts_as_unavailable(tmp_path):
    import types

    import summarizer.parser as parser_mod

    def _broken(name):
        raise RuntimeError("torch: undefined symbol")

    broken = types.ModuleType("docling.document_converter")
    broken.__getattr__ = _broken
    with (
        patch.object(parser_mod, "DocumentConverter", None),
        patch.dict(
            "sys.modules",
            {"docling": types.ModuleType("docling"), "docling.document_converter": broken},
        ),
        patch("summarizer.parser._extract_text_with_pypdf", return_value="pypdf text"),
    ):
        assert parse_pdf(_fake_pdf(tmp_path)) == "pypdf text"


def test_cache_write_failure_is_not_fatal(tmp_path, caplog):
    """Read-only PDF folders must not fail the paper after a successful extraction."""
    with (
        patch("summarizer.parser.os.replace", side_effect=PermissionError("read-only")),
        caplog.at_level(logging.WARNING, logger="summarizer.parser"),
    ):
        assert _mocked_parse(tmp_path, "text", 10_000) == "text"
    assert any("Could not write extraction cache" in r.message for r in caplog.records)
    assert not list(tmp_path.glob("*.tmp")) and not list(tmp_path.glob(".*.tmp"))


def test_pypdf_extractor_ignores_legacy_cache(tmp_path):
    pdf = _fake_pdf(tmp_path)
    (tmp_path / "paper.md").write_text("legacy", encoding="utf-8")
    with patch("summarizer.parser._extract_text_with_pypdf", return_value="pypdf text"):
        assert parse_pdf(pdf, extractor="pypdf") == "pypdf text"


def test_truncation_logs_warning(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="summarizer.parser"):
        _mocked_parse(tmp_path, text="x" * 1000, max_chars=100)
    assert any("truncated" in r.message for r in caplog.records)


def test_no_truncation_warning_when_text_fits(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="summarizer.parser"):
        _mocked_parse(tmp_path, text="short", max_chars=100)
    assert not any("truncated" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Reference section
# ---------------------------------------------------------------------------

BODY = "Intro text. " * 200


def test_strip_references_keeps_appendix_after_markdown_references():
    from summarizer.parser import strip_reference_section

    text = BODY + "\n## References\n[1] A. Author. Title. 2020.\n\n## Appendix A\nDetails.\n"
    stripped = strip_reference_section(text)
    assert "A. Author" not in stripped
    assert "## Appendix A\nDetails." in stripped and stripped.startswith(BODY)


def test_strip_references_cuts_to_end_without_headings():
    from summarizer.parser import strip_reference_section

    text = BODY + "\nREFERENCES\n[1] A. Author. Title. 2020.\n[2] B. Author."
    assert strip_reference_section(text) == BODY + "\n"


@pytest.mark.parametrize(
    "text",
    [
        "## References\n[1] early heading in the first half\n" + BODY,
        BODY + "\nSee the references in Sec. 2 for details.\n",
        BODY,
    ],
)
def test_strip_references_leaves_other_text_alone(text):
    from summarizer.parser import strip_reference_section

    assert strip_reference_section(text) == text


def test_parse_pdf_strips_before_truncating_and_caches_full_text(tmp_path):
    text = "x" * 100 + "\n## Bibliography\n" + "r" * 50
    result = _mocked_parse(tmp_path, text, max_chars=1000, extractor="docling")
    assert result == text  # off by default
    pdf = tmp_path / "paper.pdf"
    assert parse_pdf(pdf, max_chars=1000, extractor="docling", strip_references=True) == (
        "x" * 100 + "\n"
    )
    assert _cache(tmp_path, pdf, "docling").read_text(encoding="utf-8") == text


@pytest.mark.parametrize(
    "heading",
    [
        "References",
        "REFERENCES AND NOTES",
        "References Cited",
        "7 References",
        "8. References",
        "VII. REFERENCES",
        "## 6 . References",
        "## References ##",
        "**References**",
        "Bibliography:",
        "Literature Cited",
    ],
)
def test_reference_heading_variants(heading):
    from summarizer.parser import strip_reference_section

    text = BODY + f"\n{heading}\n[1] A. Author. Title. 2020.\n"
    assert "A. Author" not in strip_reference_section(text)


def test_last_reference_heading_wins():
    from summarizer.parser import strip_reference_section

    text = BODY + "\n## References\nsee below\n" + BODY + "\n## References\n[1] x\n"
    stripped = strip_reference_section(text)
    assert "see below" in stripped and "[1] x" not in stripped


def test_plain_text_keeps_sections_after_references():
    from summarizer.parser import strip_reference_section

    text = BODY + "\nReferences\n1. x 2020.\nMethods\nWe used Loihi.\n"
    assert strip_reference_section(text) == BODY + "\nMethods\nWe used Loihi.\n"


def test_strip_happens_before_truncation(tmp_path):
    body = "x" * 1000 + "\n"
    text = body + "## References\n" + "r" * 500 + "\n## Appendix\nkeep"
    _mocked_parse(tmp_path, text, max_chars=10_000, extractor="docling")
    pdf = tmp_path / "paper.pdf"
    result = parse_pdf(pdf, max_chars=1010, extractor="docling", strip_references=True)
    assert result == (body + "## Appendix\nkeep")[:1010]  # truncation after stripping


def test_cache_follows_content_not_location(tmp_path):
    """A moved PDF reuses its extraction cache (keyed by sha256)."""
    from summarizer.parser import load_text

    pdf = _fake_pdf(tmp_path)
    with patch("summarizer.parser._extract_text_with_pypdf", return_value="once") as extract:
        assert load_text(pdf, "pypdf").text == "once"
        moved = tmp_path / "moved" / "renamed.pdf"
        moved.parent.mkdir()
        pdf.rename(moved)
        parsed = load_text(moved, "pypdf")
    extract.assert_called_once()
    assert (parsed.text, parsed.extractor) == ("once", "pypdf")


def test_old_caches_next_to_the_pdf_are_still_read(tmp_path):
    from summarizer.parser import load_text

    pdf = _fake_pdf(tmp_path)
    (tmp_path / "paper.pypdf.md").write_text("old pypdf text", encoding="utf-8")
    parsed = load_text(pdf)
    assert (parsed.text, parsed.extractor) == ("old pypdf text", "pypdf")
    assert not list(tmp_path.glob("*.docling.md"))  # nothing new written next to the PDF


def test_bare_stem_cache_is_not_reused(tmp_path):
    """Regression: the original version's <stem>.md held pypdf text after any
    --extractor pypdf run, so auto served it as-is and docling never ran."""
    from summarizer.parser import load_text

    pdf = _fake_pdf(tmp_path)
    (tmp_path / "paper.md").write_text("glued pypdf text", encoding="utf-8")
    with patch("summarizer.parser._run_docling", return_value="docling text"):
        parsed = load_text(pdf)
    assert (parsed.text, parsed.extractor) == ("docling text", "docling")


def test_stale_cache_next_to_a_replaced_pdf_is_ignored(tmp_path):
    """Regression: an old <stem>.docling.md fed the previous paper's text to a new PDF."""
    import os

    pdf = _fake_pdf(tmp_path)
    cache = tmp_path / "paper.docling.md"
    cache.write_text("TEXT OF THE OLD PAPER", encoding="utf-8")
    os.utime(cache, (1_000_000, 1_000_000))  # written long before the PDF was replaced
    with patch("summarizer.parser._extract_text_with_pypdf", return_value="new paper"):
        assert load_text(pdf, "pypdf").text == "new paper"
    assert load_text(pdf, "auto").text == "new paper"  # now from the content-keyed cache


def test_extracted_newlines_are_normalized(tmp_path):
    pdf = _fake_pdf(tmp_path)
    with patch("summarizer.parser._extract_text_with_pypdf", return_value="a\r\nb\rc"):
        fresh = load_text(pdf, "pypdf").text
    assert fresh == "a\nb\nc" == load_text(pdf, "pypdf").text


def test_relative_xdg_cache_home_is_ignored(tmp_path, monkeypatch):
    from summarizer.parser import _extraction_cache_dir

    monkeypatch.setenv("XDG_CACHE_HOME", "relative/dir")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert _extraction_cache_dir() == tmp_path / ".cache" / "paper-summarizer"


def test_auto_prefers_docling_cache_in_the_cache_dir(tmp_path):
    pdf = _fake_pdf(tmp_path)
    for extractor in ("docling", "pypdf"):
        path = _cache(tmp_path, pdf, extractor)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(extractor, encoding="utf-8")
    assert load_text(pdf).text == "docling"


@pytest.mark.parametrize("extractor", ["docling", "pypdf"])
def test_explicit_extractor_reads_its_old_cache_next_to_the_pdf(tmp_path, extractor):
    pdf = _fake_pdf(tmp_path)
    (tmp_path / f"paper.{extractor}.md").write_text("old cache", encoding="utf-8")
    assert load_text(pdf, extractor).text == "old cache"
