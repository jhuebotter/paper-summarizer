"""Tests for summarizer/parser.py — docling PDF-to-markdown wrapper with cache."""

import logging
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from summarizer.models import ParseError
from summarizer.parser import parse_pdf


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
        MockConverter.return_value.convert.side_effect = Exception(
            "docling internal error"
        )
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
# Cache logic (v1.1)
# ---------------------------------------------------------------------------


def test_parse_pdf_writes_cache_on_fresh_parse(tmp_path):
    """After a fresh parse, {pdf_stem}.md is created next to the PDF."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake content")
    cache = tmp_path / "paper.md"

    mock_result = MagicMock()
    mock_result.document.export_to_markdown.return_value = "parsed content"

    with patch("summarizer.parser.DocumentConverter") as MockConverter:
        MockConverter.return_value.convert.return_value = mock_result
        parse_pdf(pdf, max_chars=10_000)

    assert cache.exists()
    assert cache.read_text(encoding="utf-8") == "parsed content"


def test_parse_pdf_reads_cache_when_present(tmp_path):
    """When {pdf_stem}.md exists and is non-empty, docling is NOT called."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake content")
    cache = tmp_path / "paper.md"
    cache.write_text("cached content", encoding="utf-8")

    with patch("summarizer.parser.DocumentConverter") as MockConverter:
        result = parse_pdf(pdf, max_chars=10_000)
        MockConverter.assert_not_called()

    assert result == "cached content"


def test_parse_pdf_truncates_cached_content(tmp_path):
    """Cached content is still subject to max_chars truncation."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake content")
    cache = tmp_path / "paper.md"
    cache.write_text("x" * 1000, encoding="utf-8")

    with patch("summarizer.parser.DocumentConverter") as MockConverter:
        result = parse_pdf(pdf, max_chars=100)
        MockConverter.assert_not_called()

    assert len(result) == 100


def test_parse_pdf_zero_byte_cache_treated_as_miss(tmp_path):
    """An empty (zero-byte) cache file causes a fresh docling run."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake content")
    cache = tmp_path / "paper.md"
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
    cache = tmp_path / "paper.md"
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
        patch(
            "summarizer.parser._extract_text_with_pypdf", return_value="fallback text"
        ),
    ):
        MockConverter.return_value.convert.side_effect = Exception(
            "PdfHyperlink url_parsing"
        )
        result = parse_pdf(pdf, max_chars=10_000)

    assert result == "fallback text"
    assert (tmp_path / "paper.md").read_text(encoding="utf-8") == "fallback text"


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


def test_parse_pdf_logs_cache_hit(tmp_path, caplog):
    """A cache hit emits an INFO message containing 'cache found'."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    (tmp_path / "paper.md").write_text("cached content", encoding="utf-8")

    with caplog.at_level(logging.INFO, logger="summarizer.parser"):
        parse_pdf(pdf, max_chars=10_000)

    assert any("cache found" in r.message.lower() for r in caplog.records)


def test_parse_pdf_logs_running_docling(tmp_path, caplog):
    """A fresh parse emits an INFO message containing 'Running ... extraction'."""
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

    assert any("Running auto extraction" in r.message for r in caplog.records)


def test_parse_pdf_logs_docling_complete(tmp_path, caplog):
    """After extraction finishes, an INFO completion message is emitted."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    mock_result = MagicMock()
    mock_result.document.export_to_markdown.return_value = "some content"

    with (
        caplog.at_level(logging.INFO, logger="summarizer.parser"),
        patch("summarizer.parser.DocumentConverter") as MockConverter,
    ):
        MockConverter.return_value.convert.return_value = mock_result
        parse_pdf(pdf, max_chars=10_000)

    assert any("Extraction complete" in r.message for r in caplog.records)


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
