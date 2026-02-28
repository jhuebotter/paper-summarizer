"""PDF to markdown parser — thin wrapper around docling, with disk cache.

The cache file ``{pdf_stem}.md`` is written next to the source PDF after a
successful parse and reused on subsequent runs to avoid re-running docling.
Zero-byte cache files are treated as a cache miss and trigger a fresh parse.
"""

import logging
import threading
from pathlib import Path

from docling.document_converter import DocumentConverter
from pypdf import PdfReader

from summarizer.models import ParseError

logger = logging.getLogger(__name__)

_DOCLING_LOCK = threading.Lock()


def parse_pdf(
    pdf_path: Path,
    max_chars: int = 200_000,
    reparse: bool = False,
    extractor: str = "auto",
) -> str:
    """Parse a PDF to markdown, using a disk cache when available.

    Cache file: ``{pdf_path.stem}.md`` in the same directory as the PDF.

    Args:
        pdf_path:  Path to the PDF file.
        max_chars: Maximum characters to return (truncates after this limit).
        reparse:   If True, ignore any existing cache and re-run extraction.
        extractor: Extraction strategy: ``auto`` (docling with pypdf fallback),
                   ``docling`` (docling only), ``pypdf`` (pypdf only).

    Returns:
        Markdown string, truncated to ``max_chars``.

    Raises:
        ParseError: if docling fails. The cache file is NOT written on failure.
    """
    cache_path = pdf_path.parent / f"{pdf_path.stem}.md"

    if not reparse and cache_path.exists() and cache_path.stat().st_size > 0:
        cached = cache_path.read_text(encoding="utf-8")
        logger.info(
            "Docling cache found: %s (%s chars)", cache_path.name, f"{len(cached):,}"
        )
        return cached[:max_chars]

    logger.info("Running %s extraction on: %s", extractor, pdf_path.name)
    text = _extract_text(pdf_path, extractor=extractor)
    cache_path.write_text(text, encoding="utf-8")
    logger.info("Extraction complete: %s chars", f"{len(text):,}")
    return text[:max_chars]


def _extract_text(pdf_path: Path, extractor: str) -> str:
    if extractor == "docling":
        with _DOCLING_LOCK:
            return _run_docling(pdf_path)
    if extractor == "pypdf":
        return _extract_text_with_pypdf(pdf_path)
    return _run_docling_with_fallback(pdf_path)


def _run_docling_with_fallback(pdf_path: Path) -> str:
    """Run docling, then fall back to pypdf text extraction on failure."""
    try:
        # Docling parsing is not always thread-safe under heavy parallel runs.
        # Serialize parse calls while keeping LLM steps parallel.
        with _DOCLING_LOCK:
            return _run_docling(pdf_path)
    except ParseError as docling_exc:
        logger.warning(
            "Docling parse failed for %s; attempting pypdf fallback: %s",
            pdf_path.name,
            docling_exc,
        )
        try:
            text = _extract_text_with_pypdf(pdf_path)
        except ParseError as fallback_exc:
            root_cause = docling_exc.__cause__ or docling_exc
            raise ParseError(
                f"Failed to parse {pdf_path}: docling and pypdf fallback failed ({fallback_exc})"
            ) from root_cause

        logger.warning(
            "Using pypdf fallback text extraction for %s (%s chars)",
            pdf_path.name,
            f"{len(text):,}",
        )
        return text


def _run_docling(pdf_path: Path) -> str:
    """Run docling on *pdf_path* and return the full markdown string.

    Raises:
        ParseError: wrapping any exception raised by docling.
    """
    try:
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        return result.document.export_to_markdown()
    except Exception as e:
        raise ParseError(f"Failed to parse {pdf_path}: {e}") from e


def _extract_text_with_pypdf(pdf_path: Path) -> str:
    """Extract text with pypdf as a robust fallback path."""
    try:
        reader = PdfReader(str(pdf_path))
        pages: list[str] = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        text = "\n\n".join(pages).strip()
        if not text:
            raise ParseError(f"Failed to parse {pdf_path}: pypdf extracted empty text")
        return text
    except ParseError:
        raise
    except Exception as e:
        raise ParseError(
            f"Failed to parse {pdf_path}: pypdf fallback error: {e}"
        ) from e
