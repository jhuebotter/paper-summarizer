"""PDF to markdown parser — docling (optional) with pypdf fallback, plus disk cache.

Cache files live next to the source PDF and are keyed by the extractor that
produced them, so switching ``--extractor`` never silently reuses text from a
different backend:

* ``{pdf_stem}.docling.md`` — written by docling
* ``{pdf_stem}.pypdf.md``   — written by pypdf (directly or as ``auto`` fallback)
* ``{pdf_stem}.md``         — legacy cache of unknown origin; honoured only by
  ``auto`` so existing libraries are not re-parsed

Zero-byte cache files are treated as a cache miss.  Cache writes are atomic and
best-effort: a read-only PDF folder only costs re-extraction on the next run.

docling is imported lazily (it costs several seconds and pulls in torch), and
is an optional dependency (``uv sync --extra docling``).  When it is missing or
fails to start, ``auto`` uses pypdf.
"""

import logging
import os
import threading
from pathlib import Path

from pypdf import PdfReader

from summarizer.models import _DEFAULT_MAX_CHARS, ParseError

logger = logging.getLogger(__name__)

#: Resolved lazily by ``_docling_converter_class``; tests patch this name.
DocumentConverter = None

_DOCLING_LOCK = threading.Lock()
_CONVERTERS: dict[object, object] = {}

_EXTRACTORS = ("auto", "docling", "pypdf")


class DoclingUnavailable(ParseError):
    """Raised when docling is requested but not installed."""


def parse_pdf(
    pdf_path: Path,
    max_chars: int = _DEFAULT_MAX_CHARS,
    reparse: bool = False,
    extractor: str = "auto",
) -> str:
    """Parse a PDF to markdown, using a disk cache when available.

    Args:
        pdf_path:  Path to the PDF file.
        max_chars: Maximum characters to return (truncates after this limit).
        reparse:   If True, ignore any existing cache and re-run extraction.
        extractor: Extraction strategy: ``auto`` (docling with pypdf fallback),
                   ``docling`` (docling only), ``pypdf`` (pypdf only).

    Returns:
        Markdown string, truncated to ``max_chars``.

    Raises:
        ParseError: if extraction fails. No cache file is written on failure.
    """
    if extractor not in _EXTRACTORS:
        raise ValueError(f"Unknown extractor {extractor!r}; expected one of {_EXTRACTORS}")

    if not reparse:
        cached = _read_cache(pdf_path, extractor)
        if cached is not None:
            return _truncate(cached, max_chars, pdf_path)

    logger.info("Running %s extraction on: %s", extractor, pdf_path.name)
    text, used = _extract_text(pdf_path, extractor=extractor)
    _write_cache(_cache_path(pdf_path, used), text)
    logger.info("Extraction complete (%s): %s chars", used, f"{len(text):,}")
    return _truncate(text, max_chars, pdf_path)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _cache_path(pdf_path: Path, extractor: str) -> Path:
    return pdf_path.parent / f"{pdf_path.stem}.{extractor}.md"


def _cache_candidates(pdf_path: Path, extractor: str) -> list[Path]:
    if extractor == "docling":
        return [_cache_path(pdf_path, "docling")]
    if extractor == "pypdf":
        return [_cache_path(pdf_path, "pypdf")]
    # auto: prefer docling output, then a previous pypdf fallback, then legacy.
    return [
        _cache_path(pdf_path, "docling"),
        _cache_path(pdf_path, "pypdf"),
        pdf_path.parent / f"{pdf_path.stem}.md",
    ]


def _read_cache(pdf_path: Path, extractor: str) -> str | None:
    for path in _cache_candidates(pdf_path, extractor):
        if path.exists() and path.stat().st_size > 0:
            cached = path.read_text(encoding="utf-8")
            logger.info("Extraction cache found: %s (%s chars)", path.name, f"{len(cached):,}")
            return cached
    return None


def _write_cache(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        logger.warning("Could not write extraction cache %s: %s", path, exc)


def _truncate(text: str, max_chars: int, pdf_path: Path) -> str:
    if len(text) > max_chars:
        logger.warning(
            "Paper text truncated for %s: keeping %s of %s chars (raise --max-chars to keep more)",
            pdf_path.name,
            f"{max_chars:,}",
            f"{len(text):,}",
        )
    return text[:max_chars]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _extract_text(pdf_path: Path, extractor: str) -> tuple[str, str]:
    """Return ``(text, extractor_actually_used)``."""
    if extractor == "docling":
        return _run_docling(pdf_path), "docling"
    if extractor == "pypdf":
        return _extract_text_with_pypdf(pdf_path), "pypdf"
    return _run_docling_with_fallback(pdf_path)


def _run_docling_with_fallback(pdf_path: Path) -> tuple[str, str]:
    """Run docling, then fall back to pypdf text extraction on failure."""
    try:
        return _run_docling(pdf_path), "docling"
    except DoclingUnavailable as exc:
        logger.warning("%s; using pypdf for %s", exc, pdf_path.name)
        return _extract_text_with_pypdf(pdf_path), "pypdf"
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
        return text, "pypdf"


def _docling_converter_class():
    """Return docling's ``DocumentConverter`` class, importing it on first use."""
    global DocumentConverter
    if DocumentConverter is None:
        try:
            from docling.document_converter import DocumentConverter as _DC
        except ImportError as exc:
            raise DoclingUnavailable("docling is not installed (uv sync --extra docling)") from exc
        except Exception as exc:  # e.g. a broken torch install
            raise DoclingUnavailable(f"docling failed to import: {exc}") from exc
        DocumentConverter = _DC
    return DocumentConverter


def _get_converter():
    """Return a process-wide converter; building one loads docling's models.

    Raises:
        DoclingUnavailable: if docling is missing or its models fail to load
            (the failure is remembered, so later papers skip straight to pypdf).
    """
    cls = _docling_converter_class()
    converter = _CONVERTERS.get(cls)
    if converter is None:
        try:
            converter = cls()
        except Exception as exc:
            converter = DoclingUnavailable(f"docling failed to start: {exc}")
        _CONVERTERS[cls] = converter
    if isinstance(converter, DoclingUnavailable):
        raise converter
    return converter


def _run_docling(pdf_path: Path) -> str:
    """Run docling on *pdf_path* and return the full markdown string.

    Calls are serialized: docling is not reliably thread-safe under parallel
    batch runs, while LLM calls stay parallel.

    Raises:
        DoclingUnavailable: if docling is not installed.
        ParseError: wrapping any exception raised by docling.
    """
    with _DOCLING_LOCK:
        converter = _get_converter()
        try:
            result = converter.convert(str(pdf_path))
            return result.document.export_to_markdown()
        except Exception as e:
            raise ParseError(f"Failed to parse {pdf_path}: {e}") from e


def _extract_text_with_pypdf(pdf_path: Path) -> str:
    """Extract plain text with pypdf (no layout/markdown structure)."""
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
        raise ParseError(f"Failed to parse {pdf_path}: pypdf error: {e}") from e
