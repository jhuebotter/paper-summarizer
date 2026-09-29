"""PDF to markdown parser — docling (optional) with pypdf fallback, plus disk cache.

Extracted text is cached per PDF content and extractor in
``$XDG_CACHE_HOME/paper-summarizer`` (default ``~/.cache/paper-summarizer``) as
``{sha256}.{extractor}.md``, so moving or renaming a PDF keeps its cache and
switching ``--extractor`` never reuses another backend's text.  Caches written
next to the PDF by earlier versions (``{stem}.docling.md``, ``{stem}.pypdf.md``,
and ``{stem}.md``, honoured only by ``auto``) are still read.

Zero-byte cache files are treated as a cache miss.  Cache writes are atomic and
best-effort.

docling is imported lazily (it costs several seconds and pulls in torch), and
is an optional dependency (``uv sync --extra docling``).  When it is missing or
fails to start, ``auto`` uses pypdf.
"""

import hashlib
import logging
import os
import re
import threading
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ParsedText:
    """Extracted (optionally reference-stripped) text of one PDF, not yet truncated."""

    text: str
    extractor: str  # "docling", "pypdf" or "legacy" (an old cache of unknown origin)
    sha256: str


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file's contents."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_text(
    pdf_path: Path,
    extractor: str = "auto",
    reparse: bool = False,
    strip_references: bool = False,
) -> ParsedText:
    """Return the PDF's full text, using the cache when available.

    Args:
        pdf_path:  Path to the PDF file.
        extractor: Extraction strategy: ``auto`` (docling with pypdf fallback),
                   ``docling`` (docling only), ``pypdf`` (pypdf only).
        reparse:   If True, ignore any existing cache and re-run extraction.
        strip_references: Remove the References/Bibliography section (see
                   ``strip_reference_section``).  Caches always keep the full text.

    Raises:
        ParseError: if extraction fails. No cache file is written on failure.
    """
    if extractor not in _EXTRACTORS:
        raise ValueError(f"Unknown extractor {extractor!r}; expected one of {_EXTRACTORS}")

    sha = sha256_file(pdf_path)
    cached = None if reparse else _read_cache(pdf_path, sha, extractor)
    if cached is not None:
        text, used = cached
    else:
        logger.info("Running %s extraction on: %s", extractor, pdf_path.name)
        text, used = _extract_text(pdf_path, extractor=extractor)
        _write_cache(_cache_path(sha, used), text)
        logger.info("Extraction complete (%s): %s chars", used, f"{len(text):,}")
    if strip_references:
        text = strip_reference_section(text, pdf_path.name)
    return ParsedText(text=text, extractor=used, sha256=sha)


def parse_pdf(
    pdf_path: Path,
    max_chars: int = _DEFAULT_MAX_CHARS,
    reparse: bool = False,
    extractor: str = "auto",
    strip_references: bool = False,
) -> str:
    """``load_text`` followed by truncation to ``max_chars`` (with a warning)."""
    parsed = load_text(pdf_path, extractor, reparse=reparse, strip_references=strip_references)
    return truncate_text(parsed.text, max_chars, pdf_path.name)


# ---------------------------------------------------------------------------
# Reference section
# ---------------------------------------------------------------------------

_REFERENCE_HEADING = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?(?:(?:\d+|[IVXLC]+)[ \t]*\.?[ \t]*)?(?:\*\*)?"
    r"(?:references(?:[ \t]+(?:and[ \t]+notes|cited))?|bibliography|literature(?:[ \t]+cited)?"
    r"|works[ \t]+cited|reference[ \t]+list)"
    r"(?:\*\*)?[ \t]*:?[ \t]*#*[ \t]*$",
    re.I | re.M,
)
_MARKDOWN_HEADING = re.compile(r"^#{1,6}[ \t]+\S", re.M)
# Sections that commonly follow the reference list in plain (pypdf) text.
_PLAIN_SECTION_AFTER = re.compile(
    r"^[ \t]*(?:[A-Z]\.?|\d+\.?)?[ \t]*(?:appendix|appendices|supplementary|supplemental"
    r"|(?:online[ \t]+)?methods|materials[ \t]+and[ \t]+methods|acknowledg(?:e)?ments?)\b.{0,80}$",
    re.I | re.M,
)


def strip_reference_section(text: str, name: str = "") -> str:
    """Remove the References/Bibliography section.

    Takes the last heading-only line named References (or a variant) in the
    second half of the text and cuts to the next markdown heading, or, in
    plain text without markdown headings, to the next Appendix / Supplementary /
    Methods / Acknowledgements line, or to the end.  Text without such a heading
    is returned unchanged.
    """
    matches = [m for m in _REFERENCE_HEADING.finditer(text) if m.start() >= len(text) / 2]
    if not matches:
        return text
    start = matches[-1].start()
    following = _MARKDOWN_HEADING if _MARKDOWN_HEADING.search(text) else _PLAIN_SECTION_AFTER
    next_section = following.search(text, matches[-1].end())
    end = next_section.start() if next_section else len(text)
    logger.info(
        "Removed reference section of %s (%s chars)", name or "document", f"{end - start:,}"
    )
    return text[:start] + text[end:]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "paper-summarizer"


def _cache_path(sha: str, extractor: str) -> Path:
    return cache_dir() / f"{sha}.{extractor}.md"


def _cache_candidates(pdf_path: Path, sha: str, extractor: str) -> list[tuple[Path, str]]:
    def beside(ext: str) -> Path:
        return pdf_path.parent / f"{pdf_path.stem}.{ext}.md"

    if extractor in ("docling", "pypdf"):
        return [(_cache_path(sha, extractor), extractor), (beside(extractor), extractor)]
    # auto: prefer docling output, then a previous pypdf fallback, then legacy.
    return [
        (_cache_path(sha, "docling"), "docling"),
        (_cache_path(sha, "pypdf"), "pypdf"),
        (beside("docling"), "docling"),
        (beside("pypdf"), "pypdf"),
        (pdf_path.parent / f"{pdf_path.stem}.md", "legacy"),
    ]


def _read_cache(pdf_path: Path, sha: str, extractor: str) -> tuple[str, str] | None:
    for path, used in _cache_candidates(pdf_path, sha, extractor):
        if path.exists() and path.stat().st_size > 0:
            cached = path.read_text(encoding="utf-8")
            logger.info("Extraction cache found: %s (%s chars)", path.name, f"{len(cached):,}")
            return cached, used
    return None


def _write_cache(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        logger.warning("Could not write extraction cache %s: %s", path, exc)


def truncate_text(text: str, max_chars: int, name: str) -> str:
    """Cut ``text`` to ``max_chars``, logging a warning when anything is lost."""
    if len(text) > max_chars:
        logger.warning(
            "Paper text truncated for %s: keeping %s of %s chars (raise --max-chars to keep more)",
            name,
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
