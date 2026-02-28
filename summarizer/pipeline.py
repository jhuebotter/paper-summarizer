"""Per-paper orchestration — converts one PDF to a validated PaperSummary.

Single LLM call per paper: metadata + Part 1 + Part 2 are requested
in one combined JSON response.
"""

import logging
import json
import re
from pathlib import Path

from pydantic import ValidationError

from summarizer.llm import call_llm, create_client
from summarizer.models import (
    Config,
    LLMResponse,
    PaperSummary,
    ParseError,
    PipelineError,
)
from summarizer.parser import parse_pdf
from summarizer.prompts import build_combined_prompt, load_references

logger = logging.getLogger(__name__)

_MAX_SCHEMA_REPAIR_RETRIES = 2


def process_pdf(pdf_path: Path, config: Config) -> PaperSummary:
    """Process a single PDF end-to-end and return a validated ``PaperSummary``.

    Steps
    -----
    1. Parse the PDF via docling (reads cache if available; see ``parser.py``).
    2. Build a combined prompt requesting metadata + Part 1 + Part 2 in one call.
    3. Call the LLM once; validate the response into ``LLMResponse``.
    4. Return a fully validated ``PaperSummary``.

    Raises:
        PipelineError: wraps any ``ParseError``, ``LLMError``,
            ``ValidationError``, or other exception that occurs.
    """
    try:
        return _run_pipeline(pdf_path, config)
    except PipelineError:
        raise
    except Exception as e:
        raise PipelineError(pdf_path, e) from e


def _run_pipeline(pdf_path: Path, config: Config) -> PaperSummary:
    # Step 1: parse PDF → markdown (uses cache if available)
    paper_text = parse_pdf(
        pdf_path,
        config.max_chars,
        reparse=config.reparse,
        extractor=config.extractor,
    )

    # Step 2: load references and build combined prompt
    references = load_references(config.skill_data_dir)
    client = create_client(config)
    prompt = build_combined_prompt(
        paper_text=paper_text,
        references=references,
        source_filename=pdf_path.name,
    )
    logger.info(
        "Building prompt (%s chars, ~%s tokens)",
        f"{len(prompt):,}",
        f"{len(prompt) // 4:,}",
    )

    # Step 3: single LLM call → parse and validate
    raw = call_llm(client, prompt)
    response = _validate_with_schema_repair(
        raw=raw,
        client=client,
        original_prompt=prompt,
        pdf_path=pdf_path,
    )

    return PaperSummary(
        metadata=response.metadata, part1=response.part1, part2=response.part2
    )


def _validate_with_schema_repair(
    raw: dict,
    client,
    original_prompt: str,
    pdf_path: Path,
) -> LLMResponse:
    """Validate response and repair schema with bounded LLM retries."""
    current = raw
    attempts = _MAX_SCHEMA_REPAIR_RETRIES + 1

    for attempt in range(1, attempts + 1):
        current = _normalize_metadata_year(current, pdf_path)
        current = _normalize_citation_key(current, pdf_path)
        try:
            return LLMResponse(**current)
        except ValidationError as exc:
            if attempt >= attempts:
                raise

            compact = _compact_validation_errors(exc)
            logger.warning(
                "Schema validation failed on attempt %d/%d; requesting repair (%s)",
                attempt,
                attempts,
                "; ".join(compact[:4]),
            )
            repair_prompt = _build_schema_repair_prompt(
                original_prompt=original_prompt,
                bad_response=current,
                validation_errors=compact,
            )
            current = call_llm(client, repair_prompt)

    raise RuntimeError("Schema validation retry loop exhausted unexpectedly")


def _compact_validation_errors(exc: ValidationError) -> list[str]:
    """Convert pydantic errors into concise 'path: message' strings."""
    compact: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", ()))
        msg = err.get("msg", "validation error")
        compact.append(f"{loc}: {msg}")
    return compact


def _build_schema_repair_prompt(
    original_prompt: str,
    bad_response: dict,
    validation_errors: list[str],
) -> str:
    """Prompt asking the LLM to repair schema validation issues only."""
    rendered_errors = "\n".join(f"- {item}" for item in validation_errors)
    bad_json = json.dumps(bad_response, ensure_ascii=False)

    return (
        "You are a JSON schema-repair assistant.\n"
        "Task: Fix the RESPONSE JSON so it satisfies the expected schema.\n"
        "Rules:\n"
        "1) Output one valid JSON object only (no markdown or explanation).\n"
        "2) Preserve existing fields and values when possible.\n"
        "3) Add/repair only what is needed to satisfy schema requirements.\n"
        "4) Do not invent unsupported facts; use 'not reported' when required and unknown.\n"
        "5) Keep top-level keys exactly: metadata, part1, part2.\n\n"
        "Validation errors:\n"
        f"{rendered_errors}\n\n"
        "Original extraction prompt (for context):\n"
        f"{original_prompt}\n\n"
        "Response JSON to repair:\n"
        f"{bad_json}"
    )


def _normalize_metadata_year(raw: dict, pdf_path: Path) -> dict:
    """Normalize non-integer metadata.year values before schema validation."""
    metadata = raw.get("metadata")
    if not isinstance(metadata, dict):
        return raw

    year = metadata.get("year")
    if isinstance(year, int):
        return raw

    normalized = _extract_year_candidate(year)
    source = "metadata.year"
    if normalized is None:
        normalized = _extract_year_candidate(metadata.get("title"))
        source = "metadata.title"
    if normalized is None:
        normalized = _extract_year_candidate(pdf_path.name)
        source = "source filename"
    if normalized is None:
        normalized = _extract_year_candidate(metadata.get("citation_key"))
        source = "citation_key"

    if normalized is None:
        normalized = 0
        source = "fallback=0"

    logger.warning(
        "LLM returned non-integer metadata.year=%r; normalized to %d using %s",
        year,
        normalized,
        source,
    )
    metadata["year"] = normalized
    return raw


def _normalize_citation_key(raw: dict, pdf_path: Path) -> dict:
    """Repair missing/invalid citation_key values before schema validation."""
    metadata = raw.get("metadata")
    if not isinstance(metadata, dict):
        return raw

    citation_key = metadata.get("citation_key")
    if _is_valid_citation_key(citation_key):
        return raw

    repaired = _build_citation_key(metadata, pdf_path)
    logger.warning(
        "LLM returned invalid citation_key=%r; repaired to %s",
        citation_key,
        repaired,
    )
    metadata["citation_key"] = repaired
    return raw


def _is_valid_citation_key(value: object) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if not stripped:
        return False
    if stripped.lower() in {"not reported", "unknown", "n/a", "na"}:
        return False
    return bool(re.fullmatch(r"[a-z][a-z0-9]*", stripped))


def _build_citation_key(metadata: dict, pdf_path: Path) -> str:
    """Synthesize a deterministic citation key: firstauthor+year+firstword."""
    year = metadata.get("year")
    if not isinstance(year, int):
        year = 0

    authors = metadata.get("authors")
    first_author_token = "paper"
    if isinstance(authors, list) and authors:
        first_author_token = (
            _author_surname_token(str(authors[0])) or first_author_token
        )

    title = metadata.get("title")
    title_token = _first_alnum_token(str(title)) if title else "paper"
    if not title_token:
        title_token = _first_alnum_token(pdf_path.stem) or "paper"

    return f"{first_author_token}{year}{title_token}".lower()


def _first_alnum_token(value: str) -> str:
    """Return first alphabetic token normalized to lowercase alnum."""
    for token in re.split(r"[^A-Za-z0-9]+", value):
        token = token.strip().lower()
        if token and re.search(r"[a-z]", token):
            return re.sub(r"[^a-z0-9]", "", token)
    return ""


def _author_surname_token(author_name: str) -> str:
    """Extract a surname-like token from an author name string."""
    tokens = [
        t.lower()
        for t in re.split(r"[^A-Za-z0-9]+", author_name)
        if t and re.search(r"[a-zA-Z]", t)
    ]
    if not tokens:
        return ""
    return re.sub(r"[^a-z0-9]", "", tokens[-1])


def _extract_year_candidate(value: object) -> int | None:
    """Extract a plausible 4-digit year from a string value."""
    if not isinstance(value, str):
        return None
    match = re.search(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", value)
    if not match:
        return None
    return int(match.group(1))
