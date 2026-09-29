"""Per-paper orchestration — converts one PDF to a validated PaperSummary.

Single LLM call per paper: metadata + Part 1 + Part 2 are requested
in one combined JSON response.
"""

import json
import logging
import re
import unicodedata
from pathlib import Path

from pydantic import ValidationError

from summarizer.llm import CostAccumulator, LLMClient, call_llm, create_client
from summarizer.models import (
    Config,
    LLMResponse,
    PaperSummary,
    PipelineError,
)
from summarizer.parser import parse_pdf
from summarizer.prompts import build_combined_prompt, load_references

logger = logging.getLogger(__name__)

_MAX_SCHEMA_REPAIR_RETRIES = 2
_CONTRACT_FILENAME = "json-output-contract.md"


def process_pdf(
    pdf_path: Path,
    config: Config,
    client: "LLMClient | None" = None,
    accumulator: "CostAccumulator | None" = None,
    references: str | None = None,
) -> PaperSummary:
    """Process a single PDF end-to-end and return a validated ``PaperSummary``.

    Steps
    -----
    1. Extract the PDF text (reads cache if available; see ``parser.py``).
    2. Build a combined prompt requesting metadata + Part 1 + Part 2 in one call.
    3. Call the LLM once; validate the response into ``LLMResponse``.
    4. Return a fully validated ``PaperSummary``.

    Args:
        pdf_path:    Path to the PDF to process.
        config:      Runtime configuration.
        client:      Optional pre-created LLM client.  When ``None`` a new
                     client is created from ``config`` (default behaviour).
        accumulator: Optional cost accumulator updated after each LLM call.
        references:  Optional pre-loaded reference text (batch mode loads it
                     once); read from ``config.skill_data_dir`` when ``None``.

    Raises:
        PipelineError: wraps any ``ParseError``, ``LLMError``,
            ``ValidationError``, or other exception that occurs.
    """
    try:
        return _run_pipeline(
            pdf_path, config, client=client, accumulator=accumulator, references=references
        )
    except PipelineError:
        raise
    except Exception as e:
        raise PipelineError(pdf_path, e) from e


def _run_pipeline(
    pdf_path: Path,
    config: Config,
    client: "LLMClient | None" = None,
    accumulator: "CostAccumulator | None" = None,
    references: str | None = None,
) -> PaperSummary:
    # Step 1: parse PDF → markdown (uses cache if available)
    paper_text = parse_pdf(
        pdf_path,
        config.max_chars,
        reparse=config.reparse,
        extractor=config.extractor,
    )

    # Step 2: load references and build combined prompt
    if references is None:
        references = load_references(config.skill_data_dir)
    if client is None:
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
    raw = call_llm(client, prompt, accumulator=accumulator)
    response = _validate_with_schema_repair(
        raw=raw,
        client=client,
        original_prompt=prompt,
        pdf_path=pdf_path,
        accumulator=accumulator,
        contract=_load_contract(config.skill_data_dir),
    )

    return PaperSummary(metadata=response.metadata, part1=response.part1, part2=response.part2)


def _load_contract(references_dir: Path) -> str:
    """Return the JSON output contract reference, or ``""`` if absent."""
    path = references_dir / _CONTRACT_FILENAME
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _validate_with_schema_repair(
    raw: dict,
    client,
    original_prompt: str,
    pdf_path: Path,
    accumulator: "CostAccumulator | None" = None,
    contract: str = "",
) -> LLMResponse:
    """Validate response and repair schema with bounded LLM retries.

    The repair prompt only includes the full original prompt (and therefore
    the paper text) when fields are *missing* — the model needs the paper to
    fill them.  Type/enum/structure errors are fixed from the JSON contract
    alone, which avoids resending ~50k tokens of paper per repair.
    """
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
            needs_paper = _needs_paper_context(exc)
            logger.warning(
                "Schema validation failed on attempt %d/%d; requesting repair "
                "(%d errors, %s paper text: %s)",
                attempt,
                attempts,
                len(compact),
                "with" if needs_paper else "without",
                "; ".join(compact[:4]),
            )
            logger.debug("All validation errors:\n%s", "\n".join(compact))
            repair_prompt = _build_schema_repair_prompt(
                original_prompt=original_prompt if needs_paper else "",
                bad_response=current,
                validation_errors=compact,
                contract=contract,
            )
            current = call_llm(client, repair_prompt, accumulator=accumulator)

    raise RuntimeError("Schema validation retry loop exhausted unexpectedly")


def _needs_paper_context(exc: ValidationError) -> bool:
    """True when repair must *add content* (missing fields / a required part2)."""
    for err in exc.errors():
        if err.get("type") == "missing" or "is required" in str(err.get("msg", "")):
            return True
    return False


def _compact_validation_errors(exc: ValidationError) -> list[str]:
    """Convert pydantic errors into concise 'path: message' strings."""
    compact: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", ()))
        msg = err.get("msg", "validation error")
        compact.append(f"{loc}: {msg}")
    return compact


_PRIMARY_PART2_FIELDS = (
    "neuron_model, network_architecture, model_scale, simulator_framework, "
    "hardware_training, controller_hardware_inference, control_task, task_type, "
    "task_complexity_scale, simulation_environment, spike_encoding, action_decoding, "
    "learning_mechanism, credit_assignment_scope, online_vs_offline, data_collection, "
    "key_training_details, comparison_to_baselines"
)

_SYNTHESIS_PART1_FIELDS = (
    "paper_type, tldr, target_papers_field, scope_coverage, taxonomy_organization, "
    "core_argument, synthesis_contribution, key_claims_narrative, key_takeaways, "
    "limitations, open_problems_future_directions, critical_assessment, notable_findings, "
    "citable_snippets, relevance"
)


def _build_schema_repair_prompt(
    original_prompt: str,
    bad_response: dict,
    validation_errors: list[str],
    contract: str = "",
) -> str:
    """Prompt asking the LLM to repair schema validation issues only.

    ``original_prompt`` may be empty (no paper context needed); ``contract``
    is the JSON output contract, included when there is no original prompt
    (which would otherwise already contain it).
    """
    rendered_errors = "\n".join(f"- {item}" for item in validation_errors)
    bad_json = json.dumps(bad_response, ensure_ascii=False)

    paper_type = (bad_response.get("metadata") or {}).get("paper_type")
    if paper_type == "primary":
        field_hint = (
            f"Required part2 fields (all must be present): {_PRIMARY_PART2_FIELDS}. "
            "Use 'not applicable' or 'not reported' for any field that cannot be filled."
        )
    elif paper_type == "synthesis":
        field_hint = (
            f"Required part1 fields for synthesis (all must be present): {_SYNTHESIS_PART1_FIELDS}. "
            "Use 'not applicable' for any field that does not apply — never omit the key."
        )
    else:
        field_hint = ""

    field_hint_block = f"\nRequired fields hint:\n{field_hint}\n" if field_hint else ""

    if original_prompt:
        context_block = f"Original extraction prompt (for context):\n{original_prompt}\n\n"
    elif contract:
        context_block = f"Expected JSON contract:\n{contract}\n\n"
    else:
        context_block = ""

    return (
        "You are a JSON schema-repair assistant.\n"
        "Task: Fix the RESPONSE JSON so it satisfies the expected schema.\n"
        "Rules:\n"
        "1) Output one valid JSON object only (no markdown or explanation).\n"
        "2) Preserve existing fields and values when possible.\n"
        "3) Add/repair only what is needed to satisfy schema requirements.\n"
        "4) Do not invent unsupported facts; use 'not reported' when required and unknown.\n"
        "5) Keep top-level keys exactly: metadata, part1, part2.\n"
        f"{field_hint_block}\n"
        "Validation errors:\n"
        f"{rendered_errors}\n\n"
        f"{context_block}"
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
        logger.warning(
            "No publication year found (LLM returned %r); leaving it unset (citation key uses 'nd')",
            year,
        )
        metadata["year"] = None
        return raw

    logger.warning(
        "LLM returned non-integer metadata.year=%r; normalized to %d using %s",
        year,
        normalized,
        source,
    )
    metadata["year"] = normalized
    return raw


def _sanitize_citation_key(value: str) -> str:
    """Strip diacritics and non-alnum chars from a citation key candidate.

    Applies NFKD Unicode decomposition to convert accented characters to their
    ASCII base (e.g. é→e, ñ→n), then removes everything outside [a-z0-9].
    """
    ascii_val = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", ascii_val.lower())


def _normalize_citation_key(raw: dict, pdf_path: Path) -> dict:
    """Repair missing/invalid citation_key values before schema validation."""
    metadata = raw.get("metadata")
    if not isinstance(metadata, dict):
        return raw

    citation_key = metadata.get("citation_key")
    if _is_valid_citation_key(citation_key):
        return raw

    # Try lightweight sanitization first (strips accents, hyphens, spaces).
    # Preserves the LLM's intent rather than rebuilding from scratch.
    if isinstance(citation_key, str) and citation_key.strip():
        sanitized = _sanitize_citation_key(citation_key.strip())
        if _is_valid_citation_key(sanitized):
            logger.info(
                "LLM citation_key=%r sanitized to %s",
                citation_key,
                sanitized,
            )
            metadata["citation_key"] = sanitized
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
    if stripped.lower() in {"not reported", "unknown", "n/a", "na", "notreported", "notapplicable"}:
        return False
    return bool(re.fullmatch(r"[a-z][a-z0-9]*", stripped))


def _build_citation_key(metadata: dict, pdf_path: Path) -> str:
    """Synthesize a deterministic citation key: firstauthor+year+firstword.

    An unknown year becomes ``nd`` (BibTeX "no date").
    """
    year = metadata.get("year")
    if not isinstance(year, int):
        year = "nd"

    authors = metadata.get("authors")
    first_author_token = "paper"
    if isinstance(authors, list) and authors:
        first_author_token = _author_surname_token(str(authors[0])) or first_author_token

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
    """Extract a surname-like token from an author name string.

    Normalizes Unicode (NFKD) before splitting so that accented characters
    are converted to their ASCII base rather than acting as separators.
    """
    ascii_name = (
        unicodedata.normalize("NFKD", author_name).encode("ascii", "ignore").decode("ascii")
    )
    tokens = [
        t.lower() for t in re.split(r"[^A-Za-z0-9]+", ascii_name) if t and re.search(r"[a-zA-Z]", t)
    ]
    if not tokens:
        return ""
    return re.sub(r"[^a-z0-9]", "", tokens[-1])


def _extract_year_candidate(value: object) -> int | None:
    """Extract a plausible 4-digit year (1900-2099) from a string value."""
    if not isinstance(value, str):
        return None
    match = re.search(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", value)
    if not match:
        return None
    return int(match.group(1))
