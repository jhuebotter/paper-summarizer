"""Per-paper orchestration — converts one PDF to a validated PaperSummary.

Single LLM call per paper: metadata + Part 1 + Part 2 are requested
in one combined JSON response.
"""

import functools
import json
import logging
import re
import subprocess
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from summarizer.llm import CostAccumulator, LLMClient, call_llm, create_client
from summarizer.models import (
    Config,
    LLMResponse,
    PaperSummary,
    PipelineError,
    Provenance,
    SummaryPart1Synthesis,
    SummaryPart2,
)
from summarizer.parser import load_text, truncate_text
from summarizer.prompts import build_combined_prompt, load_references, references_digest

logger = logging.getLogger(__name__)

_MAX_SCHEMA_REPAIR_RETRIES = 2
_CONTRACT_FILENAME = "json-output-contract.md"
_MAX_CITATION_KEY_LEN = 64  # keeps output filenames well below OS limits
_CHARS_PER_TOKEN = 3  # conservative: maths-heavy text tokenizes densely
_OUTPUT_RESERVE_TOKENS = 16_000
_MIN_CONTEXT_SHARE = 0.2


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
    # Step 1: extract text (cached), cut it to the budget and the model's context
    parsed = load_text(
        pdf_path,
        extractor=config.extractor,
        reparse=config.reparse,
        strip_references=config.strip_references,
    )
    if references is None:
        references = load_references(config.skill_data_dir)
    if client is None:
        client = create_client(config)
    paper_text = fit_to_context(
        truncate_text(parsed.text, config.max_chars, pdf_path.name),
        references,
        pdf_path.name,
        client.pricing.context_length,
        config,
    )

    # Step 2: build the combined prompt
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

    # Step 3: single LLM call → parse and validate (per-paper totals feed the caller's)
    paper_cost = CostAccumulator(parent=accumulator)
    raw = call_llm(client, prompt, accumulator=paper_cost)
    response = _validate_with_schema_repair(
        raw=raw,
        client=client,
        original_prompt=prompt,
        pdf_path=pdf_path,
        accumulator=paper_cost,
        contract=_load_contract(config.skill_data_dir),
    )

    provenance = Provenance(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        git_commit=git_commit(),
        pdf_sha256=parsed.sha256,
        source_path=str(pdf_path.resolve()),
        extractor=parsed.extractor,
        chars_full=len(parsed.text),
        chars_sent=len(paper_text),
        model=config.model,
        base_url=config.base_url,
        references_sha256=references_digest(references),
        calls=paper_cost.calls,
        json_repairs=paper_cost.json_repairs,
        schema_repairs=paper_cost.schema_repairs,
        input_tokens=paper_cost.total_input_tokens,
        output_tokens=paper_cost.total_output_tokens,
        cost_usd=paper_cost.total_cost,
    )
    return PaperSummary(
        metadata=response.metadata,
        part1=response.part1,
        part2=response.part2,
        provenance=provenance,
    )


@functools.cache
def git_commit() -> str | None:
    """Short commit hash of the checkout this code runs from, if any."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def fit_to_context(
    paper_text: str, references: str, source_filename: str, context_length: object, config: Config
) -> str:
    """Cut the paper text so prompt + reply fit the model's context window.

    Uses a conservative ~3 characters per token and reserves
    ``max_output_tokens`` (or 16k tokens) for the reply.  Unknown context
    lengths leave the text unchanged.

    Raises:
        ValueError: if less than 20% of the text would fit; a summary of the
            first pages would be misleading and would never be retried.
    """
    if not isinstance(context_length, int) or context_length <= 0:
        return paper_text
    overhead = len(build_combined_prompt("", references, source_filename)) // _CHARS_PER_TOKEN
    reserve = config.max_output_tokens or _OUTPUT_RESERVE_TOKENS
    budget = (context_length - overhead - reserve) * _CHARS_PER_TOKEN
    if budget < _MIN_CONTEXT_SHARE * len(paper_text):
        raise ValueError(
            f"The model's {context_length:,}-token context holds only "
            f"{max(budget, 0):,} of {len(paper_text):,} chars of paper text; use a larger model"
        )
    if len(paper_text) > budget:
        logger.warning(
            "Paper text cut to %s chars to fit the model's %s-token context",
            f"{budget:,}",
            f"{context_length:,}",
        )
        return paper_text[:budget]
    return paper_text


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
            if accumulator is not None:
                accumulator.note_schema_repair()
            current = call_llm(client, repair_prompt, accumulator=accumulator)

    raise RuntimeError("Schema validation retry loop exhausted unexpectedly")


def _needs_paper_context(exc: ValidationError) -> bool:
    """True when repair must *add content*: missing or null fields, or a required part2."""
    for err in exc.errors():
        err_type = str(err.get("type", ""))
        if err_type == "missing" or "is required" in str(err.get("msg", "")):
            return True
        if err_type.endswith("_type") and err.get("input", "") is None:
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


_PRIMARY_PART2_FIELDS = ", ".join(SummaryPart2.model_fields)
_SYNTHESIS_PART1_FIELDS = ", ".join(SummaryPart1Synthesis.model_fields)


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
    if isinstance(year, float) and year.is_integer():
        metadata["year"] = int(year)
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
        metadata["citation_key"] = _match_metadata(citation_key.strip(), metadata)
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
            metadata["citation_key"] = _match_metadata(sanitized, metadata)
            return raw

    repaired = _build_citation_key(metadata, pdf_path)
    logger.warning(
        "LLM returned invalid citation_key=%r; repaired to %s",
        citation_key,
        repaired,
    )
    metadata["citation_key"] = repaired[:_MAX_CITATION_KEY_LEN]
    return raw


def _match_metadata(key: str, metadata: dict) -> str:
    """Rebuild *key* when its prefix isn't taken from the first author's name or
    its year differs from the metadata.

    Models mangle surnames (``ckl2024local`` for Stöckl, ``s2024fully`` for
    Paredes-Vallés, ``apolinaro`` for Apolinario).  Any run of words from the
    name is accepted (``smith`` for "Smith JA", ``garcia`` for "García Márquez");
    the descriptive word after the year is kept.
    """
    name = _first_author_name(metadata.get("authors"))
    surname = author_surname_token(name)
    if (
        metadata.get("is_research_paper") is False
        or len(surname) < 2
        or surname in _PLACEHOLDER_NAMES
    ):
        return key[:_MAX_CITATION_KEY_LEN]
    words = _name_words(name.replace(",", " "))
    runs = {"".join(words[a:b]) for a in range(len(words)) for b in range(a + 1, len(words) + 1)}
    runs.add(surname)
    year = metadata.get("year")
    match = re.fullmatch(r"([a-z]+)(\d{4})([a-z0-9]*)", key)
    if isinstance(year, int):
        ok = bool(match) and match.group(1) in runs and int(match.group(2)) == year
    else:
        ok = any(key.startswith(run) for run in runs)
    if not ok:
        word = match.group(3) if match and match.group(3).isalpha() else ""
        word = word or _first_alnum_token(str(metadata.get("title") or "")) or "paper"
        rebuilt = f"{surname}{year if isinstance(year, int) else 'nd'}{word}"
        logger.info("Citation key %s does not match the first author/year; using %s", key, rebuilt)
        key = rebuilt
    return key[:_MAX_CITATION_KEY_LEN]


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
        first_author_token = author_surname_token(str(authors[0])) or first_author_token

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


_SURNAME_PARTICLES = {"da", "de", "del", "della", "den", "der", "di", "dos", "du", "la", "le"}
_SURNAME_PARTICLES |= {"st", "ten", "ter", "van", "von"}
_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}
_PLACEHOLDER_NAMES = {"anonymous", "na", "notreported", "reported", "unknown"}
# Letters NFKD doesn't decompose to ASCII.
_TRANSLITERATION = str.maketrans(
    {"ß": "ss", "æ": "ae", "Æ": "Ae", "œ": "oe", "Œ": "Oe", "ø": "o", "Ø": "O"}
    | {"ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ı": "i", "þ": "th", "Þ": "Th"}
)


def _first_author_name(authors: object) -> str:
    """First entry of an authors list, without "et al." or co-authors joined by "and"/"&"."""
    if not isinstance(authors, list) or not authors:
        return ""
    name = re.sub(r"\bet al\b\.?", "", str(authors[0]), flags=re.IGNORECASE)
    return re.split(r"\s+and\s+|&|;", name)[0].strip()


def _name_words(name: str) -> list[str]:
    """Lowercase ASCII words of a name; hyphenated parts joined, suffixes dropped."""
    ascii_name = unicodedata.normalize("NFKD", name.translate(_TRANSLITERATION))
    words = (
        re.sub(r"[^a-z0-9]", "", w)
        for w in ascii_name.encode("ascii", "ignore").decode().lower().split()
    )
    return [w for w in words if re.search(r"[a-z]", w) and w not in _NAME_SUFFIXES]


def author_surname_token(author_name: str) -> str:
    """Surname token of "Given Surname" or "Surname, Given", lowercase ASCII.

    Hyphenated surnames and particles are kept whole (Paredes-Vallés →
    paredesvalles, Robin Van den Berghe → vandenberghe); a first word is never a
    particle (Le Song → song).
    """
    if "," in author_name:
        return "".join(_name_words(author_name.split(",")[0]))
    words = _name_words(author_name)
    if not words:
        return ""
    surname = words[-1]
    for word in reversed(words[1:-1]):
        if word not in _SURNAME_PARTICLES:
            break
        surname = word + surname
    return surname


def _extract_year_candidate(value: object) -> int | None:
    """Extract a plausible 4-digit year (1900-2099) from a string value."""
    if not isinstance(value, str):
        return None
    match = re.search(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", value)
    if not match:
        return None
    return int(match.group(1))
