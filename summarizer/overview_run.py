"""Overview extraction: fill one ``OverviewRecord`` per paper and store it.

This is separate from the summary pipeline. A summary is prose plus a few
labels for one reader; an overview record is a set of typed facts per paper
meant to be compared across the whole library (``overview_tables.py``). Each
paper costs one LLM call: the codebook (``skill_data/overview/codebook.md``), a
JSON template generated from the schema, and the paper text. An optional
second "critic" call has the model check its own draft against the paper.

After validation (with one repair call if needed), the codebook's cross-field
rules are applied (``overview.normalize``). Evidence quotes are checked
against the paper text, and the labels used in the review are derived
(``overview.derive``). Everything is written to
``{output_dir}/records/{name}.json`` and collected in ``overview.jsonl``.

A paper is skipped when a record for the same PDF content, model and codebook
already exists (``--force`` re-runs it). Eval and summary outputs are never
touched.
"""

import hashlib
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, get_args, get_origin

from pydantic import BaseModel, ValidationError
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from summarizer.batch import StopSignal, atomic_write_text, output_dir_lock
from summarizer.llm import CostAccumulator, QuotaExhausted, call_llm, create_client
from summarizer.models import (
    _DEFAULT_MAX_CHARS,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_SKILL_DATA_DIR,
    BatchReport,
    Config,
    FailedPaper,
)
from summarizer.overview import (
    Component,
    Metrics,
    OverviewRecord,
    derive,
    missing_quotes,
    normalize,
)
from summarizer.parser import load_text, sha256_file, truncate_text
from summarizer.zotero import ZoteroRecord, lookup_all

logger = logging.getLogger(__name__)

DEFAULT_CODEBOOK = DEFAULT_SKILL_DATA_DIR.parent / "overview" / "codebook.md"
RECORDS_DIRNAME = "records"
COLLECTED_FILENAME = "overview.jsonl"
#: Overview prompts carry the codebook (~6k tokens) and the paper; most papers
#: fit well under this.
_OVERVIEW_MAX_CHARS = 120_000


@dataclass(frozen=True)
class OverviewConfig:
    """Runtime settings for ``summarize-papers overview``."""

    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    output_dir: Path = Path("output_overview")
    codebook: Path = DEFAULT_CODEBOOK
    max_chars: int = _OVERVIEW_MAX_CHARS
    extractor: Literal["auto", "docling", "pypdf"] = "auto"
    strip_references: bool = True
    critic: bool = False
    workers: int = 3
    timeout_s: int = 600
    max_cost: float | None = None
    force: bool = False
    zotero: bool = True

    def llm_config(self) -> Config:
        return Config(
            base_url=self.base_url,
            model=self.model,
            timeout_s=self.timeout_s,
            max_chars=max(self.max_chars, _DEFAULT_MAX_CHARS),
        )


class OverviewResult(BaseModel):
    """One paper's stored overview: the record plus how it was obtained."""

    sha256: str
    file: str
    citation_key: str = ""
    title: str = ""
    authors: list[str] = []
    year: int | None = None
    record: OverviewRecord
    derived: dict
    normalized: list[str] = []
    missing_quotes: list[str] = []
    draft: OverviewRecord | None = None
    model: str
    codebook_sha256: str
    extractor: str
    critic: bool = False
    cost_usd: float = 0.0
    calls: int = 0
    created: str = ""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def _template(model: type[BaseModel]) -> dict:
    """A JSON template of ``model`` listing the allowed options of every field."""
    out: dict = {}
    for name, field in model.model_fields.items():
        annotation = field.annotation
        if get_origin(annotation) is list:
            inner = get_args(annotation)[0]
            out[name] = (
                [_template(Component)]
                if inner is Component
                else ["<any of: " + " | ".join(get_args(inner)) + ">"]
            )
        elif annotation is Metrics:
            out[name] = _template(Metrics)
        elif get_origin(annotation) is Literal:
            out[name] = "<one of: " + " | ".join(get_args(annotation)) + ">"
        elif annotation is bool:
            out[name] = "<true | false>"
        else:
            out[name] = "<string>"
    return out


TEMPLATE = json.dumps(_template(OverviewRecord), indent=1, ensure_ascii=False)

_PREAMBLE = (
    "You extract structured facts about one research paper for a literature review on "
    "spiking neural networks for control."
)
_OUTPUT_RULES = (
    "Output rules: JSON only, no markdown fences or prose. Evidence strings must be verbatim "
    'quotes from the paper text below (short), or "" when there is no support.'
)


def build_prompt(paper_text: str, codebook: str) -> str:
    return f"""{_PREAMBLE}

Follow this codebook exactly:

{codebook}

---

Return exactly ONE JSON object with this structure (replace every <...> by a value; option values \
must be copied exactly; lists may have several or zero entries; `components` lists every component \
as described in section 2 of the codebook):

{TEMPLATE}

{_OUTPUT_RULES}

---

Paper text:
{paper_text}"""


def build_critic_prompt(paper_text: str, codebook: str, draft: dict) -> str:
    return f"""You are checking a draft structured record of one research paper for a literature \
review on spiking neural networks for control.

Codebook (the definitions to apply exactly):

{codebook}

---

Draft record:
{json.dumps(draft, indent=1, ensure_ascii=False)}

---

Task: check every field of the draft against the paper text and the codebook definitions, \
especially the worked edge cases. Correct every wrong value, add missing components and remove \
components that should not be listed; keep values that are right.

Return the full corrected record as ONE JSON object with the same structure. {_OUTPUT_RULES}

---

Paper text:
{paper_text}"""


def build_repair_prompt(raw: dict, error: ValidationError) -> str:
    problems = "\n".join(
        f"- {'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors()[:30]
    )
    return f"""The JSON below does not satisfy the required schema. Fix ONLY the listed problems \
(use the exact option values from the template) and return the full corrected JSON object, \
nothing else.

Problems:
{problems}

Template (allowed values):
{TEMPLATE}

JSON to fix:
{json.dumps(raw, indent=1, ensure_ascii=False)}"""


# ---------------------------------------------------------------------------
# One paper
# ---------------------------------------------------------------------------


def codebook_digest(codebook: str) -> str:
    return hashlib.sha256(codebook.encode()).hexdigest()


def _validated(client, raw: dict, accumulator: CostAccumulator) -> OverviewRecord:
    """Validate ``raw``; on failure send one repair request (without the paper)."""
    try:
        return OverviewRecord.model_validate(raw)
    except ValidationError as exc:
        logger.warning("Overview record failed validation; sending one repair request")
        accumulator.note_schema_repair()
        repaired = call_llm(client, build_repair_prompt(raw, exc), accumulator)
        return OverviewRecord.model_validate(repaired)


def extract_record(
    pdf_path: Path,
    config: OverviewConfig,
    client,
    accumulator: CostAccumulator,
    codebook: str,
    zotero: ZoteroRecord | None = None,
) -> OverviewResult:
    """Extract, validate, normalize and quote-check one paper's overview record."""
    parsed = load_text(
        pdf_path, extractor=config.extractor, strip_references=config.strip_references
    )
    text = truncate_text(parsed.text, config.max_chars, pdf_path.name)
    paper = CostAccumulator(parent=accumulator)
    record = _validated(client, call_llm(client, build_prompt(text, codebook), paper), paper)
    draft = None
    if config.critic:
        draft = record
        raw = call_llm(client, build_critic_prompt(text, codebook, record.model_dump()), paper)
        try:
            record = _validated(client, raw, paper)
        except (ValidationError, ValueError):
            logger.warning("Critic reply unusable for %s; keeping the draft", pdf_path.name)
            record = draft
    record, changes = normalize(record)
    missing = missing_quotes(record, parsed.text)
    if missing:
        logger.info("%s: %d evidence quote(s) not found in the text", pdf_path.name, len(missing))
    return OverviewResult(
        sha256=parsed.sha256,
        file=pdf_path.name,
        citation_key=zotero.citation_key if zotero else "",
        title=zotero.title if zotero else "",
        authors=zotero.authors if zotero else [],
        year=zotero.year if zotero else None,
        record=record,
        derived=derive(record),
        normalized=changes,
        missing_quotes=missing,
        draft=draft,
        model=client.model,
        codebook_sha256=codebook_digest(codebook),
        extractor=parsed.extractor,
        critic=config.critic,
        cost_usd=paper.total_cost,
        calls=paper.calls,
        created=datetime.now(UTC).isoformat(timespec="seconds"),
    )


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


def record_path(output_dir: Path, pdf_path: Path, sha: str) -> Path:
    """``records/<Zotero key or sha prefix>.json``: stable across renames of the PDF."""
    stem = pdf_path.name.split("__", 1)[0] if "__" in pdf_path.name else sha[:16]
    return output_dir / RECORDS_DIRNAME / f"{stem}.json"


def load_results(output_dir: Path) -> list[OverviewResult]:
    """Every stored record under ``output_dir`` (unreadable files are logged and skipped)."""
    results = []
    for path in sorted((output_dir / RECORDS_DIRNAME).glob("*.json")):
        try:
            results.append(OverviewResult.model_validate_json(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            logger.error("Cannot read %s: %s", path, exc)
    return results


def _is_current(path: Path, sha: str, model: str, codebook_sha: str, critic: bool) -> bool:
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        stored.get("sha256") == sha
        and stored.get("model") == model
        and stored.get("codebook_sha256") == codebook_sha
        and stored.get("critic", False) == critic
    )


def _write_collected(output_dir: Path) -> None:
    lines = [r.model_dump_json() for r in load_results(output_dir)]
    atomic_write_text(output_dir / COLLECTED_FILENAME, "\n".join(lines) + ("\n" if lines else ""))


def run_overview(pdfs: list[Path], config: OverviewConfig) -> BatchReport:
    """Extract an overview record for each PDF that has no current record yet.

    Holds the output-directory lock. Stops starting new papers when the
    backend reports an exhausted quota or ``config.max_cost`` is reached.
    """
    with output_dir_lock(config.output_dir):
        return _run_overview(pdfs, config)


def _run_overview(pdfs: list[Path], config: OverviewConfig) -> BatchReport:
    codebook = config.codebook.read_text(encoding="utf-8")
    codebook_sha = codebook_digest(codebook)
    jobs: list[tuple[Path, Path]] = []
    failed: list[FailedPaper] = []
    skipped = 0
    seen: set[str] = set()
    for pdf in pdfs:
        try:
            sha = sha256_file(pdf)
        except OSError as exc:
            failed.append(FailedPaper(pdf_path=str(pdf), error=str(exc)))
            continue
        if sha in seen:
            skipped += 1
            continue
        seen.add(sha)
        out = record_path(config.output_dir, pdf, sha)
        if not config.force and _is_current(out, sha, config.model, codebook_sha, config.critic):
            skipped += 1
            continue
        jobs.append((pdf, out))
    logger.info("Overview: %d to extract, %d already current", len(jobs), skipped)
    if not jobs:
        return BatchReport(processed=0, skipped=skipped, failed=len(failed), failed_papers=failed)

    client = create_client(config.llm_config())
    accumulator = CostAccumulator()
    zotero = lookup_all([pdf for pdf, _ in jobs]) if config.zotero else {}
    stop = StopSignal()
    processed = 0

    def work(pdf: Path) -> OverviewResult | None:
        stop.check_budget(accumulator, config.max_cost)
        if stop.is_set():
            return None
        try:
            return extract_record(pdf, config, client, accumulator, codebook, zotero.get(pdf))
        except QuotaExhausted as exc:
            stop.trip(str(exc))
            return None

    t0 = time.monotonic()
    executor = ThreadPoolExecutor(max_workers=config.workers, thread_name_prefix="overview")
    try:
        futures = {executor.submit(work, pdf): (pdf, out) for pdf, out in jobs}
        with (
            logging_redirect_tqdm(loggers=[logging.getLogger("summarizer")]),
            tqdm(
                total=len(jobs), desc="Overview", unit="pdf", disable=not sys.stderr.isatty()
            ) as progress,
        ):
            for future in as_completed(futures):
                pdf, out = futures[future]
                try:
                    result = future.result()
                    if result is None:
                        skipped += 1
                        continue
                    out.parent.mkdir(parents=True, exist_ok=True)
                    atomic_write_text(out, result.model_dump_json(indent=2))
                    processed += 1
                except Exception as exc:  # one bad paper must not stop the run
                    logger.error("Overview failed for %s: %s", pdf.name, exc)
                    failed.append(FailedPaper(pdf_path=str(pdf), error=str(exc)))
                finally:
                    progress.update(1)
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown()
    _write_collected(config.output_dir)
    logger.info("Overview finished in %.0fs", time.monotonic() - t0)
    return BatchReport(
        processed=processed,
        skipped=skipped,
        failed=len(failed),
        failed_papers=failed,
        total_cost=accumulator.total_cost,
        input_tokens=accumulator.total_input_tokens,
        output_tokens=accumulator.total_output_tokens,
        stopped_reason=stop.reason,
    )
