"""Batch processing — summarise a list of PDFs (a directory scan or a single file).

Skip detection
--------------
Two independent skip conditions:

* **Parse step** (handled in ``parser.py``): reuse a cached extraction next to
  the PDF when one exists.
* **LLM step** (handled here): skip a PDF entirely if its absolute path is in
  the processed index ``{output_dir}/processed.jsonl``.

Processed index
---------------
One JSON object per line: ``{"pdf_path": ..., "outputs": [...]}``.  A legacy
comma-separated ``processed.txt`` is read when no ``processed.jsonl`` exists;
the next save writes the new format.  Writes are atomic (temp file + rename).
Only one run should use an output directory at a time.

Output location
---------------
Summaries are written to ``{output_dir}/{paper_type}/{citekey}_summary.md``.
If that path already exists, a version suffix is appended (``_v2``, ``_v3``, ...).
"""

import hashlib
import json
import logging
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from summarizer.llm import CostAccumulator, create_client
from summarizer.models import BatchReport, Config, FailedPaper
from summarizer.pipeline import process_pdf
from summarizer.prompts import load_references
from summarizer.renderer import render_summary

logger = logging.getLogger(__name__)

INDEX_FILENAME = "processed.jsonl"
LEGACY_INDEX_FILENAME = "processed.txt"


# ---------------------------------------------------------------------------
# PDF discovery
# ---------------------------------------------------------------------------


def find_pdfs(source_dir: Path) -> list[Path]:
    """Return all PDF files found recursively under ``source_dir``, sorted.

    Matches the ``.pdf`` extension case-insensitively and ignores macOS
    AppleDouble files (``._name.pdf``) that appear on external drives.
    """
    return sorted(
        p
        for p in source_dir.rglob("*")
        if p.suffix.lower() == ".pdf" and not p.name.startswith("._") and p.is_file()
    )


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file's contents."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Processed index helpers
# ---------------------------------------------------------------------------


def load_processed_index(output_dir: Path) -> dict[str, list[str]]:
    """Return a mapping of absolute PDF path → summary paths already written.

    Reads ``processed.jsonl``; falls back to the legacy ``processed.txt``.
    Returns an empty dict if neither exists.
    """
    index_path = output_dir / INDEX_FILENAME
    if index_path.exists():
        result: dict[str, list[str]] = {}
        for lineno, line in enumerate(index_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                entry = None
            if not isinstance(entry, dict) or not isinstance(entry.get("pdf_path"), str):
                logger.warning("Ignoring malformed line %d in %s", lineno, index_path)
                continue
            result[entry["pdf_path"]] = list(entry.get("outputs") or [])
        return result

    legacy_path = output_dir / LEGACY_INDEX_FILENAME
    if legacy_path.exists():
        logger.info(
            "Reading legacy %s; it will be migrated to %s on the next save",
            LEGACY_INDEX_FILENAME,
            INDEX_FILENAME,
        )
        return _load_legacy_index(legacy_path)
    return {}


def _load_legacy_index(path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            pdf_path, outputs = _parse_legacy_line(line)
            result[pdf_path] = outputs
    return result


def _parse_legacy_line(line: str) -> tuple[str, list[str]]:
    """Parse a legacy ``pdf_path, summary1, summary2`` line.

    Fields are joined with ", ", which may also occur inside paths, so segments
    are re-joined until they end in ``.pdf`` (the source) or ``.md`` (each summary).
    """
    parts = line.strip().split(", ")
    i = next((n for n, part in enumerate(parts) if part.lower().endswith(".pdf")), None)
    if i is None:
        return line.strip(), []

    pdf_path = ", ".join(parts[: i + 1]).strip()
    outputs: list[str] = []
    current: list[str] = []
    for segment in parts[i + 1 :]:
        current.append(segment)
        if segment.endswith(".md"):
            outputs.append(", ".join(current).strip())
            current = []
    if current:
        outputs.append(", ".join(current).strip())
    return pdf_path, outputs


def save_processed_index(output_dir: Path, index: dict[str, list[str]]) -> None:
    """Atomically write ``index`` to ``output_dir/processed.jsonl``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"pdf_path": pdf_path, "outputs": index[pdf_path]}, ensure_ascii=False)
        for pdf_path in sorted(index)
    ]
    _atomic_write_text(output_dir / INDEX_FILENAME, "\n".join(lines) + "\n" if lines else "")


def _atomic_write_text(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Skip logic
# ---------------------------------------------------------------------------


def should_skip(pdf_path: Path, processed: dict[str, list[str]], force_summary: bool) -> bool:
    """Return ``True`` if the PDF should be skipped for the LLM step.

    A PDF is skipped when its absolute path is in ``processed`` and
    ``force_summary=False``.
    """
    if force_summary:
        return False
    return str(pdf_path.resolve()) in processed


# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------


def get_output_path(output_dir: Path, paper_type: str, citation_key: str) -> Path:
    """Return the output path for a summary and create its parent directory.

    Path: ``output_dir / paper_type / {citation_key}_summary.md``

    Args:
        output_dir:   Root output directory (e.g. ``output_summaries/``).
        paper_type:   ``"primary"``, ``"synthesis"`` or ``"non_research"``.
        citation_key: The citation key inferred by the LLM.
    """
    subdir = output_dir / paper_type
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir / f"{citation_key}_summary.md"


def get_versioned_output_path(path: Path) -> Path:
    """Return a non-clobbering output path by appending a version suffix.

    If ``path`` does not exist, it is returned unchanged.
    If it exists, ``_v2``, ``_v3``, ... are appended before the suffix.
    """
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    version = 2
    while True:
        candidate = path.with_name(f"{stem}_v{version}{suffix}")
        if not candidate.exists():
            return candidate
        version += 1


def _process_one_pdf(
    pdf_path: Path,
    config: Config,
    run_idx: int,
    run_total: int,
    client,
    accumulator: CostAccumulator,
    references: str,
) -> dict:
    """Worker task: process one PDF and return renderable artifacts."""
    logger.info("  Processing [%d/%d]: %s", run_idx, run_total, pdf_path.name)
    summary = process_pdf(
        pdf_path, config, client=client, accumulator=accumulator, references=references
    )
    markdown = render_summary(summary)
    return {
        "pdf_path": pdf_path,
        "summary": summary,
        "markdown": markdown,
    }


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def run_batch(source_dir: Path, config: Config) -> BatchReport:
    """Process all PDFs found recursively under ``source_dir``."""
    return run_pdfs(find_pdfs(source_dir), config)


def run_pdfs(pdfs: list[Path], config: Config) -> BatchReport:
    """Process ``pdfs`` and return an aggregate report.

    1. Load the processed index once.
    2. Drop PDFs that ``should_skip`` (unless ``force_summary``); in dry-run
       mode stop after logging what would be processed.
    3. Process the rest concurrently (``config.workers``) with one shared LLM
       client, cost accumulator and reference text.
    4. On success, write ``get_versioned_output_path(...)`` and record it in
       the index (saved after every paper, so an interrupted run keeps its
       progress).
    5. On failure, record it and continue; the index is not updated, so the
       next run retries the paper.
    """
    total = len(pdfs)

    processed_set = load_processed_index(config.output_dir)
    logger.info("Discovered PDFs: %d", total)
    logger.info("Processed-index entries loaded: %d", len(processed_set))
    if config.force_summary:
        logger.info("force-summary enabled: processed index is ignored for skip filtering")

    n_processed = 0
    n_skipped = 0
    n_failed = 0
    failed_papers: list[FailedPaper] = []

    jobs: list[Path] = []
    n_skipped_by_index = 0

    for pdf_path in pdfs:
        if should_skip(pdf_path, processed_set, config.force_summary):
            n_skipped += 1
            n_skipped_by_index += 1
            continue
        jobs.append(pdf_path)

    logger.info("Selected for processing: %d", len(jobs))
    logger.info("Skipped by processed index: %d", n_skipped_by_index)
    if config.dry_run:
        for pdf_path in jobs:
            logger.info("  would process: %s", pdf_path)
        logger.info("Dry run mode: %d files would be processed", len(jobs))
        return BatchReport(
            processed=0,
            skipped=n_skipped + len(jobs),
            failed=0,
            failed_papers=[],
        )
    if not jobs:
        return BatchReport(processed=0, skipped=n_skipped, failed=0, failed_papers=[])

    # Shared across workers: one client, one cost accumulator, one reference text.
    client = create_client(config)
    accumulator = CostAccumulator()
    references = load_references(config.skill_data_dir)

    show_progress = sys.stderr.isatty()
    run_total = len(jobs)
    executor = ThreadPoolExecutor(max_workers=config.workers, thread_name_prefix="worker")
    try:
        futures_to_path = {
            executor.submit(
                _process_one_pdf,
                pdf_path,
                config,
                run_idx,
                run_total,
                client,
                accumulator,
                references,
            ): (pdf_path, run_idx)
            for run_idx, pdf_path in enumerate(jobs, start=1)
        }
        with (
            logging_redirect_tqdm(loggers=[logging.getLogger("summarizer")]),
            tqdm(
                total=run_total,
                desc="Process",
                unit="pdf",
                disable=not show_progress,
                leave=True,
            ) as progress,
        ):
            for future in as_completed(futures_to_path):
                pdf_path, run_idx = futures_to_path[future]
                abs_path = str(pdf_path.resolve())
                try:
                    result = future.result()
                    summary = result["summary"]
                    paper_category = summary.metadata.paper_type or "non_research"
                    output_path = get_versioned_output_path(
                        get_output_path(
                            config.output_dir,
                            paper_category,
                            summary.metadata.citation_key,
                        )
                    )
                    output_path.write_text(result["markdown"], encoding="utf-8")
                    processed_set.setdefault(abs_path, []).append(str(output_path))
                    save_processed_index(config.output_dir, processed_set)
                    logger.info("  [%d/%d] Written: %s", run_idx, run_total, output_path)
                    n_processed += 1
                except Exception as exc:
                    logger.error("  [%d/%d] Failed: %s", run_idx, run_total, exc)
                    n_failed += 1
                    failed_papers.append(FailedPaper(pdf_path=str(pdf_path), error=str(exc)))
                finally:
                    progress.update(1)
                    progress.set_postfix(
                        ok=n_processed,
                        failed=n_failed,
                        cost=f"${accumulator.total_cost:.4f}",
                    )
    except BaseException:
        # Ctrl-C or a crash: cancel queued papers instead of running (and paying for) them.
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown()

    return BatchReport(
        processed=n_processed,
        skipped=n_skipped,
        failed=n_failed,
        failed_papers=failed_papers,
        total_cost=accumulator.total_cost,
        input_tokens=accumulator.total_input_tokens,
        output_tokens=accumulator.total_output_tokens,
    )
