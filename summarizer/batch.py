"""Batch processing — summarise a list of PDFs (a directory scan or a single file).

Processed index
---------------
``{output_dir}/processed.jsonl`` has one JSON object per paper:
``{"sha256": ..., "pdf_path": ..., "outputs": [...]}``.  Papers are identified
by content (sha256), so moved, renamed or duplicated PDFs are not summarized
again; entries from older versions (path only, or the comma-separated
``processed.txt``) gain their sha256 the next time their PDF is seen, unless
the PDF changed after it was summarized.  Writes are atomic, and (on POSIX) a
lock file stops two runs from sharing an output directory.

Output location
---------------
Each summary is written as ``{output_dir}/{paper_type}/{citekey}_summary.md``
plus a ``.json`` sidecar with the validated data and provenance (re-render with
``summarize-papers render``).  If the path already exists, a version suffix is
appended (``_v2``, ``_v3``, ...).
"""

import json
import logging
import os
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from pathlib import Path

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from summarizer.llm import CostAccumulator, QuotaExhausted, create_client
from summarizer.models import BatchReport, Config, FailedPaper, PaperSummary, PipelineError
from summarizer.parser import sha256_file
from summarizer.pipeline import process_pdf
from summarizer.prompts import load_references
from summarizer.renderer import render_summary

try:
    import fcntl
except ImportError:  # Windows: no advisory locks; runs must not overlap
    fcntl = None

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


# ---------------------------------------------------------------------------
# Processed index helpers
# ---------------------------------------------------------------------------


def load_processed_index(output_dir: Path) -> dict[str, dict]:
    """Return processed papers keyed by sha256 (or by path for older entries).

    Each value is ``{"pdf_path": str, "outputs": [str], "sha256": str | None}``.
    Reads ``processed.jsonl``; falls back to the legacy ``processed.txt``.
    Returns an empty dict if neither exists.
    """
    index_path = output_dir / INDEX_FILENAME
    if index_path.exists():
        result: dict[str, dict] = {}
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
            sha = entry.get("sha256") if isinstance(entry.get("sha256"), str) else None
            result[sha or entry["pdf_path"]] = _record(
                entry["pdf_path"], entry.get("outputs") or [], sha
            )
        return result

    legacy_path = output_dir / LEGACY_INDEX_FILENAME
    if legacy_path.exists():
        logger.info(
            "Reading legacy %s; it will be migrated to %s on the next save",
            LEGACY_INDEX_FILENAME,
            INDEX_FILENAME,
        )
        return {
            path: _record(path, outputs, None)
            for path, outputs in _load_legacy_index(legacy_path).items()
        }
    return {}


def _record(pdf_path: str, outputs: list[str], sha256: str | None) -> dict:
    return {"pdf_path": pdf_path, "outputs": list(outputs), "sha256": sha256}


def _migrate_path_entry(index: dict[str, dict], pdf_path: Path, sha: str) -> bool:
    """Re-key an old path-only entry to ``sha``; return True if the index changed.

    If the PDF was modified after its summaries were written, the entry is
    dropped instead, so the (probably different) paper is summarized again.
    """
    path_key = str(pdf_path.resolve())
    if sha in index or path_key not in index:
        return False
    entry = index.pop(path_key)
    outputs = [Path(o) for o in entry["outputs"] if Path(o).exists()]
    if outputs and pdf_path.stat().st_mtime > max(o.stat().st_mtime for o in outputs):
        logger.info("%s changed since it was summarized; summarizing it again", pdf_path.name)
    else:
        index[sha] = entry | {"sha256": sha}
    return True


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


def save_processed_index(output_dir: Path, index: dict[str, dict]) -> None:
    """Atomically write ``index`` to ``output_dir/processed.jsonl``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({k: v for k, v in record.items() if v is not None}, ensure_ascii=False)
        for record in sorted(index.values(), key=lambda r: r["pdf_path"])
    ]
    atomic_write_text(output_dir / INDEX_FILENAME, "\n".join(lines) + "\n" if lines else "")


def atomic_write_text(path: Path, text: str) -> None:
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


def should_skip(sha256: str, processed: dict[str, dict], force_summary: bool) -> bool:
    """Return ``True`` if a PDF with this content was already summarized."""
    return not force_summary and sha256 in processed


class OutputDirLocked(Exception):
    """Another run is using the output directory."""


@contextmanager
def output_dir_lock(output_dir: Path):
    """Hold an exclusive lock on ``output_dir`` for the duration of a run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".lock").open("w") as handle:
        if fcntl is not None:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise OutputDirLocked(f"Another run is using {output_dir}") from None
        yield


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

    If neither ``path`` nor its ``.json`` sidecar exists, ``path`` is returned
    unchanged; otherwise ``_v2``, ``_v3``, ... are appended before the suffix.
    """

    def taken(candidate: Path) -> bool:
        return candidate.exists() or candidate.with_suffix(".json").exists()

    if not taken(path):
        return path
    version = 2
    while taken(candidate := path.with_name(f"{path.stem}_v{version}{path.suffix}")):
        version += 1
    return candidate


class StopSignal:
    """Thread-safe "start no new papers" flag that keeps the first reason given."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reason: str | None = None

    def trip(self, reason: str) -> None:
        with self._lock:
            if self.reason is None:
                self.reason = reason
                logger.warning("Stopping: %s", reason)

    def is_set(self) -> bool:
        return self.reason is not None

    def check_budget(self, accumulator: CostAccumulator, max_cost: float | None) -> None:
        if max_cost is not None and accumulator.total_cost >= max_cost:
            self.trip(f"--max-cost ${max_cost:g} reached")


class _Stopped(Exception):
    """The run was stopped (quota or --max-cost) before this paper started."""


def _process_one_pdf(
    pdf_path: Path,
    config: Config,
    run_idx: int,
    run_total: int,
    client,
    accumulator: CostAccumulator,
    references: str,
    stop: StopSignal,
) -> dict:
    """Worker task: process one PDF and return renderable artifacts."""
    stop.check_budget(accumulator, config.max_cost)
    if stop.is_set():
        raise _Stopped
    logger.info("  Processing [%d/%d]: %s", run_idx, run_total, pdf_path.name)
    try:
        summary = process_pdf(
            pdf_path, config, client=client, accumulator=accumulator, references=references
        )
    except PipelineError as exc:
        if isinstance(exc.cause, QuotaExhausted):
            stop.trip(str(exc.cause))
        raise
    markdown = render_summary(summary)
    return {
        "pdf_path": pdf_path,
        "summary": summary,
        "markdown": markdown,
    }


def render_all(output_dir: Path) -> tuple[int, int]:
    """Re-render every summary's markdown from its JSON sidecar.

    No LLM calls: use this after changing the renderer or template.  Returns
    ``(rendered, failed)``; unreadable sidecars are logged and skipped.
    """
    rendered = failed = 0
    with output_dir_lock(output_dir):
        for json_path in sorted(output_dir.rglob("*_summary*.json")):
            try:
                summary = PaperSummary.model_validate_json(json_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.error("Cannot render %s: %s", json_path, exc)
                failed += 1
                continue
            atomic_write_text(json_path.with_suffix(".md"), render_summary(summary))
            rendered += 1
    return rendered, failed


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def run_batch(source_dir: Path, config: Config) -> BatchReport:
    """Process all PDFs found recursively under ``source_dir``."""
    return run_pdfs(find_pdfs(source_dir), config)


def run_pdfs(pdfs: list[Path], config: Config) -> BatchReport:
    """Process ``pdfs`` and return an aggregate report.

    1. Load the processed index once.
    2. Drop PDFs that ``should_skip`` (unless ``force_summary``) and duplicates
       (same content) within the batch; in dry-run mode stop after logging what
       would be processed.
    3. Process the rest concurrently (``config.workers``) with one shared LLM
       client, cost accumulator and reference text.
    4. On success, write ``get_versioned_output_path(...)`` and its ``.json``
       sidecar and record them in the index (saved after every paper, so an
       interrupted run keeps its progress).
    5. On failure, record it and continue; the index is not updated, so the
       next run retries the paper.
    6. Stop early (cancel queued papers, count them as skipped) when the
       backend reports an exhausted quota or ``config.max_cost`` is reached.

    Holds the output-directory lock, except in dry-run mode.

    Raises:
        OutputDirLocked: if another run is using ``config.output_dir``.
    """
    with nullcontext() if config.dry_run else output_dir_lock(config.output_dir):
        return _run_pdfs(pdfs, config)


def _run_pdfs(pdfs: list[Path], config: Config) -> BatchReport:
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
    shas: dict[Path, str] = {}
    seen: set[str] = set()
    n_skipped_by_index = 0
    migrated = False

    for pdf_path in pdfs:
        try:
            sha = sha256_file(pdf_path)
        except OSError as exc:
            logger.error("Cannot read %s: %s", pdf_path, exc)
            n_failed += 1
            failed_papers.append(FailedPaper(pdf_path=str(pdf_path), error=str(exc)))
            continue
        migrated |= _migrate_path_entry(processed_set, pdf_path, sha)
        if sha in seen:
            logger.info("Skipping %s: same content as another PDF in this batch", pdf_path.name)
            n_skipped += 1
            continue
        seen.add(sha)
        shas[pdf_path] = sha
        if should_skip(sha, processed_set, config.force_summary):
            n_skipped += 1
            n_skipped_by_index += 1
            continue
        jobs.append(pdf_path)
    if migrated and not config.dry_run:
        save_processed_index(config.output_dir, processed_set)

    logger.info("Selected for processing: %d", len(jobs))
    logger.info("Skipped by processed index: %d", n_skipped_by_index)
    if config.dry_run:
        for pdf_path in jobs:
            logger.info("  would process: %s", pdf_path)
        logger.info("Dry run mode: %d files would be processed", len(jobs))
        return BatchReport(
            processed=0,
            skipped=n_skipped + len(jobs),
            failed=n_failed,
            failed_papers=failed_papers,
        )
    if not jobs:
        return BatchReport(
            processed=0, skipped=n_skipped, failed=n_failed, failed_papers=failed_papers
        )

    # Shared across workers: one client, one cost accumulator, one reference text.
    client = create_client(config)
    accumulator = CostAccumulator()
    references = load_references(config.skill_data_dir)

    show_progress = sys.stderr.isatty()
    run_total = len(jobs)
    stop = StopSignal()
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
                stop,
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
                    atomic_write_text(
                        output_path.with_suffix(".json"), summary.model_dump_json(indent=2)
                    )
                    atomic_write_text(output_path, result["markdown"])
                    sha = shas[pdf_path]
                    record = processed_set.setdefault(sha, _record(abs_path, [], sha))
                    record["pdf_path"] = abs_path
                    record["outputs"].append(str(output_path))
                    save_processed_index(config.output_dir, processed_set)
                    logger.info("  [%d/%d] Written: %s", run_idx, run_total, output_path)
                    n_processed += 1
                except _Stopped:
                    n_skipped += 1
                except Exception as exc:
                    if isinstance(getattr(exc, "cause", None), QuotaExhausted):
                        n_skipped += 1
                        continue
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
        stopped_reason=stop.reason,
    )
