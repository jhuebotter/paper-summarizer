"""Batch processing — scan a directory of PDFs and summarise each one.

Skip detection (v1.1)
---------------------
Two independent skip conditions:

* **Parse step** (handled in ``parser.py``): skip docling re-run if
  ``{pdf_stem}.md`` exists next to the PDF and is non-empty.
* **LLM step** (handled here): skip a PDF entirely if its absolute path
  appears in ``output_summaries/processed.txt``.

Output location (v1.1)
----------------------
Summaries are written to ``{output_dir}/{paper_type}/{citekey}_summary.md``
(centralized, not colocated with the source PDF). If that path already
exists, a version suffix is appended (``_v2``, ``_v3``, ...).
"""

import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

from summarizer.models import BatchReport, Config, FailedPaper, PipelineError
from summarizer.pipeline import process_pdf
from summarizer.renderer import render_summary


# ---------------------------------------------------------------------------
# PDF discovery
# ---------------------------------------------------------------------------


def find_pdfs(source_dir: Path) -> list[Path]:
    """Return all PDF files found recursively under ``source_dir``, sorted."""
    return sorted(source_dir.rglob("*.pdf"))


# ---------------------------------------------------------------------------
# Processed index helpers
# ---------------------------------------------------------------------------


def load_processed_index(output_dir: Path) -> set[str]:
    """Read ``output_dir/processed.txt`` and return the set of recorded paths.

    Returns an empty set if the file does not exist.
    """
    index_path = output_dir / "processed.txt"
    if not index_path.exists():
        return set()
    lines = index_path.read_text(encoding="utf-8").splitlines()
    return {line for line in lines if line.strip()}


def save_processed_index(output_dir: Path, paths: set[str]) -> None:
    """Write ``paths`` to ``output_dir/processed.txt``, one path per line."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "processed.txt").write_text(
        "\n".join(sorted(paths)) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Skip logic
# ---------------------------------------------------------------------------


def should_skip(pdf_path: Path, processed: set[str], force_summary: bool) -> bool:
    """Return ``True`` if the PDF should be skipped for the LLM step.

    A PDF is skipped when its absolute path is in ``processed`` and
    ``force_summary=False``.

    Args:
        pdf_path:  Path to the PDF being evaluated.
        processed: Set of absolute path strings loaded from processed.txt.
        force_summary: If True, always return False (never skip).
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
        paper_type:   One of ``"primary"``, ``"survey"``, ``"commentary"``,
                      ``"non_research"``.
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
    pdf_path: Path, config: Config, run_idx: int, run_total: int
) -> dict:
    """Worker task: process one PDF and return renderable artifacts."""
    logger.info("  Processing [%d/%d]: %s", run_idx, run_total, pdf_path.name)
    summary = process_pdf(pdf_path, config)
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
    """Process all PDFs under ``source_dir`` and return an aggregate report.

    For each PDF:

    1. Load the processed index from ``config.output_dir/processed.txt``.
    2. Check ``should_skip`` — increment ``skipped`` and move on if True
       (or if ``config.dry_run`` is set).
    3. On ``--force-summary``, remove the PDF from the index before re-processing so
       it is re-added after a successful run.
    4. Submit eligible PDFs to a worker pool and process them concurrently.
    5. Render to markdown and write to ``get_output_path(…)``.
    6. Append the absolute PDF path to the processed index on success.
    7. On ``PipelineError``, record the failure and continue; do NOT update
       the index (so the next run retries the paper).

    Args:
        source_dir: Directory to scan for PDFs (recursive).
        config:     Runtime configuration.

    Returns:
        A ``BatchReport`` with counts and details of failed papers.
    """
    pdfs = find_pdfs(source_dir)
    total = len(pdfs)

    # Load index once at the start of the batch
    processed_set = load_processed_index(config.output_dir)
    logger.info("Discovered PDFs: %d", total)
    logger.info("Processed-index entries loaded: %d", len(processed_set))
    if config.force_summary:
        logger.info(
            "force-summary enabled: processed index is ignored for skip filtering"
        )

    n_processed = 0
    n_skipped = 0
    n_failed = 0
    failed_papers: list[FailedPaper] = []

    futures_to_path: dict[Any, tuple[Path, int]] = {}
    jobs: list[Path] = []
    n_skipped_by_index = 0

    show_progress = sys.stderr.isatty()

    for pdf_path in tqdm(
        pdfs,
        total=total,
        desc="Filter",
        unit="pdf",
        disable=not show_progress,
        leave=False,
    ):
        abs_path = str(pdf_path.resolve())

        if should_skip(pdf_path, processed_set, config.force_summary):
            n_skipped += 1
            n_skipped_by_index += 1
            continue

        # On --force-summary: remove from index so it is re-added after success
        processed_set.discard(abs_path)
        jobs.append(pdf_path)

        if config.dry_run:
            n_skipped += 1
            continue

    logger.info("Selected for processing: %d", len(jobs))
    logger.info("Skipped by processed index: %d", n_skipped_by_index)
    if config.dry_run:
        logger.info("Dry run mode: %d files would be processed", len(jobs))
        return BatchReport(
            processed=n_processed,
            skipped=n_skipped,
            failed=n_failed,
            failed_papers=failed_papers,
        )

    with ThreadPoolExecutor(
        max_workers=config.workers,
        thread_name_prefix="worker",
    ) as executor:
        run_total = len(jobs)
        for run_idx, pdf_path in enumerate(jobs, start=1):
            future = executor.submit(
                _process_one_pdf, pdf_path, config, run_idx, run_total
            )
            futures_to_path[future] = (pdf_path, run_idx)

        with tqdm(
            total=run_total,
            desc="Process",
            unit="pdf",
            disable=not show_progress,
            leave=True,
        ) as progress:
            for future in as_completed(futures_to_path):
                pdf_path, run_idx = futures_to_path[future]
                abs_path = str(pdf_path.resolve())
                try:
                    result = future.result()
                    summary = result["summary"]
                    markdown = result["markdown"]
                    paper_category = summary.metadata.paper_type or "non_research"
                    base_output_path = get_output_path(
                        config.output_dir,
                        paper_category,
                        summary.metadata.citation_key,
                    )
                    output_path = get_versioned_output_path(base_output_path)
                    output_path.write_text(markdown, encoding="utf-8")
                    processed_set.add(abs_path)
                    save_processed_index(config.output_dir, processed_set)
                    logger.info(
                        "  [%d/%d] Written: %s", run_idx, run_total, output_path
                    )
                    n_processed += 1
                except PipelineError as exc:
                    logger.error("  [%d/%d] Failed: %s", run_idx, run_total, exc)
                    n_failed += 1
                    failed_papers.append(
                        FailedPaper(pdf_path=str(pdf_path), error=str(exc))
                    )
                except Exception as exc:
                    logger.error("  [%d/%d] Failed: %s", run_idx, run_total, exc)
                    n_failed += 1
                    failed_papers.append(
                        FailedPaper(pdf_path=str(pdf_path), error=str(exc))
                    )
                finally:
                    progress.update(1)
                    progress.set_postfix(ok=n_processed, failed=n_failed)

    return BatchReport(
        processed=n_processed,
        skipped=n_skipped,
        failed=n_failed,
        failed_papers=failed_papers,
    )
