"""Command-line interface for the paper summarizer.

Entry point: ``summarize-papers`` (configured in ``pyproject.toml``).

Usage:
    summarize-papers --source DIR [options]   # batch mode
    summarize-papers --file PDF [options]     # single-file mode

Key options:
    --force-summary, --reparse, --extractor, --dry-run,
    --output-dir, --model, --base-url, --max-chars,
    --skill-data-dir, --verbose/--no-verbose,
    --log-file, --timeout, --workers, --max-output-tokens.

``--source`` and ``--file`` are mutually exclusive; exactly one must be supplied.
``--reparse`` implies ``--force-summary``.

Before processing (except in dry-run mode), the CLI performs a lightweight
reachability check against the root host of the configured ``--base-url``.
"""

import argparse
import logging
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from summarizer.batch import (
    get_output_path,
    load_processed_index,
    run_batch,
    save_processed_index,
    should_skip,
)
from summarizer.log import setup_logging
from summarizer.models import Config, PipelineError, _DEFAULT_MAX_CHARS
from summarizer.pipeline import process_pdf
from summarizer.renderer import render_summary

logger = logging.getLogger(__name__)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments, validate environment, and run the summarizer."""
    load_dotenv()

    parser = _build_parser()
    args = parser.parse_args()

    # Configure logging before any other output
    if args.log_file:
        log_file = Path(args.log_file)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path("logs") / f"run_{ts}.log"
    setup_logging(verbose=args.verbose, log_file=log_file)

    # --reparse implies --force-summary
    force_summary = args.force_summary or args.reparse

    config = Config(
        base_url=args.base_url,
        model=args.model,
        max_chars=args.max_chars,
        force_summary=force_summary,
        reparse=args.reparse,
        extractor=args.extractor,
        dry_run=args.dry_run,
        output_dir=Path(args.output_dir),
        skill_data_dir=Path(args.skill_data_dir),
        verbose=args.verbose,
        timeout_s=args.timeout,
        max_output_tokens=args.max_output_tokens,
        workers=args.workers,
    )

    # Validate LM Studio is reachable before starting any work
    if not args.dry_run:
        _check_lm_studio(config.base_url)

    if args.file:
        _run_single(Path(args.file), config)
    else:
        _run_batch(Path(args.source), config)


# ---------------------------------------------------------------------------
# Single-file mode
# ---------------------------------------------------------------------------


def _run_single(pdf_path: Path, config: Config) -> None:
    """Process a single PDF and write the output to the centralized output dir."""
    if not pdf_path.exists():
        logger.error("File not found: %s", pdf_path)
        sys.exit(1)

    processed = load_processed_index(config.output_dir)
    if should_skip(pdf_path, processed, config.force_summary):
        logger.info(
            "Already processed: %s (use --force-summary to reprocess)",
            pdf_path.name,
        )
        sys.exit(0)

    logger.info("Processing: %s", pdf_path.name)
    try:
        summary = process_pdf(pdf_path, config)
    except PipelineError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    markdown = render_summary(summary)
    paper_category = summary.metadata.paper_type or "non_research"
    output_path = get_output_path(
        config.output_dir,
        paper_category,
        summary.metadata.citation_key,
    )
    output_path.write_text(markdown, encoding="utf-8")
    processed.add(str(pdf_path.resolve()))
    save_processed_index(config.output_dir, processed)
    logger.info("Written: %s", output_path)


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------


def _run_batch(source_dir: Path, config: Config) -> None:
    """Scan ``source_dir`` for PDFs and process each one."""
    if not source_dir.exists():
        logger.error("Directory not found: %s", source_dir)
        sys.exit(1)

    report = run_batch(source_dir, config)

    logger.info(
        "Done — processed: %d, skipped: %d, failed: %d",
        report.processed,
        report.skipped,
        report.failed,
    )

    if report.failed_papers:
        logger.error("Failed papers:")
        for fp in report.failed_papers:
            logger.error("  %s: %s", fp.pdf_path, fp.error)
        sys.exit(1)


# ---------------------------------------------------------------------------
# LM Studio health check
# ---------------------------------------------------------------------------


def _check_lm_studio(base_url: str) -> None:
    """Verify that the LLM backend (LM Studio or OpenRouter) is reachable."""
    parsed = urllib.parse.urlparse(base_url)
    health_url = f"{parsed.scheme}://{parsed.netloc}"
    try:
        with urllib.request.urlopen(health_url, timeout=5):
            pass
    except urllib.error.HTTPError:
        # Any HTTP response (4xx/5xx) means the server is up; auth errors are expected
        # for cloud backends like OpenRouter when hitting the root URL unauthenticated.
        return
    except Exception as exc:
        logger.error("Cannot reach LLM backend at %s\n  Details: %s", health_url, exc)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="summarize-papers",
        description=(
            "Summarise research papers using a local LLM via LM Studio. "
            "Processes a directory of PDFs (--source) or a single PDF (--file)."
        ),
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source",
        metavar="DIR",
        help="Directory to scan recursively for PDF files.",
    )
    source_group.add_argument(
        "--file",
        metavar="PDF",
        help="Path to a single PDF file to process.",
    )

    parser.add_argument(
        "--force-summary",
        action="store_true",
        default=False,
        help="Re-run summary generation for PDFs in processed.txt; preserves extraction cache.",
    )
    parser.add_argument(
        "--reparse",
        action="store_true",
        default=False,
        help="Re-run extraction and summary generation (implies --force-summary).",
    )
    parser.add_argument(
        "--extractor",
        choices=["auto", "docling", "pypdf"],
        default="auto",
        help="PDF extraction backend strategy (default: auto).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="List PDFs that would be processed without calling the LLM.",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default="output_summaries",
        help="Root directory for centralized summary output (default: output_summaries).",
    )
    _default_model = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b:free")
    parser.add_argument(
        "--model",
        metavar="MODEL",
        default=_default_model,
        help=f"LLM model identifier (default: LLM_MODEL env var, currently {_default_model!r}).",
    )
    parser.add_argument(
        "--base-url",
        metavar="URL",
        default="https://openrouter.ai/api/v1",
        help="OpenAI-compatible API base URL (default: https://openrouter.ai/api/v1).",
    )
    parser.add_argument(
        "--max-chars",
        metavar="N",
        type=int,
        default=_DEFAULT_MAX_CHARS,
        help=(
            f"Maximum characters of paper text sent to the LLM "
            f"(default: {_DEFAULT_MAX_CHARS:,} ≈ 50k tokens)."
        ),
    )
    parser.add_argument(
        "--skill-data-dir",
        metavar="DIR",
        default="skill_data/references",
        help=(
            "Directory containing reference .md files "
            "(output-template, extraction fields, learning paradigms). "
            "Default: skill_data/references"
        ),
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable DEBUG-level logging (default: on). Use --no-verbose to suppress.",
    )
    parser.add_argument(
        "--log-file",
        metavar="FILE",
        default=None,
        help="Write log output to FILE (default: logs/run_TIMESTAMP.log).",
    )
    parser.add_argument(
        "--timeout",
        metavar="S",
        type=int,
        default=120,
        help="LLM call timeout in seconds (default: 120). Use higher values for slow local models.",
    )
    parser.add_argument(
        "--workers",
        metavar="N",
        type=_positive_int,
        default=3,
        help="Number of parallel workers for batch mode (default: 3).",
    )
    parser.add_argument(
        "--max-output-tokens",
        metavar="N",
        type=int,
        default=None,
        help=(
            "Maximum tokens the LLM may generate per call. "
            "Default: no limit (model stops on its own). "
            "Set when the backend enforces a cap or to bound cost."
        ),
    )

    return parser


if __name__ == "__main__":
    main()
