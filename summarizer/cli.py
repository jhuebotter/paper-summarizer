"""Command-line interface for the paper summarizer.

Entry point: ``summarize-papers`` (configured in ``pyproject.toml``).

Usage:
    summarize-papers --source DIR [options]     # batch mode
    summarize-papers --file PDF [options]       # single-file mode
    summarize-papers eval --source DIR ...      # evaluation (see evaluation.py)
    summarize-papers render [--output-dir DIR]  # re-render markdown from JSON sidecars

``--source`` and ``--file`` are mutually exclusive; exactly one must be supplied.
``--reparse`` implies ``--force-summary``.

Before processing (except in dry-run mode), the CLI checks that the backend
host is reachable and, for OpenRouter, that an API key is set and the model id
is listed.

Exit codes: 0 on success (or nothing to do), 1 if any paper failed or the
input/backend is unavailable, 130 when interrupted.
"""

import argparse
import importlib.util
import logging
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from summarizer.batch import OutputDirLocked, find_pdfs, render_all, run_batch, run_pdfs
from summarizer.evaluation import EvalConfig, init_gold, run_eval
from summarizer.llm import (
    fetch_openrouter_key_info,
    fetch_openrouter_models,
    is_openrouter,
    openrouter_listed_id,
)
from summarizer.log import setup_logging
from summarizer.models import (
    _DEFAULT_MAX_CHARS,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_SKILL_DATA_DIR,
    BatchReport,
    Config,
)

logger = logging.getLogger(__name__)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments, validate environment, and run the summarizer."""
    load_dotenv()
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["eval"]:
        _eval_main(argv[1:])
        return
    if argv[:1] == ["render"]:
        _render_main(argv[1:])
        return

    args = _build_parser().parse_args(argv)
    setup_logging(verbose=args.verbose, log_file=_log_file(args.log_file))

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
        strip_references=args.strip_references,
        structured_output=args.structured_output,
        max_cost=args.max_cost,
    )

    # Validate the backend is reachable and usable before starting any work
    if not args.dry_run:
        _check_backend(config.base_url)
        _check_openrouter_config(config)
        _log_key_info(config)

    try:
        if args.file:
            _run_single(Path(args.file), config)
        else:
            _run_batch(Path(args.source), config)
    except OutputDirLocked as exc:
        logger.error("%s; wait for it to finish or use another --output-dir.", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning(
            "Interrupted: queued papers were cancelled; finished papers are saved and "
            "skipped on the next run (in-flight calls may take a moment to end)."
        )
        sys.exit(130)


def _log_file(log_file: str | None) -> Path:
    if log_file:
        return Path(log_file)
    return Path("logs") / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


# ---------------------------------------------------------------------------
# Render mode
# ---------------------------------------------------------------------------


def _render_main(argv: list[str]) -> None:
    """``summarize-papers render``: rebuild markdown from the JSON sidecars."""
    parser = argparse.ArgumentParser(
        prog="summarize-papers render",
        description="Re-render every summary's markdown from its JSON sidecar (no LLM calls).",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default="output_summaries",
        help="Summary output root (default: output_summaries).",
    )
    args = parser.parse_args(argv)
    setup_logging(verbose=False, log_file=None)
    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        logger.error("Not a directory: %s", output_dir)
        sys.exit(1)
    try:
        rendered, failed = render_all(output_dir)
    except OutputDirLocked as exc:
        logger.error("%s; wait for it to finish.", exc)
        sys.exit(1)
    logger.info("Re-rendered %d summaries in %s", rendered, output_dir)
    if failed:
        logger.error("%d sidecar(s) could not be rendered (see above)", failed)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Evaluation mode
# ---------------------------------------------------------------------------


def _eval_main(argv: list[str]) -> None:
    """``summarize-papers eval``: score model × extractor configurations on a PDF set."""
    args = _build_eval_parser().parse_args(argv)
    out_dir = Path(args.out or Path("eval/runs") / datetime.now().strftime("%Y%m%d_%H%M%S"))
    gold_path = Path(args.gold)
    source = Path(args.source)

    setup_logging(verbose=args.verbose, log_file=None)  # file logging once inputs are valid
    if not source.is_dir():
        logger.error("Not a directory: %s", source)
        sys.exit(1)
    pdfs = find_pdfs(source)
    if not pdfs:
        logger.error("No PDFs found under %s", source)
        sys.exit(1)

    if args.init_gold:
        added = init_gold(gold_path, pdfs)
        logger.info("Added %d unlabelled entries to %s", added, gold_path)
        return

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    extractors = [e.strip() for e in args.extractors.split(",") if e.strip()]
    unknown = sorted(set(extractors) - {"docling", "pypdf"})
    if not models or not extractors or unknown:
        logger.error(
            "Need at least one model and extractors from docling, pypdf (got %s)",
            ", ".join(unknown) or "none",
        )
        sys.exit(1)
    if "docling" in extractors and not importlib.util.find_spec("docling"):
        logger.error(
            "docling is not installed (uv sync --extra docling), or use --extractors pypdf"
        )
        sys.exit(1)

    config = Config(
        base_url=args.base_url,
        model=models[0],
        max_chars=args.max_chars,
        skill_data_dir=Path(args.skill_data_dir),
        verbose=args.verbose,
        timeout_s=args.timeout,
        max_output_tokens=args.max_output_tokens,
        workers=args.workers,
        strip_references=args.strip_references,
        structured_output=args.structured_output,
        max_cost=args.max_cost,
    )
    _check_backend(config.base_url)
    for model in models:
        _check_openrouter_config(replace(config, model=model))
    _log_key_info(config)
    setup_logging(verbose=args.verbose, log_file=Path(args.log_file or out_dir / "eval.log"))

    configs = [EvalConfig(model=m, extractor=e) for m in models for e in extractors]
    try:
        _, stopped_reason = run_eval(
            pdfs,
            config,
            configs,
            out_dir=out_dir,
            cache_dir=Path(args.cache_dir),
            gold_path=gold_path,
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted; rerun with the same --cache-dir to resume cheaply.")
        sys.exit(130)
    logger.info("Report: %s", out_dir / "report.md")
    if stopped_reason:
        logger.warning("Stopped early (%s); rerun later to finish.", stopped_reason)
        sys.exit(1)


def _build_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="summarize-papers eval",
        description=(
            "Run model × extractor configurations over a set of PDFs and write "
            "results.jsonl and report.md (quality, reliability, cost, gold-label "
            "accuracy). Never touches output_summaries/ or the processed index."
        ),
    )
    parser.add_argument("--source", metavar="DIR", required=True, help="Directory of PDFs.")
    parser.add_argument(
        "--gold",
        metavar="FILE",
        default="eval/gold.jsonl",
        help="Gold labels, one JSON object per paper (default: eval/gold.jsonl).",
    )
    parser.add_argument(
        "--init-gold",
        action="store_true",
        help="Append unlabelled entries for PDFs missing from --gold, then exit.",
    )
    parser.add_argument(
        "--models",
        metavar="A,B",
        default=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
        help="Comma-separated model ids (default: LLM_MODEL or the default model).",
    )
    parser.add_argument(
        "--extractors",
        metavar="X,Y",
        default="docling" if importlib.util.find_spec("docling") else "pypdf",
        help="Comma-separated extractors from docling, pypdf (default: docling if installed).",
    )
    parser.add_argument(
        "--out",
        metavar="DIR",
        default=None,
        help="Run directory (default: eval/runs/TIMESTAMP).",
    )
    parser.add_argument(
        "--cache-dir",
        metavar="DIR",
        default="eval/cache",
        help="LLM response cache shared across runs (default: eval/cache).",
    )
    _add_backend_args(parser, log_default="OUT/eval.log")
    parser.set_defaults(workers=1)  # free models allow 20 requests/min and 50-1000/day
    return parser


# ---------------------------------------------------------------------------
# Single-file mode
# ---------------------------------------------------------------------------


def _run_single(pdf_path: Path, config: Config) -> None:
    """Process a single PDF through the same code path as batch mode."""
    if not pdf_path.is_file():
        logger.error("File not found: %s", pdf_path)
        sys.exit(1)
    _report_and_exit(run_pdfs([pdf_path], config))


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------


def _run_batch(source_dir: Path, config: Config) -> None:
    """Scan ``source_dir`` for PDFs and process each one."""
    if not source_dir.is_dir():
        logger.error("Not a directory: %s", source_dir)
        sys.exit(1)

    _report_and_exit(run_batch(source_dir, config))


def _report_and_exit(report: BatchReport) -> None:
    """Log the run summary; exit with status 1 if any paper failed."""
    logger.info(
        "Done — processed: %d, skipped: %d, failed: %d, tokens in=%d out=%d, cost=$%.4f",
        report.processed,
        report.skipped,
        report.failed,
        report.input_tokens,
        report.output_tokens,
        report.total_cost,
    )

    if report.failed_papers:
        logger.error("Failed papers:")
        for fp in report.failed_papers:
            logger.error("  %s: %s", fp.pdf_path, fp.error)
    if report.stopped_reason:
        logger.warning(
            "Stopped early (%s); finished papers are saved, rerun later to continue.",
            report.stopped_reason,
        )
    if report.failed_papers or report.stopped_reason:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Backend health check
# ---------------------------------------------------------------------------


def _check_backend(base_url: str) -> None:
    """Verify that the LLM backend (OpenRouter, LM Studio, ...) is reachable."""
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


def _check_openrouter_config(config: Config) -> None:
    """Fail fast on OpenRouter misconfiguration instead of failing every paper.

    Checks that an API key is set and that the model id is still listed
    (OpenRouter retires model ids, e.g. ``:free`` variants).
    """
    if not is_openrouter(config.base_url):
        return
    api_key = config.api_key or os.environ.get("LLM_API_KEY")
    if not api_key:
        logger.error("LLM_API_KEY is not set; OpenRouter requires an API key (see README).")
        sys.exit(1)
    models = fetch_openrouter_models(config.base_url)
    if models is not None:
        by_id = {m["id"]: m for m in models}
        if not _openrouter_model_listed(config.model, set(by_id)):
            logger.error(
                "Model %r is not available on OpenRouter. Pick one from "
                "https://openrouter.ai/models and pass --model or set LLM_MODEL.",
                config.model,
            )
            sys.exit(1)
        entry = by_id.get(openrouter_listed_id(config.model), {})
        if config.structured_output and "structured_outputs" not in (
            entry.get("supported_parameters") or []
        ):
            logger.error("Model %r does not support --structured-output.", config.model)
            sys.exit(1)


def _log_key_info(config: Config) -> None:
    """Log the OpenRouter key's spend and limits; warn about the free-model daily cap."""
    api_key = config.api_key or os.environ.get("LLM_API_KEY")
    if not is_openrouter(config.base_url) or not api_key:
        return
    info = fetch_openrouter_key_info(config.base_url, api_key)
    if info is None:
        return
    logger.info(
        "OpenRouter key: used $%.4f, limit %s, remaining %s",
        float(info.get("usage") or 0),
        "none" if info.get("limit") is None else f"${info['limit']}",
        "n/a" if info.get("limit_remaining") is None else f"${info['limit_remaining']}",
    )
    if info.get("is_free_tier") and config.model.endswith(":free"):
        logger.warning(
            "Free-tier key: free models allow 50 requests/day; the run stops cleanly at the "
            "cap. $10 of credits raises it to 1,000/day."
        )


def _openrouter_model_listed(model: str, model_ids: set[str]) -> bool:
    if model.startswith("@"):  # "@preset/..." ids aren't in the models list
        return True
    return openrouter_listed_id(model) in model_ids


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="summarize-papers",
        description=(
            "Summarise research papers into structured markdown using an "
            "OpenAI-compatible LLM backend (OpenRouter by default, or a local "
            "server such as LM Studio). Processes a directory of PDFs (--source) "
            "or a single PDF (--file)."
        ),
        epilog=(
            "Other commands: summarize-papers eval --help (evaluate models/extractors), "
            "summarize-papers render --help (re-render markdown from JSON)."
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
        help="Re-summarize PDFs already in the processed index; keeps the extraction cache.",
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
    _default_model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    parser.add_argument(
        "--model",
        metavar="MODEL",
        default=_default_model,
        help=f"LLM model identifier (default: LLM_MODEL env var, currently {_default_model!r}).",
    )
    _add_backend_args(parser, log_default="logs/run_TIMESTAMP.log")
    return parser


def _add_backend_args(parser: argparse.ArgumentParser, log_default: str) -> None:
    """Flags shared by the run and eval commands."""
    parser.add_argument(
        "--base-url",
        metavar="URL",
        default=DEFAULT_BASE_URL,
        help=f"OpenAI-compatible API base URL (default: {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--max-chars",
        metavar="N",
        type=int,
        default=_DEFAULT_MAX_CHARS,
        help=(
            f"Maximum characters of paper text sent to the LLM "
            f"(default: {_DEFAULT_MAX_CHARS:,} ≈ 50k tokens; longer text is "
            "truncated with a warning)."
        ),
    )
    parser.add_argument(
        "--skill-data-dir",
        metavar="DIR",
        default=str(DEFAULT_SKILL_DATA_DIR),
        help=(
            "Directory of reference .md files embedded in the prompt. "
            "Default: the skill_data/references folder of this checkout."
        ),
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "DEBUG-level logging: raw response excerpts on parse failures and "
            "full validation errors (default: off)."
        ),
    )
    parser.add_argument(
        "--log-file",
        metavar="FILE",
        default=None,
        help=f"Write log output to FILE (default: {log_default}).",
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
        help="Number of papers processed in parallel (default: %(default)s).",
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
    parser.add_argument(
        "--max-cost",
        metavar="USD",
        type=float,
        default=None,
        help=(
            "Stop starting new papers once this much has been spent (papers already "
            "running still finish)."
        ),
    )
    parser.add_argument(
        "--strip-references",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop the References/Bibliography section before sending the text (default: on).",
    )
    parser.add_argument(
        "--structured-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Constrain replies to the summary JSON schema (backend must support it; default: off)."
        ),
    )


if __name__ == "__main__":
    main()
