# CLAUDE.md

CLI that turns research PDFs into structured markdown summaries via an OpenAI-compatible LLM (OpenRouter by default). Target use case: a literature review on spiking neural networks for control. Goal: make the research field hot-swappable.

**To-do list / plan / decisions: `notes/todo.md`**. This is the single source of truth. It's gitignored (local only); read it before starting work and update it when items change.

## Commands

- Setup: `uv sync --extra docling` (omit the extra for a light, pypdf-only install; unit tests don't need it)
- Tests: `uv run pytest` — unit tests block network access (`tests/conftest.py::_no_network`); `uv run pytest -m integration` for real PDFs/OpenRouter
- Lint/format: `uv run ruff check . && uv run ruff format .`
- Run: `uv run summarize-papers --source DIR | --file PDF [--dry-run]`
- Evaluate: `uv run summarize-papers eval --source DIR [--models A,B] [--extractors pypdf,docling]` (outputs in `eval/runs/`)
- Re-render markdown from JSON sidecars: `uv run summarize-papers render [--output-dir DIR]`

Python >=3.12, developed on 3.14 (`.python-version`). Use uv, not pip/conda; commit `uv.lock` changes.

## Architecture (one LLM call per paper)

`cli` → `batch.run_pdfs` (skip index, thread pool, writes outputs on the main thread) → `pipeline.process_pdf` (parse → prompt → `llm.call_llm` → normalize year/citation key → pydantic validation with bounded schema repair) → `renderer.render_summary`.

- `skill_data/references/*.md` is the domain source of truth and is embedded verbatim in every prompt (~11k tokens). `json-output-contract.md` must agree with `models.py` (including the `Classification` label vocabularies; `test_classification_vocabularies_match_the_references` guards them), `prompts.py` (tail rules), `pipeline._PRIMARY_PART2_FIELDS`/`_SYNTHESIS_PART1_FIELDS` and `renderer.py`. Drift between these has caused bugs, and `tests/test_prompts.py` guards part of it.
- `paper_type` is exactly `primary | synthesis` (or `null` for non-research); subtypes go in `synthesis_subtype`.
- Retries live only in `llm._complete_with_retries` (the SDK has `max_retries=0`); OpenRouter provider errors inside a 200 response raise `ProviderError` (a billed `RejectedCompletion`) and are retried when transient. `finish_reason == "length"` and empty content raise `RejectedCompletion` (no repair). Daily caps / credit exhaustion raise `QuotaExhausted`, which stops `run_pdfs` and `run_eval` cleanly (worker-side `batch.StopSignal`, shared with eval); `--max-cost` uses the same stop.
- Schema repair resends the paper only for missing content (`pipeline._needs_paper_context`).
- Parser caches `$XDG_CACHE_HOME/paper-summarizer/<sha256>.<extractor>.md` (old caches next to the PDF are still read; tests point `XDG_CACHE_HOME` at `tmp_path`); docling is imported lazily, is an optional extra, and runs without OCR unless most pages have no text (scans).
- Processed index: `output_summaries/processed.jsonl`, keyed by PDF sha256 (path-only entries and legacy `processed.txt` still match by path). Each summary gets a `.json` sidecar (`PaperSummary` with `provenance`); `summarize-papers render` rebuilds markdown from them. `batch.output_dir_lock` prevents concurrent runs on one output dir.
- Zotero: `batch` looks up each `<KEY>__*.pdf` in the local Zotero API (`zotero.lookup_all`, main thread, read-only) and `process_pdf` lets the item's title/authors/year/venue/citation key override the LLM's (`pipeline._apply_zotero`). Eval never does this. Never write to `~/Zotero` (shared group library).
- Evaluation: `summarize-papers eval` (`evaluation.py` runner/cache/gold/report, `metrics.py` pure metrics) calls `process_pdf` directly and must never write to `output_summaries/`.

## Conventions

- Match the existing style: module docstrings explain the why, `logger = logging.getLogger(__name__)`, `# ---` section banners.
- Every bug fix gets a regression test.
- Real papers: the owner's library is in `input_papers/` (148 PDFs) and `test_pdfs/` (both gitignored); build eval/gold/baseline sets from it, and ask the owner before fetching PDFs from anywhere else. Gold labels need owner review.
