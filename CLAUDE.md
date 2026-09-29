# CLAUDE.md

CLI that turns research PDFs into structured markdown summaries via an OpenAI-compatible LLM (OpenRouter by default). Target use case: a literature review on spiking neural networks for control. Goal: make the research field hot-swappable.

**To-do list / plan / decisions: `notes/todo.md`**. This is the single source of truth. It's gitignored (local only); read it before starting work and update it when items change.

## Commands

- Setup: `uv sync --extra docling` (omit the extra for a light, pypdf-only install; unit tests don't need it)
- Tests: `uv run pytest` — unit tests block network access (`tests/conftest.py::_no_network`); `uv run pytest -m integration` for real PDFs/OpenRouter
- Lint/format: `uv run ruff check . && uv run ruff format .`
- Run: `uv run summarize-papers --source DIR | --file PDF [--dry-run]`

Python >=3.12, developed on 3.14 (`.python-version`). Use uv, not pip/conda; commit `uv.lock` changes.

## Architecture (one LLM call per paper)

`cli` → `batch.run_pdfs` (skip index, thread pool, writes outputs on the main thread) → `pipeline.process_pdf` (parse → prompt → `llm.call_llm` → normalize year/citation key → pydantic validation with bounded schema repair) → `renderer.render_summary`.

- `skill_data/references/*.md` is the domain source of truth and is embedded verbatim in every prompt (~11k tokens). `json-output-contract.md` must agree with `models.py`, `prompts.py` (tail rules), `pipeline._PRIMARY_PART2_FIELDS`/`_SYNTHESIS_PART1_FIELDS` and `renderer.py`. Drift between these has caused bugs, and `tests/test_prompts.py` guards part of it.
- `paper_type` is exactly `primary | synthesis` (or `null` for non-research); subtypes go in `synthesis_subtype`.
- Retries live only in `llm._complete_with_retries` (the SDK has `max_retries=0`). `finish_reason == "length"` and empty content raise `LLMError` (no repair).
- Schema repair resends the paper only for missing content (`pipeline._needs_paper_context`).
- Parser caches `<stem>.<extractor>.md` next to the PDF; docling is imported lazily and is an optional extra.
- Processed index: `output_summaries/processed.jsonl` (legacy `processed.txt` is read-only for migration).

## Conventions

- Match the existing style: module docstrings explain the why, `logger = logging.getLogger(__name__)`, `# ---` section banners.
- Every bug fix gets a regression test.
- Don't assemble a PDF dataset (gold/eval/baseline) without asking the owner first; the real Zotero library (~150 papers) lives on another machine.
