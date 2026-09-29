# Paper Summarizer

[![CI](https://github.com/jhuebotter/paper-summarizer/actions/workflows/ci.yml/badge.svg)](https://github.com/jhuebotter/paper-summarizer/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-managed-DE5FE9)](https://docs.astral.sh/uv/)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-261230)](https://docs.astral.sh/ruff/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Built by **Justus Hübotter** (2026).

`paper-summarizer` is a CLI that turns research PDFs into structured, critical markdown summaries using any OpenAI-compatible LLM backend (OpenRouter by default, or a local server such as LM Studio).

The bundled prompt references target a literature review on **spiking neural networks (SNNs) for control**. Making the research field hot-swappable is on the roadmap.

## Quick start

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/). uv installs the Python version pinned in `.python-version` (3.14) automatically.

```bash
git clone https://github.com/jhuebotter/paper-summarizer && cd paper-summarizer
uv sync --extra docling          # omit --extra docling for a light pypdf-only install
cat > .env <<'EOF'
LLM_API_KEY=your_openrouter_key_here
# LLM_MODEL=nvidia/nemotron-3-super-120b-a12b:free   # optional override
EOF
uv run summarize-papers --source input_papers
```

## What it does

- Extracts PDF text with [docling](https://github.com/docling-project/docling) (optional extra, better layout/tables) or pypdf, and caches it next to the PDF.
- Sends one combined prompt per paper (bibliographic metadata + Part 1 summary + Part 2 SNN-control extraction) and asks for a single JSON object.
- Validates the JSON with pydantic. Syntax errors get one repair call; schema errors get up to two repair calls, which resend the paper text only when content is missing.
- Sorts papers into `primary` (new experiments/methods), `synthesis` (reviews, surveys, perspectives, commentaries, …) or `non_research`.
- Processes batches in parallel, skips papers it has already done, and reports token totals and cost (priced from OpenRouter's live model list).

## Typical workflow

1. Discover PDFs under `--source` (recursive, `.pdf` in any case) or take one `--file`.
2. Extract text, or reuse the cache `<pdf_stem>.<extractor>.md` next to the PDF.
3. Build the prompt from `skill_data/references/*.md` (~11k tokens) plus the paper text (up to `--max-chars`).
4. Call the backend; validate and repair the JSON; render markdown.
5. Write `output_summaries/<paper_type>/<citation_key>_summary.md` (versioned `_v2`, `_v3`, … instead of overwriting) and record it in `output_summaries/processed.jsonl`.

## Data prep: collect PDFs from nested libraries

Use `collect_pdfs.sh` when your PDFs live in deep subfolders (for example Zotero's `storage/<KEY>/` folders):

```bash
./collect_pdfs.sh "/path/to/Zotero/storage" "./input_papers"
```

- Recursively finds all PDFs (case-insensitive; skips macOS `._*` files).
- Copies them into one folder as `<parent_folder>__<original_filename>.pdf`.
- Never overwrites: a different file with the same name gets a `__2`, `__3`, … suffix, and identical files are skipped, so re-running it is safe.

## Usage

```bash
uv run summarize-papers --source input_papers          # batch
uv run summarize-papers --file "papers/example.pdf"    # single file
uv run summarize-papers --source input_papers --dry-run  # list what would be processed
```

Local backend (LM Studio):

```bash
uv run summarize-papers --source input_papers \
  --base-url http://localhost:1234/v1 --model your-local-model-id
```

## Example output

`output_summaries/primary/doe2024spiking_summary.md` (abridged):

```markdown
# A Spiking Controller for Reaching

**Citation key:** doe2024spiking
**Authors:** Jane Doe, John Roe
**Year:** 2024
**Venue:** Preprint (arXiv:2401.00000)
**Paper Type:** primary research
**Tags:** SNN, continuous control, surrogate gradients

---

## TL;DR

The authors train a recurrent LIF controller with surrogate-gradient BPTT on a simulated 7-DOF reach task … (Source: Tbl. 2)

---

## Part 1: Paper Summary

### Problem & Motivation
…
### Notable Findings
- Tracking error within 5% of an ANN baseline (Measured) (Source: Fig. 3)
- Suitable for neuromorphic deployment (Claimed — no chip deployment attempted) (Source: Sec. 5)
…

---

## Part 2: SNN Control Extraction

**Neuron model:** LIF (Source: Sec. 3.1)

**Controller hardware (inference):** CPU/GPU
…
```

## Project structure

```text
.
├── input_papers/                 # PDFs to process (gitignored)
├── output_summaries/             # Summaries by paper type + processed.jsonl (gitignored)
├── skill_data/references/        # Prompt references: JSON contract, template, field guides
├── collect_pdfs.sh               # Flatten nested PDF libraries
└── summarizer/                   # Package source (cli, batch, pipeline, llm, parser, prompts, renderer, models)
```

## Configuration

Environment variables (a `.env` file in the working directory is loaded automatically):

- `LLM_API_KEY`: API key. Required for OpenRouter; ignored by LM Studio.
- `LLM_MODEL`: default model when `--model` is not passed.

Defaults: `--base-url https://openrouter.ai/api/v1`, `--model nvidia/nemotron-3-super-120b-a12b:free` (free, 262k context, checked 2026-09-29).

- Free models are rate-limited per day by OpenRouter, and the limit depends on your account's credit balance. Check your key's limits before a large batch, and lower `--workers` if you hit 429s.
- Free endpoints may log prompts. That's fine for published papers; don't send unpublished work.
- A paid fallback within a few-cents budget is `--model meta/muse-spark-1.3-contributor` (~$0.10 / $0.20 per million input/output tokens, about **$0.007** per paper).

Before processing (not in `--dry-run`), the CLI checks that the backend is reachable. For OpenRouter it also checks that `LLM_API_KEY` is set and that the model id is still listed; OpenRouter does retire ids, so this fails fast instead of failing every paper.

## CLI options

- `--source DIR` / `--file PDF`: batch or single-file mode.
- `--force-summary`: re-summarize papers already in the processed index (keeps the extraction cache).
- `--reparse`: also re-run extraction (implies `--force-summary`).
- `--extractor {auto,docling,pypdf}`: `auto` uses docling if installed, falling back to pypdf.
- `--output-dir DIR`: output root (default `output_summaries`).
- `--model`, `--base-url`: backend selection.
- `--max-chars N`: paper-text budget (default 200,000 ≈ 50k tokens). Longer text is truncated, and a warning is logged.
- `--max-output-tokens N`: cap generated tokens (default: no cap). If the model hits the cap, the paper fails with a clear message instead of an unrepairable half-JSON.
- `--workers N`: parallel papers (default 3). `--timeout S`: per-call timeout (default 120).
- `--verbose` / `--no-verbose`: DEBUG logging (raw response excerpts on parse failures, full validation errors); off by default.
- `--log-file FILE`: log path (default `logs/run_<timestamp>.log`).
- `--skill-data-dir DIR`: prompt reference files (default: `skill_data/references` of this checkout).

## Skip and rerun behavior

- Papers listed in `output_summaries/processed.jsonl` (keyed by absolute PDF path) are skipped. A legacy `processed.txt` is still read and migrated on the next save.
- Extraction caches are per extractor (`<stem>.docling.md`, `<stem>.pypdf.md`). A legacy `<stem>.md` is still honoured by `--extractor auto`.
- Failed papers are not recorded, so the next run retries them. The exit code is 1 if any paper failed.
- Ctrl-C cancels queued papers (exit code 130). Finished papers are kept and skipped on the next run.
- Use one run at a time per output directory.

## Troubleshooting

- **"Model … is not available on OpenRouter"**: pick a current id from <https://openrouter.ai/models>.
- **Auth errors**: check `LLM_API_KEY`.
- **"output was truncated by the token limit"**: raise `--max-output-tokens`, or choose a model with a larger output budget.
- **Poor extraction**: install the docling extra (`uv sync --extra docling`), then `--reparse`.
- **Slow runs or timeouts**: lower `--workers`, raise `--timeout`, or reduce `--max-chars`.

## Development

```bash
uv sync --extra docling              # dev tools (pytest, ruff) are included by default
uv run pytest                        # unit tests (no network, no docling needed)
uv run pytest -m integration         # real PDFs in test_pdfs/ and/or OpenRouter (needs LLM_API_KEY)
uv run ruff check . && uv run ruff format .
```

CI runs lint and tests on Python 3.12–3.14.

## Roadmap

- Evaluation harness: quote faithfulness, validity and repair rates, cost per paper, labelled accuracy, model comparison.
- Structured JSON outputs next to each summary, content-hash paper identity, DOI-based metadata.
- JSON-schema structured outputs and leaner, type-specific prompts.
- Decision models, i.e. fast classifiers that return calibrated probabilities (e.g. TypeSafe's Jev, or local models via Ollaya), for screening, typed field extraction and cross-checks.
- Domain profiles, so the research field is hot-swappable.
- Corpus-level comparison tables, BibTeX export and staged synthesis.

## License

MIT — see `LICENSE`.
