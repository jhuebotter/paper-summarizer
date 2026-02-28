# Paper Summarizer

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3110/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000?logo=python&logoColor=white)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Built by **Justus Hübotter** (February 2026).

`paper-summarizer` is a CLI tool that converts research PDFs into structured markdown summaries using an OpenAI-compatible LLM backend.

It is optimized for SNN/control literature workflows and keeps outputs consistent and reproducible.

Originally, this tool was built to help process a large volume of related research papers in one domain. The pipeline code itself (outside `skill_data/`) is topic-agnostic and can be configured for other research areas by changing prompt/reference material.

## Features

- Parse PDFs to markdown using `docling` or `pypdf` with local parse caching.
- One combined LLM prompt per paper for metadata + summary sections.
- Pydantic schema validation plus bounded JSON/schema repair retries.
- Centralized output structure by paper type: `primary`, `survey`, `commentary`, `non_research`.
- Batch processing with parallel workers and persistent processed-index skipping.
- Full test suite and comprehensive runtime logging support.

## Brief Workflow

1. Discover PDFs from `--source` recursively (or use `--file` for a single paper).
2. Parse PDF text (`<pdf_stem>.md` cache next to the source PDF).
3. Build combined prompt with references from `skill_data/references`.
4. Call the configured OpenAI-compatible backend.
5. Validate/repair JSON response and render markdown.
6. Write summary to `output_summaries/<paper_type>/<citation_key>_summary.md` and update `processed.txt`.

## Requirements

- Python `>=3.11`
- Dependencies:
  - `docling`
  - `openai`
  - `python-dotenv`
  - `pydantic`
  - `pypdf`
  - `tqdm`

Install from source:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

CLI entry point:

```bash
summarize-papers --help
```

## Environment Configuration (`.env`)

The CLI loads environment variables from `.env` (via `python-dotenv`) and from the shell.

Supported variables:

- `LLM_API_KEY`: API key for your backend.
  - Required for OpenRouter.
  - Ignored by LM Studio.
- `LLM_MODEL`: default model when `--model` is not provided.

Minimal `.env` example:

```env
LLM_API_KEY=your_openrouter_key_here
LLM_MODEL=openai/gpt-oss-120b:free
```

Notes:

- Default `--base-url` is `https://openrouter.ai/api/v1`.
- Before processing (except in `--dry-run`), the CLI checks backend reachability at the host root (for example `https://openrouter.ai`).

## Usage

Batch mode:

```bash
summarize-papers --source input_papers
```

Single-file mode:

```bash
summarize-papers --file "test_pdfs/example.pdf"
```

## CLI Options

- `--source DIR`: scan directory recursively for PDFs.
- `--file PDF`: process a single PDF.
- `--force-summary`: re-run summary step even if file is in `processed.txt`.
- `--reparse`: re-run extraction too (implies `--force-summary`).
- `--extractor {auto,docling,pypdf}`: extraction strategy.
- `--dry-run`: list files that would be processed (no LLM calls, no writes).
- `--output-dir DIR`: output root (default `output_summaries`).
- `--model MODEL`: model ID (default from `LLM_MODEL`, then `openai/gpt-oss-120b:free`).
- `--base-url URL`: OpenAI-compatible API base URL (default `https://openrouter.ai/api/v1`).
- `--max-chars N`: cap parsed text length sent to the LLM.
- `--skill-data-dir DIR`: references directory (default `skill_data/references`).
- `--verbose` / `--no-verbose`: enable/disable debug-style verbosity (default enabled).
- `--log-file FILE`: custom log file path.
- `--timeout S`: per-call timeout in seconds (default `120`).
- `--workers N`: parallel batch workers (default `3`).
- `--max-output-tokens N`: optional model output-token cap.

## Backend Examples

LM Studio:

```bash
summarize-papers \
  --source input_papers \
  --base-url http://localhost:1234/v1 \
  --model your-local-model-id
```

OpenRouter:

```bash
export LLM_API_KEY="<your_key>"
summarize-papers --source input_papers --model "openai/gpt-oss-120b:free"
```

## Skip and Re-run Behavior

- If a PDF absolute path exists in `processed.txt`, it is skipped.
- `--force-summary` bypasses that skip logic.
- Parser cache (`<pdf_stem>.md`) is reused unless `--reparse` is passed.
- `--extractor auto` tries docling first, then `pypdf` fallback.
- Existing output filename collisions are versioned (`_v2`, `_v3`, ...).

## Development

Run tests:

```bash
pytest
```

Integration tests are marked with `integration` and require real external resources.

## License

This project is licensed under the MIT License. See `LICENSE`.
