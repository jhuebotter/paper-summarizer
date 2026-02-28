# Paper Summarizer

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3110/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000?logo=python&logoColor=white)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Built by **Justus Hübotter** (February 2026).

`paper-summarizer` is a CLI that turns research PDFs into structured markdown summaries using an OpenAI-compatible backend.

It is optimized for SNN/control literature workflows, but the core pipeline is topic-agnostic and can be adapted by swapping prompt/reference material.

## Quick Start (60 seconds)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cat > .env <<'EOF'
LLM_API_KEY=your_openrouter_key_here
LLM_MODEL=openai/gpt-oss-120b:free
EOF
summarize-papers --source input_papers
```

## What it does

- Parses PDFs to markdown with local cache (`docling` or `pypdf`).
- Builds one combined LLM prompt for metadata + sectioned summary.
- Validates output with Pydantic and bounded repair retries.
- Organizes outputs by paper type (`primary`, `survey`, `commentary`, `non_research`).
- Supports batch processing with parallel workers and processed-file skipping.

## Typical workflow

1. Discover PDFs from `--source` (recursive) or `--file`.
2. Parse PDF text and cache as `<pdf_stem>.md` next to the source file.
3. Build prompt with references from `skill_data/references`.
4. Call the configured OpenAI-compatible backend.
5. Validate/repair JSON and render summary markdown.
6. Write summary to `output_summaries/<paper_type>/<citation_key>_summary.md` and update `processed.txt`.

## Data prep: collect PDFs from nested libraries

Use `collect_pdfs.sh` when your PDFs live in deep subfolders (for example a Zotero or Mendeley library export):

```bash
./collect_pdfs.sh "/path/to/library_root" "./input_papers"
```

What it does:

- Recursively finds all `*.pdf` files under the source directory.
- Copies them into one destination folder.
- Flattens names as `<parent_folder>__<original_filename>.pdf` to reduce collisions.

## Usage

Batch mode:

```bash
summarize-papers --source input_papers
```

Single-file mode:

```bash
summarize-papers --file "test_pdfs/example.pdf"
```

## Example output

`output_summaries/primary/smith2025_summary.md`:

```markdown
# Smith et al. (2025)

## Core idea
Event-driven SNN controller with reward-modulated plasticity for online adaptation.

## Methods snapshot
- Task: closed-loop motor control in simulated arm reach task
- Encoder: population spike encoding of state variables
- Learning: local plasticity + global reward signal

## Key findings
- Lower energy use than ANN baseline with comparable tracking error
- Better robustness under sensor noise
```

## Project structure

```text
.
├── input_papers/                 # PDFs to process
├── output_summaries/             # Generated summaries grouped by paper type
├── processed.txt                 # Absolute paths already processed
├── skill_data/references/        # Prompt references/domain guidance
├── collect_pdfs.sh               # Helper to flatten nested PDF libraries
└── summarizer/                   # Python package source
```

## Configuration

Environment variables:

- `LLM_API_KEY`: API key (required for OpenRouter, ignored by LM Studio).
- `LLM_MODEL`: default model if `--model` is not passed.

Notes:

- Default `--base-url`: `https://openrouter.ai/api/v1`.
- In non-`--dry-run` mode, backend reachability is checked before processing.

Backend examples:

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

## Important CLI options

- `--source DIR` / `--file PDF`: choose batch or single-file mode.
- `--force-summary`: re-run summary even if listed in `processed.txt`.
- `--reparse`: re-run extraction too (implies `--force-summary`).
- `--extractor {auto,docling,pypdf}`: parser strategy.
- `--output-dir DIR`: output root (default `output_summaries`).
- `--workers N`: parallel batch workers (default `3`).
- `--timeout S`: per-call timeout in seconds (default `120`).
- `--dry-run`: preview files without LLM calls or writes.

## Skip and rerun behavior

- Files in `processed.txt` are skipped by default.
- `--force-summary` bypasses skip logic.
- Parser cache (`<pdf_stem>.md`) is reused unless `--reparse` is used.
- `--extractor auto` tries `docling` first, then `pypdf` fallback.
- Output name collisions are versioned (`_v2`, `_v3`, ...).

## Troubleshooting

- Auth errors: verify `LLM_API_KEY` and that your backend accepts the configured model.
- Backend unreachable: check `--base-url`, local server status, and firewall/VPN.
- Parsing quality issues: retry with `--extractor pypdf` or `--reparse`.
- Slow/timeout runs: lower `--workers`, increase `--timeout`, or reduce `--max-chars`.

## Development

Run tests:

```bash
pytest
```

Integration tests are marked `integration` and require real external resources.

## Roadmap

- Add automated table generation with citations from extracted summary fields.
- Support synthesis workflows for methodological overviews across large paper sets.
- Extend the pipeline from single-paper summarization toward staged corpus distillation ("multistage RAG 2.0" without document-embedding/token-encoder indexing).

## License

This project is licensed under the MIT License. See `LICENSE`.
