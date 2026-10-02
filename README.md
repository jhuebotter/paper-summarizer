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
cp .env.example .env              # then put your OpenRouter key in .env
uv run summarize-papers --source input_papers
```

## What it does

- Extracts PDF text with [docling](https://github.com/docling-project/docling) (optional extra, better layout/tables) or pypdf, and caches it in `~/.cache/paper-summarizer/` (or `$XDG_CACHE_HOME/paper-summarizer/`), keyed by the PDF's content.
- Sends one combined prompt per paper (bibliographic metadata + Part 1 summary + Part 2 SNN-control extraction) and asks for a single JSON object.
- Validates the JSON with pydantic. Syntax errors get one repair call; schema errors get up to two repair calls, which resend the paper text only when content is missing.
- Sorts papers into `primary` (new experiments/methods), `synthesis` (reviews, surveys, perspectives, commentaries, …) or `non_research`.
- For primary papers, Part 2 also carries typed labels from the reference vocabularies (inference hardware, fully spiking vs hybrid, credit assignment, learning regime, paradigm families), ready for comparison tables.
- Writes each summary as markdown plus a JSON sidecar with the validated data and its provenance (model, extractor, text length sent, tokens, cost, commit).
- Processes batches in parallel, skips papers it has already done, and reports token totals and cost (priced from OpenRouter's live model list).

## Typical workflow

1. Discover PDFs under `--source` (recursive, `.pdf` in any case) or take one `--file`.
2. Extract text, or reuse the cache `<sha256>.<extractor>.md`; drop the reference section (`--strip-references`, on by default).
3. Build the prompt from `skill_data/references/*.md` (~11k tokens) plus the paper text (up to `--max-chars`).
4. Call the backend; validate and repair the JSON (a citation key that doesn't start with part of the first author's name plus the year is rebuilt, e.g. `ckl2024local` → `stockl2024local`); render markdown.
5. Write `output_summaries/<paper_type>/<citation_key>_summary.md` and `.json` (versioned `_v2`, `_v3`, … instead of overwriting) and record the paper in `output_summaries/processed.jsonl`.

`summarize-papers render [--output-dir DIR]` rebuilds every markdown file from its JSON sidecar without any LLM calls, e.g. after changing the template.

## Data prep: collect PDFs from nested libraries

Use `collect_pdfs.sh` when your PDFs live in deep subfolders (for example Zotero's `storage/<KEY>/` folders):

```bash
./collect_pdfs.sh "/path/to/Zotero/storage" "./input_papers"
```

- Recursively finds all PDFs (case-insensitive; skips macOS `._*` files).
- Copies them into one folder as `<parent_folder>__<original_filename>.pdf`.
- Never overwrites: a different file with the same name gets a `__2`, `__3`, … suffix, and identical files are skipped, so re-running it is safe.

### Zotero metadata

When a PDF's name starts with its Zotero attachment key (`<KEY>__…pdf`, as `collect_pdfs.sh` names them) and Zotero is running, runs take the title, authors, year, venue and citation key from the Zotero item instead of the LLM. They use Zotero's read-only local API on `localhost:23119`, which is off by default: enable *Settings → Advanced → Miscellaneous → Allow other applications on this computer to communicate with Zotero* (Zotero 7+). Your personal library and all your groups are searched.
- `--no-zotero` turns it off. When Zotero isn't reachable or stops answering, the run logs a warning and uses the LLM's metadata for the remaining papers. Items in the Zotero trash are used, with a warning.
- The sidecar records the item (`provenance.zotero_item`) and which fields Zotero changed (`provenance.zotero_fields`); each change is also logged.
- The paper type, classification and everything else still come from the LLM, and PDFs are still read from disk.
- `eval` never uses Zotero, since it measures the model.
- Metadata fixed in Zotero later reaches existing summaries only with `--force-summary`.

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

### Classification

**Inference hardware:** CPU/GPU
**Architecture:** fully spiking
**Credit assignment:** Global
**Learning regime:** Offline
**Paradigm families:** Gradient-based (surrogate gradient BPTT)

### Details

**Neuron model:** LIF (Source: Sec. 3.1)

**Controller hardware (inference):** CPU/GPU
…
```

## Project structure

```text
.
├── input_papers/                 # PDFs to process (gitignored)
├── output_summaries/             # Summaries (.md + .json) by paper type, processed.jsonl (gitignored)
├── skill_data/references/        # Prompt references: JSON contract, template, field guides
├── skill_data/overview/          # Literature-overview codebook (embedded in overview prompts)
├── output_overview/              # Overview records, overview.md/.csv tables (gitignored)
├── collect_pdfs.sh               # Flatten nested PDF libraries
├── eval/                         # Evaluation papers, gold labels, runs and cache (gitignored)
└── summarizer/                   # Package source (cli, batch, pipeline, llm, parser, prompts, renderer, models, log, evaluation, metrics, overview, overview_run, overview_tables)
```

## Configuration

Environment variables (a `.env` file in the working directory is loaded automatically):

- `LLM_API_KEY`: API key. Required for OpenRouter; ignored by LM Studio.
- `LLM_MODEL`: default model when `--model` is not passed.

Defaults: `--base-url https://openrouter.ai/api/v1`, `--model nvidia/nemotron-3-super-120b-a12b:free` (free, 262k context, checked 2026-09-29).

- Free (`:free`) models allow 20 requests per minute, and 50 per day until you've bought $10 of credits (then 1,000 per day). The preflight shows your key's spend and limits. When the daily cap or your credits run out, the run stops cleanly: queued papers are skipped, not failed, so rerun later to continue. Transient errors are retried up to 3 times: per-minute 429s wait for `Retry-After` or the reset time OpenRouter reports (up to 60 s), other 429s (free models throttled upstream) back off from 5 s, and provider errors that OpenRouter returns inside a 200 response are retried too.
- Free endpoints may log prompts. That's fine for published papers; don't send unpublished work.
- A paid fallback within a few-cents budget is `--model meta/muse-spark-1.3-contributor` (~$0.10 / $0.20 per million input/output tokens, about **$0.007** per paper).

Before processing (not in `--dry-run`), the CLI checks that the backend is reachable. For OpenRouter it also checks that `LLM_API_KEY` is set and that the model id is still listed (OpenRouter does retire ids), and logs the key's usage and limits. Reported costs are what OpenRouter actually billed when the response includes it, and otherwise an estimate from list prices.

## CLI options

- `--source DIR` / `--file PDF`: batch or single-file mode.
- `--force-summary`: re-summarize papers already in the processed index (keeps the extraction cache).
- `--reparse`: also re-run extraction (implies `--force-summary`).
- `--extractor {auto,docling,pypdf}`: `auto` uses docling if installed, falling back to pypdf. docling runs without OCR (about 1–5 s per paper) and uses OCR only for PDFs with no text layer (scans). pypdf can glue words together on some PDFs; prefer docling.
- `--output-dir DIR`: output root (default `output_summaries`).
- `--model`, `--base-url`: backend selection.
- `--max-chars N`: paper-text budget (default 200,000 ≈ 50k tokens). Longer text is truncated, and a warning is logged. For models with a smaller context window (known from OpenRouter), the text is cut further so the prompt and reply fit.
- `--strip-references` / `--no-strip-references`: drop the References/Bibliography section before sending (default on). This saves tokens and budget for the actual paper. With docling, appendices after the references are kept; with pypdf, everything after the references heading is dropped.
- `--structured-output`: ask the backend to constrain replies to the summary's JSON schema (off by default until the evaluation shows it helps; the model must support structured outputs).
- `--decider [MODEL]`: also ask a decision model for the four single-label classification fields (see [Decision model](#decision-model)). Off by default.
- `--zotero` / `--no-zotero`: bibliographic metadata from the local Zotero library (default on; see [Zotero metadata](#zotero-metadata)).
- `--max-cost USD`: stop starting new papers once this much has been spent (papers already running still finish). Also applies to `eval`.
- `--max-output-tokens N`: cap generated tokens (default: no cap). If the model hits the cap, the paper fails with a clear message instead of an unrepairable half-JSON.
- `--workers N`: parallel papers (default 3). `--timeout S`: per-call timeout (default 120).
- `--verbose` / `--no-verbose`: DEBUG logging (raw response excerpts on parse failures, full validation errors); off by default.
- `--log-file FILE`: log path (default `logs/run_<timestamp>.log`).
- `--skill-data-dir DIR`: prompt reference files (default: `skill_data/references` of this checkout).

## Skip and rerun behavior

- Papers are identified by content (SHA-256): anything in `output_summaries/processed.jsonl` is skipped even after moving or renaming the PDF, and identical copies in one batch are processed once. A changed PDF counts as a new paper. Older index entries (path only, or a legacy `processed.txt`) are migrated when their PDF is next seen, unless the file changed after it was summarized, in which case it is summarized again.
- Extraction caches are per content and extractor (`<sha256>.docling.md`, `<sha256>.pypdf.md`). Caches that earlier versions wrote next to the PDFs (`<stem>.docling.md`, `<stem>.pypdf.md`) are still read unless they are older than the PDF; the first version's bare `<stem>.md` is ignored because it may hold pypdf text under any extractor.
- Failed papers are not recorded, so the next run retries them. The exit code is 1 if any paper failed.
- Ctrl-C cancels queued papers (exit code 130). Finished papers are kept and skipped on the next run.
- A run stopped by a quota or `--max-cost` exits with 1 and says why; rerunning continues where it stopped.
- One run at a time per output directory: a second run (or `render`) on the same directory exits with an error while the first is running (POSIX; on Windows, don't overlap runs).
- `render` overwrites the markdown files, including any hand edits; invalid sidecars are reported and skipped.

## Decision model

With `--decider`, each primary paper's text is also sent to a decision model: TypeSafe's Jev on the same OpenRouter key, pinned to `typesafe/jev-1.13-20260917`. The model answers one multiple-choice question per classification field (inference hardware, architecture, credit assignment, learning regime) and returns calibrated probabilities.
- **Descriptions:** each label's description comes from the bullets in `skill_data/references/snn-extraction-fields.md`, so the LLM and the decider share definitions.
- **Storage:** the answers go into the sidecar next to the LLM's labels (`decisions`, with probabilities and the LLM's label). They don't replace them. The markdown shows a note only where the two disagree.
- **Cost:** about $0.001 per paper. It is recorded in `provenance.decider_cost_usd` (not in the LLM's `provenance.cost_usd`) and counts toward `--max-cost` and the eval's cost column.
- **Long papers:** the decider sees at most the first 60,000 characters, and less if the model reports the text is too long.
- **Failures:** a failure, including an exhausted quota, is logged and recorded (`provenance.decider_error`); the summary is kept. Papers skipped on later runs keep their missing decisions (rerun with `--force-summary` to fill them). In eval, a paper without decisions counts as wrong for `decider.*`, and the decider only runs on papers the LLM classifies as primary.
- **Privacy:** it sends the paper text to TypeSafe via OpenRouter, like the LLM call. Don't use it on unpublished work.
- **Endpoint:** it uses the System One wire format (`POST <base-url>/systemone`) on the LLM's backend. A local [Laya](https://github.com/nvkudva/laya-server) server speaks the same format, but it would need its own URL (not supported yet) and reads only 512–1024 tokens.
- **Eval:** `summarize-papers eval --decider` scores the decider's labels as `decider.*` next to the LLM's `classification.*`.

## Literature overview

`summarize-papers overview` extracts a typed **overview record** per paper. A record holds facts with evidence quotes, as defined in `skill_data/overview/codebook.md`:
- what the spiking neurons do;
- the control setting and objectives;
- each component and how its parameters were obtained (designed, solved, learned, searched, converted), with signal, update mechanism and regime;
- the controller interface;
- the platform;
- which metrics are reported.

`summarize-papers overview-tables` then builds the review's tables from the stored records, with no LLM calls:
- the analytic/learned × continuous/event-native grid;
- a training-signal + update-mechanism "chooser" table;
- objectives, analytic methods, hardware, reporting practice and a trend over years;
- a per-paper table (also as CSV);
- a list of records to check by hand.

```bash
uv run summarize-papers overview --source input_papers --model deepseek/deepseek-v4-pro
uv run summarize-papers overview --source input_papers --model nvidia/nemotron-3-ultra-550b-a55b:free --output-dir output_overview_free
uv run summarize-papers overview-tables --compare output_overview_free   # output_overview/overview.md and .csv
```

- **Prompt:** one LLM call per paper, carrying the codebook (about 8k tokens), a JSON template of the allowed values, and the paper text (up to 120k characters, references stripped).
- **Models (dev set, 26 papers, agent-made gold):**
  - `deepseek/deepseek-v4-pro`: 92% of fields at about $0.003 per paper.
  - `nvidia/nemotron-3-ultra-550b-a55b:free`: 91%, free. It is the best free model.
- **Second run:** `--compare DIR` points the tables at a second model's records. Fields where the two disagree are listed first under "Records to check by hand". On dev, labels the two agreed on were 96% right; the 9% they disagreed on held more than half of the errors. Details are in `notes/overview/` (local).
- **Checks:**
  - An invalid reply gets one repair call.
  - The codebook's cross-field rules are then applied (`overview.normalize`); each fix is recorded.
  - Every evidence quote is checked against the paper text; quotes not found are listed under "Records to check by hand".
- **Derived labels:** the review's categories (`design`, `quadrant`, `learning_pairs`, `regimes`, …) are computed from the facts by `overview.derive`. Loading records re-applies the current rules, so to change a convention you change the rule, with no new LLM calls.
- **Critic pass:** `--critic` adds a second call in which the model checks its own record.
- **Skipping and metadata:**
  - A record is skipped when one exists for the same PDF content, model, codebook and critic setting; use `--force` to redo it.
  - Records are named by Zotero key (`KEY__…pdf`) or by a sha prefix, and a PDF's content has only one record.
  - Citation keys, titles and years come from Zotero when it runs.
- **Codebook:** `--codebook` takes an edited copy of the codebook. Its option values must stay the same, because the record schema fixes them.
- **Gold scoring:** `--gold FILE` reads one `{"sha256", "record"}` object per line and logs per-field accuracy for every stored record in the output directory that has a gold entry.
- **Separate from summaries and eval:** this never touches `output_summaries/` or the eval directories.

## Evaluation

`summarize-papers eval` runs one or more model × extractor configurations over a folder of PDFs and writes a report. It uses its own run directory and never touches `output_summaries/` or the processed index.

```bash
mkdir -p eval/papers && cp /path/to/some/papers/*.pdf eval/papers/
uv run summarize-papers eval --source eval/papers --init-gold          # adds unlabelled entries to eval/gold.jsonl
# optionally fill in labels in eval/gold.jsonl, then:
uv run summarize-papers eval --source eval/papers --models nvidia/nemotron-3-super-120b-a12b:free,qwen/qwen3.8-27b:free
```

Each run writes `eval/runs/<timestamp>/`:
- `report.md`: per-configuration summary, a per-paper table, every quote that wasn't found, and label accuracy per field;
- `results.jsonl`: one row per paper × configuration, with metrics, tokens, cost, repairs, LLM seconds and provenance (git commit, reference hash, `--max-chars`);
- `summaries/<config>/<sha256>.json`: the validated summaries;
- `eval.log`.

Duplicate PDFs (same content) are evaluated once. Successful LLM responses are cached in `eval/cache/`, keyed by backend, model, output cap, structured-output mode and the full prompt, so changing the references, `--max-chars`, reference stripping or a PDF's filename gives a cache miss. Re-running is therefore free for everything that already succeeded; delete `eval/cache/` to force fresh calls.
- A run stopped by a quota or `--max-cost` writes a report marked "Stopped early" and exits 1.
- A Ctrl-C'd run writes no report; re-run it.
- Eval uses one worker by default because of free-tier limits.

**Gold labels:** `eval/gold.jsonl` holds one `{"sha256", "file", "labels"}` object per paper. The labels are `is_research_paper`, `paper_type` (`primary`/`synthesis`/`non_research`), `synthesis_subtype`, `year`, `first_author` (surname), `title`, and the typed Part 2 labels `classification.inference_hardware`, `.architecture`, `.credit_assignment`, `.learning_regime` and `.paradigm_families` (a list; compared as a set). Use the exact labels from `skill_data/references/json-output-contract.md`. `null` means not labelled, and the field is skipped.

**Metrics** (a rate with nothing to count is reported as n/a):
- **Quotes:** each `citable_snippets` quote is checked against the exact text the model saw, word by word, ignoring punctuation, hyphenation, ligatures and `[12]`-style citations. It counts as verbatim, near (≥70% of its word 3-grams found), or not found. Parts separated by an ellipsis must appear in order, and parts shorter than three words make a quote at best near. For PDFs whose extraction glues words together, parts of 20+ letters also match with spaces ignored.
- **Anchor coverage:** the share of sentences with a number that carry a `Source:` anchor in the same or the next sentence. Years in date context, references such as "Table 3", chip names such as "Loihi 2", and identifiers such as "CIFAR-10" don't count as numbers.
- **First person:** uses of "we/our/us" outside quoted text, per 1k words.
- Quality columns cover successful papers only; a failed paper counts as wrong for each of its gold labels.
- **Word budget:** Part 1 prose words ÷ the limit (600 primary, 1000 synthesis).
- **Evidence tags:** notable findings with exactly one allowed tag (no `Measured` for synthesis papers).
- **Reliability and cost:** first-try validity, JSON and schema repairs, tokens and cost (LLM seconds per paper are in `results.jsonl`).
- **Duplicate citation keys.**

**Caveats:**
- Zotero-style filenames ("Author - Year - Title.pdf") are part of the prompt, which inflates title, year and author accuracy. `paper_type` and `is_research_paper` are the informative labels.
- Free models are capped at 50 requests per day (1,000 once you've bought $10 of credits); the cache lets a set of papers be spread across days.
- With about 30 papers and one sample each, compare configurations per paper rather than by small differences in averages.

## Troubleshooting

- **"Model … is not available on OpenRouter"**: pick a current id from <https://openrouter.ai/models>.
- **Auth errors**: check `LLM_API_KEY`.
- **"output was truncated by the token limit"**: raise `--max-output-tokens`, or choose a model with a larger output budget.
- **Poor extraction**: install the docling extra (`uv sync --extra docling`), then `--reparse`.
- **Slow runs or timeouts**: lower `--workers`, raise `--timeout`, or reduce `--max-chars`.
- **"Rate limit exhausted … free-models-per-day"**: the free daily cap is reached; rerun after the reset time shown, or buy credits.
- **Errors with `--structured-output`**: the model or provider doesn't support it; run without the flag.

## Development

```bash
uv sync --extra docling              # dev tools (pytest, ruff) are included by default
uv run pytest                        # unit tests (no network, no docling needed)
uv run pytest -m integration         # real PDFs in test_pdfs/ and/or OpenRouter (needs LLM_API_KEY)
uv run ruff check . && uv run ruff format .
```

CI runs lint and tests on Python 3.12–3.14.

## Roadmap

- Leaner, type-specific prompts (and choosing the `--structured-output` default from evaluation results).
- Decision models beyond the classification labels: screening (in scope for the review?) and paradigm families; trusting them over the LLM where the evaluation supports it.
- Domain profiles, so the research field is hot-swappable.
- Corpus-level comparison tables, BibTeX export and staged synthesis.

## License

MIT — see `LICENSE`.
