# paper-summarizer — Revival Plan

Status: 2026-09-29. Based on a full read of the codebase at `9a8e222` plus a test run
(239 passed, 4 skipped, 2 errors from the gitignored sample PDF).
Phases 0–1 implemented on branch `fix/bugfix-pass` (283 tests passing); see checkboxes.

## Decisions (owner, 2026-09-29)

- **Backend:** OpenRouter. Budget: free where possible, at most a few cents per paper.
  Default model: free `nvidia/nemotron-3-super-120b-a12b:free` (owner asked for a free default;
  262k ctx, structured outputs). Paid fallback: `meta/muse-spark-1.3-contributor`
  (~$0.10/$0.20 per M tokens ≈ $0.007/paper). The non-contributor `meta/muse-spark-1.3` is
  ~$1.25/$4.25 per M ≈ $0.09/paper, which is over budget.
  The old default `openai/gpt-oss-120b:free` has been retired by OpenRouter.
- **Library:** Zotero, ~150 SNN-for-control papers, currently on another machine. **Ask the owner
  before assembling any PDF dataset** (baseline, gold set, fixtures).
- **Scope:** the SNN-for-control review is the primary use case. The tool should become
  field-agnostic, with SNN-control shipped as the example profile (Phase 6).
- **Jev / Ollaya:** separate future branch and PR (Phase 5). Sending *published* papers to hosted
  Jev is fine; never the owner's unpublished work.
- **Tooling:** uv instead of conda/pip; Python >=3.12, developed on 3.14. docling supports 3.10–3.14
  and was never a reason to stay on 3.11.

Each phase ends in a state that is shippable and measured. Phase 2 (eval harness) is deliberately
early: everything after it should be judged by numbers, not by eyeballing summaries.

Legend: `[ ]` todo · **(bug)** confirmed defect · **(debt)** cleanup · **(feat)** new capability ·
**(exp)** experiment, keep only if the eval says so.

---

## Phase 0 — Hygiene and baseline (½ day)

Goal: a clean starting point and a recorded "before" snapshot.

- [x] **(debt)** Migrate to uv: `uv.lock`, `.python-version` (3.14), `[dependency-groups] dev`
      (pytest, pytest-mock, ruff), docling as an optional extra; `requirements.txt` removed.
- [x] **(debt)** Ruff lint + format configured and applied (replaces the advertised-but-unconfigured black).
- [x] **(debt)** Integration tests skip when `test_pdfs/` sample is missing; `integration` marker
      deselected by default; OpenRouter integration tests now actually carry the marker.
- [x] **(debt)** Unit tests block network access (a `Config()` default of OpenRouter previously
      made real pricing requests during tests).
- [x] **(debt)** GitHub Actions CI: ruff, shellcheck, unit tests on 3.12/3.13/3.14 without docling.
- [x] **(debt)** `CLAUDE.md` with architecture, commands and the reference/schema drift warning.
- [ ] Record a **baseline run** on ~10 real PDFs → `eval/baseline_<date>/`. **Blocked:** waiting for
      the Zotero library to be moved to this machine (ask the owner first). Note: the "before"
      code now fails out of the box (retired default model), so the baseline is the fixed code.

---

## Phase 1 — Correctness bugs (1–2 days)

Goal: the existing single-call pipeline does what it claims. No behavioral redesign yet.

- [x] **P1.1 (bug)** `processed.txt` corrupts paths containing commas (`batch.py:66`). Verified:
      `"… - Spikes, Robots, and Control.pdf"` round-trips as `"… - Spikes"`, so the paper is
      re-summarized (and re-billed) every run. Replace with `processed.jsonl` (one JSON object per
      line: `pdf_path`, `sha256`, `outputs[]`, `model`, `timestamp`, `cost`). Include a one-time
      migration reader for the old format.
- [x] **P1.2 (bug)** Contradictory paper-type taxonomy in the prompt: `prompts.py:76` says
      `"primary", "survey", "commentary"`; the contract/schema say `"primary" | "synthesis"`.
      Fix the prompt. Also fix stale mentions in `batch.py:117`, README (output folders are
      `primary/`, `synthesis/`, `non_research/`), and the CLI help.
- [x] **P1.3 (bug)** Output-budget section says "≤600 words" for all types (`prompts.py:94`);
      synthesis allows 1000. Make it type-aware (or reference the template instead of restating).
- [x] **P1.4 (bug)** Schema repair resends the full original prompt, including up to ~50k tokens of
      paper text (`pipeline.py:209`). This makes a failing paper cost roughly 3× the input tokens.
      Send only the
      bad JSON + errors + the relevant contract section; include paper text only as an opt-in
      fallback on the last attempt.
- [x] **P1.5 (bug)** Stacked retries: the OpenAI SDK retries 2× by default (incl. timeouts) and
      `_complete_with_retries` (`llm.py:404`) adds 2× more → up to 9 attempts, ~6 min worst case per
      hung call at 120 s. Set `max_retries=0` on the SDK client and own retries in one place
      (retry timeouts/connection errors too, with jitter).
- [x] **P1.6 (bug)** Handle `message.content is None` (reasoning models / refusals) at
      `llm.py:133` — currently a `TypeError` on `len(None)`. Raise a clear `LLMError`.
- [x] **P1.7 (bug)** Detect `finish_reason == "length"`: truncated JSON is currently sent to the
      syntax-repair step, which cannot recover missing content. Instead retry with a raised
      `max_tokens` or fail fast with a clear message.
- [x] **P1.8 (bug)** Parser cache ignores extractor choice (`parser.py:45`): switching
      `--extractor` silently reuses the old text. Key the cache by extractor
      (`<stem>.<extractor>.md`) or store a small header/sidecar with extractor + version.
- [x] **P1.9 (bug/perf)** Lazy-import docling: `import summarizer.cli` takes ~3 s even for
      `--help`, `--dry-run`, or `--extractor pypdf`. Move the import inside `_run_docling`; make
      docling an optional extra (`pip install paper-summarizer[docling]`).
- [x] **P1.10 (perf)** Reuse one `DocumentConverter` per process instead of constructing one per
      PDF (`parser.py:106`) — it reloads layout models each time.
- [x] **P1.11 (bug)** `--verbose` is a no-op: default is on, but there are zero `logger.debug`
      calls. Either add debug diagnostics (prompt sizes, raw response snippet on failure, repair
      diffs) or remove the flag. Default should be INFO.
- [x] **P1.12 (bug)** Unresolvable year becomes `0` (`pipeline.py:239`) → keys like
      `smith0spiking`. Use `nd` (BibTeX convention) or fail validation into repair, and flag it.
- [x] **P1.13 (bug)** Single-file mode never aggregates cost (`cli.py:127` passes no accumulator).
      Share the batch code path for single files (a batch of one) to kill the duplicated
      write/index logic in `cli._run_single` vs `batch.run_batch`.
- [x] **P1.14 (bug)** `processed.txt` is rewritten non-atomically after every paper; a crash
      mid-write can truncate it. Write to temp + `os.replace` (or append-only JSONL, see P1.1).
- [x] **P1.15 (debt)** Stale docs/comments:
      - `models.py:~294` claims references ≈2.5k tokens — measured ≈11k (prompt overhead is
        ~44k chars). The "200k chars fits a 55k window" rationale is wrong; recompute defaults.
      - `LMStudioClient` / `_check_lm_studio` names; `__init__.py`/CLI description say "local LLM via
        LM Studio" while default is OpenRouter; `Config.base_url` default (localhost) ≠ CLI default
        (OpenRouter).
      - OpenRouter headers still say `agent-paper` (`llm.py:269`).
      - "LLM Call 1 / Call 2" docstrings and legacy two-call mock shapes in `tests/conftest.py`.
      - `output-template.md` says output is "saved as `<citationkey>.md` next to the PDF".
      - README example output doesn't match the renderer.
- [x] **P1.16 (bug)** Silent truncation: `parse_pdf` cuts at `max_chars` without logging. Log
      a warning with original vs kept length; record `truncated: true` in the output metadata.
- [x] **P1.17 (debt)** `collect_pdfs.sh`: `while read -r` breaks on names with leading/trailing
      spaces or backslashes, and same-name collisions still overwrite. Use `find -print0` /
      `read -d ''`, and `cp -n` + a counter (or hash suffix).
- [x] **P1.18 (debt)** Small cleanups: unused `abs_path`, tqdm/log interleaving
      (`logging_redirect_tqdm`), references loaded once per batch. *Kept:* dry-run still reports
      would-process files under "skipped" (now also lists them).
- [x] Regression tests for each bug above (44 new tests).

**Found and fixed during implementation (not in the original review):**
- [x] Retired default model `openai/gpt-oss-120b:free` → tool failed out of the box. Added an
      OpenRouter preflight (API key set, model id still listed) that fails fast instead of failing
      every paper.
- [x] Rendered header collapsed into one paragraph (single newlines in markdown); now hard breaks.
      Empty findings/snippets sections now say "not reported".
- [x] `find_pdfs` missed `.PDF` files and picked up macOS `._*` AppleDouble junk.
- [x] `--skill-data-dir` defaulted to a CWD-relative path, so running the CLI from any other
      directory failed with "References directory not found". Now resolved from the checkout.
      (Shipping references as package data is part of Phase 6.)
- [x] Tests that call `main()` wrote `logs/run_*.log` into the repo; tests now run in a tmp CWD.
- [x] Prompt said Part 2 fields are "exactly 1 sentence", contradicting "Learning mechanism: 2".
- [x] Weak test `test_pipeline_schema_repair_tokens_added_to_accumulator` swallows exceptions;
      superseded by new repair-path tests (could be deleted).
- [x] Verified end-to-end with real docling + real OpenAI SDK against a local fake server:
      comma paths skip correctly on rerun; `finish_reason=length` fails after 1 request; an enum
      repair prompt is ~9k chars instead of ~44k.

---

## Phase 2 — Evaluation harness (2–3 days, then ongoing)

Goal: measure quality, reliability and cost, so every later change is a reproducible
comparison instead of anecdote. This is also where "more issues found during eval" land.

### 2a. Gold set
- [ ] **Ask the owner first** — use the real Zotero library. Select 25–40 PDFs spanning the real distribution: primary (sim-only, real robot,
      neuromorphic chip), reviews/surveys, short commentaries/perspectives (the known hard case in
      `output-template.md`), non-research (slides, theses chapters), scanned/odd-layout PDFs,
      non-English if relevant.
- [ ] Hand-label the **categorical** fields only (cheap to label, easy to score):
      `is_research_paper`, `paper_type`, `synthesis_subtype`, year, first author, and the
      canonical-vocabulary Part 2 fields (inference hardware class, fully spiking vs hybrid,
      credit-assignment label, online/offline/mixed, task type, sim vs real, learning-paradigm
      families — multi-label).
- [ ] Store as `eval/gold/<sha256>.json` keyed by file hash (PDFs themselves stay out of git).

### 2b. Automatic metrics (no labels needed)
- [ ] Schema-valid on first try; # syntax repairs; # schema repairs; hard-fail rate.
- [ ] Tokens in/out, USD cost and latency per paper; cost per *valid* summary.
- [ ] **Quote faithfulness**: fuzzy-match every `citable_snippets[].quote` against the parsed
      paper text (normalize whitespace/hyphenation; e.g. `rapidfuzz.partial_ratio`). Report
      % verbatim, % near-miss, % not found (hallucinated). Cheap and very informative.
- [ ] Source-anchor coverage: % of sentences with numbers that carry `(Source: …)`.
- [ ] Voice check: first-person plural outside quotes ("we propose", "our method") → violation.
- [ ] Word/sentence-limit adherence per section.
- [ ] Evidence tag present and exactly one per notable finding.
- [ ] Citation-key format + uniqueness across the corpus.

### 2c. Labelled metrics
- [ ] Accuracy / macro-F1 per categorical field; confusion matrix for `paper_type`
      (commentary→primary misclassification is the known failure mode).
- [ ] Metadata exact-match (year, first author surname, title normalized).

### 2d. Tooling
- [ ] `summarize-papers eval --gold eval/gold --models m1,m2 --extractors docling,pypdf`
      producing a `report.md` + `results.jsonl` (one row per paper×config).
- [ ] Cache LLM responses by (prompt hash, model) so metric changes re-score without re-billing.
- [ ] Optional LLM-as-judge rubric for prose quality (critical stance, specificity) on a
      subsample — kept separate from the deterministic metrics and never the only signal.
- [ ] Compare against the Phase 0 baseline. Re-run after every phase.

### 2e. Model sweep (first real use of the harness)
- [ ] Compare 4–6 models on quality × cost × failure rate; pick the default with evidence.
      Candidates (OpenRouter, 2026-09-29, all support `structured_outputs`):
      free: `nvidia/nemotron-3-super-120b-a12b:free` (current default), `qwen/qwen3.8-27b:free`,
      `dots-studio/dots-3-note-preview:free`, `nvidia/nemotron-3-ultra-550b-a55b:free` and
      `thinkingmachines/inkling:free` (no structured outputs); paid: `meta/muse-spark-1.3-contributor`
      (~$0.007/paper); plus 1 local LM Studio model. Free tiers have daily request caps, which matters for
      a 150-paper run with repairs.
      Check what the "contributor" tier implies for data use (fine for published papers).

---

## Phase 3 — LLM robustness, prompt, and cost (2–4 days)

Goal: fewer repairs, lower tokens, same or better quality — each item validated in Phase 2.

- [ ] **(feat)** Structured outputs: send `response_format={"type": "json_schema", …}` derived from
      `LLMResponse.model_json_schema()` when the backend/model supports it (OpenRouter
      `supported_parameters`, LM Studio supports JSON schema). Keep the current parse→repair
      path as fallback. Expect first-try validity to jump and repair code to become rarely used.
- [ ] **(feat)** Two-stage or conditional prompt: decide `paper_type` first (cheap call or a
      decision model — see Phase 5), then send **only** the relevant template. Today every call
      carries primary + synthesis + non-research templates and the whole learning-paradigms
      taxonomy (~11k tokens) even for a commentary.
- [ ] **(feat)** Prompt caching: move references into a `system` message as a stable prefix so
      providers with prefix caching (OpenRouter → Anthropic/OpenAI/DeepSeek, etc.) can discount the
      ~11k repeated tokens across a batch. Measure cache-hit tokens in usage.
- [ ] **(feat)** Smarter text budgeting instead of head-truncation: strip the References /
      Bibliography section and boilerplate (acknowledgements, license footers) before truncating;
      keep appendix only if budget allows. Often saves 20–40% of tokens.
- [ ] **(feat)** Budget by tokens, not chars: use the model's `context_length` (already fetched
      from OpenRouter in `fetch_model_pricing`) to set the paper budget automatically.
- [ ] **(exp)** Split Part 2 into a separate focused extraction call for primary papers (the old
      two-call design) — only if the eval shows quality gains that justify the cost.
- [ ] **(feat)** `--max-cost` guard: stop the batch when accumulated cost exceeds a budget;
      `--estimate` mode that prices a batch using token counts before calling the LLM.
- [ ] **(debt)** Rename `LMStudioClient` → `LLMClient`; one `Backend` config object instead of
      substring-matching `"openrouter.ai"` in `create_client`.

---

## Phase 4 — Structured data, metadata, and parsing (3–5 days)

Goal: outputs that downstream tooling (tables, synthesis) can consume; trustworthy metadata.

- [ ] **(feat)** Write the validated `PaperSummary` as `<citekey>.json` next to each `.md`, plus
      provenance: pdf sha256, model, extractor, prompt/reference version hash, cost, timestamp,
      truncation flag, per-field confidence (Phase 5). The markdown becomes a *view* of the JSON.
- [ ] **(feat)** `summarize-papers render` — regenerate markdown from JSON without LLM calls
      (template changes become free).
- [ ] **(feat)** Content-hash identity: key the processed index by PDF sha256, not absolute path,
      so moving the library or re-running `collect_pdfs.sh` doesn't re-trigger everything, and
      duplicate PDFs are detected.
- [ ] **(feat)** Deterministic metadata first, LLM second: extract DOI/arXiv id from the PDF text
      and resolve via Crossref / arXiv / Semantic Scholar for title, authors, year, venue. LLM only
      fills gaps. Removes most of the year/citation-key repair code.
- [ ] **(exp)** Zotero integration: read the local Zotero SQLite (read-only) or Better BibTeX
      export to reuse the user's existing citation keys and collections/tags, and to process only
      new items.
- [ ] **(feat)** Typed categorical fields next to the free-text Part 2 fields (e.g.
      `inference_hardware_class: Literal["CPU/GPU", "Neuromorphic emulator/SDK",
      "Physical neuromorphic chip", "not reported"]`, `architecture_class`, `credit_assignment`,
      `learning_regime`, `task_action_space`, `environment_kind`, `paradigm_families: list[...]`).
      The references already define these vocabularies; the schema should enforce them. These are
      the table columns for Phase 7.
- [ ] **(feat)** Don't write caches into the user's library by default: move the parse cache to
      `~/.cache/paper-summarizer/<sha256>.<extractor>.md` (or `--cache-dir`). The current behavior
      fails on read-only source folders and pollutes Zotero storage.
- [ ] **(exp)** Evaluate alternative parsers through the harness: docling with its VLM pipeline
      (granite-docling), Marker, MinerU, olmOCR. Criteria: quote-faithfulness score, equation/table
      fidelity, speed on Apple Silicon, install weight. Keep pypdf as the light fallback.
- [ ] **(feat)** Section-aware parsing output (title/abstract/methods/results/references spans)
      — needed for conditional prompts (P3), text budgeting (P3), and decision-model states (P5).

---

## Phase 5 — Decision models: Jev / Ollaya (exploratory, 3–6 days)

Branch: separate PR after Phases 0–4 (owner decision). Hosted Jev is approved for published papers.
OpenRouter's public model list shows `typesafe/jev-router` (an LLM *router* that runs on Jev and
picks models/reasoning effort per request; variable pricing). The owner reports ~6 decision
models on OpenRouter, but they don't appear in `/api/v1/models`, so check how OpenRouter exposes them
(separate endpoint?) at the start of this phase. The router itself is also a candidate for the 2e sweep.

Background (checked 2026-09-29): **Jev** is TypeSafe's hosted "decision model". It returns
calibrated probabilities for typed questions in one forward pass and never generates text.
**Ollaya** is a new (launched ~2026-09-26), Apache-2.0, Ollama-style local runtime for open decision
models. It is wire-compatible with TypeSafe's `/v1/systemone` API, which means one client can target
either. Question types: `choice` (2–255 options), `score` (2–10 levels), `noul` (yes/no).
Up to 256 questions per request. Server state limit is 65k tokens, but the models themselves are
smaller (some are documented at 8k–16k context). `state_truncated` is reported. Models include
`winnow:e4b` (~0.72 on their typed-decision benchmark vs Jev 0.738), `laya` (≈400M, ~10 ms),
`nli` (DeBERTa/ModernBERT), `decider`, and `kev`. On macOS, only laya/nli run on the GPU (MLX);
the others run on CPU.

**Fit for this project:** a decision model cannot replace the summarizer LLM, because the summaries
are generated prose. It is a good fit for the *typed* decisions around the LLM: cheaper, faster,
calibrated, and independent of the LLM. That independence is what makes it useful as a
cross-check.

### 5a. Integration scaffold
- [ ] **(feat)** `summarizer/decide.py`: a `Decider` protocol with (1) `OllayaDecider` (HTTP to
      `localhost:11435`, `/api/decide`, handles `STATE_TRUNCATED`, `QUEUE_FULL`) and
      (2) `LLMDecider` fallback that asks the main LLM the same questions as constrained JSON. Both
      implementations are needed to evaluate the decision model against something.
- [ ] **(feat)** Question sets as versioned JSON in `skill_data/questions/` (generated from the
      same vocabularies as the references so they can't drift).
- [ ] **(feat)** `--decider {none,ollaya,jev,llm}` flag; decisions + probabilities stored in the
      output JSON under `decisions`.
- [ ] Spike first: install Ollaya on the Mac, time `laya` vs `winnow:e4b` on an abstract-sized
      state and a methods-section-sized state; note CPU latency for winnow on Apple Silicon.

### 5b. Candidate uses (each gated by the Phase 2 eval)
- [ ] **(exp) Pre-gate triage** on title + abstract + intro (fits the context limit):
      `is_research_paper` (noul), `paper_type` (choice), `synthesis_subtype` (choice), and
      **relevance to SNN control** (score 0–3). Low-relevance papers get a cheap summary path or are
      skipped. This is the biggest cost lever on a large Zotero library. It also feeds the
      conditional prompt in Phase 3.
- [ ] **(exp) Canonical Part 2 classification**: ask all vocabulary fields in one request
      (inference hardware class, fully spiking/hybrid, credit assignment, online/offline/mixed,
      task action space, sim/real, energy-evidence level) over the methods/experiments sections.
      Multi-label paradigm families → one `noul` per family.
- [ ] **(exp) Cross-check + review queue**: compare decision-model answers with the LLM's typed
      fields; disagreement or low confidence → `needs_review: [...]` in the JSON, and a
      `summarize-papers review` listing. Calibrated probabilities give principled thresholds.
- [ ] **(exp) Faithfulness via NLI**: for each notable finding / citable claim, retrieve the
      best-matching paper passage (the Phase 2 fuzzy matcher) and ask `nli` whether it
      entails the claim; also verify the evidence tag (Measured/Reported/Claimed/Attributed) with a
      `choice` question. Flags unsupported claims and upgraded evidence.
- [ ] **(exp) Calibration on own data**: use the gold labels to refit via an Ollaya Modelfile
      (`FROM laya` + `QUESTIONS` + labelled data) and compare pre/post calibration (ECE, accuracy).
- [ ] **(exp) Hosted Jev comparison** on the gold set (same client via `TYPESAFE_BASE_URL`), only
      if pricing is acceptable. Papers are published work, but note what leaves the machine.

### 5c. Decision criteria (write down before running)
- [ ] Adopt a use only if it beats the `LLMDecider` or the single-call baseline on accuracy/macro-F1,
      or matches it at a clearly lower cost or latency. Also check that calibration holds on
      *scientific* text; published benchmarks are support/triage-style text, so expect domain
      shift.
- [ ] Risks to track: project is days old (API churn, pin versions); context limits force
      section selection; Mac acceleration is partial; one more local service to run. Keep it
      optional and off by default until it wins the eval.

---

## Phase 6 — Generalization: domain profiles (2–3 days)

Goal: make the README's "topic-agnostic" claim true. SNN specifics are currently hard-coded in
four places: the `SummaryPart2` schema, the renderer headings (including the "Relevance to a review on
'spiking neural networks for control'" heading), `_PRIMARY_PART2_FIELDS` in `pipeline.py`, and
the prompt tail.

- [ ] **(feat)** A *profile* directory = references + Part 2 field spec (YAML/JSON → dynamic
      pydantic model via `create_model`) + typed vocabularies + decision questions + render
      template (Jinja2) + review topic string. `--profile snn-control` is the default.
- [ ] **(feat)** Generate the JSON contract section of the prompt from the schema, so the
      contract, schema, repair hints and renderer can't drift apart. Several Phase 1 bugs
      came from exactly this drift.
- [ ] **(feat)** Version profiles; store `profile@version` in output provenance; `render` can
      re-render old JSON with a new template.
- [ ] Ship a second tiny profile (e.g. generic ML paper) as a proof and test fixture.

---

## Phase 7 — Roadmap features: corpus tables and synthesis (open-ended)

Goal: the README roadmap, built on the Phase 4 JSON outputs.

- [ ] **(feat)** `summarize-papers table` — build comparison tables from the typed fields
      (CSV/XLSX/Markdown/LaTeX) with citation keys; filters by paper type, tags, relevance.
- [ ] **(feat)** BibTeX export for all summarized papers (from resolved metadata).
- [ ] **(feat)** Corpus aggregation: counts/trends per vocabulary (e.g. share of papers on
      physical chips, measured vs claimed energy), with every number traceable to citation keys.
- [ ] **(feat)** Staged synthesis ("multistage RAG 2.0", no embedding index): cluster by typed
      fields/tags → per-cluster synthesis from the JSON summaries (not the PDFs) → section
      drafts with citation keys, then a verification pass that each cited claim exists in the
      source summary (reuse the Phase 5 NLI check).
- [ ] **(exp)** PRISMA-style screening for systematic reviews: explicit inclusion/exclusion
      criteria as decision-model questions, logged with probabilities, giving an auditable
      screening record.

---

## Suggested sequencing

| Order | Phase | Why |
|---|---|---|
| 1 | 0 + 1 | Cheap, clear wins; stops paying for duplicate runs and repair tokens |
| 2 | 2 | Everything after this is measured |
| 3 | 4 (JSON sidecar + hash identity only) | Small change, unlocks 5–7 |
| 4 | 3 | Biggest quality/cost levers, validated by 2 |
| 5 | 5a spike → 5b triage | Try decision models where they have the best chance (short-input triage) |
| 6 | Rest of 4, 5, then 6, 7 | Build on measured, structured outputs |

## Open questions for the owner
- Should Phase 6 (domain profiles) move ahead of Phase 3, given the field-agnostic goal?
