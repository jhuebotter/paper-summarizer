# Output Template v2

Two-part summary saved as `<citationkey>.md` next to the PDF.

---

## Paper type classification

Classify as: **primary research** or **synthesis**.

- **Primary research** — new experiments, new method, empirical results. May include theoretical components, but experimental validation is central.
- **Synthesis** — primary contribution is a novel framework, synthesis, argument, or aggregation of prior work. Covers: systematic reviews, surveys, perspectives, opinions, commentaries. If the paper reviews ≥10 prior studies, proposes a conceptual framework without new experiments, or is explicitly labeled "perspective / opinion / commentary / review", classify as synthesis.

**Short perspective/focus/editorial pieces are always synthesis.** A 1–4 page article published in a high-impact journal (e.g., Science, Science Robotics, Nature, Current Biology) that comments on, contextualises, or highlights another group's paper — rather than reporting its own experiments — is **synthesis** regardless of its brevity. Signals: no Methods section, no Results section, no data figures from the authors' own experiments, text that says "in their study / in a recent paper / [Author et al.] show…" referring to others' work. Do not mistake the described work (from the paper being commented on) for the author's own contributions.

**`is_research_paper: false` is only for non-academic documents** (slides, forms, websites, software manuals, textbook chapters without a DOI, etc.). Any document published in a scientific journal, conference proceedings, or preprint server — including editorials, focus articles, opinions, commentaries, and perspectives — must be `is_research_paper: true`. If a document is an academic publication but lacks original experiments, it is **synthesis**, not non-research.

---

## Global execution rules (must follow)

- **Analytical voice — third person only:** You are a critical research assistant dissecting a paper for a review author. Never write "We propose / We show / Our method / We design" — always "The authors propose / DeWolf et al. show / Their method / The paper designs". This applies everywhere: TL;DR, all Part 1 sections, Part 2 fields. You are not a coauthor and must not adopt the paper's promotional voice.
- **Analytical stance:** Maintain critical distance. Your job is to dissect, not to endorse. If a claim is strong but evidence is thin, say so explicitly. Do not help the paper sell itself.
- **Trace anchors for specifics:** Any *specific* numeric value or concrete claim (performance, parameters, DOF, energy, dataset size, thresholds, etc.) must include a short source anchor such as `Source: Sec. X / Fig. Y / Tbl. Z / App. A` in the same sentence (or immediately after).
- **Required fields must not be empty:** If a section exists in the template, fill it. If truly unavailable, write `not reported` (or `not applicable` where appropriate). Do not leave blanks.
- **No hype / no padding:** Use precise language. Avoid vague qualifiers ("slightly", "significant" without a number). Prefer one headline number with units and comparison context.
- **Dense sentences only:** No sentence should be purely rhetorical; each must contain a concrete detail, comparison, limitation, or clearly scoped claim.
- **Sentence limits are defaults, not excuses for vagueness:** Respect limits, but if needed to avoid losing essential specificity, allow one extra sentence in **Methods** or **Results** (still obey word limit).
- **Evidence strength:** When repeating a claim that is not directly demonstrated, label it clearly (e.g., "Claimed — not experimentally demonstrated"). Do not upgrade claims to facts.
- **Citation hygiene (critical for downstream review writing):** Do not cite a claim as supported by this paper if it is only mentioned via a citation to prior work. If a statement comes from another paper, mark it explicitly as **(Attributed)** and do not treat it as this paper's result. Always cite with (Author, Year) in such cases, never e.g. [14].
- **Consistency:** Use `not reported` when the concept applies but is omitted; use `not applicable` when it genuinely does not apply.
- **Arguments:** Author arguments are perspectives ("the authors argue..."), rarely facts.
- **Author names:** Copy author names verbatim from the paper's byline or header. Do not alter spelling, capitalisation, or diacritics.
- **Citation key:** Always `firstauthorYEARfirstword` (all lowercase, letters and digits only). For the year, trust the source filename first. If the first word of the title is non-descriptive (e.g., "A", "An", "The", "On", "Towards", "Novel"), skip it and use the next descriptive word instead. Examples: "A Novel Robotic Controller…" → `marrero2024novel`; "On the Role of…" → `smith2020role`. Do not use the first name of the author in the authorname.

---

## Template: Primary Research Paper

```markdown
# <Title>
**Citation key:** firstauthorYEARfirstword
**Authors:** First Author, Second Author, et al.
**Year:** YYYY
**Venue:** Journal / Conference / Preprint (arXiv:XXXX.XXXXX if applicable)
**Paper Type:** primary research
**Tags:** comma-separated topic tags; primarily from the paper itself, add at most 3 reviewer tags for cross-paper grouping

---

## TL;DR
One or two sentences: [Method] [achieves what] [on task] [key result]. No hype words. Usable as a comparison table row.
Include (when available) one concrete metric + scope (simulation vs real robot; benchmark/task name) + trace anchor if numeric. (Source: Fig./Tbl./Sec.)

---

## Part 1: Paper Summary
**≤600 words total. ≤3 sentences per section (4 allowed in Methods or Results if needed to avoid vagueness).**

### Problem & Motivation
What specific gap does this paper address? Why hadn't it been solved? What prior work does it build on?
Include at least one concrete anchor to what was missing/blocked previously. (Source: Sec.)

### Core Contribution
Main technical proposal. What is genuinely novel vs. a combination of existing ideas? Proof-of-concept or meaningful advance?
If the novelty is mainly integration/engineering, state that explicitly. What is the Contribution type: algorithm / architecture / training procedure / benchmark/task / system integration / theory-only (with empirical demo) (these are non exhaustive examples, multiple or others may apply).

### Methods
Key components, how they work, why chosen. Highlight ablated or explicitly justified design choices. Framework/codebase if public.
If training uses multiple phases (e.g., pretrain → convert → finetune; world model → freeze → policy), state the phases explicitly. (Source: Sec./Fig.) Evaluation setting: simulation only / real robot / sim + real / offline dataset / mixed (Source: Sec.)

### Results
Headline number with unit and comparison context (e.g. "4.13% higher RMSE than analytical baseline" — not "slightly worse"). Key ablation insight if any - what is ablated/swept/controlled?
State the primary evaluation metric(s) and whether the metric matches the optimization objective/reward. Add a comparability flag: Comparable / partly comparable / not comparable (and why, in 5–10 words). (Source: Fig./Tbl./Sec.)

### Key Takeaways
Key conclusions the authors draw and how they influence the field. Was the research question answered? What changed in our understanding? (Source: Sec./App.)

### Limitations
Author-acknowledged gaps + unacknowledged ones. Negative results: what was tried and failed? Papers often bury these.
Separate "not tested" from "tested and failed" where possible. (Source: Sec./App.). Suggested considerations if present: robustness test / real-world validation / fair baseline / ablation / scaling evidence / statistical support / generalization / safety constraint handling ...

### Open Problems & Future Directions

**Future work proposed** (what the paper suggests should come next):
- ... (Paper-identified)

**Open questions** (theoretical or empirical questions left unresolved):
- ... (Paper-identified)
- ... (Reviewer-noted)

### Critical Assessment
≤2 sentences. Direct and specific: what does this paper fail to demonstrate? Per-paper only — no field-level generalizations.
If comparison is potentially unfair, name the concrete parity issue (setup mismatch, capacity mismatch, training budget mismatch). (Source: Sec./Tbl.)

### Notable findings
One finding per bullet. Include a trace anchor for any specific claim.Exactly one tag each: 
- Measured — directly observed in this paper's experiments
- Reported — paper's own results but no statistical support
- Claimed — asserted without direct experimental support
- Attributed — from another paper this paper cites
Template:
- [finding] (Measured) (Source: Fig./Tbl./Sec.)
- [finding] (Reported) (Source: Fig./Tbl./Sec.)
- ...
Example: "- 100× lower inference energy on Loihi vs. GPU (Measured) (Source: Tbl. 2)"
Example: "- Suitable for neuromorphic deployment (Claimed — no chip deployment attempted) (Source: Sec. 5)"

### Citable Snippets
2-4 bullets listing specific claims or sections where this paper would be cited (each with a trace anchor). Max 1 quotable sentence per bullet. If no suitable verbatim sentence exists, write `*no suitable quotable sentence*` and omit the blockquote. Quotes must be verbatim from this paper only — never from cited works. Each must be searchable in the PDF. Avoid generic boilerplate; pick quotes that uniquely define the method, evidence, or interpretation. Prefer at least one (Definition | Method) over pure (Claim).
- **[Specific claim or result]** (Source: Sec./Fig./Tbl.)
  > (Definition | Method | Claim | Result) "[Verbatim quote searchable in PDF]"
- **[Specific claim or result]** (Source: Sec./Fig./Tbl.)
  > (Definition | Method | Claim | Result) "[Verbatim quote]" — or: *no suitable quotable sentence*
- **[Specific claim or result]** (Source: Sec./Fig./Tbl.)
  > (Definition | Method | Claim | Result) "[Verbatim quote]"

### Relevance to a review on "spiking neural networks for control"
Primary source / background / contrasting example. Which hypothetical sections of the review is this work most interesting for?

---

## Part 2: SNN Control Extraction
**INCLUDE IN PRIMARY RESEARCH PAPERS ONLY!**

(See references/snn-extraction-fields.md for field definitions and references/learning-paradigms.md for learning mechanism guidance.)

**1 full and detailed sentence per field. Learning mechanism: 2 sentences max. Notable findings: bullet points.
Numbers: neuron/parameter counts and DOF only. No hyperparameters. No padding.
Task environment (simulated/real robot) and controller hardware (CPU/GPU/neuromorphic) are orthogonal — report both clearly and separately.**

**Neuron model:** (include source anchor if specific)
**Network architecture:** (state explicitly: fully spiking or hybrid)
**Model scale:** (neurons and/or parameters only)
**Simulator / framework:**
**Hardware (training):**
**Controller hardware (inference):** (CPU/GPU | neuromorphic emulator/SDK | physical neuromorphic chip — if a chip, specify which one; see snn-extraction-fields.md)
**Control task:**
**Task type:**
**Task complexity & scale:** (include state DOF and action DOF)
**Simulation environment:**
**Spike encoding:**
**Action decoding:**
**Learning mechanism:** Must include (1) algorithm family, (2) training signal/loss, (3) optimization structure (end-to-end vs staged/frozen components). (Source: Sec.)
**Credit assignment scope:** (if BPTT-based, state whether truncated/full and whether gradients flow through model+policy or policy-only; no numbers) (Source: Sec.)
**Online vs. offline:**
**Data collection:**
**Key training details:** (structural decisions only — no hyperparameter values)
**Comparison to baselines:** (note missing baselines; flag single-run results)
Also note parity: same task setup/observations/actions/training budget? Capacity match or mismatch? (Source: Sec./Tbl.)
```

---

## Template: Synthesis Paper

Covers: systematic reviews, surveys, perspectives, opinions, commentaries.

All synthesis fields are required in JSON. Where a section is genuinely inapplicable for the subtype, write the string `"not applicable"` as the field value — **never omit the key from the JSON object**. Never leave a section blank.

Part 2 (SNN extraction): return `null` in JSON — do not fill any extraction fields.

```markdown
# <Title>
**Citation key:** firstauthorYEARfirstword
**Authors:** First Author, Second Author, et al.
**Year:** YYYY
**Venue:** Journal / Conference / Preprint (arXiv:XXXX.XXXXX)
**Paper Type:** synthesis — [type] (for type choose the most fitting of: review/survey/perspective/commentary/opinion/tutorial/position/meta-analysis/mixed)
**Tags:** comma-separated topic tags and keywords

---

## TL;DR
One or two sentences: what the paper covers, argues, or proposes, and what it adds to the field beyond aggregation. No hype words. (Source: Sec.)

---

## Part 1: Paper Summary
**≤1000 words total. ≤3 sentences per section unless noted.**

### Target Paper(s) / Field
*For commentaries:* What result from the target paper anchors the argument? Attribute explicitly to original authors.
*For reviews/surveys:* What is general field / research focus that is being investigated?
≤2 sentences.

### Detailed Scope & Coverage
What domain, time span, N papers reviewed, selection method (systematic keyword search with explicit criteria / narrative author-discretion / not described)?
If not applicable (e.g. short commentary with no formal review scope): write `"not applicable"` as the field value. **Do not omit this field from JSON.**

### Taxonomy & Organization
How the field or argument is categorized. If the paper defines axes/categories, list them explicitly. For any summary tables cited, describe table content in one sentence each — not just the table number.
If not applicable (no explicit taxonomy proposed): write `"not applicable"`. **Do not omit this field from JSON.**

### Core Argument
The central claim or position this paper advances. "The authors argue that..." — not stated as fact. What evidence or reasoning supports the position?
If not applicable (neutral systematic review with no single central argument): write `"not applicable"`. **Do not omit this field from JSON.**

### Synthesis Contribution
What does this paper itself add beyond aggregating existing work?
- For reviews/surveys: novel taxonomy, new framing, named constructs introduced, re-interpretation of prior findings...
- For perspectives: the conceptual advance this paper originates.
- For commentaries: the specific reframing or extension of the target paper's findings.
*"Synthesizes the literature" is not acceptable — name something concrete.*
(Source: Sec.)

### Key Claims & Narrative
Main claims and the narrative arc. Note whether claims are backed by quantitative evidence from reviewed papers or are editorial judgments.
Distinguish between claims this paper makes from first principles vs. claims derived from aggregated prior evidence.

### Key Takeaways
What should a reader take away from this synthesis? What changed or should change in how the field thinks about this topic? (Source: Sec.)

### Limitations
For reviews/surveys: scope limitations, selection bias, recency.
For perspectives/opinions/commentaries: does the argument overextend from the target evidence? Are claims appropriately scoped?
≤3 sentences.

### Open Problems & Future Directions

**Gaps identified** (what the paper flags as missing in the field):
- ... (Paper-identified)

**Open questions** (theoretical or empirical questions left unresolved):
- ... (Paper-identified)
- ... (Reviewer-noted)

**Suggested research focus** (specific next experiments or directions the paper proposes):
- ... (Paper-identified)

### Critical Assessment
≤2 sentences. Direct and specific. Per-paper only — no generic field-level generalizations.
- For reviews/surveys: scope limitations, selection bias, and whether quantitative claims are backed by reviewed papers or are editorial.
- For perspectives/opinions: does the framing appropriately represent the strength of the evidence? Does it overextend?
Where the paper makes quantitative claims (counts, trends, performance summaries), add trace anchors (Source: Fig./Tbl./Sec.).

### Notable findings
One finding per bullet. Include a trace anchor for any specific claim.Exactly one tag each: 
- Reported — paper's own results but no statistical support
- Claimed — asserted without direct experimental support
- Attributed — from another paper this paper cites
Template:
- [finding] (Measured) (Source: Fig./Tbl./Sec.)
- [finding] (Reported) (Source: Fig./Tbl./Sec.)
- ...
Example: "- 100× lower inference energy on Loihi vs. GPU (Measured) (Source: Tbl. 2)"
Example: "- Suitable for neuromorphic deployment (Claimed — no chip deployment attempted) (Source: Sec. 5)"

### Citable Snippets
2-4 bullets listing specific claims or sections where this paper would be cited (each with a trace anchor). Max 1 quotable sentence per bullet. If no suitable verbatim sentence exists, write `*no suitable quotable sentence*` and omit the blockquote. Quotes must be verbatim from this paper only — never from cited works. Each must be searchable in the PDF. Avoid generic boilerplate; pick quotes that uniquely define the method, evidence, or interpretation. Prefer at least one (Definition | Method) over pure (Claim).
- **[Specific claim or result]** (Source: Sec./Fig./Tbl.)
  > (Definition | Quantitative synthesis | Editorial judgment | Attributed) "[Verbatim quote searchable in PDF]"
- **[Specific claim or result]** (Source: Sec./Fig./Tbl.)
  > (Definition | Quantitative synthesis | Editorial judgment | Attributed) "[Verbatim quote]" — or: *no suitable quotable sentence*
- ...


### Relevance to a review on "spiking neural networks for control"
Primary source / background / contrasting example. Which hypothetical sections of the review is this work most interesting for?

---

## Part 2: SNN Control Extraction
*Not applicable (synthesis). Return `null` in JSON.*
```
