# Output Template

Two-part summary saved as `<citationkey>.md` next to the PDF.

---

## Paper type classification

Classify as: **primary research** / **survey/review** / **commentary/opinion**.

**Commentary rule:** Author arguments are perspectives ("the author argues..."), never facts. Quantitative claims attributed to the target paper. Part 2 describes the target system with attribution "(as described in [author year], re [target paper])".

---

## Global execution rules (must follow)

- **Trace anchors for specifics:** Any *specific* numeric value or concrete claim (performance, parameters, DOF, energy, dataset size, thresholds, etc.) must include a short source anchor such as `Source: Sec. X / Fig. Y / Tbl. Z / App. A` in the same sentence (or immediately after).
- **Required fields must not be empty:** If a section exists in the template, fill it. If truly unavailable, write `not reported` (or `not applicable` where appropriate). Do not leave blanks (e.g., **Cite for** must have 1–3 bullets).
- **No hype / no padding:** Use precise language. Avoid vague qualifiers (“slightly”, “significant” without a number). Prefer one headline number with units and comparison context.
- **Dense sentences only:** No sentence should be purely rhetorical; each must contain a concrete detail, comparison, limitation, or clearly scoped claim.
- **Sentence limits are defaults, not excuses for vagueness:** Respect limits, but if needed to avoid losing essential specificity, allow one extra sentence in **Methods** or **Results** (still obey word limit).
- **Evidence strength:** When repeating a claim that is not directly demonstrated, label it clearly (e.g., “Claimed — not experimentally demonstrated”). Do not upgrade claims to facts.
- **Citation hygiene (critical for downstream review writing):** Do not cite a claim as supported by this paper if it is only mentioned via a citation to prior work. If a statement comes from another paper, mark it explicitly as **(Attributed)** and do not treat it as this paper’s result.
- **Consistency:** Use `not reported` when the concept applies but is omitted; use `not applicable` when it genuinely does not apply (surveys: `not applicable (survey)`).

---

## Template: Primary Research Paper

```markdown
# <Title>
**Citation key:** firstauthorYEARfirstword
**Authors:** First Author, Second Author, et al.
**Year:** YYYY
**Venue:** Journal / Conference / Preprint (arXiv:XXXX.XXXXX)
**Tags:** comma-separated topic tags (e.g. SNN, robotic control, surrogate gradients, 6-DOF) primarily using the ones of the paper itself, expanded if suitable; add at most 3 reviewer-added tags for cross-paper grouping (e.g., world model, neuromorphic deployment, actor-critic)
**Paper Type:** primary research OR survey/review OR commentary/opinion

---

## TL;DR
Two sentences: [Method] [achieves what] [on task] [key result]. No hype words. Usable as a comparison table row.
Include (when available) one concrete metric + scope (simulation vs real robot; benchmark/task name) + trace anchor if numeric. (Source: Fig./Tbl./Sec.)

---

## Part 1: Paper Summary
**≤400 words total. ≤3 sentences per section (4 allowed in Methods or Results if needed to avoid vagueness).**

### Problem & Motivation
What specific gap does this paper address? Why hadn't it been solved? What prior work does it build on?
Include at least one concrete anchor to what was missing/blocked previously. (Source: Sec.)

### Core Contribution
Main technical proposal. What is genuinely novel vs. a combination of existing ideas? Proof-of-concept or meaningful advance?
If the novelty is mainly integration/engineering, state that explicitly. (Source: Sec.)

### Methods
Key components, how they work, why chosen. Highlight ablated or explicitly justified design choices. Framework/codebase if public.
If training uses multiple phases (e.g., pretrain → convert → finetune; world model → freeze → policy), state the phases explicitly. (Source: Sec./Fig.)

### Results
Headline number with unit and comparison context (e.g. "4.13% higher RMSE than analytical baseline" — not "slightly worse"). Key ablation insight if any.
State the primary evaluation metric(s) and whether the metric matches the optimization objective/reward. Add a comparability flag: Comparable / partly comparable / not comparable (and why, in 5–10 words). (Source: Fig./Tbl./Sec.)

### Key Takeaways
Any key conclusions the authors draw from their research and how they influence the field (discussion). (Source: Sec./App.)
Was the research question answered, the problem solved, next steps suggested?

### Limitations
Author-acknowledged gaps + unacknowledged ones. Negative results: what was tried and failed? Papers often bury these.
Separate “not tested” from “tested and failed” where possible. (Source: Sec./App.)

### Critical Assessment
≤2 sentences. Direct and specific: what does this paper fail to demonstrate? Per-paper only — no field-level generalizations (e.g. not "the field lacks benchmarks" but "this paper uses a custom task with no comparison to any standard benchmark").
If comparison is potentially unfair, name the concrete parity issue (setup mismatch, capacity mismatch, training budget mismatch). (Source: Sec./Tbl.)

### Relevance to This Review
Primary source / background / contrasting example. Which section of the review.
**Cite for:** 1-3 bullets listing specific claims or sections where this paper would be cited (each with a trace anchor).
- ...
- ...
- ...

### Quotable Sentences
Max 2. Verbatim from this paper only — never from cited works. Each must be searchable in the PDF.
Avoid generic boilerplate; pick quotes that uniquely define the method, evidence, or interpretation. Prefer at least one quote that is (Definition/Method) rather than (Claim).
Tag each with one label and optionally a location.
- (Definition | Method | Claim | Result interpretation) "..." (Source: Sec./Fig./Tbl.)
- (Definition | Method | Claim | Result interpretation) "..." (Source: Sec./Fig./Tbl.)

### Notable findings
One finding per bullet. Include a trace anchor for any specific claim.Exactly one tag each: 
- Measured — directly observed in this paper's experiments
- Reported — paper's own results but no statistical support
- Claimed — asserted without direct experimental support
- Attributed — from another paper this paper cites
Template:
- [finding] (Measured) (Source: Fig./Tbl./Sec.)
- [finding] (Reported) (Source: Fig./Tbl./Sec.)
Example: "- 100× lower inference energy on Loihi vs. GPU (Measured) (Source: Tbl. 2)"
Example: "- Suitable for neuromorphic deployment (Claimed — no chip deployment attempted) (Source: Sec. 5)"

---

## Part 2: SNN Control Extraction
**INCLUDE IN PRIMARY RESEARCH PAPERS ONLY!**

(See references/snn-extraction-fields.md for field definitions and references/learning-paradigms.md for learning mechanism guidance.)

**1 sentence per field. Learning mechanism: 2 sentences max. Notable findings: bullet points.
Numbers: neuron/parameter counts and DOF only. No hyperparameters. No padding.
Task environment (simulated/real robot) and controller hardware (CPU/GPU/neuromorphic) are orthogonal — report both clearly and separately.**

**Neuron model:** (include source anchor if specific)
**Network architecture:** (state explicitly: fully spiking or hybrid)
**Model scale:** (neurons and/or parameters only)
**Simulator / framework:**
**Hardware (training):**
**Controller hardware (inference):** (CPU/GPU | neuromorphic emulator/SDK | physical neuromorphic chip, if a chip is used, be very specific which one! — these are distinct; see snn-extraction-fields.md)
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

---

## Template variant: Commentary / Opinion

Replace Core Contribution / Methods / Results / Limitations with:

### Core Argument
≤3 sentences. "The author argues that..." — not stated as fact. What evidence or reasoning supports the position?

### Target Paper(s) / Field
≤2 sentences. What result from the target paper anchors the argument? Attribute explicitly to original authors.

### Limitations
≤2 sentences. Does the argument overextend from the target evidence?

**Part 1 limit: ≤400 words.** Part 2: return `null` in the JSON response — do not fill any extraction fields.
Critical Assessment: does this commentary's framing appropriately represent the strength of the target paper's / fields evidence?
If technical properties are mentioned, attribute them to the target system explicitly and include a trace anchor to the target paper (as available).

---

## Template variant: Survey / Review Paper

Replace Core Contribution / Methods / Results / Limitations with:

### Scope & Coverage
What domain, time span, N papers reviewed, selection method (systematic keyword search with explicit criteria / narrative author-discretion / not described)?

### Taxonomy & Organization
How the field is categorized. For any summary tables cited, describe table content in one sentence each — not just the table number. 

### Key Claims & Narrative
Main argument. Note whether claims are backed by quantitative evidence from reviewed papers or are editorial judgments. Which trends are identified about the field?

### Gaps Identified
Open problems flagged by this survey. Research focus suggested for the future.

**Part 1 limit: ≤400 words.** Part 2: return `null` in the JSON response — do not fill any extraction fields.
Survey Critical Assessment: scope limitations, selection bias, recency, and whether quantitative claims are backed by the reviewed papers or are editorial.
Where the survey makes quantitative claims (counts, trends, performance summaries), add trace anchors to the survey’s own tables/figures (Source: Fig./Tbl./Sec.).