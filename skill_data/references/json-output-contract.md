# JSON Output Contract

Return exactly one JSON object with exactly these top-level keys:

- `metadata`
- `part1`
- `part2`

Hard rules:

- No markdown fences, comments, or explanation text.
- No trailing commas.
- Do not add extra keys.
- Do not omit required keys for the selected paper type.
- If a required value is unavailable, use `not reported` (or `not applicable` where truly inapplicable).

---

## Shared metadata schema (all outputs)

```json
"metadata": {
  "citation_key": "firstauthor2025firstword",
  "title": "string",
  "authors": ["First Last"],
  "year": 2025,
  "venue": "string",
  "is_research_paper": true,
  "paper_type": "primary",
  "synthesis_subtype": null,
  "rejection_reason": null,
  "tags": ["tag1", "tag2"]
}
```

- `paper_type`: **exactly** `"primary"` | `"synthesis"` | `null` — no other values are valid. Do NOT use `"commentary"`, `"review"`, `"survey"`, `"opinion"`, `"perspective"`, or any other string here.
- `synthesis_subtype`: `null` for primary; for synthesis, one of: `"review"` / `"survey"` / `"perspective"` / `"commentary"` / `"opinion"` / `"tutorial"` / `"position"` / `"meta-analysis"` / `"mixed"` — this is where the specific subtype goes.
- If `is_research_paper=false`, then `paper_type` must be `null` and `rejection_reason` must be a short reason string.
- **`is_research_paper: false` is reserved strictly for non-academic documents** (slides, forms, websites, software manuals, etc.). Any document published in a scientific journal, conference, or preprint server — including editorials, focus articles, opinions, commentaries, and perspectives — must be `is_research_paper: true`. A paper that argues a position or synthesizes prior work without new experiments is **synthesis**, not non-research.

---

## Canonical full templates

### Primary research

```json
{
  "metadata": {
    "citation_key": "firstauthor2025firstword",
    "title": "string",
    "authors": ["First Last"],
    "year": 2025,
    "venue": "string",
    "is_research_paper": true,
    "paper_type": "primary",
    "synthesis_subtype": null,
    "rejection_reason": null,
    "tags": ["tag1", "tag2"]
  },
  "part1": {
    "paper_type": "primary",
    "tldr": "string",
    "problem_motivation": "string",
    "core_contribution": "string",
    "methods": "string",
    "results": "string",
    "key_takeaways": "string",
    "limitations": "string",
    "open_problems_future_directions": {
      "future_work_proposed": ["string (Paper-identified)"],
      "open_questions": ["string (Paper-identified | Reviewer-noted)"]
    },
    "critical_assessment": "string",
    "notable_findings": ["string (Measured|Reported|Claimed|Attributed) (Source: Fig./Tbl./Sec.)"],
    "citable_snippets": [
      {
        "cite_for": "string",
        "source": "string",
        "quote_tag": "Definition | Method | Claim | Result",
        "quote": "string or null"
      }
    ],
    "relevance": "string"
  },
  "part2": {
    "neuron_model": "string",
    "network_architecture": "string",
    "model_scale": "string",
    "simulator_framework": "string",
    "hardware_training": "string",
    "controller_hardware_inference": "string",
    "control_task": "string",
    "task_type": "string",
    "task_complexity_scale": "string",
    "simulation_environment": "string",
    "spike_encoding": "string",
    "action_decoding": "string",
    "learning_mechanism": "string",
    "credit_assignment_scope": "string",
    "online_vs_offline": "string",
    "data_collection": "string",
    "key_training_details": "string",
    "comparison_to_baselines": "string"
  }
}
```

### Synthesis (review / survey / perspective / commentary / opinion / …)

```json
{
  "metadata": {
    "citation_key": "firstauthor2025firstword",
    "title": "string",
    "authors": ["First Last"],
    "year": 2025,
    "venue": "string",
    "is_research_paper": true,
    "paper_type": "synthesis",
    "synthesis_subtype": "review",
    "rejection_reason": null,
    "tags": ["tag1", "tag2"]
  },
  "part1": {
    "paper_type": "synthesis",
    "tldr": "string",
    "target_papers_field": "string",
    "scope_coverage": "string",
    "taxonomy_organization": "string",
    "core_argument": "string",
    "synthesis_contribution": "string",
    "key_claims_narrative": "string",
    "key_takeaways": "string",
    "limitations": "string",
    "open_problems_future_directions": {
      "gaps_identified": ["string (Paper-identified)"],
      "open_questions": ["string (Paper-identified | Reviewer-noted)"],
      "suggested_research_focus": ["string (Paper-identified)"]
    },
    "critical_assessment": "string",
    "notable_findings": ["string (Reported|Claimed|Attributed) (Source: Fig./Tbl./Sec.)"],
    "citable_snippets": [
      {
        "cite_for": "string",
        "source": "string",
        "quote_tag": "Definition | Quantitative synthesis | Editorial judgment | Attributed",
        "quote": "string or null"
      }
    ],
    "relevance": "string"
  },
  "part2": null
}
```

### Non-research document

```json
{
  "metadata": {
    "citation_key": "document2025slug",
    "title": "string",
    "authors": ["Unknown"],
    "year": 2025,
    "venue": "not applicable",
    "is_research_paper": false,
    "paper_type": null,
    "synthesis_subtype": null,
    "rejection_reason": "short reason",
    "tags": ["non-research"]
  },
  "part1": {
    "paper_type": "non_research",
    "note": "short reason this is not a research paper"
  },
  "part2": null
}
```
