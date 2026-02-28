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
  "rejection_reason": null,
  "tags": ["tag1", "tag2"]
}
```

If `is_research_paper=false`, then `paper_type` must be `null` and `rejection_reason` must be a short reason.

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
    "relevance": "string",
    "cite_for": ["string"],
    "critical_assessment": "string",
    "quotable_sentences": ["string"],
    "notable_findings": ["string (Measured)"]
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

### Survey / review

```json
{
  "metadata": {
    "citation_key": "firstauthor2025firstword",
    "title": "string",
    "authors": ["First Last"],
    "year": 2025,
    "venue": "string",
    "is_research_paper": true,
    "paper_type": "survey",
    "rejection_reason": null,
    "tags": ["tag1", "tag2"]
  },
  "part1": {
    "paper_type": "survey",
    "tldr": "string",
    "scope_coverage": "string",
    "taxonomy_organization": "string",
    "key_claims_narrative": "string",
    "gaps_identified": "string",
    "relevance": "string",
    "cite_for": ["string"],
    "critical_assessment": "string",
    "quotable_sentences": ["string"]
  },
  "part2": null
}
```

### Commentary / opinion

```json
{
  "metadata": {
    "citation_key": "firstauthor2025firstword",
    "title": "string",
    "authors": ["First Last"],
    "year": 2025,
    "venue": "string",
    "is_research_paper": true,
    "paper_type": "commentary",
    "rejection_reason": null,
    "tags": ["tag1", "tag2"]
  },
  "part1": {
    "paper_type": "commentary",
    "tldr": "string",
    "core_argument": "string",
    "target_papers": "string",
    "limitations": "string",
    "relevance": "string",
    "cite_for": ["string"],
    "critical_assessment": "string",
    "quotable_sentences": ["string"]
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
