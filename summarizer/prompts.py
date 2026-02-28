"""Reference file loading and LLM prompt builder.

``build_combined_prompt`` returns a self-contained prompt that requests all
three sections (metadata, Part 1, Part 2) in a single JSON response. The
prompt embeds the reference guidelines (loaded from ``skill_data/references/``)
so every call is stateless.
"""

from pathlib import Path


def load_references(references_dir: Path) -> str:
    """Load all .md files from references_dir and concatenate them.

    Files are sorted alphabetically for deterministic ordering.

    Raises:
        FileNotFoundError: if references_dir does not exist.
    """
    if not references_dir.exists():
        raise FileNotFoundError(f"References directory not found: {references_dir}")
    parts = [f.read_text(encoding="utf-8") for f in sorted(references_dir.glob("*.md"))]
    return "\n\n---\n\n".join(parts)


def build_combined_prompt(
    paper_text: str, references: str, source_filename: str
) -> str:
    """Build the combined prompt for a single LLM call.

    The LLM must return one JSON object with exactly three top-level keys:
    ``metadata``, ``part1``, and ``part2``.

    Args:
        paper_text: The (possibly truncated) paper markdown from the parser.
        references: The concatenated content of all ``skill_data/references/``
            files, as returned by ``load_references``.
        source_filename: Original PDF filename used as auxiliary metadata
            evidence for ambiguous fields like publication year.

    Returns:
        A self-contained prompt string ready to send to the LLM.
    """
    return f"""\
You are a research paper summarizer and data extractor.

Follow the reference guidelines exactly:

{references}

---

Return exactly ONE valid JSON object with exactly these top-level keys:
- "metadata"
- "part1"
- "part2"

Output rules:
- Return JSON only (no markdown fences, no prose, no comments).
- Do not include trailing commas.
- Do not include extra top-level keys.
- Keep key names exactly as specified in the references.
- Follow the JSON Output Contract template exactly for the chosen document type.
- Do not omit required keys in metadata/part1/part2 for that template.
- If a required value is unavailable, use "not reported" (or "not applicable" if truly inapplicable).

Year resolution priority (for metadata.year):
1) explicit publication year in the paper metadata/header,
2) source filename,
3) best-supported inference from paper context (e.g., references) only if needed.
- metadata.year must be an integer (never "not reported").

Research gate rules:
- If the document is a research paper:
  - metadata.is_research_paper = true
  - metadata.paper_type is one of "primary", "survey", "commentary"
  - metadata.rejection_reason = null
  - part1.paper_type must match metadata.paper_type
- If the document is not a research paper:
  - metadata.is_research_paper = false
  - metadata.paper_type = null
  - metadata.rejection_reason is a short reason
  - part1 = {{"paper_type": "non_research", "note": "..."}}

Part 2 rules:
- For metadata.paper_type == "primary": part2 must be a full Part 2 object.
- For primary papers, do NOT omit Part 2 keys even if no control task or no learning setup is present; use "not applicable" (or "not reported") for those fields.
- For metadata.paper_type in {{"survey", "commentary"}}: part2 must be null.
- For non-research documents: part2 must be null.

Variant-specific part1 required keys:
- primary: paper_type, tldr, problem_motivation, core_contribution, methods, results, key_takeaways, limitations, relevance, cite_for, critical_assessment, quotable_sentences, notable_findings
- survey: paper_type, tldr, scope_coverage, taxonomy_organization, key_claims_narrative, gaps_identified, relevance, cite_for, critical_assessment, quotable_sentences
- commentary: paper_type, tldr, core_argument, target_papers, limitations, relevance, cite_for, critical_assessment, quotable_sentences
- non_research: paper_type, note

---

Source filename:
{source_filename}

---

Paper text:
{paper_text}"""
