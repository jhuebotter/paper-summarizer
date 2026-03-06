"""Tests for summarizer/prompts.py — reference loading and prompt building."""

import pytest
from pathlib import Path

from summarizer.prompts import load_references, build_combined_prompt

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCES_DIR = PROJECT_ROOT / "skill_data" / "references"


# ---------------------------------------------------------------------------
# load_references
# ---------------------------------------------------------------------------


def test_load_references_returns_nonempty_string():
    content = load_references(REFERENCES_DIR)
    assert isinstance(content, str)
    assert len(content) > 100


def test_load_references_includes_all_three_files():
    content = load_references(REFERENCES_DIR)
    # Each file has unique distinctive content
    assert "Output Template" in content  # output-template.md
    assert "Neuron model" in content  # snn-extraction-fields.md
    assert "learning" in content.lower()  # learning-paradigms.md
    assert "JSON Output Contract" in content  # json-output-contract.md


def test_load_references_raises_on_missing_dir(tmp_path):
    missing = tmp_path / "nonexistent"
    with pytest.raises(FileNotFoundError):
        load_references(missing)


def test_load_references_empty_dir_returns_empty_string(tmp_path):
    result = load_references(tmp_path)
    assert result == ""


def test_load_references_uses_only_md_files(tmp_path):
    (tmp_path / "keep.md").write_text("markdown content")
    (tmp_path / "ignore.txt").write_text("text content should not appear")
    result = load_references(tmp_path)
    assert "markdown content" in result
    assert "text content should not appear" not in result


# ---------------------------------------------------------------------------
# build_combined_prompt (v1.1 — single call)
# ---------------------------------------------------------------------------


def test_build_combined_prompt_contains_paper_text():
    paper_text = "This paper introduces a novel SNN approach to motor control."
    references = "Some reference content."
    prompt = build_combined_prompt(
        paper_text=paper_text,
        references=references,
        source_filename="paper-2025.pdf",
    )
    assert paper_text in prompt


def test_build_combined_prompt_contains_references():
    paper_text = "Paper text here."
    references = "Distinctive reference material xyz."
    prompt = build_combined_prompt(
        paper_text=paper_text,
        references=references,
        source_filename="paper-2025.pdf",
    )
    assert references in prompt


def test_build_combined_prompt_requests_json_output():
    prompt = build_combined_prompt(
        paper_text="text", references="refs", source_filename="paper-2025.pdf"
    )
    assert "JSON" in prompt or "json" in prompt


def test_build_combined_prompt_mentions_metadata_key():
    """Combined prompt must instruct the LLM to return a 'metadata' top-level key."""
    prompt = build_combined_prompt(
        paper_text="text", references="refs", source_filename="paper-2025.pdf"
    )
    assert "metadata" in prompt


def test_build_combined_prompt_mentions_part1_key():
    """Combined prompt must instruct the LLM to return a 'part1' top-level key."""
    prompt = build_combined_prompt(
        paper_text="text", references="refs", source_filename="paper-2025.pdf"
    )
    assert "part1" in prompt


def test_build_combined_prompt_mentions_part2_key():
    """Combined prompt must instruct the LLM to return a 'part2' top-level key."""
    prompt = build_combined_prompt(
        paper_text="text", references="refs", source_filename="paper-2025.pdf"
    )
    assert "part2" in prompt


def test_build_combined_prompt_mentions_non_research_gate():
    """Prompt must include research/non-research gating instructions."""
    prompt = build_combined_prompt(
        paper_text="text", references="refs", source_filename="paper-2025.pdf"
    )
    assert "is_research_paper" in prompt
    assert "non_research" in prompt


def test_build_combined_prompt_mentions_primary_only_part2():
    """Prompt must require part2 only for primary papers."""
    prompt = build_combined_prompt(
        paper_text="text", references="refs", source_filename="paper-2025.pdf"
    )
    assert 'paper_type == "primary"' in prompt
    assert "part2 must be null" in prompt


def test_build_combined_prompt_contains_source_filename_context():
    prompt = build_combined_prompt(
        paper_text="text",
        references="refs",
        source_filename="ABC123__Doe - 2024 - Title.pdf",
    )
    assert "Source filename:" in prompt
    assert "ABC123__Doe - 2024 - Title.pdf" in prompt


def test_build_combined_prompt_mentions_year_priority_rules():
    prompt = build_combined_prompt(
        paper_text="text",
        references="refs",
        source_filename="paper.pdf",
    )
    assert "Year resolution priority" in prompt
    assert "source filename" in prompt


# ---------------------------------------------------------------------------
# load_references — structural tests
# ---------------------------------------------------------------------------


def test_load_references_joins_with_separator(tmp_path):
    """Multiple .md files are joined with a '---' separator."""
    (tmp_path / "a.md").write_text("content A", encoding="utf-8")
    (tmp_path / "b.md").write_text("content B", encoding="utf-8")
    result = load_references(tmp_path)
    assert "---" in result
    assert "content A" in result
    assert "content B" in result


def test_load_references_alphabetical_order(tmp_path):
    """Files are loaded in alphabetical order (deterministic)."""
    (tmp_path / "b.md").write_text("SECOND", encoding="utf-8")
    (tmp_path / "a.md").write_text("FIRST", encoding="utf-8")
    result = load_references(tmp_path)
    assert result.index("FIRST") < result.index("SECOND")


# ---------------------------------------------------------------------------
# build_combined_prompt — required keys per variant
# ---------------------------------------------------------------------------

# Build the prompt once at module level; the function is pure.
_PROMPT = build_combined_prompt("text", "refs", "paper.pdf")

_PRIMARY_PART1_KEYS = [
    "tldr", "problem_motivation", "core_contribution", "methods", "results",
    "key_takeaways", "limitations", "open_problems_future_directions",
    "critical_assessment", "notable_findings", "citable_snippets", "relevance",
]
_SYNTHESIS_PART1_KEYS = [
    "tldr", "target_papers_field", "scope_coverage", "taxonomy_organization",
    "core_argument", "synthesis_contribution", "key_claims_narrative",
    "key_takeaways", "limitations", "open_problems_future_directions",
    "critical_assessment", "notable_findings", "citable_snippets", "relevance",
]
_NON_RESEARCH_PART1_KEYS = ["paper_type", "note"]


@pytest.mark.parametrize("key", _PRIMARY_PART1_KEYS)
def test_build_combined_prompt_lists_primary_part1_key(key):
    """Every required primary part1 key appears in the prompt."""
    assert key in _PROMPT, f"Primary part1 key '{key}' missing from prompt"


@pytest.mark.parametrize("key", _SYNTHESIS_PART1_KEYS)
def test_build_combined_prompt_lists_synthesis_part1_key(key):
    """Every required synthesis part1 key appears in the prompt."""
    assert key in _PROMPT, f"Synthesis part1 key '{key}' missing from prompt"


@pytest.mark.parametrize("key", _NON_RESEARCH_PART1_KEYS)
def test_build_combined_prompt_lists_non_research_part1_key(key):
    """Every required non_research part1 key appears in the prompt."""
    assert key in _PROMPT, f"Non-research part1 key '{key}' missing from prompt"


# ---------------------------------------------------------------------------
# build_combined_prompt — critical instructions
# ---------------------------------------------------------------------------


def test_build_combined_prompt_forbids_markdown_fences():
    """Prompt explicitly forbids markdown fences in the output."""
    assert "no markdown fences" in _PROMPT


def test_build_combined_prompt_mentions_not_reported_fallback():
    """Prompt instructs the LLM to use 'not reported' for unavailable values."""
    assert "not reported" in _PROMPT


def test_build_combined_prompt_mentions_output_budget_warning():
    """Prompt contains the output budget / completeness warning."""
    assert "Output budget" in _PROMPT


def test_build_combined_prompt_part2_requires_full_object_for_primary():
    """Prompt requires a full Part 2 object for primary papers (not just null rules)."""
    assert "part2 must be a full Part 2 object" in _PROMPT
