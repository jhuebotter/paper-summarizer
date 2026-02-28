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
