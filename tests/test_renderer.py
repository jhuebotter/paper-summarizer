"""Tests for summarizer/renderer.py — PaperSummary → markdown string."""

import pytest
from pathlib import Path

from summarizer.models import (
    PaperMetadata,
    PaperSummary,
    SummaryPart1Commentary,
    SummaryPart1NonResearch,
    SummaryPart1Primary,
    SummaryPart1Survey,
    SummaryPart2,
)
from summarizer.renderer import render_summary

# ---------------------------------------------------------------------------
# Helpers — reuse conftest mock data split logic
# ---------------------------------------------------------------------------

_METADATA_ONLY_KEYS = frozenset(
    {"citation_key", "title", "authors", "year", "venue", "tags"}
)


def _make_summary(mock_part1_dict, mock_part2_dict) -> PaperSummary:
    """Construct a PaperSummary from the standard mock dicts (primary paper)."""
    meta_data = {k: mock_part1_dict[k] for k in (*_METADATA_ONLY_KEYS, "paper_type")}
    meta_data["is_research_paper"] = True
    meta_data["rejection_reason"] = None
    meta = PaperMetadata(**meta_data)
    part1_data = {
        k: v for k, v in mock_part1_dict.items() if k not in _METADATA_ONLY_KEYS
    }
    part1 = SummaryPart1Primary(**part1_data)
    part2 = SummaryPart2(**mock_part2_dict)
    return PaperSummary(metadata=meta, part1=part1, part2=part2)


# ---------------------------------------------------------------------------
# Primary paper rendering
# ---------------------------------------------------------------------------


def test_render_primary_contains_title(mock_part1_dict, mock_part2_dict):
    md = render_summary(_make_summary(mock_part1_dict, mock_part2_dict))
    assert mock_part1_dict["title"] in md


def test_render_primary_contains_citation_key(mock_part1_dict, mock_part2_dict):
    md = render_summary(_make_summary(mock_part1_dict, mock_part2_dict))
    assert mock_part1_dict["citation_key"] in md


def test_render_primary_contains_tldr(mock_part1_dict, mock_part2_dict):
    md = render_summary(_make_summary(mock_part1_dict, mock_part2_dict))
    assert mock_part1_dict["tldr"] in md


def test_render_primary_has_part1_sections(mock_part1_dict, mock_part2_dict):
    md = render_summary(_make_summary(mock_part1_dict, mock_part2_dict))
    assert "## Part 1: Paper Summary" in md
    assert "### Problem & Motivation" in md
    assert "### Core Contribution" in md
    assert "### Methods" in md
    assert "### Results" in md
    assert "### Key Takeaways" in md
    assert "### Limitations" in md
    assert "### Relevance to This Review" in md
    assert "### Critical Assessment" in md
    assert "### Quotable Sentences" in md


def test_render_primary_has_part2_section(mock_part1_dict, mock_part2_dict):
    md = render_summary(_make_summary(mock_part1_dict, mock_part2_dict))
    assert "## Part 2: SNN Control Extraction" in md
    assert "**Neuron model:**" in md
    assert "**Network architecture:**" in md
    assert "**Comparison to baselines:**" in md


def test_render_primary_cite_for_bullets(mock_part1_dict, mock_part2_dict):
    md = render_summary(_make_summary(mock_part1_dict, mock_part2_dict))
    assert "**Cite for:**" in md
    for item in mock_part1_dict["cite_for"]:
        assert f"- {item}" in md


def test_render_primary_quotable_sentences(mock_part1_dict, mock_part2_dict):
    md = render_summary(_make_summary(mock_part1_dict, mock_part2_dict))
    for sentence in mock_part1_dict["quotable_sentences"]:
        assert sentence in md


def test_render_primary_notable_findings_bullets(mock_part1_dict, mock_part2_dict):
    md = render_summary(_make_summary(mock_part1_dict, mock_part2_dict))
    for finding in mock_part1_dict["notable_findings"]:
        assert finding in md


def test_render_header_contains_paper_type(mock_part1_dict, mock_part2_dict):
    md = render_summary(_make_summary(mock_part1_dict, mock_part2_dict))
    assert "**Paper Type:** primary research" in md


def test_render_primary_has_separators(mock_part1_dict, mock_part2_dict):
    md = render_summary(_make_summary(mock_part1_dict, mock_part2_dict))
    assert md.count("---") >= 2


# ---------------------------------------------------------------------------
# Survey paper rendering
# ---------------------------------------------------------------------------


def _make_survey_summary() -> PaperSummary:
    meta = PaperMetadata(
        citation_key="oikonomou2025reinforcement",
        title="Reinforcement Learning with SNNs: A Survey",
        authors=["A. Oikonomou"],
        year=2025,
        venue="Preprint",
        is_research_paper=True,
        paper_type="survey",
        rejection_reason=None,
        tags=["SNN", "survey", "RL"],
    )
    part1 = SummaryPart1Survey(
        paper_type="survey",
        tldr="A survey of RL methods using spiking neural networks.",
        scope_coverage="80 papers, 2018–2024, keyword search.",
        taxonomy_organization="Organized by learning algorithm.",
        key_claims_narrative="SNNs are increasingly viable for RL.",
        gaps_identified="No standardized benchmarks.",
        relevance="Background for Section 2.",
        cite_for=["Overview of SNN-based RL approaches"],
        critical_assessment="Selection bias toward arXiv; no systematic criteria.",
        quotable_sentences=["The field is rapidly maturing."],
    )
    return PaperSummary(metadata=meta, part1=part1, part2=None)


def test_render_survey_has_survey_sections():
    md = render_summary(_make_survey_summary())
    assert "### Scope & Coverage" in md
    assert "### Taxonomy & Organization" in md
    assert "### Key Claims & Narrative" in md
    assert "### Gaps Identified" in md


def test_render_survey_does_not_have_primary_sections():
    md = render_summary(_make_survey_summary())
    assert "### Problem & Motivation" not in md
    assert "### Core Contribution" not in md
    assert "### Methods" not in md
    assert "### Results" not in md


def test_render_survey_has_no_part2():
    md = render_summary(_make_survey_summary())
    assert "## Part 2: SNN Control Extraction" not in md


# ---------------------------------------------------------------------------
# Commentary paper rendering
# ---------------------------------------------------------------------------


def _make_commentary_summary() -> PaperSummary:
    meta = PaperMetadata(
        citation_key="dewolf2021spiking",
        title="Spiking Neural Networks Take Control",
        authors=["T. W. Dewolf"],
        year=2021,
        venue="Science Robotics",
        is_research_paper=True,
        paper_type="commentary",
        rejection_reason=None,
        tags=["SNN", "commentary", "robotics"],
    )
    part1 = SummaryPart1Commentary(
        paper_type="commentary",
        tldr="The author argues SNNs are ready for real robotic deployment.",
        core_argument="Recent advances make SNNs viable for autonomous robots.",
        target_papers="Stagsted et al. 2020",
        limitations="Evidence from a single paper; claims may overextend.",
        relevance="Opinion context for the review.",
        cite_for=["Optimistic framing of SNN maturity"],
        critical_assessment="Argument is plausible but rests on limited evidence.",
        quotable_sentences=["Spiking neural networks are finally ready."],
    )
    return PaperSummary(metadata=meta, part1=part1, part2=None)


def test_render_commentary_has_commentary_sections():
    md = render_summary(_make_commentary_summary())
    assert "### Core Argument" in md
    assert "### Target Paper(s)" in md


def test_render_commentary_does_not_have_primary_sections():
    md = render_summary(_make_commentary_summary())
    assert "### Core Contribution" not in md
    assert "### Methods" not in md
    assert "### Results" not in md


def test_render_commentary_has_no_part2():
    md = render_summary(_make_commentary_summary())
    assert "## Part 2: SNN Control Extraction" not in md


# ---------------------------------------------------------------------------
# Non-research document rendering
# ---------------------------------------------------------------------------


def _make_non_research_summary() -> PaperSummary:
    meta = PaperMetadata(
        citation_key="document2025foo",
        title="Some Scanned Form",
        authors=["Unknown Author"],
        year=2025,
        venue="not applicable",
        is_research_paper=False,
        paper_type=None,
        rejection_reason="Scanned form",
        tags=["non-research"],
    )
    part1 = SummaryPart1NonResearch(
        paper_type="non_research",
        note="Document appears to be a scanned administrative form, not a research paper.",
    )
    return PaperSummary(metadata=meta, part1=part1, part2=None)


def test_render_non_research_contains_note():
    md = render_summary(_make_non_research_summary())
    assert "scanned administrative form" in md


def test_render_non_research_has_no_part1_prose_sections():
    md = render_summary(_make_non_research_summary())
    assert "### Problem & Motivation" not in md
    assert "### Core Contribution" not in md
    assert "## Part 1: Paper Summary" not in md


def test_render_non_research_has_no_part2():
    md = render_summary(_make_non_research_summary())
    assert "## Part 2: SNN Control Extraction" not in md


# ---------------------------------------------------------------------------
# Word-count warning (Part 1 prose ≥ 150% of type limit → stderr warning)
# ---------------------------------------------------------------------------


def _make_bloated_primary(mock_part1_dict, mock_part2_dict) -> PaperSummary:
    """Return a primary PaperSummary whose Part 1 prose is far over the 400-word limit."""
    bloated = (
        "word " * 700
    )  # 700 words — exceeds 150% of the 400-word limit (threshold: 600)
    meta_data = {k: mock_part1_dict[k] for k in (*_METADATA_ONLY_KEYS, "paper_type")}
    meta_data["is_research_paper"] = True
    meta_data["rejection_reason"] = None
    meta = PaperMetadata(**meta_data)
    part1_data = {
        k: v for k, v in mock_part1_dict.items() if k not in _METADATA_ONLY_KEYS
    }
    part1_data["problem_motivation"] = bloated
    part1 = SummaryPart1Primary(**part1_data)
    part2 = SummaryPart2(**mock_part2_dict)
    return PaperSummary(metadata=meta, part1=part1, part2=part2)


def test_render_warns_when_part1_exceeds_word_limit(
    mock_part1_dict, mock_part2_dict, caplog
):
    import logging

    with caplog.at_level(logging.WARNING, logger="summarizer.renderer"):
        render_summary(_make_bloated_primary(mock_part1_dict, mock_part2_dict))
    assert any("word" in r.message.lower() for r in caplog.records)
    assert any(r.levelno == logging.WARNING for r in caplog.records)
