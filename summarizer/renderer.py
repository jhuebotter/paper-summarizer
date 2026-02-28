"""Render a validated PaperSummary to the final markdown string.

The output format matches the ``output-template.md`` reference exactly.
No file I/O is performed here — the caller (``batch.py`` / ``cli.py``) is
responsible for writing the returned string to disk.
"""

import logging
from typing import Sequence

logger = logging.getLogger(__name__)

from summarizer.models import (
    PaperMetadata,
    PaperSummary,
    SummaryPart1Commentary,
    SummaryPart1NonResearch,
    SummaryPart1Primary,
    SummaryPart1Survey,
    SummaryPart2,
)

# ---------------------------------------------------------------------------
# Word-limit constants (per output-template.md)
# ---------------------------------------------------------------------------

_WORD_LIMITS: dict[str, int] = {
    "primary": 400,
    "survey": 400,
    "commentary": 400,
}

# Warn when Part 1 prose exceeds the limit by more than this fraction.
_WARN_THRESHOLD = 0.5  # 50% over limit triggers a warning


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_summary(summary: PaperSummary) -> str:
    """Convert a ``PaperSummary`` to the final markdown string.

    Dispatches to the appropriate variant renderer based on ``paper_type``.
    If Part 1 prose exceeds the per-type word limit by more than 50%, a
    warning is printed to stderr (but rendering continues unchanged — no
    truncation is applied).

    Args:
        summary: A fully validated ``PaperSummary`` produced by the pipeline.

    Returns:
        The complete markdown content ready to be written to ``<citation_key>.md``.
    """
    paper_type = summary.metadata.paper_type

    if paper_type == "primary":
        assert isinstance(summary.part1, SummaryPart1Primary)
        assert summary.part2 is not None
        return _render_primary(summary.metadata, summary.part1, summary.part2)
    if paper_type == "survey":
        assert isinstance(summary.part1, SummaryPart1Survey)
        return _render_survey(summary.metadata, summary.part1, summary.part2)
    if paper_type == "commentary":
        assert isinstance(summary.part1, SummaryPart1Commentary)
        return _render_commentary(summary.metadata, summary.part1, summary.part2)
    # Non-research document path
    assert isinstance(summary.part1, SummaryPart1NonResearch)
    return _render_non_research(summary.metadata, summary.part1)


# ---------------------------------------------------------------------------
# Word-count helper
# ---------------------------------------------------------------------------


def _count_words(*texts: str) -> int:
    """Return the total word count across all provided strings."""
    return sum(len(t.split()) for t in texts)


def _check_word_limit(paper_type: str, word_count: int) -> None:
    """Emit a stderr warning if ``word_count`` is >50% over the type limit."""
    limit = _WORD_LIMITS.get(paper_type)
    if limit is None:
        return
    threshold = int(limit * (1 + _WARN_THRESHOLD))
    if word_count > threshold:
        logger.warning(
            "Part 1 prose is %d words (limit: %d; >50%% over — consider refining the prompt)",
            word_count,
            limit,
        )


# ---------------------------------------------------------------------------
# Shared sub-renderers
# ---------------------------------------------------------------------------


def _render_header(meta: PaperMetadata) -> str:
    """Render the YAML-style metadata header block."""
    paper_type_label = {
        "primary": "primary research",
        "survey": "survey/review",
        "commentary": "commentary/opinion",
        None: "non-research",
    }[meta.paper_type]

    return (
        f"# {meta.title}\n\n"
        f"**Citation key:** {meta.citation_key}\n"
        f"**Authors:** {', '.join(meta.authors)}\n"
        f"**Year:** {meta.year}\n"
        f"**Venue:** {meta.venue}\n"
        f"**Paper Type:** {paper_type_label}\n"
        f"**Tags:** {', '.join(meta.tags)}"
    )


def _render_bullets(items: Sequence[str]) -> str:
    """Render a list of strings as markdown bullet points."""
    return "\n".join(f"- {item}" for item in items)


def _render_part2(part2: SummaryPart2) -> str:
    """Render the Part 2 SNN extraction section."""
    return (
        "## Part 2: SNN Control Extraction\n\n"
        f"**Neuron model:** {part2.neuron_model}\n\n"
        f"**Network architecture:** {part2.network_architecture}\n\n"
        f"**Model scale:** {part2.model_scale}\n\n"
        f"**Simulator / framework:** {part2.simulator_framework}\n\n"
        f"**Hardware (training):** {part2.hardware_training}\n\n"
        f"**Controller hardware (inference):** {part2.controller_hardware_inference}\n\n"
        f"**Control task:** {part2.control_task}\n\n"
        f"**Task type:** {part2.task_type}\n\n"
        f"**Task complexity & scale:** {part2.task_complexity_scale}\n\n"
        f"**Simulation environment:** {part2.simulation_environment}\n\n"
        f"**Spike encoding:** {part2.spike_encoding}\n\n"
        f"**Action decoding:** {part2.action_decoding}\n\n"
        f"**Learning mechanism:** {part2.learning_mechanism}\n\n"
        f"**Credit assignment scope:** {part2.credit_assignment_scope}\n\n"
        f"**Online vs. offline:** {part2.online_vs_offline}\n\n"
        f"**Data collection:** {part2.data_collection}\n\n"
        f"**Key training details:** {part2.key_training_details}\n\n"
        f"**Comparison to baselines:** {part2.comparison_to_baselines}"
    )


# ---------------------------------------------------------------------------
# Variant renderers
# ---------------------------------------------------------------------------


def _render_primary(
    meta: PaperMetadata,
    part1: SummaryPart1Primary,
    part2: SummaryPart2,
) -> str:
    """Render a primary research paper (≤400 words for Part 1 prose)."""
    prose_words = _count_words(
        part1.tldr,
        part1.problem_motivation,
        part1.core_contribution,
        part1.methods,
        part1.results,
        part1.key_takeaways,
        part1.limitations,
        part1.relevance,
        part1.critical_assessment,
    )
    _check_word_limit("primary", prose_words)

    cite_for = _render_bullets(part1.cite_for)
    quotable = _render_bullets([f'"{s}"' for s in part1.quotable_sentences])
    notable = _render_bullets(part1.notable_findings)

    return (
        f"{_render_header(meta)}\n\n"
        "---\n\n"
        f"## TL;DR\n\n{part1.tldr}\n\n"
        "---\n\n"
        "## Part 1: Paper Summary\n\n"
        f"### Problem & Motivation\n\n{part1.problem_motivation}\n\n"
        f"### Core Contribution\n\n{part1.core_contribution}\n\n"
        f"### Methods\n\n{part1.methods}\n\n"
        f"### Results\n\n{part1.results}\n\n"
        f"### Key Takeaways\n\n{part1.key_takeaways}\n\n"
        f"### Limitations\n\n{part1.limitations}\n\n"
        f"### Relevance to This Review\n\n{part1.relevance}\n\n"
        f"**Cite for:**\n{cite_for}\n\n"
        f"### Critical Assessment\n\n{part1.critical_assessment}\n\n"
        f"### Quotable Sentences\n\n{quotable}\n\n"
        f"### Notable Findings\n\n{notable}\n\n"
        "---\n\n"
        f"{_render_part2(part2)}"
    )


def _render_survey(
    meta: PaperMetadata,
    part1: SummaryPart1Survey,
    part2: SummaryPart2 | None,
) -> str:
    """Render a survey or review paper (≤400 words for Part 1 prose)."""
    prose_words = _count_words(
        part1.tldr,
        part1.scope_coverage,
        part1.taxonomy_organization,
        part1.key_claims_narrative,
        part1.gaps_identified,
        part1.relevance,
        part1.critical_assessment,
    )
    _check_word_limit("survey", prose_words)

    cite_for = _render_bullets(part1.cite_for)
    quotable = _render_bullets([f'"{s}"' for s in part1.quotable_sentences])

    return (
        f"{_render_header(meta)}\n\n"
        "---\n\n"
        f"## TL;DR\n\n{part1.tldr}\n\n"
        "---\n\n"
        "## Part 1: Paper Summary\n\n"
        f"### Scope & Coverage\n\n{part1.scope_coverage}\n\n"
        f"### Taxonomy & Organization\n\n{part1.taxonomy_organization}\n\n"
        f"### Key Claims & Narrative\n\n{part1.key_claims_narrative}\n\n"
        f"### Gaps Identified\n\n{part1.gaps_identified}\n\n"
        f"### Relevance to This Review\n\n{part1.relevance}\n\n"
        f"**Cite for:**\n{cite_for}\n\n"
        f"### Critical Assessment\n\n{part1.critical_assessment}\n\n"
        f"### Quotable Sentences\n\n{quotable}\n\n"
        + (f"---\n\n{_render_part2(part2)}" if part2 is not None else "")
    )


def _render_commentary(
    meta: PaperMetadata,
    part1: SummaryPart1Commentary,
    part2: SummaryPart2 | None,
) -> str:
    """Render a commentary or opinion paper (≤400 words for Part 1 prose)."""
    prose_words = _count_words(
        part1.tldr,
        part1.core_argument,
        part1.target_papers,
        part1.limitations,
        part1.relevance,
        part1.critical_assessment,
    )
    _check_word_limit("commentary", prose_words)

    cite_for = _render_bullets(part1.cite_for)
    quotable = _render_bullets([f'"{s}"' for s in part1.quotable_sentences])

    return (
        f"{_render_header(meta)}\n\n"
        "---\n\n"
        f"## TL;DR\n\n{part1.tldr}\n\n"
        "---\n\n"
        "## Part 1: Paper Summary\n\n"
        f"### Core Argument\n\n{part1.core_argument}\n\n"
        f"### Target Paper(s)\n\n{part1.target_papers}\n\n"
        f"### Limitations\n\n{part1.limitations}\n\n"
        f"### Relevance to This Review\n\n{part1.relevance}\n\n"
        f"**Cite for:**\n{cite_for}\n\n"
        f"### Critical Assessment\n\n{part1.critical_assessment}\n\n"
        f"### Quotable Sentences\n\n{quotable}\n\n"
        + (f"---\n\n{_render_part2(part2)}" if part2 is not None else "")
    )


def _render_non_research(meta: PaperMetadata, part1: SummaryPart1NonResearch) -> str:
    """Render a non-research document with a rejection note only."""
    return (
        f"{_render_header(meta)}\n\n"
        "---\n\n"
        "## Note\n\n"
        "This document was detected as non-research and excluded from paper-type classification.\n\n"
        f"{part1.note}"
    )
