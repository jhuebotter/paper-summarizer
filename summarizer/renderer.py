"""Render a validated PaperSummary to the final markdown string.

The output format matches the ``output-template.md`` reference exactly.
No file I/O is performed here — the caller (``batch.py`` / ``cli.py``) is
responsible for writing the returned string to disk.
"""

import logging
from typing import Sequence

logger = logging.getLogger(__name__)

from summarizer.models import (
    CitableSnippet,
    OpenProblemsPrimary,
    OpenProblemsSynthesis,
    PaperMetadata,
    PaperSummary,
    SummaryPart1NonResearch,
    SummaryPart1Primary,
    SummaryPart1Synthesis,
    SummaryPart2,
)

# ---------------------------------------------------------------------------
# Word-limit constants (per output-template.md)
# ---------------------------------------------------------------------------

_WORD_LIMITS: dict[str, int] = {
    "primary": 600,
    "synthesis": 1000,
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
    if paper_type == "synthesis":
        assert isinstance(summary.part1, SummaryPart1Synthesis)
        return _render_synthesis(summary.metadata, summary.part1)
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
    if meta.paper_type == "primary":
        paper_type_label = "primary research"
    elif meta.paper_type == "synthesis":
        if meta.synthesis_subtype:
            paper_type_label = f"synthesis — {meta.synthesis_subtype}"
        else:
            paper_type_label = "synthesis"
    else:
        paper_type_label = "non-research"

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
    if not items:
        return ""
    return "\n".join(f"- {item}" for item in items)


def _render_citable_snippets(snippets: list[CitableSnippet]) -> str:
    """Render citable snippets with co-located optional quotable sentences."""
    if not snippets:
        return ""
    lines = []
    for s in snippets:
        lines.append(f"- **{s.cite_for}** (Source: {s.source})")
        if s.quote:
            tag_prefix = f"({s.quote_tag}) " if s.quote_tag else ""
            lines.append(f'  > {tag_prefix}"{s.quote}"')
        else:
            lines.append("  > *no suitable quotable sentence*")
    return "\n".join(lines)


def _render_open_problems_primary(op: OpenProblemsPrimary) -> str:
    """Render the Open Problems & Future Directions section for primary papers."""
    future = _render_bullets(op.future_work_proposed) or "- not reported"
    questions = _render_bullets(op.open_questions) or "- not reported"
    return (
        "**Future work proposed** (what the paper suggests should come next):\n"
        f"{future}\n\n"
        "**Open questions** (theoretical or empirical questions left unresolved):\n"
        f"{questions}"
    )


def _render_open_problems_synthesis(op: OpenProblemsSynthesis) -> str:
    """Render the Open Problems & Future Directions section for synthesis papers."""
    gaps = _render_bullets(op.gaps_identified) or "- not reported"
    questions = _render_bullets(op.open_questions) or "- not reported"
    focus = _render_bullets(op.suggested_research_focus) or "- not reported"
    return (
        "**Gaps identified** (what the paper flags as missing in the field):\n"
        f"{gaps}\n\n"
        "**Open questions** (theoretical or empirical questions left unresolved):\n"
        f"{questions}\n\n"
        "**Suggested research focus** (specific next experiments or directions the paper proposes):\n"
        f"{focus}"
    )


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
    """Render a primary research paper (≤600 words for Part 1 prose)."""
    prose_words = _count_words(
        part1.tldr,
        part1.problem_motivation,
        part1.core_contribution,
        part1.methods,
        part1.results,
        part1.key_takeaways,
        part1.limitations,
        part1.critical_assessment,
        part1.relevance,
    )
    _check_word_limit("primary", prose_words)

    open_problems = _render_open_problems_primary(part1.open_problems_future_directions)
    notable = _render_bullets(part1.notable_findings)
    citable = _render_citable_snippets(part1.citable_snippets)

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
        f"### Open Problems & Future Directions\n\n{open_problems}\n\n"
        f"### Critical Assessment\n\n{part1.critical_assessment}\n\n"
        f"### Notable Findings\n\n{notable}\n\n"
        f"### Citable Snippets\n\n{citable}\n\n"
        f'### Relevance to a review on "spiking neural networks for control"\n\n{part1.relevance}\n\n'
        "---\n\n"
        f"{_render_part2(part2)}"
    )


def _render_synthesis(
    meta: PaperMetadata,
    part1: SummaryPart1Synthesis,
) -> str:
    """Render a synthesis paper (≤1000 words for Part 1 prose)."""
    prose_words = _count_words(
        part1.tldr,
        part1.target_papers_field,
        part1.scope_coverage,
        part1.taxonomy_organization,
        part1.core_argument,
        part1.synthesis_contribution,
        part1.key_claims_narrative,
        part1.key_takeaways,
        part1.limitations,
        part1.critical_assessment,
        part1.relevance,
    )
    _check_word_limit("synthesis", prose_words)

    open_problems = _render_open_problems_synthesis(part1.open_problems_future_directions)
    notable = _render_bullets(part1.notable_findings)
    citable = _render_citable_snippets(part1.citable_snippets)

    return (
        f"{_render_header(meta)}\n\n"
        "---\n\n"
        f"## TL;DR\n\n{part1.tldr}\n\n"
        "---\n\n"
        "## Part 1: Paper Summary\n\n"
        f"### Target Paper(s) / Field\n\n{part1.target_papers_field}\n\n"
        f"### Detailed Scope & Coverage\n\n{part1.scope_coverage}\n\n"
        f"### Taxonomy & Organization\n\n{part1.taxonomy_organization}\n\n"
        f"### Core Argument\n\n{part1.core_argument}\n\n"
        f"### Synthesis Contribution\n\n{part1.synthesis_contribution}\n\n"
        f"### Key Claims & Narrative\n\n{part1.key_claims_narrative}\n\n"
        f"### Key Takeaways\n\n{part1.key_takeaways}\n\n"
        f"### Limitations\n\n{part1.limitations}\n\n"
        f"### Open Problems & Future Directions\n\n{open_problems}\n\n"
        f"### Critical Assessment\n\n{part1.critical_assessment}\n\n"
        f"### Notable Findings\n\n{notable}\n\n"
        f"### Citable Snippets\n\n{citable}\n\n"
        f'### Relevance to a review on "spiking neural networks for control"\n\n{part1.relevance}'
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
