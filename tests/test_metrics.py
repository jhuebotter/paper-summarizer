"""Tests for summarizer/metrics.py — deterministic per-summary quality metrics."""

import pytest

from summarizer.metrics import (
    Rate,
    anchor_coverage,
    compute_metrics,
    duplicate_keys,
    evidence_tags,
    first_person_count,
    normalize_text,
    quote_faithfulness,
    split_sentences,
)
from summarizer.models import (
    CitableSnippet,
    PaperMetadata,
    PaperSummary,
    SummaryPart1NonResearch,
    SummaryPart1Primary,
    SummaryPart1Synthesis,
    SummaryPart2,
)

PAPER = (
    "Abstract. We train a spik-\ning neural netﬁts controller on a **7-DOF** arm. "
    "The controller reaches 95.2% success, i.e. it generalizes across tasks and seeds."
)


def _primary(**part1_overrides) -> PaperSummary:
    part1 = dict(
        paper_type="primary",
        tldr="The authors train an SNN controller.",
        problem_motivation="p",
        core_contribution="c",
        methods="m",
        results="It reaches 95.2% success (Source: Tbl. 2).",
        key_takeaways="k",
        limitations="l",
        critical_assessment="a",
        relevance="r",
        notable_findings=["95.2% success (Measured) (Source: Tbl. 2)"],
        citable_snippets=[
            CitableSnippet(cite_for="x", source="Sec. 1", quote="spiking neural netfits controller")
        ],
    )
    part1.update(part1_overrides)
    return PaperSummary(
        metadata=PaperMetadata(
            citation_key="doe2024spiking",
            title="T",
            authors=["Jane Doe"],
            year=2024,
            venue="v",
            is_research_paper=True,
            paper_type="primary",
            tags=[],
        ),
        part1=SummaryPart1Primary(**part1),
        part2=SummaryPart2(**dict.fromkeys(SummaryPart2.model_fields, "not reported")),
    )


# ---------------------------------------------------------------------------
# Rate
# ---------------------------------------------------------------------------


def test_rate_with_nothing_to_count_is_not_applicable():
    assert Rate(0, 0).value is None
    assert Rate(1, 4).as_dict() == {"n": 1, "of": 4, "value": 0.25}


# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------


def test_normalize_text_handles_ligatures_hyphenation_quotes_and_markup():
    assert normalize_text("spik-\ning neﬁts **bold** “quoted”") == ("spiking nefits bold quoted")
    assert normalize_text("state-of-the-art") == normalize_text("state of the art")


@pytest.mark.parametrize(
    "quote,status",
    [
        ("spiking neural netfits controller on a 7-DOF arm", "verbatim"),
        ("We train a spiking … generalizes across tasks and seeds", "verbatim"),
        (
            "The controller reaches 95.2% success, i.e. it generalizes across all tasks and seeds.",
            "near",
        ),
        ("The controller fails on every task we tried in the real world", "not_found"),
    ],
)
def test_quote_status(quote, status):
    summary = _primary(citable_snippets=[CitableSnippet(cite_for="x", source="s", quote=quote)])
    result = quote_faithfulness(summary, PAPER)
    assert result[status] == 1
    assert result["not_found_quotes"] == ([quote] if status == "not_found" else [])


def test_snippets_without_quote_are_not_counted():
    summary = _primary(citable_snippets=[CitableSnippet(cite_for="x", source="s", quote=None)])
    assert quote_faithfulness(summary, PAPER)["total"] == 0


# ---------------------------------------------------------------------------
# Prose checks
# ---------------------------------------------------------------------------


def test_split_sentences_keeps_abbreviations_together():
    text = "See Sec. 3 and Fig. 2 for details. Smith et al. argue X. It works, e.g. here."
    assert split_sentences(text) == [
        "See Sec. 3 and Fig. 2 for details.",
        "Smith et al. argue X.",
        "It works, e.g. here.",
    ]


def test_anchor_coverage_ignores_years_and_named_numbers():
    texts = [
        "In 2021 the authors ran it on the Loihi 2 chip.",  # no number that needs an anchor
        "It reaches 95.2% success (Source: Tbl. 2).",  # covered
        "It uses 1000 neurons. (Source: Sec. 3)",  # covered by the next sentence
        "It is 3x faster.",  # not covered
    ]
    assert anchor_coverage(texts) == Rate(2, 3)


def test_first_person_outside_quotes_only():
    count, words = first_person_count(['We propose a model; "we show" is quoted.', "Tested by us."])
    assert count == 2
    assert words == 9


def test_evidence_tags_exactly_one_and_no_measured_in_synthesis():
    primary = _primary(
        notable_findings=[
            "a (Measured) (Source: x)",
            "b (Claimed — no chip) (Source: y)",
            "c without tag",
            "d (Measured) (Reported)",
        ]
    )
    assert evidence_tags(primary) == Rate(2, 4)

    synthesis = PaperSummary(
        metadata=primary.metadata.model_copy(update={"paper_type": "synthesis"}),
        part1=SummaryPart1Synthesis(
            paper_type="synthesis",
            **dict.fromkeys(
                [
                    "tldr",
                    "target_papers_field",
                    "scope_coverage",
                    "taxonomy_organization",
                    "core_argument",
                    "synthesis_contribution",
                    "key_claims_narrative",
                    "key_takeaways",
                    "limitations",
                    "critical_assessment",
                    "relevance",
                ],
                "x",
            ),
            notable_findings=["a (Measured)", "b (Reported)"],
        ),
        part2=None,
    )
    assert evidence_tags(synthesis) == Rate(1, 2)


def test_compute_metrics_for_primary_paper():
    metrics = compute_metrics(_primary(methods="We designed it."), PAPER)
    assert metrics["quotes"]["verbatim"] == 1
    assert metrics["first_person"]["count"] == 1
    assert metrics["word_budget"]["limit"] == 600
    assert 0 < metrics["word_budget"]["ratio"] < 1
    assert metrics["evidence_tags"]["value"] == 1.0


def test_compute_metrics_is_empty_for_non_research():
    summary = PaperSummary(
        metadata=PaperMetadata(
            citation_key="x2020y",
            title="t",
            authors=[],
            year=2020,
            venue="v",
            is_research_paper=False,
            paper_type=None,
            rejection_reason="slides",
            tags=[],
        ),
        part1=SummaryPart1NonResearch(paper_type="non_research", note="slides"),
        part2=None,
    )
    assert compute_metrics(summary, PAPER) == {}


def test_duplicate_keys():
    assert duplicate_keys(["a", "b", "a", "c", "b"]) == ["a", "b"]
