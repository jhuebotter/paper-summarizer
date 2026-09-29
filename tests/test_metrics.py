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


_PART2 = dict.fromkeys(SummaryPart2.model_fields, "not reported") | {
    "classification": {
        "inference_hardware": "not reported",
        "architecture": "not reported",
        "credit_assignment": "not reported",
        "learning_regime": "not reported",
        "paradigm_families": [],
    }
}


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
        part2=SummaryPart2(**_PART2),
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


def test_normalize_text_nfkc_citations_and_hyphens():
    assert normalize_text("x\u00b2 \uff46\uff55\uff4c\uff4c ne\ufb01ts") == "x2 full nefits"
    assert normalize_text("prior work [12], [3\u20135] shows") == "prior work  ,   shows"
    # Hyphenated words are joined the same way in the paper and in the quote.
    assert normalize_text("spike-\ntiming") == normalize_text("spike-timing") == "spiketiming"
    assert normalize_text("A \u2014 B") == "a   b"


def _status(quote: str, paper: str = PAPER) -> str:
    summary = _primary(citable_snippets=[CitableSnippet(cite_for="x", source="s", quote=quote)])
    result = quote_faithfulness(summary, paper)
    return next(k for k in ("verbatim", "near", "not_found") if result[k])


@pytest.mark.parametrize(
    "quote,status",
    [
        ("spiking neural netfits controller on a 7-DOF arm", "verbatim"),
        ("We train a spiking ... generalizes across tasks and seeds", "verbatim"),
        ("We train a spiking [...] generalizes across tasks and seeds", "verbatim"),
        ("We train a spiking \u2026 generalizes across tasks and seeds", "verbatim"),
        # 12 words, one inserted: 7 of 10 three-grams still match (0.7) -> near
        (
            "The controller reaches 95.2% success, i.e. it generalizes across all tasks and seeds.",
            "near",
        ),
        # two inserted words: 6 of 11 (0.55) -> not found
        (
            "The controller reaches 95.2% success, i.e. it clearly generalizes across all tasks.",
            "not_found",
        ),
        ("The controller fails on every task we tried in the real world", "not_found"),
        # fragments that exist but in the wrong order are a stitched quote
        ("generalizes across tasks and seeds ... We train a spiking", "not_found"),
        # a made-up second fragment
        ("We train a spiking ... and beats every baseline by far", "not_found"),
        # fragments under three words can't be checked
        ("We train a spiking ... controller", "near"),
        ("the ... of", "not_found"),
    ],
)
def test_quote_status(quote, status):
    assert _status(quote) == status


def test_quote_matches_text_extracted_with_glued_words():
    """Regression: pypdf glued the words of some PDFs ("Thedevelopmentprocess..."), so
    correctly spaced quotes of them were reported as not found."""
    paper = "oftheendeffector.Thedevelopmentprocessinvolvedfourstages:(1)Designing"
    assert _status("The development process involved four stages", paper) == "verbatim"
    assert _status("The design process involved four stages", paper) == "not_found"


def test_quote_ignores_punctuation_and_numeric_citations():
    paper = "As shown in prior work [12], spiking networks can control robot arms."
    assert _status("As shown in prior work, spiking networks can control robot arms", paper) == (
        "verbatim"
    )


def test_quote_matches_whole_words_only():
    assert _status("ron model is", "The neuron model is simple.") == "not_found"


def test_not_found_quotes_are_listed():
    quote = "The controller fails on every task we tried in the real world"
    summary = _primary(citable_snippets=[CitableSnippet(cite_for="x", source="s", quote=quote)])
    assert quote_faithfulness(summary, PAPER)["not_found_quotes"] == [quote]


def test_snippets_without_quote_are_not_counted():
    summary = _primary(citable_snippets=[CitableSnippet(cite_for="x", source="s", quote=None)])
    assert quote_faithfulness(summary, PAPER)["total"] == 0


# ---------------------------------------------------------------------------
# Prose checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "See Sec. 3 and Fig. 2 for details. Smith et al. argue X. It works, e.g. here.",
            ["See Sec. 3 and Fig. 2 for details.", "Smith et al. argue X.", "It works, e.g. here."],
        ),
        ("Smith et al. Show X. Next.", ["Smith et al. Show X.", "Next."]),
        ("Chips, e.g. Loihi, help.", ["Chips, e.g. Loihi, help."]),
        ("see fig. 2 and (Source: p. 5, Alg. 1).", ["see fig. 2 and (Source: p. 5, Alg. 1)."]),
        ('It is fast. 95% pass. "Quoted" next.', ["It is fast.", "95% pass.", '"Quoted" next.']),
    ],
)
def test_split_sentences(text, expected):
    assert split_sentences(text) == expected


@pytest.mark.parametrize(
    "sentence,needs_anchor",
    [
        ("It reaches 95.2% success.", True),
        ("SNN 94.8% vs. ANN 95.1% accuracy.", True),  # not hidden by capitalized words
        ("It uses 2048 neurons.", True),  # a number, not a year
        ("It uses 7-DOF arms and is 3x faster.", True),
        ("In 2021 the authors ran it on the Loihi 2 chip.", False),
        ("(Smith et al., 2020) show it; see Table 2 and Phase 3.", False),
        ("Table 2 lists results.", False),
        ("CIFAR-10 and ResNet-18 in 3D.", False),
    ],
)
def test_needs_anchor(sentence, needs_anchor):
    assert (anchor_coverage([sentence]).of == 1) is needs_anchor


def test_anchor_coverage_counts_same_or_next_sentence():
    texts = [
        "It reaches 95.2% success (Source: Tbl. 2).",  # covered
        "It uses 1000 neurons. (Source: Sec. 3)",  # covered by the next sentence
        "Latency is 3 ms (Source: p. 5).",  # "p." must not split the anchor
        "It is 3x faster.",  # not covered
    ]
    assert anchor_coverage(texts) == Rate(3, 4)


def test_citable_snippet_anchor_is_its_source_field():
    summary = _primary(
        results="r",
        citable_snippets=[CitableSnippet(cite_for="Energy is 12 mJ per step", source="Tbl. 3")],
    )
    assert compute_metrics(summary, PAPER)["anchors"]["value"] == 1.0


def test_first_person_outside_quotes_only():
    count, words = first_person_count(
        [
            'We propose a model; "we show" is quoted.',
            "\u2018we find\u2019 is quoted too; \u201cour model\u201d as well.",
            "It takes 5 us per step; tested by us.",
        ]
    )
    assert count == 2  # "We propose" and "by us"
    assert words > 0


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
