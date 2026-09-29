"""Tests for summarizer/models.py — pydantic models, dataclass Config, exceptions."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from summarizer.models import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_SKILL_DATA_DIR,
    BatchReport,
    Config,
    FailedPaper,
    LLMResponse,
    PaperMetadata,
    PaperSummary,
    PipelineError,
    SummaryPart1NonResearch,
    SummaryPart1Primary,
    SummaryPart1Synthesis,
    SummaryPart2,
)

# ---------------------------------------------------------------------------
# Helpers — split the flat conftest mock dict into metadata vs. part1 keys
# ---------------------------------------------------------------------------

_METADATA_ONLY_KEYS = frozenset({"citation_key", "title", "authors", "year", "venue", "tags"})


def _meta_fields(d: dict) -> dict:
    """Return only PaperMetadata fields (includes paper_type)."""
    return {k: d[k] for k in (*_METADATA_ONLY_KEYS, "paper_type")}


def _part1_fields(d: dict) -> dict:
    """Return Part1 fields (paper_type + prose fields; excludes metadata-only keys)."""
    return {k: v for k, v in d.items() if k not in _METADATA_ONLY_KEYS}


# ---------------------------------------------------------------------------
# PaperMetadata
# ---------------------------------------------------------------------------


def test_paper_metadata_valid(mock_part1_dict):
    data = _meta_fields(mock_part1_dict)
    data["is_research_paper"] = True
    data["rejection_reason"] = None
    meta = PaperMetadata(**data)
    assert meta.citation_key == "huebotter2025spiking"
    assert meta.year == 2025
    assert "SNN" in meta.tags


def test_paper_metadata_invalid_year():
    with pytest.raises(ValidationError):
        PaperMetadata(
            citation_key="test2025foo",
            title="Test",
            authors=["Test Author"],
            year="not-a-year",
            venue="Preprint",
            is_research_paper=True,
            paper_type="primary",
            rejection_reason=None,
            tags=[],
        )


def test_paper_metadata_invalid_paper_type():
    with pytest.raises(ValidationError):
        PaperMetadata(
            citation_key="test2025foo",
            title="Test",
            authors=["Test Author"],
            year=2025,
            venue="Preprint",
            is_research_paper=True,
            paper_type="bogus",
            rejection_reason=None,
            tags=[],
        )


def test_paper_metadata_non_research_requires_null_paper_type():
    with pytest.raises(ValidationError):
        PaperMetadata(
            citation_key="x2025doc",
            title="Slides",
            authors=["Unknown"],
            year=2025,
            venue="not applicable",
            is_research_paper=False,
            paper_type="primary",
            rejection_reason="Not a research paper",
            tags=["non-research"],
        )


def test_paper_metadata_non_research_requires_rejection_reason():
    with pytest.raises(ValidationError):
        PaperMetadata(
            citation_key="x2025doc",
            title="Slides",
            authors=["Unknown"],
            year=2025,
            venue="not applicable",
            is_research_paper=False,
            paper_type=None,
            rejection_reason=None,
            tags=["non-research"],
        )


# ---------------------------------------------------------------------------
# SummaryPart1 variants
# ---------------------------------------------------------------------------


def test_summary_part1_primary_valid(mock_part1_dict):
    part1 = SummaryPart1Primary(**_part1_fields(mock_part1_dict))
    assert part1.paper_type == "primary"
    assert part1.tldr == mock_part1_dict["tldr"]
    assert isinstance(part1.citable_snippets, list)


def test_summary_part1_primary_wrong_type_rejected():
    with pytest.raises(ValidationError):
        SummaryPart1Primary(
            paper_type="synthesis",  # wrong literal for this class
            tldr="Some summary.",
            problem_motivation="gap",
            core_contribution="contribution",
            methods="BPTT",
            results="good",
            key_takeaways="Key finding.",
            limitations="sim only",
            relevance="high",
            critical_assessment="OK.",
        )


def test_summary_part1_synthesis_valid():
    part1 = SummaryPart1Synthesis(
        paper_type="synthesis",
        tldr="A review of SNNs in robotics.",
        target_papers_field="SNNs for robot control.",
        scope_coverage="SNNs for control, 2018–2024, ~80 papers, narrative selection.",
        taxonomy_organization="Organized by learning mechanism: STDP, surrogate gradients, reward-based.",
        core_argument="The authors argue that SNNs are increasingly competitive with ANNs for energy-constrained applications.",
        synthesis_contribution="Introduces a three-axis taxonomy (neuron model, learning rule, deployment) not previously formalized.",
        key_claims_narrative="SNNs achieve competitive performance on simulated tasks but no real-robot benchmarks exist.",
        key_takeaways="The field needs standardized benchmarks and more real-hardware evaluations.",
        limitations="Narrative selection bias; no systematic inclusion criteria stated.",
        relevance="Background for Section 2 of the SNN control review.",
        critical_assessment="Narrative selection bias is unacknowledged; quantitative claims lack trace anchors.",
    )
    assert part1.paper_type == "synthesis"
    assert isinstance(part1.citable_snippets, list)
    assert isinstance(part1.notable_findings, list)


def test_summary_part1_non_research_valid():
    part1 = SummaryPart1NonResearch(
        paper_type="non_research",
        note="Document appears to be a scanned form, not a research paper.",
    )
    assert part1.paper_type == "non_research"
    assert "scanned" in part1.note


# ---------------------------------------------------------------------------
# SummaryPart2
# ---------------------------------------------------------------------------


def test_summary_part2_valid(mock_part2_dict):
    part2 = SummaryPart2(**mock_part2_dict)
    assert part2.neuron_model == mock_part2_dict["neuron_model"]


def test_summary_part2_missing_required_field(mock_part2_dict):
    incomplete = {k: v for k, v in mock_part2_dict.items() if k != "neuron_model"}
    with pytest.raises(ValidationError):
        SummaryPart2(**incomplete)


# ---------------------------------------------------------------------------
# PaperSummary (composition)
# ---------------------------------------------------------------------------


def test_paper_summary_composition(mock_part1_dict, mock_part2_dict):
    meta_data = _meta_fields(mock_part1_dict)
    meta_data["is_research_paper"] = True
    meta_data["rejection_reason"] = None
    metadata = PaperMetadata(**meta_data)
    part1 = SummaryPart1Primary(**_part1_fields(mock_part1_dict))
    part2 = SummaryPart2(**mock_part2_dict)
    summary = PaperSummary(metadata=metadata, part1=part1, part2=part2)
    assert summary.metadata.citation_key == "huebotter2025spiking"
    assert summary.part1.paper_type == "primary"
    assert summary.part2 is not None
    assert summary.part2.neuron_model == mock_part2_dict["neuron_model"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_defaults():
    config = Config()
    assert config.base_url == DEFAULT_BASE_URL == "https://openrouter.ai/api/v1"
    assert config.model == DEFAULT_MODEL
    assert config.max_chars == 200_000
    assert config.force_summary is False
    assert config.reparse is False
    assert config.dry_run is False
    assert config.skill_data_dir == DEFAULT_SKILL_DATA_DIR
    assert (config.skill_data_dir / "json-output-contract.md").exists()
    assert config.output_dir == Path("output_summaries")
    assert config.verbose is False
    assert config.api_key is None
    assert config.timeout_s == 120
    assert config.max_output_tokens is None
    assert config.workers == 3
    assert config.extractor == "auto"


def test_config_api_key_and_timeout():
    config = Config(api_key="sk-test-key", timeout_s=300)
    assert config.api_key == "sk-test-key"
    assert config.timeout_s == 300


def test_config_custom_values():
    config = Config(
        base_url="http://custom:5678/v1",
        model="custom-model",
        max_chars=10_000,
        force_summary=True,
        reparse=True,
        skill_data_dir=Path("/custom/skill_data"),
        output_dir=Path("/custom/output"),
        workers=7,
        extractor="pypdf",
    )
    assert config.base_url == "http://custom:5678/v1"
    assert config.max_chars == 10_000
    assert config.force_summary is True
    assert config.reparse is True
    assert config.skill_data_dir == Path("/custom/skill_data")
    assert config.output_dir == Path("/custom/output")
    assert config.workers == 7
    assert config.extractor == "pypdf"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


def test_pipeline_error_stores_path_and_cause():
    cause = ValueError("root cause")
    path = Path("/data/paper.pdf")
    err = PipelineError(pdf_path=path, cause=cause)
    assert err.pdf_path == path
    assert err.cause is cause
    assert isinstance(err, Exception)
    assert "paper.pdf" in str(err)


# ---------------------------------------------------------------------------
# BatchReport / FailedPaper
# ---------------------------------------------------------------------------


def test_failed_paper_valid():
    fp = FailedPaper(pdf_path="/some/paper.pdf", error="parse failed")
    assert fp.pdf_path == "/some/paper.pdf"
    assert fp.error == "parse failed"


def test_batch_report_valid():
    report = BatchReport(
        processed=5,
        skipped=2,
        failed=1,
        failed_papers=[FailedPaper(pdf_path="/foo.pdf", error="error")],
    )
    assert report.processed == 5
    assert report.failed == 1
    assert len(report.failed_papers) == 1


def test_batch_report_total_cost_defaults_to_zero():
    report = BatchReport(processed=0, skipped=0, failed=0, failed_papers=[])
    assert report.total_cost == 0.0


def test_batch_report_total_cost_accepts_value():
    report = BatchReport(
        processed=3,
        skipped=0,
        failed=0,
        failed_papers=[],
        total_cost=0.0345,
    )
    assert report.total_cost == pytest.approx(0.0345)


# ---------------------------------------------------------------------------
# LLMResponse (combined response model)
# ---------------------------------------------------------------------------


def test_llm_response_primary_paper(mock_part1_dict, mock_part2_dict):
    """LLMResponse parses correctly for a primary paper (part2 present)."""
    meta_keys = frozenset(
        {"citation_key", "title", "authors", "year", "venue", "paper_type", "tags"}
    )
    metadata = {k: mock_part1_dict[k] for k in meta_keys}
    metadata["is_research_paper"] = True
    metadata["rejection_reason"] = None
    part1 = {
        k: mock_part1_dict[k] for k in mock_part1_dict if k not in (meta_keys - {"paper_type"})
    }
    response = LLMResponse(
        metadata=metadata,
        part1=part1,
        part2=mock_part2_dict,
    )
    assert response.metadata.citation_key == "huebotter2025spiking"
    assert response.part1.paper_type == "primary"
    assert response.part2 is not None
    assert response.part2.neuron_model == mock_part2_dict["neuron_model"]


def test_llm_response_non_research_part2_none():
    """LLMResponse accepts non-research docs with part2=None."""
    response = LLMResponse(
        metadata={
            "citation_key": "doc2025foo",
            "title": "Scanned Form",
            "authors": ["Unknown"],
            "year": 2025,
            "venue": "not applicable",
            "is_research_paper": False,
            "paper_type": None,
            "rejection_reason": "Scanned form",
            "tags": ["non-research"],
        },
        part1={"paper_type": "non_research", "note": "Not a research paper."},
        part2=None,
    )
    assert response.part1.paper_type == "non_research"
    assert response.part2 is None


def test_llm_response_invalid_metadata_rejected(mock_part2_dict):
    """LLMResponse raises ValidationError when metadata fields are invalid."""
    with pytest.raises(ValidationError):
        LLMResponse(
            metadata={"paper_type": "primary"},  # missing required fields
            part1={"paper_type": "primary", "tldr": "x"},
            part2=mock_part2_dict,
        )


# ---------------------------------------------------------------------------
# Classification (typed Part 2 labels)
# ---------------------------------------------------------------------------


def test_classification_labels_are_matched_leniently():
    from summarizer.models import Classification

    c = Classification(
        inference_hardware="cpu / gpu",
        architecture="Fully  Spiking",
        credit_assignment="semi-local",
        learning_regime="ONLINE",
        paradigm_families=["reinforcement learning (model-free)", "Hybrid/multi-phase"],
    )
    assert c.model_dump() == {
        "inference_hardware": "CPU/GPU",
        "architecture": "fully spiking",
        "credit_assignment": "Semi-local",
        "learning_regime": "Online",
        "paradigm_families": ["Reinforcement learning (model-free)", "Hybrid / multi-phase"],
    }


def test_classification_rejects_unknown_labels():
    from pydantic import ValidationError

    from summarizer.models import Classification

    with pytest.raises(ValidationError):
        Classification(
            inference_hardware="TPU",
            architecture="hybrid",
            credit_assignment="Global",
            learning_regime="Offline",
            paradigm_families=[],
        )


def test_classification_vocabularies_match_the_references():
    """Drift guard: every allowed label is spelled out in the prompt references."""
    from typing import get_args

    from summarizer.models import Classification

    refs = Path(__file__).parent.parent / "skill_data" / "references"
    contract = (refs / "json-output-contract.md").read_text(encoding="utf-8")
    paradigms = (refs / "learning-paradigms.md").read_text(encoding="utf-8")
    for name, field in Classification.model_fields.items():
        labels = get_args(field.annotation)
        if name == "paradigm_families":
            for label in get_args(labels[0]):
                assert f"`{label}`" in paradigms, label
        else:
            for label in labels:
                assert label in contract, (name, label)


def test_classification_placeholders_and_list_forms():
    from summarizer.models import Classification

    c = Classification(
        inference_hardware="N/A",
        architecture="Not applicable",
        credit_assignment="not applicable",
        learning_regime="Not Applicable",
        paradigm_families="hybrid/multi-phase",
    )
    assert (c.inference_hardware, c.architecture) == ("not reported", "not reported")
    assert (c.credit_assignment, c.learning_regime) == ("Not applicable", "not applicable")
    assert c.paradigm_families == ["Hybrid / multi-phase"]
    none = Classification(**{**c.model_dump(), "paradigm_families": None})
    assert none.paradigm_families == []
    dupes = Classification(
        **{
            **c.model_dump(),
            "paradigm_families": ["hybrid/multi-phase", "Hybrid / Multi-phase", "not reported"],
        }
    )
    assert dupes.paradigm_families == ["Hybrid / multi-phase"]


def test_classification_contract_lists_exactly_the_allowed_labels():
    """Two-way drift guard: the contract's `a | b | c` line per field equals the Literal."""
    import re

    from summarizer.models import Classification, labels

    refs = Path(__file__).parent.parent / "skill_data" / "references"
    contract = (refs / "json-output-contract.md").read_text(encoding="utf-8")
    for name, field in Classification.model_fields.items():
        if name == "paradigm_families":
            continue  # defined in learning-paradigms.md, checked separately
        line = re.search(rf'"{name}": "([^"]+)"', contract).group(1)
        assert set(line.split(" | ")) == set(labels(field.annotation)), name


@pytest.mark.parametrize(
    "value, expected",
    [
        (
            [{"finding": "SNN matches ANN", "source": "Fig. 2"}],
            ["SNN matches ANN (Source: Fig. 2)"],
        ),
        (
            [{"finding": "27 uJ per inference", "evidence": "Measured", "source": "Tbl. 2"}],
            ["27 uJ per inference (Measured) (Source: Tbl. 2)"],
        ),
        ("one finding (Claimed) (Source: Sec. 1)", ["one finding (Claimed) (Source: Sec. 1)"]),
        (  # the word "Measured" in the text is not the evidence tag
            [{"text": "Measured energy drops", "evidence": "Measured", "source": "Fig. 3"}],
            ["Measured energy drops (Measured) (Source: Fig. 3)"],
        ),
        (  # tags already written into the text are not repeated
            [
                {
                    "finding": "x (Reported) (Source: Tbl. 1)",
                    "evidence": "Reported",
                    "source": "Tbl. 1",
                }
            ],
            ["x (Reported) (Source: Tbl. 1)"],
        ),
        (None, []),
    ],
)
def test_notable_findings_accept_objects_and_bare_strings(mock_part1_dict, value, expected):
    """Regression: models returned {finding, source} objects or a bare string, each
    costing a schema-repair call."""
    part1 = SummaryPart1Primary(**{**_part1_fields(mock_part1_dict), "notable_findings": value})
    assert part1.notable_findings == expected


def test_synthesis_notable_findings_accept_objects():
    text_fields = [
        name
        for name, field in SummaryPart1Synthesis.model_fields.items()
        if field.annotation is str
    ]
    part1 = SummaryPart1Synthesis(
        paper_type="synthesis",
        notable_findings=[{"finding": "72 papers surveyed", "source": "Sec. 2"}],
        **dict.fromkeys(text_fields, "x"),
    )
    assert part1.notable_findings == ["72 papers surveyed (Source: Sec. 2)"]


def test_notable_findings_reject_unrecognized_objects(mock_part1_dict):
    with pytest.raises(ValidationError):
        SummaryPart1Primary(
            **{**_part1_fields(mock_part1_dict), "notable_findings": [{"note": "x"}]}
        )
