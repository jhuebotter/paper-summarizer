"""Tests for summarizer/models.py — pydantic models, dataclass Config, exceptions."""

import pytest
from pathlib import Path
from pydantic import ValidationError

from summarizer.models import (
    LLMResponse,
    PaperMetadata,
    SummaryPart1Primary,
    SummaryPart1Survey,
    SummaryPart1Commentary,
    SummaryPart1NonResearch,
    SummaryPart2,
    PaperSummary,
    Config,
    FailedPaper,
    BatchReport,
    ParseError,
    LLMError,
    PipelineError,
)

# ---------------------------------------------------------------------------
# Helpers — split the flat conftest mock dict into metadata vs. part1 keys
# ---------------------------------------------------------------------------

_METADATA_ONLY_KEYS = frozenset(
    {"citation_key", "title", "authors", "year", "venue", "tags"}
)


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
    assert isinstance(part1.cite_for, list)


def test_summary_part1_primary_wrong_type_rejected():
    with pytest.raises(ValidationError):
        SummaryPart1Primary(
            paper_type="survey",  # wrong literal for this class
            tldr="Some summary.",
            problem_motivation="gap",
            core_contribution="contribution",
            methods="BPTT",
            results="good",
            limitations="sim only",
            relevance="high",
            cite_for=["thing"],
            critical_assessment="OK.",
            quotable_sentences=["sentence"],
        )


def test_summary_part1_survey_valid():
    part1 = SummaryPart1Survey(
        paper_type="survey",
        tldr="A review of SNNs in robotics.",
        scope_coverage="SNNs for control, 2018–2024, ~80 papers.",
        taxonomy_organization="Organized by learning mechanism.",
        key_claims_narrative="SNNs are increasingly competitive with ANNs.",
        gaps_identified="No real-robot benchmarks exist.",
        relevance="Background for Section 2.",
        cite_for=["Overview of SNN learning paradigms"],
        critical_assessment="Narrative selection bias; no systematic inclusion criteria.",
        quotable_sentences=["The field is rapidly maturing."],
    )
    assert part1.paper_type == "survey"


def test_summary_part1_commentary_valid():
    part1 = SummaryPart1Commentary(
        paper_type="commentary",
        tldr="The author argues SNNs are underutilized in robotics.",
        core_argument="Current SNN benchmarks underestimate capability.",
        target_papers="Huebotter et al. 2025",
        limitations="Argument lacks quantitative support.",
        relevance="Contextual opinion on the field.",
        cite_for=["Critique of SNN benchmarking practices"],
        critical_assessment="Overstates implications from limited evidence.",
        quotable_sentences=["As the author argues..."],
    )
    assert part1.paper_type == "commentary"


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
    assert config.base_url == "http://localhost:1234/v1"
    assert config.model == "openai/gpt-oss-120b:free"
    assert config.max_chars == 200_000
    assert config.force_summary is False
    assert config.reparse is False
    assert config.dry_run is False
    assert config.skill_data_dir == Path("skill_data/references")
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


def test_parse_error_is_exception():
    err = ParseError("docling failed")
    assert isinstance(err, Exception)
    with pytest.raises(ParseError):
        raise err


def test_llm_error_is_exception():
    err = LLMError("llm failed")
    assert isinstance(err, Exception)
    with pytest.raises(LLMError):
        raise err


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


# ---------------------------------------------------------------------------
# LLMResponse (v1.1 combined response model)
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
        k: mock_part1_dict[k]
        for k in mock_part1_dict
        if k not in (meta_keys - {"paper_type"})
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
