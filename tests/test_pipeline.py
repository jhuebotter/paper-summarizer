"""Tests for summarizer/pipeline.py — end-to-end per-paper orchestration (mocked LLM)."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from summarizer.llm import CostAccumulator, ModelPricing, UsageStats
from summarizer.models import Config, LLMError, ParseError, PipelineError, PaperSummary
from summarizer.pipeline import (
    process_pdf,
    _sanitize_citation_key,
    _author_surname_token,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_path):
    """Config pointing at the real skill_data/references/ for prompt building."""
    references_dir = Path(__file__).parent.parent / "skill_data" / "references"
    return Config(
        base_url="http://localhost:1234/v1",
        model="test-model",
        skill_data_dir=references_dir,
        output_dir=tmp_path / "output_summaries",
    )


@pytest.fixture
def fake_pdf(tmp_path):
    """A dummy file that stands in for a PDF (docling is mocked)."""
    pdf = tmp_path / "test_paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    return pdf


def _mock_parse(paper_text: str):
    """Return a patcher that makes parse_pdf return the given text."""
    return patch("summarizer.pipeline.parse_pdf", return_value=paper_text)


def _make_combined_dict(part1_dict: dict, part2_dict: dict) -> dict:
    """Build the combined LLMResponse dict as the LLM would return it."""
    meta_keys = frozenset(
        {"citation_key", "title", "authors", "year", "venue", "paper_type", "tags"}
    )
    metadata = {k: part1_dict[k] for k in meta_keys}
    metadata["is_research_paper"] = True
    metadata["rejection_reason"] = None
    part1 = {
        k: part1_dict[k] for k in part1_dict if k not in (meta_keys - {"paper_type"})
    }
    return {"metadata": metadata, "part1": part1, "part2": part2_dict}


def _mock_llm_combined(combined_dict: dict):
    """Return a patcher that makes call_llm return the combined dict once."""
    mock_client = MagicMock()
    mock_client.complete.return_value = MagicMock(text=json.dumps(combined_dict))
    return patch(
        "summarizer.pipeline.create_client", return_value=mock_client
    ), mock_client


# ---------------------------------------------------------------------------
# Happy path — primary paper
# ---------------------------------------------------------------------------


def test_process_pdf_returns_paper_summary(
    fake_pdf, config, mock_part1_dict, mock_part2_dict
):
    """Full pipeline run returns a validated PaperSummary for a primary paper."""
    combined = _make_combined_dict(mock_part1_dict, mock_part2_dict)
    client_patcher, mock_client = _mock_llm_combined(combined)

    with _mock_parse("Some paper text."), client_patcher:
        summary = process_pdf(fake_pdf, config)

    assert isinstance(summary, PaperSummary)
    assert summary.metadata.citation_key == mock_part1_dict["citation_key"]
    assert summary.part1.paper_type == "primary"
    assert summary.part2 is not None
    assert summary.part2.neuron_model == mock_part2_dict["neuron_model"]


def test_process_pdf_makes_exactly_one_llm_call(
    fake_pdf, config, mock_part1_dict, mock_part2_dict
):
    """Exactly one LLM call is made per paper (combined call)."""
    combined = _make_combined_dict(mock_part1_dict, mock_part2_dict)
    client_patcher, mock_client = _mock_llm_combined(combined)

    with _mock_parse("Some paper text."), client_patcher:
        process_pdf(fake_pdf, config)

    assert mock_client.complete.call_count == 1


def test_process_pdf_prompt_contains_paper_text(
    fake_pdf, config, mock_part1_dict, mock_part2_dict
):
    """The single prompt contains the paper text."""
    combined = _make_combined_dict(mock_part1_dict, mock_part2_dict)
    client_patcher, mock_client = _mock_llm_combined(combined)

    with _mock_parse("UNIQUE_PAPER_TEXT_XYZ"), client_patcher:
        process_pdf(fake_pdf, config)

    prompt_sent = mock_client.complete.call_args[0][0]
    assert "UNIQUE_PAPER_TEXT_XYZ" in prompt_sent


def test_process_pdf_schema_repair_retry_recovers_missing_primary_fields(
    fake_pdf, config, mock_part1_dict, mock_part2_dict
):
    """A schema repair retry is attempted when primary fields are missing."""
    broken = _make_combined_dict(mock_part1_dict, mock_part2_dict)
    broken["part2"] = None
    broken["part1"].pop("results")
    fixed = _make_combined_dict(mock_part1_dict, mock_part2_dict)

    mock_client = MagicMock()
    mock_client.complete.side_effect = [
        MagicMock(text=json.dumps(broken)),
        MagicMock(text=json.dumps(fixed)),
    ]

    with (
        _mock_parse("Some paper text."),
        patch("summarizer.pipeline.create_client", return_value=mock_client),
    ):
        summary = process_pdf(fake_pdf, config)

    assert summary.part2 is not None
    assert summary.part1.results == mock_part1_dict["results"]
    assert mock_client.complete.call_count == 2
    repair_prompt = mock_client.complete.call_args_list[1].args[0]
    assert "Validation errors" in repair_prompt
    assert "part1.primary.results" in repair_prompt


def test_process_pdf_normalizes_non_integer_metadata_year(
    fake_pdf, config, mock_part1_dict, mock_part2_dict
):
    """Non-integer metadata.year is normalized from source filename when possible."""
    bad_part1 = dict(mock_part1_dict)
    bad_part1["year"] = "not reported"
    bad_part1["citation_key"] = "invalidkey"
    combined = _make_combined_dict(bad_part1, mock_part2_dict)
    client_patcher, _ = _mock_llm_combined(combined)
    pdf_with_year = fake_pdf.with_name("Doe - 2024 - test_paper.pdf")
    pdf_with_year.write_bytes(b"%PDF-1.4 fake")

    with _mock_parse("Some paper text."), client_patcher:
        summary = process_pdf(pdf_with_year, config)

    assert isinstance(summary.metadata.year, int)
    assert summary.metadata.year == 2024


def test_process_pdf_normalizes_invalid_citation_key(
    fake_pdf, config, mock_part1_dict, mock_part2_dict
):
    """Invalid citation keys are deterministically repaired before validation."""
    bad_part1 = dict(mock_part1_dict)
    bad_part1["citation_key"] = "not reported"
    combined = _make_combined_dict(bad_part1, mock_part2_dict)
    client_patcher, _ = _mock_llm_combined(combined)

    with _mock_parse("Some paper text."), client_patcher:
        summary = process_pdf(fake_pdf, config)

    assert summary.metadata.citation_key == "huebotter2025spiking"


def test_process_pdf_sanitizes_accented_citation_key(
    fake_pdf, config, mock_part1_dict, mock_part2_dict
):
    """Accented + hyphenated keys like 'paredes-vallés2024fully' are sanitized to ASCII alnum."""
    bad_part1 = dict(mock_part1_dict)
    bad_part1["citation_key"] = "paredes-vallés2024fully"
    combined = _make_combined_dict(bad_part1, mock_part2_dict)
    client_patcher, _ = _mock_llm_combined(combined)

    with _mock_parse("Some paper text."), client_patcher:
        summary = process_pdf(fake_pdf, config)

    assert summary.metadata.citation_key == "paredesvalles2024fully"


# ---------------------------------------------------------------------------
# Unit tests for citation key helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("paredes-vallés2024fully", "paredesvalles2024fully"),
        ("müller2023control", "muller2023control"),
        ("smith2024neural", "smith2024neural"),  # already clean — no change
        ("Ño2025spiking", "no2025spiking"),
        ("de-wolf2021spiking", "dewolf2021spiking"),
    ],
)
def test_sanitize_citation_key(value, expected):
    assert _sanitize_citation_key(value) == expected


@pytest.mark.parametrize(
    "author_name, expected",
    [
        ("F. Paredes-Vallés", "valles"),   # "First. Surname" → last token is surname
        ("Müller J.", "j"),                # "Surname I." → last token is initial (by design)
        ("J. Müller", "muller"),           # "I. Surname" → last token is surname
        ("Travis DeWolf", "dewolf"),
        ("Smith", "smith"),
    ],
)
def test_author_surname_token_unicode(author_name, expected):
    assert _author_surname_token(author_name) == expected


# ---------------------------------------------------------------------------
# Non-research document path
# ---------------------------------------------------------------------------


def test_process_pdf_non_research(fake_pdf, config):
    """Non-research documents: part2 is None and a PaperSummary is returned."""
    combined = {
        "metadata": {
            "citation_key": "doc2025foo",
            "title": "Scanned Form",
            "authors": ["Unknown"],
            "year": 2025,
            "venue": "not applicable",
            "is_research_paper": False,
            "paper_type": None,
            "rejection_reason": "Not a research paper.",
            "tags": ["non-research"],
        },
        "part1": {"paper_type": "non_research", "note": "Not a research paper."},
        "part2": None,
    }
    mock_client = MagicMock()
    mock_client.complete.return_value = MagicMock(text=json.dumps(combined))

    with (
        _mock_parse("Some text."),
        patch("summarizer.pipeline.create_client", return_value=mock_client),
    ):
        summary = process_pdf(fake_pdf, config)

    assert mock_client.complete.call_count == 1
    assert isinstance(summary, PaperSummary)
    assert summary.part1.paper_type == "non_research"
    assert summary.part2 is None


# ---------------------------------------------------------------------------
# reparse flag is forwarded to parse_pdf
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config_attr,config_value,kwarg_name,expected", [
    ("reparse", True, "reparse", True),
    ("extractor", "pypdf", "extractor", "pypdf"),
])
def test_process_pdf_forwards_parser_config(
    fake_pdf, config, mock_part1_dict, mock_part2_dict, config_attr, config_value, kwarg_name, expected
):
    """Config parser flags are forwarded to parse_pdf as keyword arguments."""
    setattr(config, config_attr, config_value)
    combined = _make_combined_dict(mock_part1_dict, mock_part2_dict)
    client_patcher, _ = _mock_llm_combined(combined)

    with (
        patch("summarizer.pipeline.parse_pdf", return_value="text") as mock_parse,
        client_patcher,
    ):
        process_pdf(fake_pdf, config)

    _, kwargs = mock_parse.call_args
    assert kwargs.get(kwarg_name) == expected


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


def test_process_pdf_parse_error_raises_pipeline_error(fake_pdf, config):
    """ParseError from the PDF parser is wrapped in PipelineError."""
    with patch("summarizer.pipeline.parse_pdf", side_effect=ParseError("corrupt PDF")):
        with pytest.raises(PipelineError) as exc_info:
            process_pdf(fake_pdf, config)
    assert exc_info.value.pdf_path == fake_pdf
    assert isinstance(exc_info.value.cause, ParseError)


_BAD_RESPONSE_JSON = json.dumps({"metadata": {"paper_type": "primary"}, "part1": {}, "part2": {}})


@pytest.mark.parametrize("side_effect,return_text,expected_calls", [
    (Exception("connection refused"), None, 1),
    (None, _BAD_RESPONSE_JSON, 3),
])
def test_process_pdf_llm_errors_raise_pipeline_error(
    fake_pdf, config, side_effect, return_text, expected_calls
):
    """LLM and validation errors are wrapped in PipelineError."""
    mock_client = MagicMock()
    if side_effect is not None:
        mock_client.complete.side_effect = side_effect
    else:
        mock_client.complete.return_value = MagicMock(text=return_text)

    with (
        _mock_parse("text."),
        patch("summarizer.pipeline.create_client", return_value=mock_client),
    ):
        with pytest.raises(PipelineError):
            process_pdf(fake_pdf, config)

    assert mock_client.complete.call_count == expected_calls


# ---------------------------------------------------------------------------
# Phase 4: shared client + accumulator
# ---------------------------------------------------------------------------


def _make_mock_client_for_combined(combined_dict: dict) -> MagicMock:
    """Build a mock client whose complete() returns a proper response with usage."""
    mock_response = MagicMock()
    mock_response.text = json.dumps(combined_dict)
    mock_response.usage = UsageStats(input_tokens=500, output_tokens=200, reasoning_tokens=0)
    mock_client = MagicMock()
    mock_client.pricing = ModelPricing()
    mock_client.complete.return_value = mock_response
    return mock_client


def test_process_pdf_uses_provided_client_and_skips_create_client(
    fake_pdf, config, mock_part1_dict, mock_part2_dict
):
    """When a client is passed, create_client is NOT called."""
    combined = _make_combined_dict(mock_part1_dict, mock_part2_dict)
    mock_client = _make_mock_client_for_combined(combined)

    with (
        _mock_parse("text"),
        patch("summarizer.pipeline.create_client") as mock_create,
    ):
        process_pdf(fake_pdf, config, client=mock_client)

    mock_create.assert_not_called()
    mock_client.complete.assert_called_once()


def test_process_pdf_creates_client_when_none_provided(
    fake_pdf, config, mock_part1_dict, mock_part2_dict
):
    """When client=None, create_client is called exactly once."""
    combined = _make_combined_dict(mock_part1_dict, mock_part2_dict)
    mock_client = _make_mock_client_for_combined(combined)

    with (
        _mock_parse("text"),
        patch("summarizer.pipeline.create_client", return_value=mock_client) as mock_create,
    ):
        process_pdf(fake_pdf, config)  # no client kwarg

    mock_create.assert_called_once_with(config)


def test_process_pdf_accumulator_is_updated(
    fake_pdf, config, mock_part1_dict, mock_part2_dict
):
    """When accumulator is provided, it is updated after the LLM call."""
    combined = _make_combined_dict(mock_part1_dict, mock_part2_dict)
    mock_client = _make_mock_client_for_combined(combined)
    mock_client.pricing = ModelPricing(prompt=1e-6, completion=3e-6)

    acc = CostAccumulator()
    with (
        _mock_parse("text"),
        patch("summarizer.pipeline.create_client", return_value=mock_client),
    ):
        process_pdf(fake_pdf, config, accumulator=acc)

    assert acc.total_input_tokens == 500
    assert acc.total_output_tokens == 200
    assert acc.total_cost > 0
