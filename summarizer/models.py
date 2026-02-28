"""Pydantic models, dataclass Config, and exceptions for the summarizer pipeline.

All domain-specific knowledge lives in ``skill_data/references/`` and in the
prompt builders (``prompts.py``). This module only defines the *schema* of the
data that flows through the pipeline — validation of LLM output, batch
reporting, and runtime configuration.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

PaperType = Literal["primary", "survey", "commentary"]
"""The three supported research-paper classifications."""

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class PaperMetadata(BaseModel):
    """Bibliographic metadata extracted from the paper in LLM Call 1.

    The ``citation_key`` follows the convention ``firstauthorYEARfirstword``
    (all lowercase), e.g. ``huebotter2025spiking``.
    """

    citation_key: str
    title: str
    authors: list[str]
    year: int
    venue: str
    is_research_paper: bool
    paper_type: PaperType | None
    rejection_reason: str | None = None
    tags: list[str]

    @model_validator(mode="after")
    def _validate_research_gate(self) -> "PaperMetadata":
        if self.is_research_paper:
            if self.paper_type is None:
                raise ValueError("paper_type is required when is_research_paper=true")
            if self.rejection_reason is not None:
                raise ValueError(
                    "rejection_reason must be null when is_research_paper=true"
                )
            return self

        # Non-research document path
        if self.paper_type is not None:
            raise ValueError("paper_type must be null when is_research_paper=false")
        if not self.rejection_reason:
            raise ValueError(
                "rejection_reason is required when is_research_paper=false"
            )
        return self


# ---------------------------------------------------------------------------
# Part 1 variants (discriminated union on paper_type)
# ---------------------------------------------------------------------------


class SummaryPart1Primary(BaseModel):
    """Part 1 summary fields for a primary research paper (≤250 words total).

    ``cite_for`` and ``quotable_sentences`` default to empty lists so that
    local LLMs which occasionally omit these fields do not hard-fail validation.
    """

    paper_type: Literal["primary"]
    tldr: str
    problem_motivation: str
    core_contribution: str
    methods: str
    results: str
    key_takeaways: str
    limitations: str
    relevance: str
    cite_for: list[str] = Field(default_factory=list)
    critical_assessment: str
    quotable_sentences: list[str] = Field(default_factory=list)
    notable_findings: list[str] = Field(default_factory=list)


class SummaryPart1Survey(BaseModel):
    """Part 1 summary fields for a survey or review paper (≤200 words total).

    ``cite_for`` and ``quotable_sentences`` default to empty lists so that
    local LLMs which occasionally omit these fields do not hard-fail validation.
    """

    paper_type: Literal["survey"]
    tldr: str
    scope_coverage: str
    taxonomy_organization: str
    key_claims_narrative: str
    gaps_identified: str
    relevance: str
    cite_for: list[str] = Field(default_factory=list)
    critical_assessment: str
    quotable_sentences: list[str] = Field(default_factory=list)


class SummaryPart1Commentary(BaseModel):
    """Part 1 summary fields for a commentary or opinion paper (≤120 words total).

    ``cite_for`` and ``quotable_sentences`` default to empty lists so that
    local LLMs which occasionally omit these fields do not hard-fail validation.
    """

    paper_type: Literal["commentary"]
    tldr: str
    core_argument: str
    target_papers: str
    limitations: str
    relevance: str
    cite_for: list[str] = Field(default_factory=list)
    critical_assessment: str
    quotable_sentences: list[str] = Field(default_factory=list)


class SummaryPart1NonResearch(BaseModel):
    """Fallback summary for non-research documents.

    The ``note`` should explain why the document is rejected as a research
    paper (e.g. scanned form, slides, brochure, etc.).
    """

    paper_type: Literal["non_research"]
    note: str


SummaryPart1 = Annotated[
    Union[
        SummaryPart1Primary,
        SummaryPart1Survey,
        SummaryPart1Commentary,
        SummaryPart1NonResearch,
    ],
    Field(discriminator="paper_type"),
]
"""Discriminated union: pydantic selects the correct variant by ``paper_type``."""

# ---------------------------------------------------------------------------
# Part 2 (SNN extraction fields)
# ---------------------------------------------------------------------------


class SummaryPart2(BaseModel):
    """Structured extraction of SNN-specific technical fields (LLM Call 2).

    Field values should use ``"not reported"`` when a concept applies but the
    paper omits it, and ``"not applicable"`` only when the concept genuinely
    does not apply (e.g. ``"not applicable (survey)"``).

    Part 2 is produced only for ``paper_type="primary"`` research papers.
    Survey/commentary and non-research documents use ``part2=null``.
    """

    neuron_model: str
    network_architecture: str
    model_scale: str
    simulator_framework: str
    hardware_training: str
    controller_hardware_inference: str
    control_task: str
    task_type: str
    task_complexity_scale: str
    simulation_environment: str
    spike_encoding: str
    action_decoding: str
    learning_mechanism: str
    credit_assignment_scope: str
    online_vs_offline: str
    data_collection: str
    key_training_details: str
    comparison_to_baselines: str


# ---------------------------------------------------------------------------
# Combined LLM response (v1.1 — single call returns all three sections)
# ---------------------------------------------------------------------------


class LLMResponse(BaseModel):
    """The raw validated response from a single combined LLM call (v1.1).

    The LLM returns a JSON object with three top-level keys:
    ``metadata``, ``part1``, and ``part2``.

    - For primary papers, ``part2`` is required.
    - For survey/commentary and non-research documents, ``part2`` is ``null``.
    """

    metadata: PaperMetadata
    part1: SummaryPart1
    part2: SummaryPart2 | None

    @model_validator(mode="after")
    def _validate_consistency(self) -> "LLMResponse":
        if not self.metadata.is_research_paper:
            if self.part1.paper_type != "non_research":
                raise ValueError(
                    "part1.paper_type must be 'non_research' when is_research_paper=false"
                )
            if self.part2 is not None:
                raise ValueError("part2 must be null when is_research_paper=false")
            return self

        # Research paper path
        assert self.metadata.paper_type is not None
        if self.part1.paper_type != self.metadata.paper_type:
            raise ValueError("metadata.paper_type must match part1.paper_type")

        if self.metadata.paper_type == "primary" and self.part2 is None:
            raise ValueError("part2 is required for primary papers")

        if (
            self.metadata.paper_type in {"survey", "commentary"}
            and self.part2 is not None
        ):
            raise ValueError("part2 must be null for survey/commentary papers")

        return self


# ---------------------------------------------------------------------------
# Composed summary
# ---------------------------------------------------------------------------


class PaperSummary(BaseModel):
    """The complete validated output for one paper, composed of all three parts.

    Constructed by ``pipeline.process_pdf()`` after the LLM call succeeds and
    all pydantic validation passes. Passed to ``renderer.render_summary()`` to
    produce the final markdown string.

    ``part2`` is ``None`` for survey/commentary and non-research documents.
    Part 2 is generated only for primary research papers.
    """

    metadata: PaperMetadata
    part1: SummaryPart1
    part2: SummaryPart2 | None


# ---------------------------------------------------------------------------
# Batch reporting
# ---------------------------------------------------------------------------


class FailedPaper(BaseModel):
    """Records a single paper that could not be processed during a batch run."""

    pdf_path: str
    error: str


class BatchReport(BaseModel):
    """Aggregate result of a batch run over a directory of PDFs."""

    processed: int
    skipped: int
    failed: int
    failed_papers: list[FailedPaper]


# ---------------------------------------------------------------------------
# Config (dataclass — not pydantic; holds runtime settings)
# ---------------------------------------------------------------------------

#: Estimate: 1 token ≈ 4 characters for English text.  At 200 000 chars the
#: paper text budget is ~50 000 tokens, which leaves headroom for the
#: references context (~2 500 tokens) and LLM output (~2 500 tokens) inside a
#: 55 000-token context window.  Increase with ``--max-chars`` for models with
#: larger context windows, or decrease for small local models.
_DEFAULT_MAX_CHARS = 200_000


@dataclass
class Config:
    """Runtime configuration for the summarizer pipeline.

    All fields correspond to CLI flags.  Defaults are chosen to fit comfortably
    inside a 50k-token context window with the standard skill reference files.

    Attributes:
        base_url:       OpenAI-compatible API base URL.  Use
                        ``http://localhost:1234/v1`` for LM Studio or
                        ``https://openrouter.ai/api/v1`` for OpenRouter.
        model:          Model identifier passed to the API.
        max_chars:      Maximum characters of paper markdown sent to the LLM.
                        Default (~200k chars ≈ 50k tokens) suits models with a
                        55k+ token context window.  Lower this for 4k/8k models.
        force_summary:  If True, re-run summary generation even for PDFs already
                        in processed.txt (preserves extraction cache unless
                        ``reparse`` is set).
        reparse:        If True, also re-run docling (ignores cached .md files).
                        Implies summary regeneration for selected files.
        extractor:      PDF text extraction strategy: ``auto`` (docling with
                        pypdf fallback), ``docling`` (docling-only), or
                        ``pypdf`` (pypdf-only).
        dry_run:        If True, list PDFs that would be processed without
                        making any LLM calls or writing any files.
        output_dir:     Root directory for centralized summary output.
                        Subdirs ``primary/``, ``survey/``, ``commentary/``,
                        and ``non_research/`` are created automatically.
        skill_data_dir: Path to the directory containing reference .md files
                        (output-template, extraction fields, learning paradigms).
        verbose:        If True, print context size diagnostics (chars, estimated
                        tokens) to stderr before each LLM call.
        api_key:        API key for the LLM backend.  ``None`` means the key is
                        read from the ``LLM_API_KEY`` environment variable; if
                        that is also unset the dummy ``"lm-studio"`` string is
                        used (LM Studio ignores the value).
        timeout_s:         Seconds before an LLM call is killed.  Prevents the
                           pipeline from hanging indefinitely on slow or unresponsive
                           backends.
        max_output_tokens: Maximum tokens the LLM may generate per call.  ``None``
                           (default) imposes no limit — the model stops on its own.
                           Set explicitly (e.g. ``--max-output-tokens 8192``) when
                           the backend enforces a cap or to bound cost.
        workers:           Number of concurrent workers used in batch mode.
                           Each worker processes full PDFs end-to-end.
    """

    base_url: str = "http://localhost:1234/v1"
    model: str = "openai/gpt-oss-120b:free"
    max_chars: int = _DEFAULT_MAX_CHARS
    force_summary: bool = False
    reparse: bool = False
    extractor: Literal["auto", "docling", "pypdf"] = "auto"
    dry_run: bool = False
    output_dir: Path = Path("output_summaries")
    skill_data_dir: Path = Path("skill_data/references")
    verbose: bool = False
    api_key: str | None = None
    timeout_s: int = 120
    max_output_tokens: int | None = None
    workers: int = 3


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ParseError(Exception):
    """Raised when docling fails to parse a PDF (corrupt, password-protected, etc.)."""


class LLMError(Exception):
    """Raised when an LLM call fails or returns output that cannot be parsed as JSON."""


class PipelineError(Exception):
    """Wraps any sub-error that occurs during per-paper processing.

    Attributes:
        pdf_path: Path to the PDF that failed.
        cause:    The original exception that triggered the failure.
    """

    def __init__(self, pdf_path: Path, cause: Exception) -> None:
        self.pdf_path = pdf_path
        self.cause = cause
        super().__init__(f"Pipeline failed for {pdf_path}: {cause}")
