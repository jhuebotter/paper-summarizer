"""Pydantic models, dataclass Config, and exceptions for the summarizer pipeline.

All domain-specific knowledge lives in ``skill_data/references/`` and in the
prompt builders (``prompts.py``). This module only defines the *schema* of the
data that flows through the pipeline — validation of LLM output, batch
reporting, and runtime configuration.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, get_args, get_origin

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

PaperType = Literal["primary", "synthesis"]
"""The two supported research-paper classifications."""

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class PaperMetadata(BaseModel):
    """Bibliographic metadata extracted from the paper.

    The ``citation_key`` follows the convention ``firstauthorYEARfirstword``
    (all lowercase), e.g. ``huebotter2025spiking``.  ``year`` is ``None`` only
    when no year could be found anywhere (the key then uses ``nd``, BibTeX's
    "no date" convention).
    """

    citation_key: str
    title: str
    authors: list[str]
    year: int | None
    venue: str
    is_research_paper: bool
    paper_type: PaperType | None
    synthesis_subtype: str | None = None
    rejection_reason: str | None = None
    tags: list[str]

    @model_validator(mode="after")
    def _validate_research_gate(self) -> "PaperMetadata":
        if self.is_research_paper:
            if self.paper_type is None:
                raise ValueError("paper_type is required when is_research_paper=true")
            if self.rejection_reason is not None:
                raise ValueError("rejection_reason must be null when is_research_paper=true")
            return self

        # Non-research document path
        if self.paper_type is not None:
            raise ValueError("paper_type must be null when is_research_paper=false")
        if not self.rejection_reason:
            raise ValueError("rejection_reason is required when is_research_paper=false")
        return self


# ---------------------------------------------------------------------------
# Nested helper models
# ---------------------------------------------------------------------------


class CitableSnippet(BaseModel):
    """A single citable snippet with an optional verbatim quotable sentence."""

    cite_for: str
    source: str
    quote_tag: str | None = None
    quote: str | None = None


class OpenProblemsPrimary(BaseModel):
    """Open problems and future directions for primary research papers."""

    future_work_proposed: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class OpenProblemsSynthesis(BaseModel):
    """Open problems and future directions for synthesis papers."""

    gaps_identified: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    suggested_research_focus: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Part 1 variants (discriminated union on paper_type)
# ---------------------------------------------------------------------------


def _findings_as_strings(value: object) -> object:
    """Accept a bare string or ``{finding, evidence, source}`` objects for
    ``notable_findings``; some models return these instead of strings, and a
    schema repair for it costs a full extra call."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return value
    out = []
    for item in value:
        text = (item.get("finding") or item.get("text")) if isinstance(item, dict) else None
        if not isinstance(text, str):
            out.append(item)
            continue
        for key, template in (("evidence", " ({})"), ("source", " (Source: {})")):
            if isinstance(item.get(key), str) and item[key]:
                tag = template.format(item[key])
                text += "" if tag.strip() in text else tag
        out.append(text)
    return out


class SummaryPart1Primary(BaseModel):
    """Part 1 summary fields for a primary research paper (≤600 words total)."""

    paper_type: Literal["primary"]
    tldr: str
    problem_motivation: str
    core_contribution: str
    methods: str
    results: str
    key_takeaways: str
    limitations: str
    open_problems_future_directions: OpenProblemsPrimary = Field(
        default_factory=OpenProblemsPrimary
    )
    critical_assessment: str
    notable_findings: list[str] = Field(default_factory=list)
    citable_snippets: list[CitableSnippet] = Field(default_factory=list)
    relevance: str

    _findings = field_validator("notable_findings", mode="before")(_findings_as_strings)


class SummaryPart1Synthesis(BaseModel):
    """Part 1 summary fields for a synthesis paper (≤1000 words total).

    Covers reviews, surveys, perspectives, opinions, commentaries.
    Use ``not applicable`` for sections that genuinely do not apply.
    """

    paper_type: Literal["synthesis"]
    tldr: str
    target_papers_field: str
    scope_coverage: str
    taxonomy_organization: str
    core_argument: str
    synthesis_contribution: str
    key_claims_narrative: str
    key_takeaways: str
    limitations: str
    open_problems_future_directions: OpenProblemsSynthesis = Field(
        default_factory=OpenProblemsSynthesis
    )
    critical_assessment: str
    notable_findings: list[str] = Field(default_factory=list)
    citable_snippets: list[CitableSnippet] = Field(default_factory=list)
    relevance: str

    _findings = field_validator("notable_findings", mode="before")(_findings_as_strings)


class SummaryPart1NonResearch(BaseModel):
    """Fallback summary for non-research documents.

    The ``note`` should explain why the document is rejected as a research
    paper (e.g. scanned form, slides, brochure, etc.).
    """

    paper_type: Literal["non_research"]
    note: str


SummaryPart1 = Annotated[
    SummaryPart1Primary | SummaryPart1Synthesis | SummaryPart1NonResearch,
    Field(discriminator="paper_type"),
]
"""Discriminated union: pydantic selects the correct variant by ``paper_type``."""

# ---------------------------------------------------------------------------
# Part 2 (SNN extraction fields)
# ---------------------------------------------------------------------------

InferenceHardware = Literal[
    "CPU/GPU", "Neuromorphic emulator/SDK", "Physical neuromorphic chip", "not reported"
]
Architecture = Literal["fully spiking", "hybrid", "not reported"]
CreditAssignment = Literal[
    "Global", "Semi-local", "Local", "Analytical", "Hybrid", "Not applicable", "not reported"
]
LearningRegime = Literal["Offline", "Online", "Mixed", "not applicable", "not reported"]
ParadigmFamily = Literal[
    "Gradient-based (surrogate gradient BPTT)",
    "Gradient-based (online approximation: e-prop/FPTT/OSTL)",
    "ANN-to-SNN conversion",
    "Predictive coding / prediction error learning",
    "Reinforcement learning (model-free)",
    "Reinforcement learning (model-based)",
    "Local plasticity (STDP / R-STDP / three-factor)",
    "Homeostatic / intrinsic plasticity (auxiliary)",
    "Evolutionary / black-box optimization",
    "Analytical / closed-form (NEF / reservoir / control law)",
    "Hybrid / multi-phase",
]


def _canonical_key(value: str) -> str:
    return re.sub(r"\s*/\s*", "/", re.sub(r"\s+", " ", value.strip())).casefold()


def labels(annotation: object) -> tuple[str, ...]:
    """Allowed labels of a ``Literal`` or ``list[Literal]`` annotation."""
    if get_origin(annotation) is list:
        annotation = get_args(annotation)[0]
    return get_args(annotation)


_NOT_APPLICABLE = {"not applicable", "n/a", "na", "none"}


class Classification(BaseModel):
    """Typed labels from the controlled vocabularies in the prompt references.

    These are the columns for comparison tables and gold-label scoring.  Values
    are matched case- and whitespace-insensitively to the allowed labels;
    "not applicable" becomes "not reported" where a field has no such label,
    and ``paradigm_families`` tolerates a bare string, null, placeholders and
    duplicates.
    """

    inference_hardware: InferenceHardware
    architecture: Architecture
    credit_assignment: CreditAssignment
    learning_regime: LearningRegime
    paradigm_families: list[ParadigmFamily]

    @model_validator(mode="before")
    @classmethod
    def _canonicalize(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        for name, field in cls.model_fields.items():
            allowed = {_canonical_key(label): label for label in labels(field.annotation)}

            def canonical(value: object, allowed: dict = allowed) -> object:
                if not isinstance(value, str):
                    return value
                key = _canonical_key(value)
                if key in _NOT_APPLICABLE and key not in allowed:
                    key = "not reported"
                return allowed.get(key, value)

            value = data.get(name)
            if name == "paradigm_families":
                items = [] if value is None else [value] if isinstance(value, str) else value
                if isinstance(items, list):
                    kept = [canonical(v) for v in items]
                    kept = [
                        v
                        for v in kept
                        if not (
                            isinstance(v, str)
                            and _canonical_key(v) in _NOT_APPLICABLE | {"not reported"}
                        )
                    ]
                    hashable = all(isinstance(v, str) for v in kept)
                    data[name] = list(dict.fromkeys(kept)) if hashable else kept
            else:
                data[name] = canonical(value)
        return data


class SummaryPart2(BaseModel):
    """Structured extraction of SNN-specific technical fields.

    Field values should use ``"not reported"`` when a concept applies but the
    paper omits it, and ``"not applicable"`` only when the concept genuinely
    does not apply.

    Part 2 is produced only for ``paper_type="primary"`` research papers.
    Synthesis and non-research documents use ``part2=null``.
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
    classification: Classification


# ---------------------------------------------------------------------------
# Combined LLM response (single call returns all three sections)
# ---------------------------------------------------------------------------


class LLMResponse(BaseModel):
    """The validated response from the single combined LLM call.

    The LLM returns a JSON object with three top-level keys:
    ``metadata``, ``part1``, and ``part2``.

    - For primary papers, ``part2`` is required.
    - For synthesis and non-research documents, ``part2`` is ``null``.
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

        if self.metadata.paper_type == "synthesis" and self.part2 is not None:
            raise ValueError("part2 must be null for synthesis papers")

        return self


# ---------------------------------------------------------------------------
# Composed summary
# ---------------------------------------------------------------------------


class Provenance(BaseModel):
    """How a summary was produced; stored in its JSON sidecar."""

    schema_version: int = 1
    created_at: str
    git_commit: str | None
    pdf_sha256: str
    source_path: str
    extractor: str
    chars_full: int  # after reference stripping
    chars_sent: int
    model: str
    base_url: str
    references_sha256: str
    calls: int
    json_repairs: int
    schema_repairs: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    zotero_item: str | None = None  # e.g. "groups/5824653/items/YXWFBPTP"
    zotero_fields: list[str] = Field(default_factory=list)  # fields where Zotero replaced the LLM's


class PaperSummary(BaseModel):
    """The complete validated output for one paper, composed of all three parts.

    Constructed by ``pipeline.process_pdf()`` after the LLM call succeeds and
    all pydantic validation passes. Passed to ``renderer.render_summary()`` to
    produce the final markdown string.

    ``part2`` is ``None`` for synthesis and non-research documents.
    Part 2 is generated only for primary research papers.
    """

    metadata: PaperMetadata
    part1: SummaryPart1
    part2: SummaryPart2 | None
    provenance: Provenance | None = None


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
    total_cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    stopped_reason: str | None = None  # set when the run stopped before all papers


# ---------------------------------------------------------------------------
# Config (dataclass — not pydantic; holds runtime settings)
# ---------------------------------------------------------------------------

#: Estimate: 1 token ≈ 4 characters for English text.  200 000 chars of paper
#: text is ~50 000 tokens.  The fixed prompt (references + output rules) adds
#: ~11 000 tokens and the response ~3 000-5 000, so a call needs a context
#: window of roughly 70 000 tokens.  Lower ``--max-chars`` for small local
#: models.
_DEFAULT_MAX_CHARS = 200_000

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

#: A free OpenRouter model with a 262k-token context.  Free models are
#: rate-limited and may be retired; the CLI's preflight check reports a retired
#: id.  Paid fallback within a few-cents budget: ``meta/muse-spark-1.3-contributor``.
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

#: Resolved from the source checkout (not the CWD) so the CLI works from any
#: directory.  Requires an editable/source install.
DEFAULT_SKILL_DATA_DIR = Path(__file__).resolve().parent.parent / "skill_data" / "references"


@dataclass
class Config:
    """Runtime configuration for the summarizer pipeline.

    All fields correspond to CLI flags.  The defaults need a context window of
    roughly 70k tokens (see ``_DEFAULT_MAX_CHARS``).

    Attributes:
        base_url:       OpenAI-compatible API base URL.  Use
                        ``http://localhost:1234/v1`` for LM Studio or
                        ``https://openrouter.ai/api/v1`` for OpenRouter.
        model:          Model identifier passed to the API.
        max_chars:      Maximum characters of paper text sent to the LLM
                        (~200k chars ≈ 50k tokens).  Lower this for small
                        local models.
        force_summary:  If True, re-summarize PDFs already in the processed
                        index (keeps the extraction cache unless ``reparse``).
        reparse:        If True, also re-run extraction (ignores cached text).
                        Implies summary regeneration for selected files.
        extractor:      PDF text extraction strategy: ``auto`` (docling with
                        pypdf fallback), ``docling`` (docling-only), or
                        ``pypdf`` (pypdf-only).
        dry_run:        If True, list PDFs that would be processed without
                        making any LLM calls or writing any files.
        output_dir:     Root directory for centralized summary output.
                        Subdirs ``primary/``, ``synthesis/``, and
                        ``non_research/`` are created automatically.
        skill_data_dir: Directory of reference .md files embedded in the prompt.
        verbose:        If True, log at DEBUG level (prompt sizes, raw response
                        excerpts on failures, full validation errors).
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
        strip_references:  Drop the References/Bibliography section before the
                           text is truncated and sent.
        structured_output: Ask the backend to constrain replies to the
                           ``LLMResponse`` JSON schema.
        max_cost:          Stop starting new papers once this many USD are spent.
        zotero:            Take bibliographic metadata from the local Zotero library
                           (batch runs only; see ``zotero.py``).
    """

    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    max_chars: int = _DEFAULT_MAX_CHARS
    force_summary: bool = False
    reparse: bool = False
    extractor: Literal["auto", "docling", "pypdf"] = "auto"
    dry_run: bool = False
    output_dir: Path = Path("output_summaries")
    skill_data_dir: Path = DEFAULT_SKILL_DATA_DIR
    verbose: bool = False
    api_key: str | None = None
    timeout_s: int = 120
    max_output_tokens: int | None = None
    workers: int = 3
    strip_references: bool = True
    structured_output: bool = False
    max_cost: float | None = None
    zotero: bool = True


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ParseError(Exception):
    """Raised when PDF text extraction fails (corrupt, password-protected, etc.)."""


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
