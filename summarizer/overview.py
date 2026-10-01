"""Literature-overview records: typed facts per paper, and labels derived from them.

The summary's ``Classification`` asks the model for judgement labels such as
``Hybrid`` or ``Mixed``. Those turned out to be the least reliable output,
because each one bundles several facts with a convention. An overview record
instead asks for observable facts, each with a quote: which parts spike, how
each component got its parameters, where the controller runs, which metrics
are reported. The categories used in the review are then computed from those
facts by explicit rules (``derive``), so changing a convention changes a rule
rather than every label.

The option strings are the ones in ``skill_data/overview/codebook.md``, which is
embedded in the extraction prompt;
``test_codebook_lists_every_option`` keeps the two in sync.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vocabularies (must match the codebook)
# ---------------------------------------------------------------------------

PaperKind = Literal["primary research", "review or survey", "other"]
SpikingRole = Literal["sensing", "state estimation", "controller", "other"]
ControlLevel = Literal[
    "closed-loop", "open-loop actuation", "perception for control", "no control task"
]
PlantSetting = Literal["simulated", "real", "simulated and real", "none"]
PlantDynamics = Literal["linear", "nonlinear", "both", "not applicable"]
PlantModelUse = Literal[
    "known model used in design", "model learned", "model-free", "not applicable"
]
Objective = Literal[
    "regulation",
    "tracking",
    "disturbance rejection",
    "adaptive control",
    "locomotion / pattern generation",
    "navigation / decision-making",
    "state estimation / prediction",
    "other",
]
Role = Literal[
    "controller",
    "critic / value",
    "state estimation / world model",
    "perception / encoder",
    "readout / decoder",
    "other",
]
ObtainedBy = Literal[
    "hand-designed", "solved", "learned", "searched", "converted", "random / fixed"
]
Signal = Literal[
    "supervised / imitation",
    "reinforcement",
    "self-supervised / system identification",
    "unsupervised",
    "not applicable",
]
Mechanism = Literal[
    "backprop / BPTT",
    "forward-mode / e-prop",
    "eligibility + modulator (three-factor)",
    "local error (LMS-like)",
    "Hebbian / STDP (two-factor)",
    "perturbation",
    "evolutionary / black-box",
    "ANN-to-SNN conversion",
    "other",
    "not applicable",
]
Regime = Literal["offline", "interleaved", "online", "not applicable"]
AnalyticMethod = Literal[
    "control-theoretic",
    "NEF",
    "spike coding network",
    "hand-wired circuit",
    "reservoir",
    "other",
]
Interface = Literal["continuous", "event-native", "mixed", "not applicable", "not reported"]
Platform = Literal[
    "neuromorphic chip (digital)",
    "neuromorphic chip (mixed-signal / analog)",
    "FPGA / digital accelerator",
    "neuromorphic emulator / SDK",
    "embedded CPU / microcontroller",
    "CPU/GPU",
    "software simulation (machine not stated)",
    "not reported",
]
Coverage = Literal["whole network", "partial", "not applicable"]
Reported = Literal["reported", "not reported"]
Latency = Literal["measured", "estimated", "claimed", "not reported"]
Energy = Literal["measured", "estimated", "claimed", "not reported"]
Stability = Literal["formal", "empirical", "not addressed"]
Robustness = Literal["tested", "not tested"]
SimToReal = Literal["transferred", "real only", "simulation only", "not applicable"]


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------


class Component(BaseModel):
    name: str
    role: Role
    spiking: bool
    deployed: bool
    obtained_by: ObtainedBy
    signal: Signal
    mechanism: Mechanism
    regime: Regime
    adapts_during_evaluation: bool
    evidence: str = ""


class Metrics(BaseModel):
    tracking_error: Reported
    latency: Latency
    energy: Energy
    spike_activity: Reported
    stability: Stability
    robustness: Robustness
    sim_to_real: SimToReal
    metrics_evidence: str = ""


class OverviewRecord(BaseModel):
    """The facts the codebook asks for, as returned by the extraction model."""

    paper_kind: PaperKind = "primary research"
    spiking_roles: list[SpikingRole]
    control_level: ControlLevel
    control_evidence: str = ""
    plant_setting: PlantSetting
    plant_dynamics: PlantDynamics
    plant_model_use: PlantModelUse
    objectives: list[Objective]
    task: str
    components: list[Component] = Field(default_factory=list)
    analytic_methods: list[AnalyticMethod]
    interface: Interface
    interface_evidence: str = ""
    platform: Platform
    platform_name: str = ""
    platform_coverage: Coverage
    hardware_in_loop: bool
    platform_evidence: str = ""
    metrics: Metrics
    notes: str = ""


_ANALYTIC = {"hand-designed", "solved"}
_LEARNED = {"learned", "searched", "converted"}

# ---------------------------------------------------------------------------
# Consistency rules (implications stated in the codebook)
# ---------------------------------------------------------------------------

_SOFTWARE_PLATFORMS = {
    "CPU/GPU",
    "embedded CPU / microcontroller",
    "software simulation (machine not stated)",
    "not reported",
}
_NO_PLANT_LEVELS = {"perception for control", "no control task"}


def normalize(record: OverviewRecord) -> tuple[OverviewRecord, list[str]]:
    """Apply the codebook's cross-field implications; return the record and what changed.

    Models sometimes answer two fields inconsistently (a CPU run with
    ``platform_coverage = whole network``, a perception-only paper with a
    plant model). These rules only enforce what the codebook already implies.
    """
    r = record.model_copy(deep=True)
    changes: list[str] = []

    def put(obj, field: str, value) -> None:
        if getattr(obj, field) != value:
            changes.append(f"{field}: {getattr(obj, field)!r} -> {value!r}")
            setattr(obj, field, value)

    if r.platform in _SOFTWARE_PLATFORMS:
        put(r, "platform_coverage", "not applicable")
        if r.platform != "embedded CPU / microcontroller":
            put(r, "hardware_in_loop", False)
    elif r.platform_coverage == "not applicable":
        put(r, "platform_coverage", "whole network")
    if r.control_level in _NO_PLANT_LEVELS:
        put(r, "plant_dynamics", "not applicable")
        put(r, "plant_model_use", "not applicable")
        put(r, "interface", "not applicable")
        put(r.metrics, "sim_to_real", "not applicable")
        put(r, "hardware_in_loop", False)
    elif r.plant_setting == "none":
        put(r.metrics, "sim_to_real", "not applicable")
    elif r.plant_setting == "simulated":
        put(r.metrics, "sim_to_real", "simulation only")
    elif r.plant_setting == "real" and r.metrics.sim_to_real in (
        "simulation only",
        "not applicable",
    ):
        put(r.metrics, "sim_to_real", "real only")
    for c in r.components:
        if c.obtained_by not in _LEARNED:
            for field in ("signal", "mechanism", "regime"):
                put(c, field, "not applicable")
            put(c, "adapts_during_evaluation", False)
    return r, changes


# ---------------------------------------------------------------------------
# Derived labels (the review's categories, computed from the facts)
# ---------------------------------------------------------------------------

SIGNAL_SHORT = {
    "supervised / imitation": "supv",
    "reinforcement": "rl",
    "self-supervised / system identification": "self",
    "unsupervised": "unsup",
}
MECHANISM_SHORT = {
    "backprop / BPTT": "BPTT",
    "forward-mode / e-prop": "eprop",
    "eligibility + modulator (three-factor)": "3F",
    "local error (LMS-like)": "LMS",
    "Hebbian / STDP (two-factor)": "Hebb",
    "perturbation": "perturb",
    "evolutionary / black-box": "ES",
    "ANN-to-SNN conversion": "conv",
    "other": "other",
}


def learning_pair(component: Component) -> str:
    """``signal+mechanism`` tag of a component, e.g. ``rl+BPTT`` (the draft's pair notation)."""
    signal = SIGNAL_SHORT.get(component.signal, "?")
    return f"{signal}+{MECHANISM_SHORT.get(component.mechanism, '?')}"


def derive(record: OverviewRecord) -> dict:
    """Compute the review's categories from a record's facts.

    - ``design``: analytic / learned / analytic + learned (the 2×2's first
      axis), from the ``obtained_by`` of the components that make up the
      controller: deployed ones, or ones in the ``controller`` role. A teacher
      or training-only model doesn't count. ``analytic_methods`` (other than a
      reservoir) decide only when no such component is listed.
    - ``quadrant``: design × interface.
    - ``learning_pairs``: sorted ``signal+mechanism`` tags of the learned parts.
    - ``regimes``: the set of learning regimes; ``adapts_online`` if any
      learned component keeps learning during evaluation.
    - ``fully_spiking_deployed``: something spikes at deployment and every
      deployed component spikes, except readouts.
    - ``nonspiking_training_only``: a non-spiking part used only in training
      (e.g. an ANN critic).
    """
    comps = record.components
    system = [c for c in comps if c.deployed or c.role == "controller"]
    analytic = any(c.obtained_by in _ANALYTIC for c in system) or (
        not system and any(m != "reservoir" for m in record.analytic_methods)
    )
    learned = any(c.obtained_by in _LEARNED for c in system)
    if analytic and learned:
        design = "analytic + learned"
    elif learned:
        design = "learned"
    elif analytic:
        design = "analytic"
    else:
        design = "not determinable"

    learned_parts = [c for c in comps if c.obtained_by in _LEARNED]
    pairs = sorted({learning_pair(c) for c in learned_parts if c.mechanism != "not applicable"})
    regimes = sorted({c.regime for c in learned_parts if c.regime != "not applicable"})
    deployed = [c for c in comps if c.deployed]
    fully_spiking = any(c.spiking for c in deployed) and all(
        c.spiking or c.role == "readout / decoder" for c in deployed
    )
    nonspiking_training_only = any(not c.spiking and not c.deployed for c in comps)
    interface = record.interface
    quadrant = (
        f"{design} × {interface}"
        if design != "not determinable" and interface in ("continuous", "event-native", "mixed")
        else "n/a"
    )
    return {
        "design": design,
        "quadrant": quadrant,
        "learning_pairs": pairs,
        "regimes": regimes,
        "adapts_online": any(c.adapts_during_evaluation for c in learned_parts),
        "fully_spiking_deployed": fully_spiking,
        "nonspiking_training_only": nonspiking_training_only,
    }


# ---------------------------------------------------------------------------
# Evidence quotes
# ---------------------------------------------------------------------------


#: A quote shorter than this (letters and digits) would match almost anywhere,
#: so it can't count as evidence.
_MIN_QUOTE_CHARS = 12


def _squash(text: str) -> str:
    """Lowercase letters and digits only: quotes survive line breaks, hyphenation, markup."""
    return re.sub(r"[^0-9a-z]+", "", text.casefold())


def evidence_quotes(record: OverviewRecord) -> list[tuple[str, str]]:
    """``(field, quote)`` for every non-empty evidence string in ``record``."""
    quotes = [
        ("control_evidence", record.control_evidence),
        ("interface_evidence", record.interface_evidence),
        ("platform_evidence", record.platform_evidence),
        ("metrics.metrics_evidence", record.metrics.metrics_evidence),
    ]
    quotes += [(f"components[{i}].evidence", c.evidence) for i, c in enumerate(record.components)]
    return [(field, quote) for field, quote in quotes if quote.strip()]


def missing_quotes(record: OverviewRecord, paper_text: str) -> list[str]:
    """Fields whose evidence quote does not occur in the paper text.

    A quote the model invented or paraphrased is the cheapest available signal
    that the value it supports needs checking by hand.
    """
    text = _squash(paper_text)
    missing = []
    for field, quote in evidence_quotes(record):
        squashed = _squash(quote)
        if len(squashed) < _MIN_QUOTE_CHARS or squashed not in text:
            missing.append(field)
    return missing


# ---------------------------------------------------------------------------
# Scoring against gold records
# ---------------------------------------------------------------------------

SCORED_FIELDS = (
    "control_level",
    "plant_setting",
    "plant_dynamics",
    "plant_model_use",
    "interface",
    "platform",
    "platform_coverage",
    "hardware_in_loop",
)
SCORED_METRICS = (
    "tracking_error",
    "latency",
    "energy",
    "spike_activity",
    "stability",
    "robustness",
    "sim_to_real",
)
SCORED_SETS = ("spiking_roles", "objectives", "analytic_methods")


def comparable_facts(record: OverviewRecord) -> dict:
    """The record's facts and derived labels as hashable values, keyed by scored field."""
    facts: dict = {field: getattr(record, field) for field in SCORED_FIELDS}
    facts.update({f"metrics.{m}": getattr(record.metrics, m) for m in SCORED_METRICS})
    facts.update({field: tuple(sorted(getattr(record, field))) for field in SCORED_SETS})
    for key, value in derive(record).items():
        facts[f"derived.{key}"] = tuple(value) if isinstance(value, list) else value
    return facts


def score_record(predicted: OverviewRecord, gold: OverviewRecord) -> dict[str, bool]:
    """Exact-match correctness of every scored field (sets compare as sets)."""
    pred, ref = comparable_facts(predicted), comparable_facts(gold)
    return {field: pred[field] == ref[field] for field in ref}


# ---------------------------------------------------------------------------
# Stored results
# ---------------------------------------------------------------------------

RECORDS_DIRNAME = "records"


class OverviewResult(BaseModel):
    """One paper's stored overview: the record plus how it was obtained."""

    sha256: str
    file: str
    citation_key: str = ""
    title: str = ""
    authors: list[str] = []
    year: int | None = None
    record: OverviewRecord
    derived: dict
    normalized: list[str] = []
    missing_quotes: list[str] = []
    draft: OverviewRecord | None = None
    model: str
    codebook_sha256: str
    extractor: str
    critic: bool = False
    cost_usd: float = 0.0
    calls: int = 0
    created: str = ""


def load_results(output_dir: Path) -> list[OverviewResult]:
    """Every stored result under ``output_dir``, one per PDF content.

    Labels are recomputed with the current rules (``normalize`` then
    ``derive``), so a rule change shows up in the tables without new LLM
    calls. If two files hold the same paper (e.g. a PDF stored with and
    without its Zotero key prefix), the newest one wins. Unreadable files are
    logged and skipped.
    """
    by_sha: dict[str, OverviewResult] = {}
    for path in sorted((output_dir / RECORDS_DIRNAME).glob("*.json")):
        try:
            result = OverviewResult.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.error("Cannot read %s: %s", path, exc)
            continue
        result.record, changes = normalize(result.record)
        result.normalized = result.normalized + [c for c in changes if c not in result.normalized]
        result.derived = derive(result.record)
        known = by_sha.get(result.sha256)
        if known is None or result.created > known.created:
            by_sha[result.sha256] = result
    return list(by_sha.values())
