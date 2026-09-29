"""Shared pytest fixtures for the snn_summarizer test suite."""

import copy
import logging
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Logger isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_summarizer_logger():
    """Clear the summarizer logger between tests.

    Tests that call ``main()`` trigger ``setup_logging()``, which attaches
    handlers and sets ``propagate=False``.  Without this fixture the state
    leaks into subsequent tests and breaks ``caplog`` capture.
    """
    logger = logging.getLogger("summarizer")
    for h in logger.handlers[:]:
        try:
            h.close()
        except Exception:
            pass
        logger.removeHandler(h)
    logger.propagate = True
    yield
    for h in logger.handlers[:]:
        try:
            h.close()
        except Exception:
            pass
        logger.removeHandler(h)
    logger.propagate = True


@pytest.fixture(autouse=True)
def _no_network(request, monkeypatch):
    """Unit tests must not hit the network (e.g. OpenRouter pricing lookups).

    Tests that need a response patch ``urlopen`` themselves (their patch wins);
    tests marked ``integration`` are exempt.
    """
    if request.node.get_closest_marker("integration"):
        return

    def _blocked(*args, **kwargs):
        raise OSError("network access is disabled in unit tests")

    monkeypatch.setattr("urllib.request.urlopen", _blocked)


@pytest.fixture(autouse=True)
def _isolated_env(request, monkeypatch):
    """Unit tests must not see the developer's ``.env`` (API key, model override)."""
    if request.node.get_closest_marker("integration"):
        return
    monkeypatch.setattr("summarizer.cli.load_dotenv", lambda *args, **kwargs: False)
    for var in ("LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _extraction_cache_in_tmp(tmp_path, monkeypatch):
    """Extraction caches go to $XDG_CACHE_HOME; keep them per test, out of $HOME."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))


@pytest.fixture(autouse=True)
def _default_log_dir_in_tmp(tmp_path, monkeypatch):
    """``main()`` writes ``logs/run_<ts>.log`` relative to the CWD by default.

    Run each test from its own tmp dir so tests never litter the repo. Tests
    that need repo files use absolute paths (see ``PROJECT_ROOT``).
    """
    monkeypatch.chdir(tmp_path)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture
def sample_pdf_path() -> Path:
    """Path to a real PDF in test_pdfs/ for smoke/integration tests."""
    path = (
        PROJECT_ROOT
        / "test_pdfs"
        / "Huebotter et al. - 2025 - Spiking Neural Networks for Continuous Control via End-to-End Model-Based Learning.pdf"
    )
    if not path.exists():
        pytest.skip(f"Sample PDF not found at {path} (test_pdfs/ is gitignored)")
    return path


# ---------------------------------------------------------------------------
# Mock LLM data
#
# MOCK_PART1_DICT is a *flat* dict holding metadata and Part 1 fields together
# (a leftover of the old two-call design). Tests split it into the combined
# ``{"metadata", "part1", "part2"}`` shape with their ``_make_combined_dict``
# helpers.
# ---------------------------------------------------------------------------

MOCK_PART1_DICT = {
    "citation_key": "huebotter2025spiking",
    "title": "Spiking Neural Networks for Continuous Control via End-to-End Model-Based Learning",
    "authors": ["Jan Huebotter", "Sirko Straube"],
    "year": 2025,
    "venue": "Preprint (arXiv:2501.00000)",
    "paper_type": "primary",
    "tags": ["SNN", "continuous control", "model-based learning", "robotic control"],
    "tldr": "An end-to-end model-based learning approach for continuous robot control using SNNs.",
    "problem_motivation": "Standard ANNs dominate robot control; SNNs offer energy efficiency but lack end-to-end training methods for continuous tasks.",
    "core_contribution": "First end-to-end model-based SNN training pipeline for continuous motor control.",
    "methods": "Surrogate gradient training with BPTT through a differentiable SNN model.",
    "results": "Matches ANN baseline on simulated arm tasks with lower spike rate.",
    "key_takeaways": "Shows primary feasibility of end-to-end model-based SNN control with competitive performance.",
    "limitations": "Evaluated in simulation only; sim-to-real gap not addressed.",
    "relevance": "Directly relevant as a primary example of SNN-based continuous control with end-to-end learning.",
    "open_problems_future_directions": {
        "future_work_proposed": ["Extend to physical robot; address sim-to-real transfer"],
        "open_questions": [
            "Does the approach scale to higher-DOF tasks? (Reviewer-noted)",
        ],
    },
    "critical_assessment": "Strong contribution; simulation-only evaluation limits generalizability.",
    "notable_findings": [
        "SNN matches ANN baseline reward within 5% (Measured)",
        "SNN achieves 40% lower average spike rate than equivalent rate-coded baseline (Reported)",
    ],
    "citable_snippets": [
        {
            "cite_for": "End-to-end model-based SNN training for continuous control",
            "source": "Sec. 3",
            "quote_tag": "Method",
            "quote": None,
        },
    ],
}

MOCK_PART2_DICT = {
    "neuron_model": "Leaky Integrate-and-Fire (LIF)",
    "network_architecture": "Multi-layer feedforward SNN",
    "model_scale": "~10k neurons",
    "simulator_framework": "Custom PyTorch-based SNN simulator",
    "hardware_training": "GPU (NVIDIA A100)",
    "controller_hardware_inference": "not reported",
    "control_task": "Simulated 7-DOF robotic arm reaching",
    "task_type": "Continuous motor control",
    "task_complexity_scale": "Medium — single arm, 7 DOF, simulated",
    "simulation_environment": "MuJoCo",
    "spike_encoding": "Rate coding",
    "action_decoding": "Linear readout from output layer spike counts",
    "learning_mechanism": "Surrogate gradient descent (BPTT)",
    "credit_assignment_scope": "Full network, end-to-end",
    "online_vs_offline": "Offline (batch training)",
    "data_collection": "Simulated rollouts in MuJoCo",
    "key_training_details": "BPTT with surrogate gradients; 500 training episodes",
    "comparison_to_baselines": "Compared against ANN with same architecture; SNN achieves comparable reward",
    "classification": {
        "inference_hardware": "CPU/GPU",
        "architecture": "fully spiking",
        "credit_assignment": "Global",
        "learning_regime": "Offline",
        "paradigm_families": ["Gradient-based (surrogate gradient BPTT)"],
    },
}


@pytest.fixture
def mock_part1_dict() -> dict:
    """Flat metadata + Part 1 dict (copy) for building models and combined responses."""
    return copy.deepcopy(MOCK_PART1_DICT)


@pytest.fixture
def mock_part2_dict() -> dict:
    """Part 2 (SNN extraction) dict (copy)."""
    return copy.deepcopy(MOCK_PART2_DICT)
