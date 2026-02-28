"""Shared pytest fixtures for the snn_summarizer test suite."""

import json
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
    assert path.exists(), f"Sample PDF not found at {path}"
    return path


# ---------------------------------------------------------------------------
# Mock LLM responses (JSON strings the LLM would return)
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
    "cite_for": [
        "End-to-end model-based SNN training for continuous control",
        "Comparison of SNN vs ANN energy efficiency in robotics",
    ],
    "critical_assessment": "Strong contribution; simulation-only evaluation limits generalizability.",
    "quotable_sentences": [
        "We demonstrate that SNNs can match ANN performance on continuous control tasks.",
    ],
    "notable_findings": [
        "SNN matches ANN baseline reward within 5% (Measured)",
        "SNN achieves 40% lower average spike rate than equivalent rate-coded baseline (Reported)",
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
}


@pytest.fixture
def mock_part1_response() -> str:
    """Raw JSON string the LLM would return for Call 1 (metadata + Part 1)."""
    return json.dumps(MOCK_PART1_DICT)


@pytest.fixture
def mock_part2_response() -> str:
    """Raw JSON string the LLM would return for Call 2 (Part 2 extraction)."""
    return json.dumps(MOCK_PART2_DICT)


@pytest.fixture
def mock_part1_dict() -> dict:
    """Parsed dict for mock Part 1 response — useful for building pydantic models."""
    return MOCK_PART1_DICT.copy()


@pytest.fixture
def mock_part2_dict() -> dict:
    """Parsed dict for mock Part 2 response — useful for building pydantic models."""
    return MOCK_PART2_DICT.copy()
