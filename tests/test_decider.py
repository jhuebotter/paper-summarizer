"""Tests for summarizer/decider.py and its pipeline, eval and renderer hooks."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from summarizer.decider import DECIDER_MAX_CHARS, QUESTIONS, decide, label_descriptions
from summarizer.models import Classification, Config, Decision, PaperSummary, labels

REFERENCES = Path(__file__).parent.parent / "skill_data" / "references"


def _reply(choices: dict | None = None, model="typesafe/jev-1.13-20260917", cost=0.0005):
    """A System One reply choosing ``choices`` (field -> option key) or each field's first label."""
    descriptions = label_descriptions(REFERENCES)
    answers = {}
    for field, labels_ in descriptions.items():
        keys = [k.lower().replace("/", "_").replace(" ", "_").replace("-", "_") for k in labels_]
        keys = ["cpu_gpu" if k == "cpu_gpu" else k for k in keys]
        choice = (choices or {}).get(field, keys[0])
        answers[field] = {
            "type": "choice",
            "choice": choice,
            "confidence": 0.9,
            "probabilities": {k: (0.9 if k == choice else 0.1 / (len(keys) - 1)) for k in keys},
        }
    return {"model": model, "answers": answers, "usage": {"input_tokens": 100, "cost": cost}}


def _client(reply):
    client = MagicMock()
    client.decide.side_effect = reply if callable(reply) else lambda body: reply
    return client


def test_every_classification_label_has_a_description_in_the_references():
    """Drift guard: the decider reads its label definitions from snn-extraction-fields.md."""
    descriptions = label_descriptions(REFERENCES)
    assert set(descriptions) == set(QUESTIONS)
    for field, labels_ in descriptions.items():
        assert list(labels_) == list(labels(Classification.model_fields[field].annotation))
        assert all(text.strip() for text in labels_.values())


def test_a_label_missing_from_the_references_is_an_error(tmp_path):
    text = (REFERENCES / "snn-extraction-fields.md").read_text()
    (tmp_path / "snn-extraction-fields.md").write_text(text.replace("- **hybrid** —", "- hybrid —"))
    with pytest.raises(ValueError, match="architecture"):
        label_descriptions(tmp_path)


def test_decide_maps_answers_back_to_labels_and_cuts_the_text():
    client = _client(_reply({"inference_hardware": "physical_neuromorphic_chip"}))
    decisions, cost, served = decide(
        client, "typesafe/jev-1.13-20260917", "T", "x" * 200_000, label_descriptions(REFERENCES)
    )
    body = client.decide.call_args.args[0]
    assert set(body["questions"]) == set(QUESTIONS)
    assert all(q["type"] == "choice" for q in body["questions"].values())
    assert body["state"]["title"] == "T"
    assert len(body["state"]["paper"]) == DECIDER_MAX_CHARS
    assert decisions["inference_hardware"].label == "Physical neuromorphic chip"
    assert decisions["inference_hardware"].probabilities["Physical neuromorphic chip"] == 0.9
    assert (cost, served) == (0.0005, "typesafe/jev-1.13-20260917")


def test_a_different_snapshot_than_the_pinned_one_is_logged(caplog):
    client = _client(_reply(model="typesafe/jev-1.13-20261201"))
    decide(client, "typesafe/jev-1.13-20260917", "T", "text", label_descriptions(REFERENCES))
    assert any("answered as" in r.message for r in caplog.records)


def test_decide_retries_transient_errors():
    error = Exception("Error code: 503 - overloaded")
    calls = iter([error, _reply()])

    def flaky(body):
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return item

    with patch("summarizer.llm.time.sleep"):
        decisions, _, _ = decide(_client(flaky), "m", "T", "text", label_descriptions(REFERENCES))
    assert set(decisions) == set(QUESTIONS)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_path):
    return Config(
        base_url="http://localhost:1234/v1",
        model="m",
        skill_data_dir=REFERENCES,
        output_dir=tmp_path / "out",
        decider="typesafe/jev-1.13-20260917",
    )


def _run(config, combined, decide_side_effect, accumulator=None):
    from summarizer.llm import ModelPricing
    from summarizer.parser import ParsedText
    from summarizer.pipeline import process_pdf

    client = MagicMock()
    client.complete.return_value = MagicMock(text=json.dumps(combined), usage=None)
    client.decide.side_effect = decide_side_effect
    client.pricing = ModelPricing()
    with (
        patch(
            "summarizer.pipeline.load_text",
            return_value=ParsedText("paper text", "pypdf", "0" * 64),
        ),
        patch("summarizer.pipeline.create_client", return_value=client),
    ):
        return process_pdf(Path("p.pdf"), config, accumulator=accumulator), client


def _combined(mock_part1_dict, mock_part2_dict):
    meta_keys = {"citation_key", "title", "authors", "year", "venue", "paper_type", "tags"}
    metadata = {k: mock_part1_dict[k] for k in meta_keys} | {
        "is_research_paper": True,
        "rejection_reason": None,
    }
    part1 = {k: v for k, v in mock_part1_dict.items() if k not in meta_keys - {"paper_type"}}
    return {"metadata": metadata, "part1": part1, "part2": mock_part2_dict}


def test_pipeline_stores_decisions_next_to_the_llm_labels(config, mock_part1_dict, mock_part2_dict):
    from summarizer.llm import CostAccumulator

    run_total = CostAccumulator()
    summary, _ = _run(
        config,
        _combined(mock_part1_dict, mock_part2_dict),
        lambda body: _reply({"architecture": "hybrid"}),
        accumulator=run_total,
    )
    assert summary.decisions["architecture"].label == "hybrid"
    assert summary.decisions["architecture"].llm_label == "fully spiking"
    assert summary.part2.classification.architecture == "fully spiking"  # store only
    assert summary.provenance.decider_model == "typesafe/jev-1.13-20260917"
    assert summary.provenance.decider_cost_usd == 0.0005
    assert summary.provenance.calls == 1  # the decider is not an LLM call
    assert run_total.total_cost == pytest.approx(0.0005)  # but it counts toward --max-cost


def test_a_failing_decider_does_not_fail_the_paper(config, mock_part1_dict, mock_part2_dict):
    summary, _ = _run(
        config,
        _combined(mock_part1_dict, mock_part2_dict),
        ValueError("bad reply"),
    )
    assert summary.decisions is None
    assert "bad reply" in summary.provenance.decider_error


def test_exhausted_quota_in_the_decider_keeps_the_summary(config, mock_part1_dict, mock_part2_dict):
    """Review finding: re-raising threw away an LLM summary that was already paid for."""
    from summarizer.llm import QuotaExhausted

    summary, _ = _run(config, _combined(mock_part1_dict, mock_part2_dict), QuotaExhausted("402"))
    assert summary.decisions is None and "402" in summary.provenance.decider_error


def test_too_long_text_is_retried_shorter():
    """Review finding: Jev rejects over-long input (400 max_tokens_exceeded), it doesn't truncate."""
    import httpx2 as httpx
    import openai

    request = httpx.Request("POST", "https://x/systemone")
    too_long = openai.BadRequestError(
        "max_tokens_exceeded", response=httpx.Response(400, request=request), body=None
    )
    sizes = []

    def reply(body):
        sizes.append(len(body["state"]["paper"]))
        if len(sizes) == 1:
            raise too_long
        return _reply()

    decisions, _, _ = decide(
        _client(reply), "m", "T", "x" * 100_000, label_descriptions(REFERENCES)
    )
    assert sizes == [DECIDER_MAX_CHARS, int(DECIDER_MAX_CHARS * 0.6)] and decisions


def test_a_reply_without_answers_is_an_error():
    with pytest.raises(ValueError, match="without answers"):
        decide(_client({"model": "m"}), "m", "T", "text", label_descriptions(REFERENCES))


def test_unpinned_model_ids_are_not_checked(caplog):
    decide(
        _client(_reply(model="typesafe/jev-1.13-20261201")),
        "typesafe/jev-1.13",
        "T",
        "x",
        label_descriptions(REFERENCES),
    )
    assert not any("answered as" in r.message for r in caplog.records)


def test_no_decider_call_without_the_flag(config, mock_part1_dict, mock_part2_dict):
    config.decider = None
    summary, client = _run(config, _combined(mock_part1_dict, mock_part2_dict), lambda b: _reply())
    client.decide.assert_not_called()
    assert summary.decisions is None


def test_no_decider_call_for_papers_without_part2(config):
    synthesis = {
        "metadata": {
            "citation_key": "doe2020survey",
            "title": "A Survey",
            "authors": ["J. Doe"],
            "year": 2020,
            "venue": "V",
            "is_research_paper": True,
            "paper_type": "synthesis",
            "rejection_reason": None,
            "tags": [],
        },
        "part1": {
            "paper_type": "synthesis",
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
        },
        "part2": None,
    }
    summary, client = _run(config, synthesis, lambda b: _reply())
    client.decide.assert_not_called()
    assert summary.decisions is None


def test_decider_gets_the_title_and_truncated_text(config, mock_part1_dict, mock_part2_dict):
    config.max_chars = 5
    summary, client = _run(
        config,
        _combined(mock_part1_dict, mock_part2_dict),
        lambda b: _reply(model="typesafe/jev-1.13-20261201"),
    )
    state = client.decide.call_args.args[0]["state"]
    assert state == {"title": summary.metadata.title, "paper": "paper"}  # "paper text"[:5]
    assert summary.provenance.decider_model == "typesafe/jev-1.13-20261201"  # the served one


# ---------------------------------------------------------------------------
# Eval and renderer
# ---------------------------------------------------------------------------


def _summary_with(mock_part1_dict, mock_part2_dict, decisions):
    summary = PaperSummary(**_combined(mock_part1_dict, mock_part2_dict))
    return summary.model_copy(update={"decisions": decisions})


def test_decisions_are_scored_against_the_gold_classification(mock_part1_dict, mock_part2_dict):
    from summarizer.evaluation import score_decisions

    gold = {"classification.architecture": "hybrid", "classification.learning_regime": "Offline"}
    decisions = {"architecture": Decision(label="hybrid", probabilities={"hybrid": 1.0})}
    scores = score_decisions(_summary_with(mock_part1_dict, mock_part2_dict, decisions), gold)
    assert scores == {"decider.architecture": True, "decider.learning_regime": False}
    no_decisions = _summary_with(mock_part1_dict, mock_part2_dict, None)
    assert score_decisions(no_decisions, gold) == {
        "decider.architecture": False,
        "decider.learning_regime": False,
    }


def test_caching_client_replays_decisions_per_request(tmp_path):
    from summarizer.evaluation import CachingClient

    inner = MagicMock(model="m", base_url="u", pricing=None)
    inner.decide.side_effect = lambda body: {"answers": {"q": body["q"]}}
    cached = CachingClient(inner, tmp_path)
    assert cached.decide({"q": 1}) == cached.decide({"q": 1}) == {"answers": {"q": 1}}
    assert cached.decide({"q": 2}) == {"answers": {"q": 2}}
    assert inner.decide.call_count == 2
    assert (cached.hits, cached.misses) == (1, 2)


def test_malformed_decider_replies_are_not_cached(tmp_path):
    from summarizer.evaluation import CachingClient

    inner = MagicMock(model="m", base_url="u", pricing=None)
    inner.decide.return_value = "not json"
    cached = CachingClient(inner, tmp_path)
    cached.decide({"q": 1})
    cached.decide({"q": 1})
    assert inner.decide.call_count == 2


def test_add_cost_reaches_the_run_total_but_not_the_call_count():
    from summarizer.llm import CostAccumulator

    run = CostAccumulator()
    paper = CostAccumulator(parent=run)
    paper.add_cost(0.5)
    assert (run.total_cost, run.calls, paper.calls) == (0.5, 0, 0)


@pytest.mark.parametrize("command", [[], ["eval"]])
def test_decider_flag_reaches_the_config(tmp_path, command):
    from summarizer.cli import main

    report = MagicMock(processed=0, skipped=0, failed=0, failed_papers=[], total_cost=0.0)
    report.stopped_reason = None
    argv = [*command, "--source", str(tmp_path), "--decider"]
    if not command:
        argv.append("--dry-run")
    with (
        patch("summarizer.cli.run_batch", return_value=report) as run_batch,
        patch("summarizer.cli.run_eval", return_value=([], None)) as run_eval,
        patch("summarizer.cli._check_backend"),
        patch("summarizer.cli._check_openrouter_config"),
        patch("summarizer.cli._log_key_info"),
        patch("sys.exit"),
    ):
        (tmp_path / "a.pdf").write_bytes(b"%PDF")
        main(argv)
    called = run_eval if command else run_batch
    config = called.call_args.args[1]
    assert config.decider == "typesafe/jev-1.13-20260917"


def test_markdown_notes_only_disagreeing_decisions(mock_part1_dict, mock_part2_dict):
    from summarizer.renderer import render_summary

    decisions = {
        "architecture": Decision(label="hybrid", probabilities={"hybrid": 0.81}),
        "learning_regime": Decision(label="Offline", probabilities={"Offline": 0.99}),
    }
    md = render_summary(_summary_with(mock_part1_dict, mock_part2_dict, decisions))
    assert "**Architecture:** fully spiking (decision model: hybrid, p=0.81)" in md
    assert "**Learning regime:** Offline  " in md
