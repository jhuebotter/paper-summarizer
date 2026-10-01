"""Tests for literature-overview records: schema, rules, extraction, tables and CLI."""

import json
import re
from pathlib import Path
from typing import Literal, get_args, get_origin
from unittest.mock import MagicMock, patch

import pytest

from summarizer import overview
from summarizer.cli import main
from summarizer.llm import CompletionResponse, ModelPricing, QuotaExhausted, UsageStats
from summarizer.overview import (
    Component,
    OverviewRecord,
    derive,
    missing_quotes,
    normalize,
    score_record,
)
from summarizer.overview_run import (
    DEFAULT_CODEBOOK,
    OverviewConfig,
    OverviewResult,
    build_prompt,
    extract_record,
    load_results,
    run_overview,
)
from summarizer.overview_tables import label, papers_csv, render_report
from summarizer.parser import ParsedText, sha256_file

PAPER = (
    "We train a spiking actor with TD3 and a non-spiking critic.\n"
    "The actor runs on a desk-\ntop GPU (RTX 3090) and outputs joint torques."
)


def _component(**kw) -> Component:
    base = dict(
        name="spiking actor",
        role="controller",
        spiking=True,
        deployed=True,
        obtained_by="learned",
        signal="reinforcement",
        mechanism="backprop / BPTT",
        regime="interleaved",
        adapts_during_evaluation=False,
        evidence="",
    )
    return Component(**{**base, **kw})


def _record(**kw) -> OverviewRecord:
    base = dict(
        spiking_roles=["controller"],
        control_level="closed-loop",
        control_evidence="",
        plant_setting="simulated",
        plant_dynamics="nonlinear",
        plant_model_use="model-free",
        objectives=["locomotion / pattern generation"],
        task="MuJoCo HalfCheetah",
        components=[
            _component(),
            _component(name="ANN critic", role="critic / value", spiking=False, deployed=False),
        ],
        analytic_methods=[],
        interface="continuous",
        platform="CPU/GPU",
        platform_name="RTX 3090",
        platform_coverage="not applicable",
        hardware_in_loop=False,
        metrics=dict(
            tracking_error="reported",
            latency="not reported",
            energy="estimated",
            spike_activity="reported",
            stability="not addressed",
            robustness="not tested",
            sim_to_real="simulation only",
        ),
    )
    return OverviewRecord.model_validate({**base, **kw})


# ---------------------------------------------------------------------------
# Codebook drift
# ---------------------------------------------------------------------------


def _literal_options():
    for name in dir(overview):
        value = getattr(overview, name)
        if get_origin(value) is Literal:
            yield name, get_args(value)


def test_codebook_lists_every_option():
    """Drift guard: every allowed value in the schema is defined in the codebook."""
    codebook = DEFAULT_CODEBOOK.read_text(encoding="utf-8")
    missing = [
        (name, option)
        for name, opts in _literal_options()
        for option in opts
        if f"`{option}`" not in codebook
    ]
    assert not missing


def test_codebook_names_every_record_field():
    codebook = DEFAULT_CODEBOOK.read_text(encoding="utf-8")
    fields = [
        *OverviewRecord.model_fields,
        *Component.model_fields,
        *overview.Metrics.model_fields,
    ]
    assert [f for f in fields if f"`{f}`" not in codebook] == []


# ---------------------------------------------------------------------------
# Derived labels
# ---------------------------------------------------------------------------


def test_deep_rl_with_ann_critic():
    d = derive(_record())
    assert d["design"] == "learned"
    assert d["learning_pairs"] == ["rl+BPTT"]
    assert d["regimes"] == ["interleaved"]
    assert d["fully_spiking_deployed"] is True  # the critic is training-only
    assert d["nonspiking_training_only"] is True
    assert d["quadrant"] == "learned × continuous"


def test_nef_with_pes_adaptation_is_analytic_and_learned():
    rec = _record(
        components=[
            _component(
                name="decoders",
                role="readout / decoder",
                obtained_by="solved",
                signal="not applicable",
                mechanism="not applicable",
                regime="not applicable",
            ),
            _component(
                name="adaptive population",
                signal="supervised / imitation",
                mechanism="local error (LMS-like)",
                regime="online",
                adapts_during_evaluation=True,
            ),
        ],
        analytic_methods=["NEF"],
    )
    d = derive(rec)
    assert d["design"] == "analytic + learned"
    assert d["learning_pairs"] == ["supv+LMS"]
    assert d["regimes"] == ["online"]
    assert d["adapts_online"] is True


def test_hand_designed_pid_has_no_learning():
    rec = _record(
        components=[
            _component(
                name="PID",
                obtained_by="hand-designed",
                signal="not applicable",
                mechanism="not applicable",
                regime="not applicable",
            )
        ],
        analytic_methods=["control-theoretic"],
        interface="event-native",
    )
    d = derive(rec)
    assert (d["design"], d["learning_pairs"], d["regimes"]) == ("analytic", [], [])
    assert d["quadrant"] == "analytic × event-native"


def test_a_reservoir_alone_does_not_make_a_design_analytic():
    rec = _record(
        components=[
            _component(
                name="reservoir",
                obtained_by="random / fixed",
                signal="not applicable",
                mechanism="not applicable",
                regime="not applicable",
            ),
            _component(
                name="readout",
                role="readout / decoder",
                spiking=False,
                signal="supervised / imitation",
                mechanism="local error (LMS-like)",
                regime="offline",
            ),
        ],
        analytic_methods=["reservoir"],
    )
    assert derive(rec)["design"] == "learned"


def test_a_deployed_trained_ann_makes_the_system_hybrid():
    rec = _record(
        components=[
            _component(),
            _component(name="CNN encoder", spiking=False, role="perception / encoder"),
        ]
    )
    assert derive(rec)["fully_spiking_deployed"] is False


def test_regimes_ignore_components_that_were_not_learned():
    rec = _record(components=[_component(obtained_by="hand-designed", regime="offline")])
    assert derive(rec)["regimes"] == []


# ---------------------------------------------------------------------------
# Consistency rules
# ---------------------------------------------------------------------------


def test_software_platforms_have_no_coverage_and_no_hardware_loop():
    rec, changes = normalize(_record(platform_coverage="whole network", hardware_in_loop=True))
    assert (rec.platform_coverage, rec.hardware_in_loop) == ("not applicable", False)
    assert len(changes) == 2


def test_a_chip_without_coverage_defaults_to_the_whole_network():
    rec, _ = normalize(_record(platform="neuromorphic chip (digital)"))
    assert rec.platform_coverage == "whole network"


def test_perception_papers_have_no_plant_model_or_interface():
    rec, _ = normalize(_record(control_level="perception for control", plant_setting="real"))
    assert (rec.plant_dynamics, rec.plant_model_use, rec.interface) == ("not applicable",) * 3
    assert rec.metrics.sim_to_real == "not applicable"


@pytest.mark.parametrize(
    ("setting", "claimed", "expected"),
    [
        ("simulated", "transferred", "simulation only"),
        ("real", "simulation only", "real only"),
        ("simulated and real", "transferred", "transferred"),
        ("none", "real only", "not applicable"),
    ],
)
def test_sim_to_real_follows_the_plant_setting(setting, claimed, expected):
    metrics = {**_record().metrics.model_dump(), "sim_to_real": claimed}
    rec, _ = normalize(_record(plant_setting=setting, metrics=metrics))
    assert rec.metrics.sim_to_real == expected


def test_components_that_were_not_learned_lose_their_learning_fields():
    rec, _ = normalize(_record(components=[_component(obtained_by="solved")]))
    c = rec.components[0]
    assert (c.signal, c.mechanism, c.regime, c.adapts_during_evaluation) == (
        "not applicable",
        "not applicable",
        "not applicable",
        False,
    )


def test_a_consistent_record_is_unchanged():
    assert normalize(_record())[1] == []


# ---------------------------------------------------------------------------
# Quotes and scoring
# ---------------------------------------------------------------------------


def test_quotes_survive_line_breaks_and_hyphenation():
    rec = _record(platform_evidence="The actor runs on a desktop GPU (RTX 3090)")
    assert missing_quotes(rec, PAPER) == []


def test_an_invented_quote_is_reported():
    comps = [_component(evidence="We use Loihi for all experiments.")]
    rec = _record(components=comps, control_evidence="outputs joint torques")
    assert missing_quotes(rec, PAPER) == ["components[0].evidence"]


def test_scores_compare_sets_regardless_of_order():
    a = _record(objectives=["tracking", "regulation"])
    b = _record(objectives=["regulation", "tracking"], interface="event-native")
    scores = score_record(a, b)
    assert scores["objectives"] is True
    assert scores["interface"] is False
    assert scores["derived.quadrant"] is False


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _client(*replies: str) -> MagicMock:
    client = MagicMock()
    client.model = "test/model"
    client.base_url = "http://localhost:1234/v1"
    client.pricing = ModelPricing(prompt=1e-6, completion=1e-6)
    client.complete.side_effect = [
        CompletionResponse(text=r, usage=UsageStats(input_tokens=1000, output_tokens=100))
        for r in replies
    ]
    return client


def _pdf(
    tmp_path: Path, name="ABCD1234__Doe et al. - 2024 - A paper.pdf", content=b"%PDF a"
) -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def _extract(tmp_path, client, **kw):
    config = OverviewConfig(output_dir=tmp_path / "out", zotero=False, **kw)
    with patch(
        "summarizer.overview_run.load_text", return_value=ParsedText(PAPER, "docling", "f" * 64)
    ):
        return extract_record(_pdf(tmp_path), config, client, overview_accumulator(), "CODEBOOK")


def overview_accumulator():
    from summarizer.llm import CostAccumulator

    return CostAccumulator()


def test_prompt_carries_codebook_template_and_paper():
    prompt = build_prompt("PAPER TEXT", "CODEBOOK TEXT")
    assert "CODEBOOK TEXT" in prompt and prompt.rstrip().endswith("PAPER TEXT")
    assert "<one of: closed-loop | open-loop actuation" in prompt
    assert '"components": [' in prompt


def test_extract_record_validates_normalizes_and_checks_quotes(tmp_path):
    raw = _record(platform_coverage="whole network").model_dump()
    raw["control_evidence"] = "We use Loihi."
    result = _extract(tmp_path, _client(json.dumps(raw)))
    assert result.record.platform_coverage == "not applicable"
    assert result.normalized == ["platform_coverage: 'whole network' -> 'not applicable'"]
    assert result.missing_quotes == ["control_evidence"]
    assert result.derived["design"] == "learned"
    assert result.calls == 1 and result.cost_usd == pytest.approx(0.0011)
    assert result.model == "test/model" and len(result.codebook_sha256) == 64


def test_an_invalid_record_gets_one_repair_call(tmp_path):
    bad = _record().model_dump()
    bad["interface"] = "spikes"
    client = _client(json.dumps(bad), json.dumps(_record().model_dump()))
    result = _extract(tmp_path, client)
    assert result.record.interface == "continuous"
    repair_prompt = client.complete.call_args_list[1].args[0]
    assert "interface" in repair_prompt and PAPER not in repair_prompt


def test_the_critic_pass_replaces_the_draft(tmp_path):
    fixed = _record(interface="event-native").model_dump()
    client = _client(json.dumps(_record().model_dump()), json.dumps(fixed))
    result = _extract(tmp_path, client, critic=True)
    assert result.record.interface == "event-native"
    assert result.draft.interface == "continuous"
    assert "Draft record" in client.complete.call_args_list[1].args[0]


def test_an_unusable_critic_reply_keeps_the_draft(tmp_path):
    client = _client(json.dumps(_record().model_dump()), '{"interface": "x"}', '{"still": "bad"}')
    result = _extract(tmp_path, client, critic=True)
    assert result.record.interface == "continuous"


# ---------------------------------------------------------------------------
# Batch runs
# ---------------------------------------------------------------------------


def _run(tmp_path, pdfs, client, **kw):
    config = OverviewConfig(
        output_dir=tmp_path / "out", zotero=False, workers=1, model="test/model", **kw
    )
    with (
        patch("summarizer.overview_run.create_client", return_value=client),
        patch(
            "summarizer.overview_run.load_text",
            side_effect=lambda p, **_: ParsedText(PAPER, "docling", sha256_file(p)),
        ),
    ):
        return run_overview(pdfs, config)


def test_run_writes_records_and_skips_current_ones(tmp_path):
    pdfs = [_pdf(tmp_path), _pdf(tmp_path, "WXYZ9876__Roe - 2020 - B.pdf", b"%PDF b")]
    reply = json.dumps(_record().model_dump())
    report = _run(tmp_path, pdfs, _client(reply, reply))
    assert (report.processed, report.failed) == (2, 0)
    out = tmp_path / "out"
    assert sorted(p.name for p in (out / "records").iterdir()) == ["ABCD1234.json", "WXYZ9876.json"]
    assert len((out / "overview.jsonl").read_text().splitlines()) == 2

    again = _run(tmp_path, pdfs, _client())
    assert (again.processed, again.skipped) == (0, 2)
    forced = _run(tmp_path, pdfs[:1], _client(reply), force=True)
    assert forced.processed == 1


def test_a_changed_codebook_makes_records_stale(tmp_path):
    pdfs = [_pdf(tmp_path)]
    reply = json.dumps(_record().model_dump())
    _run(tmp_path, pdfs, _client(reply))
    other = tmp_path / "codebook.md"
    other.write_text("a different codebook")
    assert _run(tmp_path, pdfs, _client(reply), codebook=other).processed == 1


def test_a_failing_paper_does_not_stop_the_run(tmp_path):
    pdfs = [_pdf(tmp_path), _pdf(tmp_path, "WXYZ9876__Roe - 2020 - B.pdf", b"%PDF b")]
    report = _run(
        tmp_path,
        pdfs,
        _client("not json at all", "still not json", json.dumps(_record().model_dump())),
    )
    assert report.failed == 1 and report.processed == 1


def test_an_exhausted_quota_stops_the_run(tmp_path):
    pdfs = [_pdf(tmp_path), _pdf(tmp_path, "WXYZ9876__Roe - 2020 - B.pdf", b"%PDF b")]
    client = _client()
    client.complete.side_effect = QuotaExhausted("daily cap")
    report = _run(tmp_path, pdfs, client)
    assert report.processed == 0 and report.stopped_reason == "daily cap"


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _result(file: str, record: OverviewRecord, **kw) -> OverviewResult:
    return OverviewResult(
        sha256=file,
        file=file,
        record=record,
        derived=derive(record),
        model="m",
        codebook_sha256="c",
        extractor="docling",
        **kw,
    )


def test_labels_come_from_zotero_or_the_filename():
    rec = _record()
    assert label(_result("ABCD1234__Doe et al. - 2024 - T.pdf", rec)) == "Doe 2024"
    assert label(_result("x.pdf", rec, citation_key="doe2024t")) == "doe2024t"


def test_report_counts_primary_papers_and_flags_the_rest():
    pid = _record(
        components=[
            _component(
                name="PID",
                obtained_by="hand-designed",
                signal="not applicable",
                mechanism="not applicable",
                regime="not applicable",
            )
        ],
        analytic_methods=["control-theoretic"],
        interface="event-native",
        platform="neuromorphic chip (digital)",
        platform_coverage="whole network",
        hardware_in_loop=True,
    )
    results = [
        _result("AAAA1111__Doe - 2024 - RL.pdf", _record()),
        _result("BBBB2222__Roe - 2019 - PID.pdf", pid, missing_quotes=["control_evidence"]),
        _result("CCCC3333__Poe - 2021 - Review.pdf", _record(paper_kind="review or survey")),
    ]
    report = render_report(results)
    grid = report.split("## 1.")[1].split("## 2.")[0]
    assert re.search(r"\| learned \| \*\*1\*\* \(Doe 2024\) \| – \| – \|", grid)
    assert re.search(r"\| analytic \| – \| \*\*1\*\* \(Roe 2019\) \| – \|", grid)
    assert "`rl+BPTT`" in report and "Poe 2021" not in grid
    flags = report.split("## 9.")[1]
    assert "Roe 2019" in flags and "quote not in text" in flags
    assert "paper kind: review or survey" in flags
    rows = papers_csv(results).splitlines()
    assert rows[0].startswith("paper,year") and len(rows) == 3


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _store(output_dir: Path, results: list[OverviewResult]) -> None:
    (output_dir / "records").mkdir(parents=True)
    for i, r in enumerate(results):
        (output_dir / "records" / f"{i}.json").write_text(r.model_dump_json())


def test_overview_tables_command_writes_markdown_and_csv(tmp_path):
    _store(tmp_path / "ov", [_result("AAAA1111__Doe - 2024 - RL.pdf", _record())])
    main(["overview-tables", "--input", str(tmp_path / "ov")])
    assert "## 1. Design × interface" in (tmp_path / "ov" / "overview.md").read_text()
    assert (tmp_path / "ov" / "overview.csv").read_text().startswith("paper,")


def test_overview_tables_command_needs_records(tmp_path):
    with pytest.raises(SystemExit):
        main(["overview-tables", "--input", str(tmp_path)])


def test_overview_command_builds_the_config_and_scores_gold(tmp_path, monkeypatch, capfd):
    pdf = _pdf(tmp_path)
    out = tmp_path / "ov"
    _store(out, [_result("f" * 64, _record())])
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        json.dumps({"sha256": "f" * 64, "record": _record(interface="mixed").model_dump()})
    )
    monkeypatch.setenv("LLM_API_KEY", "k")
    seen = {}

    def fake_run(pdfs, config):
        seen.update(pdfs=pdfs, config=config)
        from summarizer.models import BatchReport

        return BatchReport(processed=1, skipped=0, failed=0, failed_papers=[])

    with (
        patch("summarizer.cli.run_overview", side_effect=fake_run),
        patch("summarizer.cli._check_backend"),
        patch("summarizer.cli._check_openrouter_config"),
        patch("summarizer.cli._log_key_info"),
    ):
        main(
            [
                "overview",
                "--file",
                str(pdf),
                "--output-dir",
                str(out),
                "--model",
                "x/y",
                "--critic",
                "--no-zotero",
                "--gold",
                str(gold),
            ]
        )
    assert seen["pdfs"] == [pdf]
    assert (seen["config"].model, seen["config"].critic, seen["config"].zotero) == (
        "x/y",
        True,
        False,
    )
    logged = "".join(capfd.readouterr())
    assert re.search(r"interface\s+0/1", logged)
    assert re.search(r"control_level\s+1/1", logged)


def test_load_results_skips_unreadable_records(tmp_path, caplog):
    _store(tmp_path, [_result("a.pdf", _record())])
    (tmp_path / "records" / "bad.json").write_text("{")
    assert len(load_results(tmp_path)) == 1
    assert "Cannot read" in caplog.text


def test_a_second_run_flags_the_fields_it_disagrees_on(tmp_path):
    from summarizer.overview_tables import disagreements

    first = [_result("AAAA1111__Doe - 2024 - RL.pdf", _record())]
    second = [_result("AAAA1111__Doe - 2024 - RL.pdf", _record(interface="event-native"))]
    assert disagreements(first, second) == {
        "AAAA1111__Doe - 2024 - RL.pdf": ["interface", "derived.quadrant"]
    }
    _store(tmp_path / "a", first)
    _store(tmp_path / "b", second)
    main(["overview-tables", "--input", str(tmp_path / "a"), "--compare", str(tmp_path / "b")])
    report = (tmp_path / "a" / "overview.md").read_text()
    assert "disagrees on 2 labels in 1 papers" in report
    assert "second run disagrees on: interface, derived.quadrant" in report
