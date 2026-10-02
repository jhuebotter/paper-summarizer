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


# ---------------------------------------------------------------------------
# Regression tests from the code review
# ---------------------------------------------------------------------------


def _extract_with(tmp_path, client, accumulator=None, **kw):
    from summarizer.llm import CostAccumulator

    config = OverviewConfig(output_dir=tmp_path / "out", zotero=False, **kw)
    acc = accumulator or CostAccumulator()
    with patch(
        "summarizer.overview_run.load_text", return_value=ParsedText(PAPER, "docling", "f" * 64)
    ):
        return extract_record(_pdf(tmp_path), config, client, acc, "CODEBOOK"), acc


def test_a_critic_reply_without_json_keeps_the_paid_draft(tmp_path):
    client = _client(json.dumps(_record().model_dump()), "sorry, no json here", "still none")
    result, _ = _extract_with(tmp_path, client, critic=True)
    assert result.record.interface == "continuous"
    assert result.draft is not None


def test_an_exhausted_quota_in_the_critic_still_stops_the_run(tmp_path):
    client = _client(json.dumps(_record().model_dump()))
    client.complete.side_effect = [
        CompletionResponse(text=json.dumps(_record().model_dump()), usage=None),
        QuotaExhausted("daily cap"),
    ]
    with pytest.raises(QuotaExhausted):
        _extract_with(tmp_path, client, critic=True)


def test_quotes_are_checked_on_the_final_record_and_repairs_are_counted(tmp_path):
    bad = _record().model_dump()
    bad["interface"] = "spikes"
    fixed = _record(control_evidence="We use Loihi for all experiments.").model_dump()
    result, acc = _extract_with(tmp_path, _client(json.dumps(bad), json.dumps(fixed)))
    assert result.missing_quotes == ["control_evidence"]
    assert (result.calls, acc.schema_repairs) == (2, 1)


def test_the_same_pdf_under_two_names_gets_one_record(tmp_path):
    reply = json.dumps(_record().model_dump())
    keyed = _pdf(tmp_path, "ABCD1234__Doe - 2024 - A.pdf", b"%PDF same")
    _run(tmp_path, [keyed], _client(reply))
    plain = _pdf(tmp_path, "Doe 2024 copy.pdf", b"%PDF same")
    again = _run(tmp_path, [plain, keyed], _client())
    assert (again.processed, again.skipped) == (0, 2)
    assert [p.name for p in (tmp_path / "out" / "records").iterdir()] == ["ABCD1234.json"]
    assert len(load_results(tmp_path / "out")) == 1


def test_names_sharing_a_prefix_do_not_overwrite_each_other(tmp_path):
    reply = json.dumps(_record().model_dump())
    pdfs = [_pdf(tmp_path, "draft__v1.pdf", b"%PDF 1"), _pdf(tmp_path, "draft__v2.pdf", b"%PDF 2")]
    assert _run(tmp_path, pdfs, _client(reply, reply)).processed == 2
    assert len(list((tmp_path / "out" / "records").iterdir())) == 2
    assert _run(tmp_path, pdfs, _client()).processed == 0


def test_a_record_without_a_key_is_named_by_its_sha(tmp_path):
    pdf = _pdf(tmp_path, "plain.pdf", b"%PDF plain")
    _run(tmp_path, [pdf], _client(json.dumps(_record().model_dump())))
    assert (tmp_path / "out" / "records" / f"{sha256_file(pdf)[:16]}.json").exists()


def test_a_changed_model_or_critic_setting_makes_records_stale(tmp_path):
    pdfs = [_pdf(tmp_path)]
    reply = json.dumps(_record().model_dump())
    _run(tmp_path, pdfs, _client(reply))
    assert _run(tmp_path, pdfs, _client(reply, reply), critic=True).processed == 1
    client = _client(reply)
    client.model = "other/model"
    config = OverviewConfig(
        output_dir=tmp_path / "out", zotero=False, workers=1, model="other/model"
    )
    with (
        patch("summarizer.overview_run.create_client", return_value=client),
        patch(
            "summarizer.overview_run.load_text",
            side_effect=lambda p, **_: ParsedText(PAPER, "docling", sha256_file(p)),
        ),
    ):
        assert run_overview(pdfs, config).processed == 1


def test_the_collected_file_is_rebuilt_even_when_nothing_is_extracted(tmp_path):
    pdfs = [_pdf(tmp_path)]
    _run(tmp_path, pdfs, _client(json.dumps(_record().model_dump())))
    collected = tmp_path / "out" / "overview.jsonl"
    collected.unlink()
    _run(tmp_path, pdfs, _client())
    assert len(collected.read_text().splitlines()) == 1


def test_max_cost_stops_before_any_call(tmp_path):
    pdfs = [_pdf(tmp_path), _pdf(tmp_path, "WXYZ9876__Roe - 2020 - B.pdf", b"%PDF b")]
    client = _client()
    report = _run(tmp_path, pdfs, client, max_cost=0.0)
    assert (report.processed, report.skipped) == (0, 2)
    assert report.stopped_reason and client.complete.call_count == 0


def test_run_reports_tokens_and_cost(tmp_path):
    reply = json.dumps(_record().model_dump())
    report = _run(tmp_path, [_pdf(tmp_path)], _client(reply))
    assert report.input_tokens == 1000 and report.total_cost == pytest.approx(0.0011)


def test_zotero_metadata_is_stored(tmp_path):
    from summarizer.zotero import ZoteroRecord

    pdf = _pdf(tmp_path)
    zr = ZoteroRecord(
        item="groups/1/items/X",
        citation_key="doe2024",
        title="T",
        authors=["J Doe"],
        year=2024,
        venue="V",
    )
    config = OverviewConfig(output_dir=tmp_path / "out", workers=1, model="test/model")
    with (
        patch(
            "summarizer.overview_run.create_client",
            return_value=_client(json.dumps(_record().model_dump())),
        ),
        patch(
            "summarizer.overview_run.load_text",
            side_effect=lambda p, **_: ParsedText(PAPER, "docling", sha256_file(p)),
        ),
        patch("summarizer.overview_run.lookup_all", return_value={pdf: zr}),
    ):
        run_overview([pdf], config)
    (result,) = load_results(tmp_path / "out")
    assert (result.citation_key, result.year, label(result)) == ("doe2024", 2024, "doe2024")


def test_loading_recomputes_labels_with_the_current_rules(tmp_path):
    stale = _result("AAAA1111__Doe - 2024 - RL.pdf", _record())
    stale.derived = {**stale.derived, "design": "analytic"}
    _store(tmp_path, [stale])
    (loaded,) = load_results(tmp_path)
    assert loaded.derived["design"] == "learned"


def test_duplicate_records_keep_the_newest(tmp_path):
    old = _result("same", _record(), created="2026-01-01T00:00:00+00:00")
    new = _result("same", _record(interface="event-native"), created="2026-02-01T00:00:00+00:00")
    _store(tmp_path, [old, new])
    (loaded,) = load_results(tmp_path)
    assert loaded.record.interface == "event-native"


def test_a_non_spiking_hand_designed_controller_is_not_fully_spiking():
    rec = _record(
        spiking_roles=[],
        components=[
            _component(
                name="control law",
                spiking=False,
                obtained_by="hand-designed",
                signal="not applicable",
                mechanism="not applicable",
                regime="not applicable",
            )
        ],
    )
    assert derive(rec)["fully_spiking_deployed"] is False


def test_a_teacher_outside_the_controller_does_not_make_it_analytic():
    rec = _record(
        components=[
            _component(
                signal="supervised / imitation",
                mechanism="eligibility + modulator (three-factor)",
                regime="online",
            ),
            _component(
                name="teacher PID",
                role="other",
                spiking=False,
                deployed=False,
                obtained_by="hand-designed",
                signal="not applicable",
                mechanism="not applicable",
                regime="not applicable",
            ),
        ],
        analytic_methods=["control-theoretic"],
    )
    assert derive(rec)["design"] == "learned"


def test_analytic_methods_decide_only_without_controller_components():
    assert (
        derive(_record(components=[], analytic_methods=["control-theoretic"]))["design"]
        == "analytic"
    )
    assert (
        derive(_record(components=[], analytic_methods=["reservoir"]))["design"]
        == "not determinable"
    )


def test_searched_components_count_as_learned():
    rec = _record(
        components=[_component(obtained_by="searched", mechanism="evolutionary / black-box")]
    )
    d = derive(rec)
    assert (d["design"], d["learning_pairs"]) == ("learned", ["rl+ES"])


def test_a_learned_component_without_mechanism_has_no_pair():
    assert (
        derive(_record(components=[_component(mechanism="not applicable")]))["learning_pairs"] == []
    )


def test_quadrant_covers_mixed_and_excludes_unreported_interfaces():
    assert derive(_record(interface="mixed"))["quadrant"] == "learned × mixed"
    assert derive(_record(interface="not reported"))["quadrant"] == "n/a"


def test_adapts_online_ignores_components_that_were_not_learned():
    comp = _component(obtained_by="hand-designed", adapts_during_evaluation=True)
    assert derive(_record(components=[comp]))["adapts_online"] is False


def test_trained_readouts_keep_a_system_fully_spiking():
    rec = _record(
        components=[
            _component(),
            _component(name="readout", role="readout / decoder", spiking=False),
        ]
    )
    d = derive(rec)
    assert d["fully_spiking_deployed"] is True and d["nonspiking_training_only"] is False


def test_embedded_platforms_keep_their_hardware_loop():
    rec, _ = normalize(_record(platform="embedded CPU / microcontroller", hardware_in_loop=True))
    assert rec.hardware_in_loop is True


def test_partial_coverage_on_a_chip_is_kept():
    rec, changes = normalize(
        _record(platform="neuromorphic chip (digital)", platform_coverage="partial")
    )
    assert rec.platform_coverage == "partial" and changes == []


def test_no_control_task_clears_plant_fields_and_the_hardware_loop():
    rec, _ = normalize(
        _record(
            control_level="no control task",
            platform="neuromorphic chip (digital)",
            platform_coverage="whole network",
            hardware_in_loop=True,
        )
    )
    assert (rec.hardware_in_loop, rec.plant_model_use, rec.interface) == (
        False,
        "not applicable",
        "not applicable",
    )


def test_simulated_and_real_plants_keep_their_sim_to_real_value():
    metrics = {**_record().metrics.model_dump(), "sim_to_real": "simulation only"}
    rec, _ = normalize(_record(plant_setting="simulated and real", metrics=metrics))
    assert rec.metrics.sim_to_real == "simulation only"


def test_real_plants_without_sim_to_real_become_real_only():
    metrics = {**_record().metrics.model_dump(), "sim_to_real": "not applicable"}
    rec, _ = normalize(_record(plant_setting="real", metrics=metrics))
    assert rec.metrics.sim_to_real == "real only"


def test_fixed_components_lose_learning_fields():
    rec, _ = normalize(
        _record(
            components=[_component(obtained_by="random / fixed", adapts_during_evaluation=True)]
        )
    )
    c = rec.components[0]
    assert (c.signal, c.regime, c.adapts_during_evaluation) == (
        "not applicable",
        "not applicable",
        False,
    )


def test_metrics_evidence_is_checked_and_short_quotes_do_not_count():
    metrics = {**_record().metrics.model_dump(), "metrics_evidence": "an invented energy number"}
    assert missing_quotes(_record(metrics=metrics), PAPER) == ["metrics.metrics_evidence"]
    assert missing_quotes(_record(control_evidence="the"), PAPER) == ["control_evidence"]


def test_scored_fields_are_complete():
    assert set(score_record(_record(), _record())) == {
        "control_level",
        "plant_setting",
        "plant_dynamics",
        "plant_model_use",
        "interface",
        "platform",
        "platform_coverage",
        "hardware_in_loop",
        *(f"metrics.{m}" for m in overview.SCORED_METRICS),
        *overview.SCORED_SETS,
        "derived.design",
        "derived.quadrant",
        "derived.learning_pairs",
        "derived.regimes",
        "derived.adapts_online",
        "derived.fully_spiking_deployed",
        "derived.nonspiking_training_only",
    }


# ---------------------------------------------------------------------------
# Tables: every section
# ---------------------------------------------------------------------------


def _section(report: str, n: int) -> str:
    return report.split(f"## {n}.")[1].split(f"## {n + 1}.")[0]


def test_every_table_counts_the_right_papers():
    from summarizer.overview_tables import year_of

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
        objectives=[],
        platform="neuromorphic chip (digital)",
        platform_coverage="whole network",
        hardware_in_loop=True,
        task="Drone | hover\nreal",
        metrics={**_record().metrics.model_dump(), "latency": "measured"},
    )
    two = _record(
        components=[
            _component(),
            _component(
                name="world model",
                role="state estimation / world model",
                signal="self-supervised / system identification",
                regime="offline",
            ),
        ],
        interface="not reported",
    )
    results = [
        _result("AAAA1111__Doe and Roe - 2024 - RL.pdf", _record()),
        _result("BBBB2222__Roe - 2020 - PID.pdf", pid, normalized=["x: 'a' -> 'b'"]),
        _result("CCCC3333__Poe - 2021 - Two.pdf", two),
    ]
    results[2].record.notes = "check this"
    report = render_report(results)
    assert "Poe 2021" in _section(report, 1).split("Not in the grid")[1]
    chooser = _section(report, 2)
    rl_row = next(line for line in chooser.splitlines() if line.startswith("| `rl+BPTT`"))
    assert "| interleaved 3 |" in rl_row and "continuous 1" in rl_row and "not reported 1" in rl_row
    assert re.search(r"\| `self\+BPTT` \|.*\| offline 1 \|", chooser)
    assert "| (none) | **1** (Roe 2020) |" in _section(report, 3)
    assert "| control-theoretic | **1** (Roe 2020) |" in _section(report, 4)
    assert "| neuromorphic chip (digital) | **1** (Roe 2020) | 1 | 0 | 0 |" in _section(report, 5)
    reporting = _section(report, 6)
    assert "energy | estimated: 3 (100%)" in reporting
    assert "Measured energy or latency: **1** (Roe 2020)" in reporting
    assert "| 2016–20 | 1 | 1 | 0 | 0 | 1 |" in _section(report, 7)
    assert "Drone \\| hover real" in _section(report, 8)
    flags = _section(report, 9) if "## 10." in report else report.split("## 9.")[1]
    assert "fixed by consistency rules: x: 'a' -> 'b'" in flags and "note: check this" in flags
    assert (
        label(results[0]) == "Doe 2024" and year_of(_result("x.pdf", _record(), year=2019)) == 2019
    )
    assert label(_result("x.pdf", _record())) == "x.pdf"


def test_disagreements_ignore_identical_and_missing_records():
    from summarizer.overview_tables import disagreements

    first = [_result("a.pdf", _record()), _result("b.pdf", _record())]
    assert disagreements(first, [_result("a.pdf", _record())]) == {}
    assert render_report(first, []).count("disagrees") == 0


def test_overview_tables_rejects_a_csv_output_and_an_empty_comparison(tmp_path):
    _store(tmp_path / "ov", [_result("a.pdf", _record())])
    with pytest.raises(SystemExit):
        main(
            [
                "overview-tables",
                "--input",
                str(tmp_path / "ov"),
                "--output",
                str(tmp_path / "x.csv"),
            ]
        )
    with pytest.raises(SystemExit):
        main(
            [
                "overview-tables",
                "--input",
                str(tmp_path / "ov"),
                "--compare",
                str(tmp_path / "none"),
            ]
        )


def test_overview_command_checks_its_inputs_before_running(tmp_path):
    pdf = _pdf(tmp_path)
    with patch("summarizer.cli.run_overview") as run:
        for argv in (
            ["--gold", str(tmp_path / "missing.jsonl")],
            ["--codebook", str(tmp_path / "missing.md")],
        ):
            with pytest.raises(SystemExit):
                main(["overview", "--file", str(pdf), *argv])
        with pytest.raises(SystemExit):
            main(["overview", "--source", str(tmp_path / "nope")])
    run.assert_not_called()


def test_checked_fields_are_no_longer_flagged():
    from summarizer.overview_tables import review_flags

    first = [_result("a.pdf", _record(), checked=["interface"])]
    second = [_result("a.pdf", _record(interface="event-native"))]
    flags = render_report(first, second).split("## 9.")[1]
    assert "second run disagrees on: derived.quadrant" in flags
    assert "interface," not in flags and "checked by hand: 1 field(s)" in flags
    assert review_flags([_result("b.pdf", _record())]) == "None."
