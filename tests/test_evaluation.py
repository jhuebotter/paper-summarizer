"""Tests for summarizer/evaluation.py — cache, gold labels, runner and report."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from summarizer.evaluation import (
    CachingClient,
    EvalConfig,
    config_summary,
    init_gold,
    load_gold,
    render_report,
    run_eval,
    score_gold,
)
from summarizer.llm import CompletionResponse, LLMError, ModelPricing, UsageStats
from summarizer.models import Config, PaperSummary
from summarizer.parser import ParsedText

REFERENCES_DIR = Path(__file__).parent.parent / "skill_data" / "references"


def _parsed(text: str) -> ParsedText:
    return ParsedText(text=text, extractor="pypdf", sha256="0" * 64)


PAPER_TEXT = "The controller reaches 95.2% success on the reaching task."


def _combined(mock_part1_dict, mock_part2_dict) -> dict:
    meta_keys = {"citation_key", "title", "authors", "year", "venue", "paper_type", "tags"}
    metadata = {k: mock_part1_dict[k] for k in meta_keys}
    metadata |= {"is_research_paper": True, "rejection_reason": None}
    part1 = {k: v for k, v in mock_part1_dict.items() if k not in meta_keys - {"paper_type"}}
    return {"metadata": metadata, "part1": part1, "part2": mock_part2_dict}


def _inner_client(reply) -> MagicMock:
    """A fake LLM client; ``reply(prompt)`` returns the response text."""
    client = MagicMock()
    client.model = "m"
    client.base_url = "http://localhost:1234/v1"
    client.max_output_tokens = None
    client.pricing = ModelPricing()
    client.complete.side_effect = lambda prompt: CompletionResponse(
        text=reply(prompt), usage=UsageStats(input_tokens=100, output_tokens=10)
    )
    return client


def _replies(*rules: tuple[str, str]):
    """Stateless fake LLM: the first rule whose marker occurs in the prompt wins."""

    def reply(prompt: str) -> str:
        for marker, text in rules:
            if marker in prompt:
                return text
        raise AssertionError("unexpected LLM call")

    return reply


REPAIR = "Payload to repair"  # marker of the JSON syntax-repair prompt


# ---------------------------------------------------------------------------
# CachingClient
# ---------------------------------------------------------------------------


def test_cache_hit_makes_no_inner_call_and_returns_same_usage(tmp_path):
    inner = _inner_client(lambda p: '{"a": 1}')
    first = CachingClient(inner, tmp_path).complete("prompt")
    second_client = CachingClient(inner, tmp_path)
    second = second_client.complete("prompt")
    assert inner.complete.call_count == 1
    assert (second.text, second.usage) == (first.text, first.usage)
    assert (second_client.hits, second_client.misses) == (1, 0)


@pytest.mark.parametrize(
    "attr,value", [("model", "other"), ("base_url", "http://x/v1"), ("max_output_tokens", 99)]
)
def test_cache_key_covers_backend_settings(tmp_path, attr, value):
    inner = _inner_client(lambda p: '{"a": 1}')
    CachingClient(inner, tmp_path).complete("prompt")
    setattr(inner, attr, value)
    CachingClient(inner, tmp_path).complete("prompt")
    assert inner.complete.call_count == 2


def test_failed_calls_are_not_cached(tmp_path):
    inner = _inner_client(lambda p: "ok")
    inner.complete.side_effect = [LLMError("boom"), CompletionResponse(text="ok", usage=None)]
    with pytest.raises(LLMError):
        CachingClient(inner, tmp_path).complete("prompt")
    assert CachingClient(inner, tmp_path).complete("prompt").text == "ok"


# ---------------------------------------------------------------------------
# Gold labels
# ---------------------------------------------------------------------------


def test_init_gold_appends_stubs_and_never_overwrites(tmp_path):
    a, b, a_copy = tmp_path / "a.pdf", tmp_path / "b.pdf", tmp_path / "a_copy.pdf"
    a.write_bytes(b"%PDF a")
    b.write_bytes(b"%PDF b")
    a_copy.write_bytes(b"%PDF a")
    gold = tmp_path / "gold.jsonl"
    assert init_gold(gold, [a, a_copy]) == 1  # identical content is one paper

    record = json.loads(gold.read_text().splitlines()[0])
    record["labels"]["paper_type"] = "primary"
    gold.write_text(json.dumps(record) + "\n")

    assert init_gold(gold, [a, b]) == 1  # only b is new
    labels = load_gold(gold)
    assert labels[record["sha256"]]["paper_type"] == "primary"
    assert len(labels) == 2


def test_score_gold_skips_null_labels_and_normalizes(mock_part1_dict, mock_part2_dict):
    summary = PaperSummary(**_combined(mock_part1_dict, mock_part2_dict))
    labels = {
        "paper_type": "primary",
        "year": 2024,  # the mock says 2025
        "first_author": "Huebotter, Jan",
        "title": "spiking neural networks for continuous control via end-to-end "
        "model-based learning",
        "synthesis_subtype": None,
    }
    assert score_gold(summary, labels) == {
        "paper_type": True,
        "year": False,
        "first_author": True,
        "title": True,
    }


@pytest.mark.parametrize(
    "gold_author,expected",
    [
        ("Huebotter", True),
        ("Jan Huebotter Jr.", True),
        # Known limitation: transliterations differ ("ü" -> "u", not "ue").
        ("Hübotter", False),
    ],
)
def test_score_gold_first_author_forms(mock_part1_dict, mock_part2_dict, gold_author, expected):
    summary = PaperSummary(**_combined(mock_part1_dict, mock_part2_dict))
    assert score_gold(summary, {"first_author": gold_author}) == {"first_author": expected}


def test_score_gold_non_research_label(mock_part1_dict, mock_part2_dict):
    summary = PaperSummary(**_combined(mock_part1_dict, mock_part2_dict))
    assert score_gold(summary, {"paper_type": "non_research"}) == {"paper_type": False}


# ---------------------------------------------------------------------------
# run_eval
# ---------------------------------------------------------------------------


@pytest.fixture
def pdfs(tmp_path):
    papers = tmp_path / "papers"
    papers.mkdir()
    for name in ("a.pdf", "b.pdf"):
        (papers / name).write_bytes(f"%PDF {name}".encode())
    return sorted(papers.iterdir())


@pytest.fixture
def good(mock_part1_dict, mock_part2_dict) -> str:
    mock_part1_dict["citable_snippets"] = [
        {"cite_for": "x", "source": "s", "quote": "reaches 95.2% success"},
        {"cite_for": "y", "source": "s", "quote": "an invented sentence that is nowhere in it"},
    ]
    return json.dumps(_combined(mock_part1_dict, mock_part2_dict))


def _run(tmp_path, pdfs, reply, configs=None, paper_text=PAPER_TEXT, max_chars=200_000, **kw):
    config = Config(
        base_url="http://localhost:1234/v1",
        skill_data_dir=REFERENCES_DIR,
        workers=2,
        max_chars=max_chars,
    )
    with (
        patch("summarizer.evaluation.load_text", return_value=_parsed(paper_text)),
        patch("summarizer.pipeline.load_text", return_value=_parsed(paper_text)),
        patch("summarizer.evaluation.create_client", return_value=_inner_client(reply)),
    ):
        rows, _ = run_eval(
            pdfs,
            config,
            configs or [EvalConfig(model="m", extractor="pypdf")],
            out_dir=tmp_path / "run",
            cache_dir=tmp_path / "cache",
            **kw,
        )
        return rows


def test_run_eval_writes_results_report_and_summaries(tmp_path, pdfs, good):
    rows = _run(
        tmp_path, pdfs, _replies((REPAIR, "still not json"), ("a.pdf", good), ("b.pdf", "not json"))
    )

    by_file = {r["file"]: r for r in rows}
    ok, failed = by_file["a.pdf"], by_file["b.pdf"]
    assert ok["ok"] and ok["first_try_valid"] and ok["calls"] == 1
    quotes = ok["metrics"]["quotes"]
    assert (quotes["verbatim"], quotes["not_found"]) == (1, 1)
    assert (failed["ok"], failed["first_try_valid"], failed["json_repairs"]) == (False, False, 1)
    assert "JSON" in failed["error"]

    run = tmp_path / "run"
    assert len((run / "results.jsonl").read_text().splitlines()) == 2
    assert (run / "summaries" / "m__pypdf" / f"{ok['sha256']}.json").exists()
    assert dict(config_summary(rows))["ok"] == "50% (1/2)"
    report = (run / "report.md").read_text()
    assert "a.pdf (`m__pypdf`): “an invented sentence that is nowhere in it”" in report
    assert ok["provenance"]["references_sha256"]


def test_first_try_valid_is_false_after_a_repair(tmp_path, pdfs, good):
    rows = _run(tmp_path, pdfs[:1], _replies((REPAIR, good), ("a.pdf", "not json")))
    assert (rows[0]["ok"], rows[0]["first_try_valid"], rows[0]["json_repairs"]) == (
        True,
        False,
        1,
    )


def test_parse_failure_row(tmp_path, pdfs):
    config = Config(base_url="http://localhost:1234/v1", skill_data_dir=REFERENCES_DIR)
    with (
        patch("summarizer.pipeline.load_text", side_effect=RuntimeError("corrupt")),
        patch("summarizer.evaluation.create_client", return_value=_inner_client(lambda p: "")),
    ):
        rows, _ = run_eval(
            pdfs[:1],
            config,
            [EvalConfig(model="m", extractor="pypdf")],
            out_dir=tmp_path / "run",
            cache_dir=tmp_path / "cache",
        )
    assert rows[0]["ok"] is False and "corrupt" in rows[0]["error"]
    assert rows[0].get("first_try_valid", False) is False


def test_metrics_use_the_text_that_was_sent(tmp_path, pdfs, good):
    """A quote from beyond --max-chars was never shown to the model."""
    paper = "x " * 50 + "The controller reaches 95.2% success on the reaching task."
    rows = _run(tmp_path, pdfs[:1], _replies(("a.pdf", good)), paper_text=paper, max_chars=60)
    assert rows[0]["truncated"] is True
    assert rows[0]["metrics"]["quotes"]["verbatim"] == 0


def test_run_eval_second_run_uses_cache_and_overwrites_results(tmp_path, pdfs, good):
    _run(tmp_path, pdfs, _replies((".pdf", good)))
    rows = _run(tmp_path, pdfs, _replies())  # no replies left: must come from the cache
    assert all(r["ok"] and r["cache_hits"] == 1 and r["cache_misses"] == 0 for r in rows)
    assert all(r["input_tokens"] == 100 for r in rows)
    assert len((tmp_path / "run" / "results.jsonl").read_text().splitlines()) == 2


def test_duplicate_pdfs_are_evaluated_once(tmp_path, pdfs, good):
    (pdfs[0].parent / "a_copy.pdf").write_bytes(pdfs[0].read_bytes())
    rows = _run(tmp_path, sorted(pdfs[0].parent.iterdir()), _replies((".pdf", good)))
    assert sorted(r["file"] for r in rows) == ["a.pdf", "b.pdf"]


def test_run_eval_scores_gold_labels_and_failures_count_as_wrong(tmp_path, pdfs, good):
    gold = tmp_path / "gold.jsonl"
    init_gold(gold, pdfs)
    records = [json.loads(line) for line in gold.read_text().splitlines()]
    for record in records:
        record["labels"]["paper_type"] = "primary"
    gold.write_text("".join(json.dumps(r) + "\n" for r in records))

    rows = _run(
        tmp_path,
        pdfs,
        _replies((REPAIR, "still not json"), ("a.pdf", good), ("b.pdf", "not json")),
        gold_path=gold,
    )
    by_file = {r["file"]: r for r in rows}
    assert by_file["a.pdf"]["gold"] == {"paper_type": True}
    assert by_file["b.pdf"]["gold"] == {"paper_type": False}
    assert dict(config_summary(rows))["gold labels"] == "50% (1/2)"


def test_keyboard_interrupt_propagates(tmp_path, pdfs):
    import threading

    def reply(prompt):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, pdfs[:1], reply)
    for thread in threading.enumerate():  # no worker may outlive the test's patches
        if thread.name.startswith("eval"):
            thread.join(5)


def test_report_with_several_configs(good):
    def row(config, file, ok, key="doe2024x"):
        base = {"config": config, "file": file, "sha256": file, "ok": ok, "error": None}
        if not ok:
            return base
        return base | {
            "paper_type": "primary",
            "citation_key": key,
            "json_repairs": 0,
            "schema_repairs": 0,
            "first_try_valid": True,
            "metrics": {},
        }

    configs = [EvalConfig("a", "pypdf"), EvalConfig("b", "pypdf")]
    rows = [
        row("a__pypdf", "one.pdf", True),
        row("a__pypdf", "two.pdf", True),  # same citation key -> duplicate
        row("b__pypdf", "one.pdf", False),
    ]
    report = render_report(
        rows, configs, {"git_commit": "c", "references_sha256": "r", "max_chars": 1}
    )
    assert "| one.pdf | primary | FAILED |" in report
    assert "| two.pdf | primary | — |" in report
    assert "- `a__pypdf`: doe2024x" in report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_eval_dispatch(tmp_path, pdfs):
    from summarizer.cli import main

    with (
        patch("summarizer.cli._check_backend"),
        patch("summarizer.cli._check_openrouter_config") as check,
        patch("summarizer.cli.importlib.util.find_spec", return_value=True),
        patch("summarizer.cli.run_eval", return_value=([], None)) as run,
    ):
        main(
            [
                "eval",
                "--source",
                str(pdfs[0].parent),
                "--models",
                "a/x,b/y",
                "--extractors",
                "pypdf,docling",
                "--out",
                str(tmp_path / "run"),
            ]
        )
    configs = run.call_args.args[2]
    assert [(c.model, c.extractor) for c in configs] == [
        ("a/x", "pypdf"),
        ("a/x", "docling"),
        ("b/y", "pypdf"),
        ("b/y", "docling"),
    ]
    assert [c.args[0].model for c in check.call_args_list] == ["a/x", "b/y"]
    assert run.call_args.args[1].workers == 1  # eval defaults to one worker (free-tier caps)


def test_cli_eval_init_gold(tmp_path, pdfs):
    from summarizer.cli import main

    gold = tmp_path / "gold.jsonl"
    main(["eval", "--source", str(pdfs[0].parent), "--init-gold", "--gold", str(gold)])
    assert len(load_gold(gold)) == 2
    assert not (tmp_path / "eval").exists()


@pytest.mark.parametrize("extractors", ["auto", "pypdf,ocr", ""])
def test_cli_eval_rejects_unknown_extractors(tmp_path, pdfs, extractors):
    from summarizer.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["eval", "--source", str(pdfs[0].parent), "--extractors", extractors])
    assert exc_info.value.code == 1


def test_cli_eval_requires_docling_when_requested(tmp_path, pdfs):
    from summarizer.cli import main

    with (
        patch("summarizer.cli.importlib.util.find_spec", return_value=None),
        pytest.raises(SystemExit) as exc_info,
    ):
        main(["eval", "--source", str(pdfs[0].parent), "--extractors", "docling"])
    assert exc_info.value.code == 1


def test_cli_eval_bad_source_leaves_no_run_dir(tmp_path):
    from summarizer.cli import main

    with pytest.raises(SystemExit):
        main(["eval", "--source", str(tmp_path / "missing"), "--out", str(tmp_path / "run")])
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("argv", [["--source", "{tmp}"], ["--source", "{tmp}/eval"]])
def test_non_eval_invocations_route_to_run(tmp_path, argv):
    """Only a leading `eval` selects evaluation; a folder named eval does not."""
    from summarizer.cli import main

    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "paper.pdf").write_bytes(b"%PDF")
    with (
        patch("summarizer.cli.run_batch") as run_batch,
        patch("summarizer.cli.run_eval") as run_eval_mock,
    ):
        run_batch.return_value = MagicMock(
            processed=0,
            skipped=1,
            failed=0,
            failed_papers=[],
            total_cost=0.0,
            input_tokens=0,
            output_tokens=0,
            stopped_reason=None,
        )
        main([a.format(tmp=tmp_path) for a in argv] + ["--dry-run"])
    run_batch.assert_called_once()
    run_eval_mock.assert_not_called()


def test_quota_exhaustion_stops_eval_without_scoring_failures(tmp_path, good):
    from summarizer.llm import QuotaExhausted

    papers = tmp_path / "papers"
    papers.mkdir()
    for name in ("a.pdf", "b.pdf", "c.pdf", "d.pdf"):
        (papers / name).write_bytes(f"%PDF {name}".encode())
    calls = []

    def reply(prompt):
        calls.append(prompt)
        if "a.pdf" in prompt:
            return good
        raise QuotaExhausted("free-models-per-day")

    config = Config(base_url="http://localhost:1234/v1", skill_data_dir=REFERENCES_DIR, workers=1)
    with (
        patch("summarizer.evaluation.load_text", return_value=_parsed(PAPER_TEXT)),
        patch("summarizer.pipeline.load_text", return_value=_parsed(PAPER_TEXT)),
        patch("summarizer.evaluation.create_client", return_value=_inner_client(reply)),
    ):
        rows, reason = run_eval(
            sorted(papers.iterdir()),
            config,
            [EvalConfig("m", "pypdf"), EvalConfig("n", "pypdf")],
            out_dir=tmp_path / "run",
            cache_dir=tmp_path / "cache",
        )
    assert reason == "free-models-per-day"
    assert [r["file"] for r in rows] == ["a.pdf"]  # the second config never ran
    assert len(calls) == 2  # a.pdf, then the quota hit; c.pdf and d.pdf never started
    assert "Stopped early: free-models-per-day" in (tmp_path / "run" / "report.md").read_text()


def test_max_cost_stops_eval(tmp_path, pdfs, good):
    client = _inner_client(_replies((".pdf", good)))
    client.complete.side_effect = lambda prompt: CompletionResponse(
        text=good, usage=UsageStats(input_tokens=1, cost=0.01)
    )
    config = Config(
        base_url="http://localhost:1234/v1", skill_data_dir=REFERENCES_DIR, workers=1, max_cost=0.01
    )
    with (
        patch("summarizer.evaluation.load_text", return_value=_parsed(PAPER_TEXT)),
        patch("summarizer.pipeline.load_text", return_value=_parsed(PAPER_TEXT)),
        patch("summarizer.evaluation.create_client", return_value=client),
    ):
        rows, reason = run_eval(
            pdfs,
            config,
            [EvalConfig("m", "pypdf")],
            out_dir=tmp_path / "run",
            cache_dir=tmp_path / "cache",
        )
    assert [r["file"] for r in rows] == ["a.pdf"]
    assert reason == "--max-cost $0.01 reached"


def test_eval_scores_against_the_stripped_text(tmp_path, pdfs, good):
    with patch("summarizer.evaluation.load_text", return_value=_parsed(PAPER_TEXT)) as load:
        config = Config(base_url="http://localhost:1234/v1", skill_data_dir=REFERENCES_DIR)
        with (
            patch("summarizer.pipeline.load_text", return_value=_parsed(PAPER_TEXT)),
            patch(
                "summarizer.evaluation.create_client", return_value=_inner_client(lambda p: good)
            ),
        ):
            run_eval(
                pdfs[:1],
                config,
                [EvalConfig("m", "pypdf")],
                out_dir=tmp_path / "run",
                cache_dir=tmp_path / "cache",
            )
    assert load.call_args.kwargs["strip_references"] is True


def test_cache_key_covers_structured_output(tmp_path):
    inner = _inner_client(lambda p: '{"a": 1}')
    inner.response_format = None
    CachingClient(inner, tmp_path).complete("prompt")
    inner.response_format = {"type": "json_schema"}
    CachingClient(inner, tmp_path).complete("prompt")
    assert inner.complete.call_count == 2


def test_gold_scores_classification_labels(mock_part1_dict, mock_part2_dict):
    summary = PaperSummary(**_combined(mock_part1_dict, mock_part2_dict))
    labels = {
        "classification.architecture": "Fully spiking",
        "classification.paradigm_families": ["gradient-based (surrogate gradient BPTT)"],
        "classification.inference_hardware": "Physical neuromorphic chip",
    }
    assert score_gold(summary, labels) == {
        "classification.architecture": True,
        "classification.paradigm_families": True,
        "classification.inference_hardware": False,
    }


def test_init_gold_stub_has_classification_fields(tmp_path, pdfs):
    gold = tmp_path / "gold.jsonl"
    init_gold(gold, pdfs[:1])
    labels = json.loads(gold.read_text())["labels"]
    assert "classification.learning_regime" in labels and labels["title"] is None
