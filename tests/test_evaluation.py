"""Tests for summarizer/evaluation.py — cache, gold labels, runner and report."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from summarizer.evaluation import (
    CachingClient,
    EvalConfig,
    init_gold,
    load_gold,
    run_eval,
    score_gold,
)
from summarizer.llm import LLMError, ModelPricing, UsageStats, _CompletionResponse
from summarizer.models import Config, PaperSummary

REFERENCES_DIR = Path(__file__).parent.parent / "skill_data" / "references"
PAPER_TEXT = "The controller reaches 95.2% success on the reaching task."


def _combined(mock_part1_dict, mock_part2_dict) -> dict:
    meta_keys = {"citation_key", "title", "authors", "year", "venue", "paper_type", "tags"}
    metadata = {k: mock_part1_dict[k] for k in meta_keys}
    metadata |= {"is_research_paper": True, "rejection_reason": None}
    part1 = {k: v for k, v in mock_part1_dict.items() if k not in meta_keys - {"paper_type"}}
    return {"metadata": metadata, "part1": part1, "part2": mock_part2_dict}


def _inner_client(texts: list[str]) -> MagicMock:
    client = MagicMock()
    client.model = "m"
    client.base_url = "http://localhost:1234/v1"
    client.max_output_tokens = None
    client.pricing = ModelPricing()
    client.complete.side_effect = [
        _CompletionResponse(text=t, usage=UsageStats(input_tokens=100, output_tokens=10))
        for t in texts
    ]
    return client


# ---------------------------------------------------------------------------
# CachingClient
# ---------------------------------------------------------------------------


def test_cache_hit_makes_no_inner_call_and_returns_same_usage(tmp_path):
    inner = _inner_client(['{"a": 1}'])
    first = CachingClient(inner, tmp_path).complete("prompt")
    second_client = CachingClient(inner, tmp_path)
    second = second_client.complete("prompt")
    assert inner.complete.call_count == 1
    assert second.text == first.text
    assert second.usage == first.usage
    assert (second_client.hits, second_client.misses) == (1, 0)


def test_cache_key_includes_model(tmp_path):
    inner = _inner_client(['{"a": 1}', '{"a": 2}'])
    CachingClient(inner, tmp_path).complete("prompt")
    inner.model = "other"
    assert CachingClient(inner, tmp_path).complete("prompt").text == '{"a": 2}'
    assert inner.complete.call_count == 2


def test_failed_calls_are_not_cached(tmp_path):
    inner = _inner_client([])
    inner.complete.side_effect = [LLMError("boom"), _CompletionResponse(text="ok", usage=None)]
    with pytest.raises(LLMError):
        CachingClient(inner, tmp_path).complete("prompt")
    assert CachingClient(inner, tmp_path).complete("prompt").text == "ok"


# ---------------------------------------------------------------------------
# Gold labels
# ---------------------------------------------------------------------------


def test_init_gold_appends_stubs_and_never_overwrites(tmp_path):
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    a.write_bytes(b"%PDF a")
    b.write_bytes(b"%PDF b")
    gold = tmp_path / "gold.jsonl"
    assert init_gold(gold, [a]) == 1

    lines = gold.read_text().splitlines()
    record = json.loads(lines[0])
    record["labels"]["paper_type"] = "primary"
    gold.write_text(json.dumps(record) + "\n")

    assert init_gold(gold, [a, b]) == 1  # only b is new
    labels = load_gold(gold)
    assert labels[record["sha256"]]["paper_type"] == "primary"
    assert len(labels) == 2
    assert json.loads(gold.read_text().splitlines()[1])["file"] == "b.pdf"


def test_score_gold_skips_null_labels_and_normalizes(mock_part1_dict, mock_part2_dict):
    summary = PaperSummary(**_combined(mock_part1_dict, mock_part2_dict))
    labels = {
        "paper_type": "primary",
        "year": 2024,  # the mock says 2025
        "first_author": "Hübotter",  # mock: "Jan Huebotter"; diacritics differ
        "title": "spiking neural networks for continuous control via end-to-end "
        "model-based learning",
        "synthesis_subtype": None,
    }
    assert score_gold(summary, labels) == {
        "paper_type": True,
        "year": False,
        "first_author": False,  # "hubotter" vs "huebotter": a real spelling difference
        "title": True,
    }


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


def _run(tmp_path, pdfs, responses, **kwargs):
    config = Config(base_url="http://localhost:1234/v1", skill_data_dir=REFERENCES_DIR, workers=1)
    with (
        patch("summarizer.evaluation.parse_pdf", return_value=PAPER_TEXT),
        patch("summarizer.pipeline.parse_pdf", return_value=PAPER_TEXT),
        patch("summarizer.evaluation.create_client", return_value=_inner_client(responses)),
    ):
        return run_eval(
            pdfs,
            config,
            [EvalConfig(model="m", extractor="pypdf")],
            out_dir=tmp_path / "run",
            cache_dir=tmp_path / "cache",
            **kwargs,
        )


def test_run_eval_writes_results_report_and_summaries(
    tmp_path, pdfs, mock_part1_dict, mock_part2_dict
):
    mock_part1_dict["citable_snippets"] = [
        {"cite_for": "x", "source": "s", "quote": "reaches 95.2% success"},
        {"cite_for": "y", "source": "s", "quote": "an invented sentence that is nowhere in it"},
    ]
    good = json.dumps(_combined(mock_part1_dict, mock_part2_dict))
    rows = _run(tmp_path, pdfs, [good, "not json", "still not json"])

    by_file = {r["file"]: r for r in rows}
    ok, failed = by_file["a.pdf"], by_file["b.pdf"]
    assert ok["ok"] and ok["first_try_valid"] and ok["calls"] == 1
    quotes = ok["metrics"]["quotes"]
    assert (quotes["verbatim"], quotes["not_found"]) == (1, 1)
    assert (failed["ok"], failed["json_repairs"], failed["calls"]) == (False, 1, 2)
    assert "JSON" in failed["error"]

    run = tmp_path / "run"
    assert len((run / "results.jsonl").read_text().splitlines()) == 2
    assert (run / "summaries" / "m__pypdf" / f"{ok['sha256']}.json").exists()
    report = (run / "report.md").read_text()
    assert "| `m__pypdf` | 50% (1/2) |" in report
    assert "a.pdf (`m__pypdf`): “an invented sentence that is nowhere in it”" in report
    assert ok["provenance"]["references_sha256"]


def test_run_eval_second_run_uses_cache(tmp_path, pdfs, mock_part1_dict, mock_part2_dict):
    good = json.dumps(_combined(mock_part1_dict, mock_part2_dict))
    _run(tmp_path, pdfs, [good, good])
    rows = _run(tmp_path, pdfs, [])  # no responses available: must come from the cache
    assert all(r["ok"] and r["cache_hits"] == 1 and r["cache_misses"] == 0 for r in rows)
    assert all(r["input_tokens"] == 100 for r in rows)


def test_run_eval_scores_gold_labels(tmp_path, pdfs, mock_part1_dict, mock_part2_dict):
    gold = tmp_path / "gold.jsonl"
    init_gold(gold, pdfs)
    records = [json.loads(line) for line in gold.read_text().splitlines()]
    for record in records:
        record["labels"]["paper_type"] = "primary"
    gold.write_text("".join(json.dumps(r) + "\n" for r in records))

    good = json.dumps(_combined(mock_part1_dict, mock_part2_dict))
    rows = _run(tmp_path, pdfs, [good, good], gold_path=gold)
    assert all(r["gold"] == {"paper_type": True} for r in rows)
    assert "| paper_type | 100% (2/2) |" in (tmp_path / "run" / "report.md").read_text()


def test_run_eval_leaves_output_summaries_untouched(
    tmp_path, pdfs, mock_part1_dict, mock_part2_dict, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    good = json.dumps(_combined(mock_part1_dict, mock_part2_dict))
    _run(tmp_path, pdfs, [good, good])
    assert not (tmp_path / "output_summaries").exists()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_eval_dispatch(tmp_path, pdfs):
    from summarizer.cli import main

    with (
        patch("summarizer.cli._check_backend"),
        patch("summarizer.cli._check_openrouter_config") as check,
        patch("summarizer.cli.run_eval") as run,
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


def test_cli_eval_init_gold(tmp_path, pdfs):
    from summarizer.cli import main

    gold = tmp_path / "gold.jsonl"
    main(["eval", "--source", str(pdfs[0].parent), "--init-gold", "--gold", str(gold)])
    assert len(load_gold(gold)) == 2


@pytest.mark.parametrize("extractors", ["auto", "pypdf,ocr", ""])
def test_cli_eval_rejects_unknown_extractors(tmp_path, pdfs, extractors):
    from summarizer.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["eval", "--source", str(pdfs[0].parent), "--extractors", extractors])
    assert exc_info.value.code == 1


def test_legacy_invocation_still_routes_to_run(tmp_path):
    from summarizer.cli import main

    (tmp_path / "paper.pdf").write_bytes(b"%PDF")
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
        )
        main(["--source", str(tmp_path), "--dry-run"])
    run_batch.assert_called_once()
    run_eval_mock.assert_not_called()
