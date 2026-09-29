"""Evaluation harness: run model × extractor configurations over PDFs and report metrics.

Outputs go to a run directory (``results.jsonl``, ``report.md``, one JSON per
summary) and never touch ``output_summaries/`` or the processed index.  LLM
responses are cached on disk, so re-running (to re-score, or to resume an
interrupted free-tier run) costs nothing for calls that already succeeded.
"""

import hashlib
import json
import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path

from summarizer.batch import StopSignal, atomic_write_text
from summarizer.llm import (
    CompletionResponse,
    CostAccumulator,
    QuotaExhausted,
    UsageStats,
    create_client,
)
from summarizer.metrics import compute_metrics, duplicate_keys
from summarizer.models import Classification, Config, PaperSummary, PipelineError
from summarizer.parser import load_text, sha256_file
from summarizer.pipeline import author_surname_token, git_commit, process_pdf
from summarizer.prompts import load_references

logger = logging.getLogger(__name__)

GOLD_FIELDS = (
    "is_research_paper",
    "paper_type",
    "synthesis_subtype",
    "year",
    "first_author",
    "title",
    *(f"classification.{name}" for name in Classification.model_fields),
)


@dataclass(frozen=True)
class EvalConfig:
    model: str
    extractor: str

    @property
    def name(self) -> str:
        return f"{self.model.replace('/', '_').replace(':', '_')}__{self.extractor}"


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------


class CachingClient:
    """Wraps an LLM client with an on-disk response cache.

    One instance per paper, so ``hits`` / ``misses`` / ``llm_seconds`` describe
    that paper.  Failed calls are never cached.  On a hit, the stored usage
    (including the billed cost, when the backend reported one) and the original
    call's duration are replayed.  The key covers backend,
    model, output cap, structured-output mode and the full prompt (so a renamed
    PDF misses).
    """

    def __init__(self, inner, cache_dir: Path) -> None:
        self._inner = inner
        self._cache_dir = cache_dir
        self.model = inner.model
        self.base_url = inner.base_url
        self.pricing = inner.pricing
        self.response_format = getattr(inner, "response_format", None)
        self.hits = 0
        self.misses = 0
        self.llm_seconds = 0.0

    def _path(self, prompt: str) -> Path:
        key = json.dumps(
            [
                self.base_url,
                self.model,
                getattr(self._inner, "max_output_tokens", None),
                self.response_format is not None,
                prompt,
            ]
        )
        return self._cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.json"

    def complete(self, prompt: str) -> CompletionResponse:
        path = self._path(prompt)
        if path.exists():
            entry = json.loads(path.read_text(encoding="utf-8"))
            self.hits += 1
            self.llm_seconds += entry.get("elapsed_s", 0.0)
            usage = UsageStats(**entry["usage"]) if entry.get("usage") else None
            return CompletionResponse(text=entry["text"], usage=usage)

        t0 = time.monotonic()
        response = self._inner.complete(prompt)
        elapsed = time.monotonic() - t0
        self.misses += 1
        self.llm_seconds += elapsed
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "text": response.text,
            "usage": asdict(response.usage) if response.usage else None,
            "elapsed_s": elapsed,
        }
        atomic_write_text(path, json.dumps(entry))
        return response


# ---------------------------------------------------------------------------
# Gold labels
# ---------------------------------------------------------------------------


def load_gold(path: Path) -> dict[str, dict]:
    """Return ``sha256 -> labels`` from a gold JSONL file (missing file → empty)."""
    if not path.exists():
        return {}
    gold = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            gold[record["sha256"]] = record.get("labels") or {}
    return gold


def init_gold(path: Path, pdfs: list[Path]) -> int:
    """Append an unlabelled stub for every PDF not yet in ``path``; return how many."""
    known = set(load_gold(path))
    stubs = []
    for pdf in pdfs:
        sha = sha256_file(pdf)
        if sha not in known:
            known.add(sha)
            labels = dict.fromkeys(GOLD_FIELDS)
            stubs.append(json.dumps({"sha256": sha, "file": pdf.name, "labels": labels}))
    if stubs:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write("\n".join(stubs) + "\n")
    return len(stubs)


def _norm(value: object) -> str:
    return "".join(ch for ch in str(value).casefold() if ch.isalnum())


_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def _surname(name: str) -> str:
    """Surname token of "Given Surname" or "Surname, Given" (suffixes like Jr. ignored)."""
    if "," in name:
        name = name.split(",")[0]
    words = [w for w in name.split() if _norm(w) not in _NAME_SUFFIXES]
    return author_surname_token(" ".join(words))


def score_gold(summary: PaperSummary, labels: dict) -> dict[str, bool]:
    """Compare predictions with the labelled (non-null) gold fields."""
    meta = summary.metadata
    predicted = {
        "is_research_paper": meta.is_research_paper,
        "paper_type": meta.paper_type or "non_research",
        "synthesis_subtype": meta.synthesis_subtype,
        "year": meta.year,
        "first_author": _surname(meta.authors[0]) if meta.authors else None,
        "title": meta.title,
    }
    classification = summary.part2.classification if summary.part2 else None
    for name in Classification.model_fields:
        predicted[f"classification.{name}"] = getattr(classification, name, None)
    scores = {}
    for field in GOLD_FIELDS:
        expected = labels.get(field)
        if expected is None:
            continue
        got = predicted[field]
        if field == "first_author":
            expected = _surname(str(expected))
        if isinstance(expected, list):
            scores[field] = {_norm(v) for v in got or []} == {_norm(v) for v in expected}
        elif isinstance(expected, str) and got is not None:
            scores[field] = _norm(got) == _norm(expected)
        else:
            scores[field] = got == expected
    return scores


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _eval_one(
    pdf: Path,
    sha: str,
    *,
    config: Config,
    cfg: EvalConfig,
    client,
    references: str,
    cache_dir: Path,
    gold: dict[str, dict],
    summaries_dir: Path,
    stop: StopSignal,
    total: CostAccumulator,
) -> dict | None:
    """Evaluate one paper; ``None`` if the run was stopped before or during it.

    A failed paper counts as wrong on every labelled gold field.
    """
    stop.check_budget(total, config.max_cost)
    if stop.is_set():
        return None
    labelled = {f: False for f, v in gold.get(sha, {}).items() if v is not None}
    row = {
        "config": cfg.name,
        "model": cfg.model,
        "extractor": cfg.extractor,
        "file": pdf.name,
        "sha256": sha,
        "ok": False,
        "error": None,
        "gold": labelled or None,
    }
    accumulator = CostAccumulator(parent=total)
    cached_client = CachingClient(client, cache_dir)
    summary = None
    try:
        summary = process_pdf(
            pdf, config, client=cached_client, accumulator=accumulator, references=references
        )
    except PipelineError as exc:
        if isinstance(exc.cause, QuotaExhausted):
            stop.trip(str(exc.cause))
            return None
        row["error"] = str(exc.cause)

    row |= {
        "ok": summary is not None,
        "calls": accumulator.calls,
        "json_repairs": accumulator.json_repairs,
        "schema_repairs": accumulator.schema_repairs,
        "first_try_valid": summary is not None
        and accumulator.json_repairs == 0
        and accumulator.schema_repairs == 0,
        "input_tokens": accumulator.total_input_tokens,
        "output_tokens": accumulator.total_output_tokens,
        "cost_usd": accumulator.total_cost,
        "cache_hits": cached_client.hits,
        "cache_misses": cached_client.misses,
        "llm_s": round(cached_client.llm_seconds, 3),
    }
    if summary is None:
        return row

    # Score against exactly the text the model was shown (a prefix of the stripped text).
    provenance = summary.provenance
    paper_text = load_text(pdf, cfg.extractor, strip_references=config.strip_references).text[
        : provenance.chars_sent
    ]
    summaries_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(summaries_dir / f"{sha}.json", summary.model_dump_json(indent=2))
    row |= {
        "chars_full": provenance.chars_full,
        "chars_sent": provenance.chars_sent,
        "truncated": provenance.chars_full > provenance.chars_sent,
        "paper_type": summary.metadata.paper_type or "non_research",
        "citation_key": summary.metadata.citation_key,
        "metrics": compute_metrics(summary, paper_text),
        "gold": score_gold(summary, gold[sha]) if sha in gold else None,
    }
    return row


def run_eval(
    pdfs: list[Path],
    base_config: Config,
    configs: list[EvalConfig],
    out_dir: Path,
    cache_dir: Path,
    gold_path: Path | None = None,
) -> tuple[list[dict], str | None]:
    """Evaluate every configuration on every PDF; write results and a report.

    Returns the result rows and, if the run stopped early (exhausted quota or
    ``--max-cost``), the reason.  Papers hit by the stop are not scored.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    gold = load_gold(gold_path) if gold_path else {}
    references = load_references(base_config.skill_data_dir)
    provenance = {
        "git_commit": git_commit(),
        "references_sha256": hashlib.sha256(references.encode()).hexdigest()[:12],
        "max_chars": base_config.max_chars,
        "strip_references": base_config.strip_references,
        "structured_output": base_config.structured_output,
    }
    shas: dict[Path, str] = {}
    for pdf in pdfs:
        sha = sha256_file(pdf)
        if sha in shas.values():
            logger.warning("Skipping %s: same content as another PDF in the set", pdf.name)
        else:
            shas[pdf] = sha
    results_path = out_dir / "results.jsonl"
    results_path.write_text("", encoding="utf-8")
    rows: list[dict] = []
    stop = StopSignal()
    total = CostAccumulator()

    for cfg in configs:
        config = replace(base_config, model=cfg.model, extractor=cfg.extractor, dry_run=False)
        client = create_client(config)
        evaluate = partial(
            _eval_one,
            config=config,
            cfg=cfg,
            client=client,
            references=references,
            cache_dir=cache_dir,
            gold=gold,
            summaries_dir=out_dir / "summaries" / cfg.name,
            stop=stop,
            total=total,
        )
        logger.info("Evaluating %s on %d PDFs", cfg.name, len(shas))
        executor = ThreadPoolExecutor(max_workers=config.workers, thread_name_prefix="eval")
        try:
            futures = [executor.submit(evaluate, pdf, sha) for pdf, sha in shas.items()]
            for future in as_completed(futures):
                if future.cancelled():
                    continue
                result = future.result()
                if result is None:  # stopped
                    for pending in futures:
                        pending.cancel()
                    continue
                row = result | {"provenance": provenance}
                with results_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                rows.append(row)
                logger.info(
                    "  %s %s: %s", cfg.name, row["file"], "ok" if row["ok"] else row["error"]
                )
        except BaseException:
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        executor.shutdown()
        if stop.is_set():
            break

    report = render_report(rows, configs, provenance, stop.reason)
    atomic_write_text(out_dir / "report.md", report)
    return rows, stop.reason


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _pct(n: int, of: int) -> str:
    return f"{n / of:.0%} ({n}/{of})" if of else "n/a"


def _sum_rate(rows: list[dict], key: str) -> tuple[int, int]:
    rates = [r["metrics"][key] for r in rows if r.get("metrics")]
    return sum(x["n"] for x in rates), sum(x["of"] for x in rates)


def config_summary(rows: list[dict]) -> list[tuple[str, str]]:
    """``(column, value)`` pairs for one configuration's summary row.

    Quality columns use successful papers only; the ``ok`` column says how many.
    """
    ok = [r for r in rows if r["ok"]]
    scored = [r for r in ok if r.get("metrics")]
    quotes = [r["metrics"]["quotes"] for r in scored]
    q_total = sum(q["total"] for q in quotes)
    voice = [r["metrics"]["first_person"] for r in scored]
    voice_words = sum(v["words"] for v in voice)
    budgets = [r["metrics"]["word_budget"]["ratio"] for r in scored]
    gold = [s for r in rows if r.get("gold") for s in r["gold"].values()]
    return [
        ("ok", _pct(len(ok), len(rows))),
        ("first-try valid", _pct(sum(r.get("first_try_valid", False) for r in rows), len(rows))),
        ("repairs", str(sum(r.get("json_repairs", 0) + r.get("schema_repairs", 0) for r in rows))),
        (
            "tokens in / out",
            f"{sum(r.get('input_tokens', 0) for r in rows):,} / "
            f"{sum(r.get('output_tokens', 0) for r in rows):,}",
        ),
        ("cost", f"${sum(r.get('cost_usd', 0.0) for r in rows):.4f}"),
        ("quotes verbatim", _pct(sum(q["verbatim"] for q in quotes), q_total)),
        ("near", _pct(sum(q["near"] for q in quotes), q_total)),
        ("not found", _pct(sum(q["not_found"] for q in quotes), q_total)),
        ("anchor coverage", _pct(*_sum_rate(scored, "anchors"))),
        (
            "first person /1k words",
            f"{1000 * sum(v['count'] for v in voice) / voice_words:.1f}" if voice_words else "n/a",
        ),
        ("word budget (median)", f"{statistics.median(budgets):.2f}" if budgets else "n/a"),
        ("evidence tags", _pct(*_sum_rate(scored, "evidence_tags"))),
        ("gold labels", _pct(sum(gold), len(gold))),
    ]


def render_report(
    rows: list[dict],
    configs: list[EvalConfig],
    provenance: dict,
    stopped_reason: str | None = None,
) -> str:
    """Markdown report: per-config summary, per-paper table, missing quotes, labels."""
    by_config = {c.name: [r for r in rows if r["config"] == c.name] for c in configs}
    summaries = {name: config_summary(config_rows) for name, config_rows in by_config.items()}
    columns = [column for column, _ in next(iter(summaries.values()), [])]
    lines = [
        "# Evaluation report",
        "",
        f"Commit `{provenance['git_commit']}` · references `{provenance['references_sha256']}` · "
        f"max_chars {provenance['max_chars']:,} · {len({r['sha256'] for r in rows})} PDFs",
        "",
        *(
            [f"**Stopped early: {stopped_reason}.** Re-run to finish (cached calls are free).", ""]
            if stopped_reason
            else []
        ),
        "## Per configuration",
        "",
        "Quality columns cover successful papers only; failed papers count as wrong for gold "
        "labels.",
        "",
        "| config | " + " | ".join(columns) + " |",
        "|" + "---|" * (len(columns) + 1),
    ]
    for name, summary in summaries.items():
        lines.append(f"| `{name}` | " + " | ".join(value for _, value in summary) + " |")

    lines += ["", "## Per paper", "", "| paper | " + " | ".join(by_config) + " |"]
    lines.append("|" + "---|" * (len(by_config) + 1))
    for sha in dict.fromkeys(r["sha256"] for r in rows):
        cells = []
        for config_rows in by_config.values():
            row = next((r for r in config_rows if r["sha256"] == sha), None)
            cells.append(_paper_cell(row))
        file = next(r["file"] for r in rows if r["sha256"] == sha)
        lines.append(f"| {file} | " + " | ".join(cells) + " |")

    missing = [
        (r["file"], r["config"], q)
        for r in rows
        if r.get("metrics")
        for q in r["metrics"]["quotes"]["not_found_quotes"]
    ]
    lines += ["", "## Quotes not found in the paper text", ""]
    lines += [f"- {f} (`{c}`): “{q}”" for f, c, q in missing] or ["None."]

    lines += ["", "## Gold label accuracy by field", ""]
    labelled = [r for r in rows if r.get("gold")]
    if labelled:
        lines += ["| field | " + " | ".join(by_config) + " |", "|" + "---|" * (len(by_config) + 1)]
        for field in GOLD_FIELDS:
            cells = []
            for config_rows in by_config.values():
                scores = [r["gold"][field] for r in config_rows if field in (r.get("gold") or {})]
                cells.append(_pct(sum(scores), len(scores)))
            lines.append(f"| {field} | " + " | ".join(cells) + " |")
    else:
        lines.append("No labelled papers.")

    lines += ["", "## Duplicate citation keys", ""]
    dupes = {
        name: duplicate_keys([r["citation_key"] for r in config_rows if r["ok"]])
        for name, config_rows in by_config.items()
    }
    lines += [f"- `{n}`: {', '.join(d)}" for n, d in dupes.items() if d] or ["None."]
    return "\n".join(lines) + "\n"


def _paper_cell(row: dict | None) -> str:
    if row is None:
        return "—"
    if not row["ok"]:
        return "FAILED"
    cell = [row["paper_type"]]
    quotes = row["metrics"].get("quotes") if row.get("metrics") else None
    if quotes and quotes["total"]:
        cell.append(f"quotes {quotes['verbatim']}/{quotes['near']}/{quotes['not_found']}")
    repairs = row["json_repairs"] + row["schema_repairs"]
    if repairs:
        cell.append(f"{repairs} repair(s)")
    return ", ".join(cell)
