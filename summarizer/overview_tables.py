"""Corpus tables for the review, built from stored overview records (no LLM calls).

Each table answers one question the review draft asks of the literature:
- the analytic/learned × continuous/event-native framing (2×2, or 3×3 with mixed);
- the chooser table: training signal + update mechanism, with regimes and interfaces;
- methods by control objective;
- hardware and deployment;
- reporting practice (which metrics papers report, and how);
- a trend over publication years.

Every cell lists the papers it counts, so each number can be traced back. Only
``primary research`` records are counted. A per-paper table (also as CSV) and
a list of records worth checking by hand close the report.
"""

import csv
import io
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable

from summarizer.overview import (
    MECHANISM_SHORT,
    SCORED_METRICS,
    SIGNAL_SHORT,
    OverviewResult,
    comparable_facts,
    learning_pair,
)

_DESIGNS = ("analytic", "learned", "analytic + learned")
_INTERFACES = ("continuous", "event-native", "mixed")
_YEAR_BINS = (
    (0, 2010, "≤2010"),
    (2011, 2015, "2011–15"),
    (2016, 2020, "2016–20"),
    (2021, 2023, "2021–23"),
    (2024, 9999, "2024+"),
)
_FILENAME = re.compile(r"^(?:[A-Z0-9]{8}__)?(?P<authors>.+?) - (?P<year>\d{4}) - ")


def label(result: OverviewResult) -> str:
    """Citation key if known, else "Author Year" from the ``KEY__Author - Year - Title`` name."""
    if result.citation_key:
        return result.citation_key
    m = _FILENAME.match(result.file)
    if m:
        first = m.group("authors").split(" et al.")[0].split(" and ")[0]
        return f"{first} {m.group('year')}"
    return result.file[:30]


def year_of(result: OverviewResult) -> int | None:
    if result.year:
        return result.year
    m = _FILENAME.match(result.file)
    return int(m.group("year")) if m else None


def primary(results: Iterable[OverviewResult]) -> list[OverviewResult]:
    return sorted(
        (r for r in results if r.record.paper_kind == "primary research"),
        key=lambda r: (year_of(r) or 0, label(r)),
    )


def _cell(papers: list[OverviewResult]) -> str:
    if not papers:
        return "–"
    return f"**{len(papers)}** ({', '.join(label(p) for p in papers)})"


def _escape(cell: str) -> str:
    """Keep model-written text from breaking a markdown table row."""
    return " ".join(str(cell).split()).replace("|", "\\|")


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_escape(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _group(results: list[OverviewResult], key: Callable[[OverviewResult], Iterable[str]]):
    groups: dict[str, list[OverviewResult]] = defaultdict(list)
    for r in results:
        for k in key(r):
            groups[k].append(r)
    return groups


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def framing_table(results: list[OverviewResult]) -> str:
    """Design (analytic / learned / both) × controller interface."""
    rows = []
    for design in _DESIGNS:
        row = [design]
        for interface in _INTERFACES:
            cell = [
                r
                for r in results
                if r.derived["design"] == design and r.record.interface == interface
            ]
            row.append(_cell(cell))
        rows.append(row)
    other = [
        r
        for r in results
        if r.derived["design"] not in _DESIGNS or r.record.interface not in _INTERFACES
    ]
    table = _table(["design \\ interface", *_INTERFACES], rows)
    return (
        table
        + f"\n\nNot in the grid (no actuation, interface not reported, or design not determinable): {_cell(other)}"
    )


def chooser_table(results: list[OverviewResult]) -> str:
    """One row per training signal + update mechanism pair found in the corpus."""
    signal_long = {v: k for k, v in SIGNAL_SHORT.items()}
    mechanism_long = {v: k for k, v in MECHANISM_SHORT.items()}
    groups = _group(results, lambda r: r.derived["learning_pairs"])
    rows = []
    for pair, papers in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        signal, _, mechanism = pair.partition("+")
        regimes = Counter()
        interfaces = Counter()
        for r in papers:
            for c in r.record.components:
                if learning_pair(c) == pair and c.regime != "not applicable":
                    regimes[c.regime] += 1
            interfaces[r.record.interface] += 1
        rows.append(
            [
                f"`{pair}`",
                signal_long.get(signal, signal),
                mechanism_long.get(mechanism, mechanism),
                ", ".join(f"{k} {v}" for k, v in regimes.most_common()),
                ", ".join(f"{k} {v}" for k, v in interfaces.most_common()),
                _cell(papers),
            ]
        )
    return _table(
        ["approach", "training signal", "update mechanism", "regime", "interface", "papers"], rows
    )


def objectives_table(results: list[OverviewResult]) -> str:
    """Control objective × design."""
    groups = _group(results, lambda r: r.record.objectives or ["(none)"])
    rows = []
    for objective, papers in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        rows.append(
            [objective] + [_cell([p for p in papers if p.derived["design"] == d]) for d in _DESIGNS]
        )
    return _table(["objective \\ design", *_DESIGNS], rows)


def analytic_table(results: list[OverviewResult]) -> str:
    """Analytic method × objective, for the closed-form section."""
    groups = _group(results, lambda r: r.record.analytic_methods)
    rows = [
        [
            m,
            _cell(p),
            ", ".join(
                f"{k} {v}"
                for k, v in Counter(o for x in p for o in x.record.objectives).most_common()
            ),
        ]
        for m, p in sorted(groups.items(), key=lambda kv: -len(kv[1]))
    ]
    return _table(["analytic method", "papers", "objectives (counts)"], rows)


def hardware_table(results: list[OverviewResult]) -> str:
    groups = _group(results, lambda r: [r.record.platform])
    rows = []
    for platform, papers in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        rows.append(
            [
                platform,
                _cell(papers),
                str(sum(p.record.hardware_in_loop for p in papers)),
                str(sum(p.record.platform_coverage == "partial" for p in papers)),
                str(sum(p.record.metrics.energy == "measured" for p in papers)),
            ]
        )
    return _table(["platform", "papers", "in closed loop", "partial", "energy measured"], rows)


def reporting_table(results: list[OverviewResult]) -> str:
    """How many papers report each metric, at each evidence level."""
    n = len(results) or 1
    rows = []
    for metric in SCORED_METRICS:
        counts = Counter(getattr(r.record.metrics, metric) for r in results)
        rows.append(
            [metric, ", ".join(f"{k}: {v} ({100 * v / n:.0f}%)" for k, v in counts.most_common())]
        )
    strong = [
        r
        for r in results
        if r.record.metrics.energy == "measured" or r.record.metrics.latency == "measured"
    ]
    return (
        _table(["metric", "papers by level"], rows)
        + f"\n\nMeasured energy or latency: {_cell(strong)}"
    )


def trend_table(results: list[OverviewResult]) -> str:
    rows = []
    for lo, hi, name in _YEAR_BINS:
        papers = [r for r in results if (y := year_of(r)) is not None and lo <= y <= hi]
        designs = Counter(r.derived["design"] for r in papers)
        chips = sum(r.record.platform.startswith(("neuromorphic chip", "FPGA")) for r in papers)
        rows.append(
            [name, str(len(papers))] + [str(designs.get(d, 0)) for d in _DESIGNS] + [str(chips)]
        )
    return _table(["years", "papers", *_DESIGNS, "on chip/FPGA"], rows)


def paper_rows(results: list[OverviewResult]) -> tuple[list[str], list[list[str]]]:
    header = [
        "paper",
        "year",
        "task",
        "spiking roles",
        "control",
        "design",
        "interface",
        "learning",
        "regimes",
        "adapts online",
        "platform",
        "in loop",
        "energy",
        "latency",
    ]
    rows = []
    for r in results:
        rec = r.record
        rows.append(
            [
                label(r),
                str(year_of(r) or ""),
                rec.task,
                ", ".join(rec.spiking_roles),
                rec.control_level,
                r.derived["design"],
                rec.interface,
                ", ".join(r.derived["learning_pairs"]),
                ", ".join(r.derived["regimes"]),
                "yes" if r.derived["adapts_online"] else "",
                rec.platform + (f" ({rec.platform_name})" if rec.platform_name else ""),
                "yes" if rec.hardware_in_loop else "",
                rec.metrics.energy,
                rec.metrics.latency,
            ]
        )
    return header, rows


def disagreements(
    results: list[OverviewResult], second: list[OverviewResult]
) -> dict[str, list[str]]:
    """Per paper (sha256), the scored fields on which a second extraction run differs.

    Labels that two different models agree on are far more often right than the
    ones they disagree on (see notes/overview/), so these are the fields to
    check first.
    """
    others = {r.sha256: r for r in second}
    out = {}
    for r in results:
        other = others.get(r.sha256)
        if other is None:
            continue
        mine, theirs = comparable_facts(r.record), comparable_facts(other.record)
        fields = [field for field in mine if mine[field] != theirs[field]]
        if fields:
            out[r.sha256] = fields
    return out


def review_flags(
    results: list[OverviewResult], disagree: dict[str, list[str]] | None = None
) -> str:
    """Records worth a manual check: model disagreements, invented quotes, rule fixes, notes."""
    disagree = disagree or {}
    lines = []
    for r in sorted(results, key=label):
        flags = []
        if r.sha256 in disagree:
            flags.append("second run disagrees on: " + ", ".join(disagree[r.sha256]))
        if r.record.paper_kind != "primary research":
            flags.append(f"paper kind: {r.record.paper_kind} (not counted)")
        if r.missing_quotes:
            flags.append("quote not in text: " + ", ".join(r.missing_quotes))
        if r.normalized:
            flags.append("fixed by consistency rules: " + "; ".join(r.normalized))
        if r.record.notes:
            flags.append("note: " + r.record.notes)
        if flags:
            lines.append(f"- **{label(r)}** — " + " · ".join(flags))
    return "\n".join(lines) or "None."


def render_report(results: list[OverviewResult], second: list[OverviewResult] | None = None) -> str:
    """The full markdown report; ``second`` (another model's records) adds disagreement flags."""
    papers = primary(results)
    disagree = disagreements(results, second) if second else {}
    models = Counter(r.model for r in results)
    header, rows = paper_rows(papers)
    sections = [
        "# Literature overview: spiking neural networks for control",
        f"{len(papers)} primary papers (of {len(results)} records). Extraction model(s): "
        + ", ".join(f"`{m}` ({n})" for m, n in models.most_common())
        + ". Labels are model-extracted: check the flagged records before citing numbers."
        + (
            f" A second run (`{second[0].model}`) disagrees on {sum(map(len, disagree.values()))}"
            f" labels in {len(disagree)} papers; see section 9."
            if second
            else ""
        ),
        "## 1. Design × interface (the 2×2 framing)",
        framing_table(papers),
        "## 2. Learning approaches (training signal + update mechanism)",
        chooser_table(papers),
        "## 3. Control objectives by design",
        objectives_table(papers),
        "## 4. Analytic methods",
        analytic_table(papers),
        "## 5. Hardware and deployment",
        hardware_table(papers),
        "## 6. Reporting practice",
        reporting_table(papers),
        "## 7. Over time",
        trend_table(papers),
        "## 8. Per paper",
        _table(header, rows),
        "## 9. Records to check by hand",
        review_flags(results, disagree),
    ]
    return "\n\n".join(sections) + "\n"


def papers_csv(results: list[OverviewResult]) -> str:
    header, rows = paper_rows(primary(results))
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue()
