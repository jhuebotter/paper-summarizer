"""Deterministic quality metrics for one validated summary.

Rates are ``Rate(n, of)``.  When nothing is countable (``of == 0``), the value
is ``None`` ("n/a"), never a perfect score.
"""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from summarizer.models import PaperSummary, SummaryPart1Primary, SummaryPart1Synthesis
from summarizer.renderer import _WORD_LIMITS, part1_prose

_NGRAM = 3
_NEAR_THRESHOLD = 0.7  # share of a quote's word 3-grams found in the paper


@dataclass(frozen=True)
class Rate:
    n: int
    of: int

    @property
    def value(self) -> float | None:
        return self.n / self.of if self.of else None

    def as_dict(self) -> dict:
        return {"n": self.n, "of": self.of, "value": self.value}


# ---------------------------------------------------------------------------
# Quote faithfulness
# ---------------------------------------------------------------------------

_DASHES = dict.fromkeys(map(ord, "-\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), " ")
_QUOTES = str.maketrans(
    {"\u2018": "'", "\u2019": "'", "\u201c": "", "\u201d": "", '"': "", "\u00ad": ""}
)
_LINE_BREAK_HYPHEN = re.compile(r"[-\u2010\u2011\u00ad]\s*\n\s*")
_ELLIPSIS = re.compile(r"\[\s*(?:\.\.\.|…)\s*\]|\.\.\.|…")


def normalize_text(text: str) -> str:
    """Normalize for quote matching.

    Applies NFKC (ligatures), joins words hyphenated across line breaks, turns
    other dashes into spaces, drops double quotes and markdown emphasis,
    collapses whitespace and casefolds.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _LINE_BREAK_HYPHEN.sub("", text).translate(_DASHES).translate(_QUOTES)
    text = re.sub(r"[*_`]+", "", text)
    return re.sub(r"\s+", " ", text).casefold().strip()


def _ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


class _PaperText:
    """Normalized paper text with lazily built word n-gram sets."""

    def __init__(self, text: str) -> None:
        self.norm = normalize_text(text)
        self._words = self.norm.split()
        self._grams: dict[int, set[tuple[str, ...]]] = {}

    def ngrams(self, n: int) -> set[tuple[str, ...]]:
        if n not in self._grams:
            self._grams[n] = _ngrams(self._words, n)
        return self._grams[n]


def quote_status(quote: str, paper: _PaperText) -> str:
    """Classify one quote as ``verbatim``, ``near`` or ``not_found``.

    Ellipses split a quote into fragments that must each match.
    """
    fragments = [normalize_text(f) for f in _ELLIPSIS.split(quote)]
    fragments = [f for f in fragments if f]
    if not fragments:
        return "not_found"
    if all(f in paper.norm for f in fragments):
        return "verbatim"
    for fragment in fragments:
        words = fragment.split()
        n = min(_NGRAM, len(words))
        grams = _ngrams(words, n)
        if len(grams & paper.ngrams(n)) / len(grams) < _NEAR_THRESHOLD:
            return "not_found"
    return "near"


def quote_faithfulness(summary: PaperSummary, paper_text: str) -> dict:
    """Check every citable-snippet quote against the text the model was given."""
    quotes = [s.quote for s in getattr(summary.part1, "citable_snippets", []) if s.quote]
    paper = _PaperText(paper_text)
    statuses = [quote_status(q, paper) for q in quotes]
    counts = Counter(statuses)
    return {
        "verbatim": counts["verbatim"],
        "near": counts["near"],
        "not_found": counts["not_found"],
        "total": len(quotes),
        "not_found_quotes": [
            q for q, st in zip(quotes, statuses, strict=True) if st == "not_found"
        ],
    }


# ---------------------------------------------------------------------------
# Prose checks
# ---------------------------------------------------------------------------

_ABBREVIATIONS = re.compile(
    r"\b(Secs?|Figs?|Tbls?|Tabs?|Eqs?|Apps?|Refs?|No|approx|vs|al|e\.g|i\.e|cf)\.", re.I
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])")
_ANCHOR = re.compile(r"\(?\s*Source:[^)]*\)?", re.I)
_NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?")
# A number right after a capitalized word mid-sentence is a name ("Loihi 2", "Table 3").
_NAMED_NUMBER = re.compile(r"(?<=\S\s)[A-Z][\w-]*\s+\d+(?:\.\d+)?\b")
_YEAR = re.compile(r"^(19|20)\d{2}$")
_QUOTED = re.compile(r"\"[^\"]*\"|“[^”]*”")
_FIRST_PERSON = re.compile(r"\b(?:[Ww]e|[Oo]urs?|us)\b")
_EVIDENCE_TAG = re.compile(r"\((Measured|Reported|Claimed|Attributed)\b")


def split_sentences(text: str) -> list[str]:
    protected = _ABBREVIATIONS.sub(lambda m: m.group(0).replace(".", "\0"), text)
    return [s.replace("\0", ".").strip() for s in _SENTENCE_END.split(protected) if s.strip()]


def _needs_anchor(sentence: str) -> bool:
    stripped = _NAMED_NUMBER.sub(" ", _ANCHOR.sub(" ", sentence))
    return any(not _YEAR.match(num) for num in _NUMBER.findall(stripped))


def anchor_coverage(texts: list[str]) -> Rate:
    """Share of number-bearing sentences with a ``Source:`` anchor in or right after them.

    Years and numbers that belong to names ("Loihi 2") don't count as numbers.
    """
    needing = covered = 0
    for text in texts:
        sentences = split_sentences(text)
        for i, sentence in enumerate(sentences):
            if not _needs_anchor(sentence):
                continue
            needing += 1
            window = sentence + " " + (sentences[i + 1] if i + 1 < len(sentences) else "")
            covered += "source:" in window.casefold()
    return Rate(covered, needing)


def first_person_count(texts: list[str]) -> tuple[int, int]:
    """Return (first-person-plural uses outside quoted segments, words checked)."""
    unquoted = [_QUOTED.sub(" ", t) for t in texts]
    return (
        sum(len(_FIRST_PERSON.findall(t)) for t in unquoted),
        sum(len(t.split()) for t in unquoted),
    )


def evidence_tags(summary: PaperSummary) -> Rate:
    """Share of notable findings with exactly one allowed evidence tag."""
    part1 = summary.part1
    findings = getattr(part1, "notable_findings", [])
    allowed = {"Measured", "Reported", "Claimed", "Attributed"}
    if isinstance(part1, SummaryPart1Synthesis):
        allowed.discard("Measured")
    valid = 0
    for finding in findings:
        tags = _EVIDENCE_TAG.findall(finding)
        valid += len(tags) == 1 and tags[0] in allowed
    return Rate(valid, len(findings))


def _analytic_texts(summary: PaperSummary) -> list[str]:
    """Prose written by the model (not verbatim quotes)."""
    part1 = summary.part1
    if not isinstance(part1, SummaryPart1Primary | SummaryPart1Synthesis):
        return []
    texts = part1_prose(part1) + list(part1.notable_findings)
    texts += [s.cite_for for s in part1.citable_snippets]
    for items in part1.open_problems_future_directions.model_dump().values():
        texts += items
    if summary.part2 is not None:
        texts += list(summary.part2.model_dump().values())
    return texts


def compute_metrics(summary: PaperSummary, paper_text: str) -> dict:
    """All per-paper metrics; empty for non-research documents."""
    part1 = summary.part1
    if not isinstance(part1, SummaryPart1Primary | SummaryPart1Synthesis):
        return {}
    texts = _analytic_texts(summary)
    voice, words = first_person_count(texts)
    prose_words = sum(len(t.split()) for t in part1_prose(part1))
    limit = _WORD_LIMITS[part1.paper_type]
    return {
        "quotes": quote_faithfulness(summary, paper_text),
        "anchors": anchor_coverage(texts).as_dict(),
        "first_person": {"count": voice, "words": words},
        "word_budget": {"words": prose_words, "limit": limit, "ratio": prose_words / limit},
        "evidence_tags": evidence_tags(summary).as_dict(),
    }


def duplicate_keys(citation_keys: list[str]) -> list[str]:
    """Citation keys that occur more than once."""
    return sorted(k for k, n in Counter(citation_keys).items() if n > 1)
