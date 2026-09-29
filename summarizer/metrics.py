"""Deterministic quality metrics for one validated summary.

Rates are ``Rate(n, of)``.  When nothing is countable (``of == 0``), the value
is ``None`` ("n/a"), never a perfect score.
"""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from summarizer.models import PaperSummary, SummaryPart1Primary, SummaryPart1Synthesis
from summarizer.renderer import WORD_LIMITS, _count_words, part1_prose

_NGRAM = 3
_NEAR_THRESHOLD = 0.7  # share of a quote's word 3-grams found in the paper
_MIN_FRAGMENT_WORDS = 3


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

_HYPHEN = re.compile(r"[-‐‑­]\s*")  # joined, incl. across line breaks
_SPACED_DASHES = dict.fromkeys(map(ord, "‒–—―−"), " ")
_NUMERIC_CITATION = re.compile(r"\[\s*\d+(?:\s*[,;–-]\s*\d+)*\s*\]")
_ELLIPSIS = re.compile(r"\[\s*(?:\.\.\.|…)\s*\]|\.\.\.|…")
_WORD = re.compile(r"\w+")


def normalize_text(text: str) -> str:
    """Normalize for quote matching.

    NFKC (ligatures, fullwidth forms), numeric citation brackets removed,
    hyphenated words joined (also across line breaks), other dashes turned into
    spaces, casefolded.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _NUMERIC_CITATION.sub(" ", text).translate(_SPACED_DASHES)
    return _HYPHEN.sub("", text).casefold()


def _tokens(text: str) -> list[str]:
    return _WORD.findall(normalize_text(text))


def _ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


class _PaperText:
    """Tokenized paper text with its word 3-grams."""

    def __init__(self, text: str) -> None:
        tokens = _tokens(text)
        self.joined = " " + " ".join(tokens) + " "
        self.ngrams = _ngrams(tokens, _NGRAM)


def quote_status(quote: str, paper: _PaperText) -> str:
    """Classify one quote as ``verbatim``, ``near`` or ``not_found``.

    Words are compared after normalization (punctuation ignored).  Ellipses split
    a quote into fragments that must appear in order; fragments shorter than
    three words can't be checked, so a quote with any of them is at best
    ``near``.
    """
    fragments = [f for f in (_tokens(part) for part in _ELLIPSIS.split(quote)) if f]
    checkable = [f for f in fragments if len(f) >= _MIN_FRAGMENT_WORDS]
    if not checkable:
        return "not_found"

    position = 0
    for fragment in checkable:
        found = paper.joined.find(" " + " ".join(fragment) + " ", position)
        if found < 0:
            break
        position = found + 1
    else:
        return "verbatim" if len(checkable) == len(fragments) else "near"

    if all(" " + " ".join(f) + " " in paper.joined for f in checkable):
        return "not_found"  # every fragment is verbatim but they're stitched out of order
    for fragment in checkable:
        grams = _ngrams(fragment, _NGRAM)
        if len(grams & paper.ngrams) / len(grams) < _NEAR_THRESHOLD:
            return "not_found"
    return "near"


def quote_faithfulness(summary: PaperSummary, paper_text: str) -> dict:
    """Check every citable-snippet quote against the text the model was given."""
    quotes = [s.quote for s in summary.part1.citable_snippets if s.quote]
    paper = _PaperText(paper_text)
    statuses = [quote_status(q, paper) for q in quotes]
    counts = Counter(statuses)
    return {
        "verbatim": counts["verbatim"],
        "near": counts["near"],
        "not_found": counts["not_found"],
        "total": len(quotes),
        "not_found_quotes": [
            q for q, status in zip(quotes, statuses, strict=True) if status == "not_found"
        ],
    }


# ---------------------------------------------------------------------------
# Prose checks
# ---------------------------------------------------------------------------

_ABBREVIATIONS = re.compile(
    r"\b(Secs?|Sects?|Figs?|Tbls?|Tabs?|Eqs?|Apps?|Algs?|Refs?|Ch|Suppl|No|pp?|approx|vs|al"
    r"|e\.g|i\.e|cf)\.",
    re.I,
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])")
_ANCHOR = re.compile(r"\(?\s*Source:[^)]*\)?", re.I)
# References and names followed by a number: "Table 3", "Phase 2", "Loihi 2".
_NAMED_NUMBER = re.compile(
    r"\b(?:Tables?|Tbls?|Tabs?|Figs?|Figures?|Secs?|Sects?|Sections?|Eqs?|Equations?|Apps?"
    r"|Appendix|Algs?|Algorithms?|Chapters?|Parts?|Phases?|Stages?|Steps?|Versions?|Levels?"
    r"|Loihi|SpiNNaker|BrainScaleS|TrueNorth)\.?\s*\d+(?:\.\d+)?",
    re.I,
)
_YEAR_MENTION = re.compile(
    r"(?:\(|,\s*|\b(?:in|since|from|until|by|al\.)\s+)(?:19|20)\d{2}\b(?:[a-z]\b)?", re.I
)
_NUMBER = re.compile(r"(?<![\w.-])\d+(?:[.,]\d+)?(?![A-Z]\b)")
_QUOTED = re.compile(r"\"[^\"]*\"|“[^”]*”|‘[^’]*’")
_FIRST_PERSON = re.compile(r"\b(?:[Ww]e|[Oo]urs?)\b|(?<!\d)(?<!\d )\bus\b")
_EVIDENCE_TAG = re.compile(r"\((Measured|Reported|Claimed|Attributed)\b")


def split_sentences(text: str) -> list[str]:
    protected = _ABBREVIATIONS.sub(lambda m: m.group(0).replace(".", "\0"), text)
    return [s.replace("\0", ".").strip() for s in _SENTENCE_END.split(protected) if s.strip()]


def _needs_anchor(sentence: str) -> bool:
    stripped = _ANCHOR.sub(" ", sentence)
    stripped = _YEAR_MENTION.sub(" ", _NAMED_NUMBER.sub(" ", stripped))
    return bool(_NUMBER.search(stripped))


def anchor_coverage(texts: list[str]) -> Rate:
    """Share of number-bearing sentences with a ``Source:`` anchor in or right after them.

    Years in citation/date context, references ("Table 3") and chip names
    ("Loihi 2") don't count as numbers; neither do numbers inside identifiers
    ("CIFAR-10", "3D").  "7-DOF" and "3x" do count.
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
    return sum(len(_FIRST_PERSON.findall(t)) for t in unquoted), _count_words(*unquoted)


def evidence_tags(summary: PaperSummary) -> Rate:
    """Share of notable findings with exactly one allowed evidence tag."""
    allowed = {"Measured", "Reported", "Claimed", "Attributed"}
    if isinstance(summary.part1, SummaryPart1Synthesis):
        allowed.discard("Measured")
    findings = summary.part1.notable_findings
    valid = 0
    for finding in findings:
        tags = _EVIDENCE_TAG.findall(finding)
        valid += len(tags) == 1 and tags[0] in allowed
    return Rate(valid, len(findings))


def _analytic_texts(summary: PaperSummary) -> list[str]:
    """Prose written by the model (not verbatim quotes)."""
    part1 = summary.part1
    texts = part1_prose(part1) + list(part1.notable_findings)
    texts += [f"{s.cite_for} (Source: {s.source})" for s in part1.citable_snippets]
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
    prose_words = _count_words(*part1_prose(part1))
    limit = WORD_LIMITS[part1.paper_type]
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
