"""Typed classification from a decision model.

Decision models (TypeSafe's Jev, served by OpenRouter) answer typed questions
with calibrated probabilities instead of generating text.  Given the paper text, one ``choice`` question per single-label
``Classification`` field is asked; the label descriptions are read from
``snn-extraction-fields.md``, so the LLM and the decider work from the same
definitions.  The answers are stored next to the LLM's labels, not instead of
them.
"""

import functools
import logging
import re
from pathlib import Path

import openai

from summarizer.llm import LLMClient, with_retries
from summarizer.models import Classification, Decision, labels

logger = logging.getLogger(__name__)

DEFAULT_DECIDER = "typesafe/jev-1.13-20260917"
# Jev's context is 32k tokens including the questions; dense pypdf text can take
# 2-3 characters per token, so over-long requests are retried with less text.
DECIDER_MAX_CHARS = 60_000
_SHRINK = 0.6

#: field -> (heading in snn-extraction-fields.md, question)
QUESTIONS = {
    "inference_hardware": (
        "Controller hardware (inference)",
        "Where does the paper's own controller network execute at runtime (not the task environment)?",
    ),
    "architecture": (
        "Network architecture",
        "Is the paper's own controller network fully spiking or hybrid?",
    ),
    "credit_assignment": (
        "Credit assignment scope",
        "How is credit assigned when the paper's own network is trained?",
    ),
    "learning_regime": (
        "Online vs. offline",
        "Are the paper's own network weights updated during task execution (online) or in a "
        "separate training phase (offline)?",
    ),
}
_NOT_REPORTED = "The paper does not say."
_BULLET = re.compile(r"^- \*\*(.+?)\*\* — (.+)$", re.MULTILINE)
_PINNED = re.compile(r"-\d{8}$")  # a dated snapshot, e.g. typesafe/jev-1.13-20260917


@functools.cache
def label_descriptions(references_dir: Path) -> dict[str, dict[str, str]]:
    """``field -> label -> description`` from the reference's label bullets.

    Raises:
        ValueError: if a label of a ``Classification`` field has no description.
    """
    text = (references_dir / "snn-extraction-fields.md").read_text(encoding="utf-8")
    out = {}
    for field, (heading, _) in QUESTIONS.items():
        if f"## {heading}\n" not in text:
            raise ValueError(f"No '## {heading}' section in snn-extraction-fields.md")
        section = text.split(f"## {heading}\n", 1)[1].split("\n## ", 1)[0]
        bullets = dict(_BULLET.findall(section))
        bullets.setdefault("not reported", _NOT_REPORTED)
        allowed = labels(Classification.model_fields[field].annotation)
        missing = [label for label in allowed if label not in bullets]
        if missing:
            raise ValueError(f"No description for {field} labels {missing} under '## {heading}'")
        if len({_key(label) for label in allowed}) < len(allowed):
            raise ValueError(f"Two {field} labels map to the same question key")
        out[field] = {label: bullets[label] for label in allowed}
    return out


def _key(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def decide(
    client: LLMClient,
    model: str,
    title: str,
    paper_text: str,
    descriptions: dict[str, dict[str, str]],
) -> tuple[dict[str, Decision], float, str | None]:
    """Ask the decision model; return ``(decisions, cost_usd, served model)``."""
    questions = {
        field: {
            "type": "choice",
            "instructions": QUESTIONS[field][1],
            "criteria": {_key(label): text for label, text in labels_.items()},
        }
        for field, labels_ in descriptions.items()
    }
    text = paper_text[:DECIDER_MAX_CHARS]
    while True:
        body = {"model": model, "state": {"title": title, "paper": text}, "questions": questions}
        try:
            reply = with_retries(lambda body=body: client.decide(body), client)
            break
        except Exception as exc:
            cause = exc.__cause__
            too_long = isinstance(cause, openai.BadRequestError) and "max_tokens" in str(cause)
            if not too_long or len(text) < 5_000:
                raise
            logger.info("Paper too long for the decision model; retrying with less text")
            text = text[: int(len(text) * _SHRINK)]
    answers = reply.get("answers") if isinstance(reply, dict) else None
    if not isinstance(answers, dict) or any(f not in answers for f in descriptions):
        raise ValueError(f"Decision model reply without answers: {str(reply)[:200]}")
    served = reply.get("model")
    if _PINNED.search(model) and served and served != model:
        logger.warning("Decision model %s answered as %s", model, served)
    decisions = {}
    for field, labels_ in descriptions.items():
        answer = answers[field]
        by_key = {_key(label): label for label in labels_}
        decisions[field] = Decision(
            label=by_key[answer["choice"]],
            probabilities={by_key[k]: p for k, p in answer.get("probabilities", {}).items()},
            confidence=answer.get("confidence"),
        )
    return decisions, float(reply.get("usage", {}).get("cost") or 0.0), served
