"""LLM client setup and inference — wraps the openai SDK.

Supports any OpenAI-compatible backend: LM Studio (local) or OpenRouter
(cloud).  The ``create_client`` factory handles API-key resolution and
injects the extra headers required by OpenRouter when the base URL matches.

The public interface is ``LMStudioClient.complete(prompt)`` returning an
object with a ``.text`` attribute, keeping all call sites and mocks stable.
"""

import json
import logging
import os
import re
import time

import openai as _openai

from summarizer.models import Config, LLMError

logger = logging.getLogger(__name__)

_MAX_TRANSIENT_RETRIES = 2


# ---------------------------------------------------------------------------
# Client wrapper
# ---------------------------------------------------------------------------


class _CompletionResponse:
    """Thin wrapper presenting an openai chat response as ``response.text``."""

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text


class LMStudioClient:
    """OpenAI-compatible client for LM Studio or OpenRouter.

    Wraps ``openai.OpenAI`` so that the model name is stored at construction
    time and call sites use ``client.complete(prompt)``.

    Attributes:
        model: The model identifier passed to every completion request.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "lm-studio",
        extra_headers: dict | None = None,
        timeout_s: int = 120,
        max_output_tokens: int | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.timeout_s = timeout_s
        self.max_output_tokens = max_output_tokens
        self._client = _openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers=extra_headers or {},
        )

    def complete(self, prompt: str) -> _CompletionResponse:
        """Send a chat completion request and return the model's reply."""
        kwargs: dict = dict(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            timeout=self.timeout_s,
        )
        if self.max_output_tokens is not None:
            kwargs["max_tokens"] = self.max_output_tokens
        response = self._client.chat.completions.create(**kwargs)
        return _CompletionResponse(text=response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def create_client(config: Config) -> LMStudioClient:
    """Create a client from configuration, resolving API key and headers.

    API key resolution order:
        1. ``config.api_key`` (explicit)
        2. ``LLM_API_KEY`` environment variable
        3. ``"lm-studio"`` fallback (LM Studio ignores the value)

    OpenRouter headers are injected automatically when ``config.base_url``
    contains ``"openrouter.ai"``.
    """
    api_key = config.api_key or os.environ.get("LLM_API_KEY") or "lm-studio"

    extra_headers: dict = {}
    if "openrouter.ai" in config.base_url:
        extra_headers = {
            "HTTP-Referer": "https://github.com/agent-paper",
            "X-Title": "agent-paper",
        }

    return LMStudioClient(
        model=config.model,
        base_url=config.base_url,
        api_key=api_key,
        extra_headers=extra_headers,
        timeout_s=config.timeout_s,
        max_output_tokens=config.max_output_tokens,
    )


def call_llm(client: LMStudioClient, prompt: str) -> dict:
    """Send a prompt to the LLM and return the parsed JSON response.

    Raises:
        LLMError: if the LLM call fails or the response is not valid JSON.
    """
    logger.info("Calling LLM  model=%s  backend=%s", client.model, client.base_url)
    logger.info("Awaiting response...")
    t0 = time.monotonic()
    text = _complete_with_retries(client, prompt)
    elapsed = time.monotonic() - t0
    logger.info("Response received (%.1fs, %s chars)", elapsed, f"{len(text):,}")

    try:
        return _extract_json(text)
    except LLMError as parse_exc:
        logger.warning("Initial JSON parse failed; running one syntax-repair retry")
        try:
            repaired = _repair_json_once(client, text)
        except Exception:
            raise parse_exc
        try:
            return _extract_json(repaired)
        except LLMError:
            raise parse_exc


def _extract_json(text: str) -> dict:
    """Extract a JSON object from text that may include markdown code fences.

    Locates the first ``{`` and last ``}`` to find the JSON object boundary,
    then parses the substring.

    Raises:
        LLMError: if no JSON object is found or if ``json.loads`` fails.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise LLMError(f"No JSON object found in LLM response: {text[:200]!r}")
    json_str = text[start : end + 1]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise LLMError(f"Failed to parse LLM response as JSON: {e}") from e


def _repair_json_once(client: LMStudioClient, bad_text: str) -> str:
    """Attempt one JSON syntax repair call and return repaired text.

    The model is instructed to preserve meaning and output valid JSON only.
    """
    repair_prompt = (
        "You are a JSON repair assistant.\n"
        "Task: Repair the JSON syntax in the payload below.\n"
        "Rules:\n"
        "1) Output valid JSON only (no markdown, no comments, no explanation).\n"
        "2) Preserve all original keys and values whenever possible.\n"
        "3) Fix only syntax/escaping/quoting/comma/bracket issues.\n"
        "4) Do not invent new facts.\n\n"
        "Payload to repair:\n"
        f"{bad_text}"
    )
    response = client.complete(repair_prompt)
    return response.text


def _complete_with_retries(client: LMStudioClient, prompt: str) -> str:
    """Run one completion with retry/backoff on transient 429/5xx errors."""
    attempts = _MAX_TRANSIENT_RETRIES + 1
    for attempt in range(1, attempts + 1):
        try:
            response = client.complete(prompt)
            return response.text
        except Exception as exc:
            if attempt >= attempts or not _is_retryable_status_error(exc):
                raise LLMError(f"LLM call failed: {exc}") from exc

            delay_s = _retry_delay_seconds(attempt)
            logger.warning(
                "Transient LLM error on attempt %d/%d (%s); retrying in %.1fs",
                attempt,
                attempts,
                exc,
                delay_s,
            )
            time.sleep(delay_s)

    raise LLMError("LLM call failed after retries")


def _retry_delay_seconds(attempt: int) -> float:
    """Exponential backoff delay: 1.0s, 2.0s, ..."""
    return float(2 ** (attempt - 1))


def _is_retryable_status_error(exc: Exception) -> bool:
    """Return True for transient API errors that should be retried."""
    status_code = _extract_status_code(exc)
    if status_code == 429:
        return True
    if status_code is not None and 500 <= status_code <= 599:
        return True
    return False


def _extract_status_code(exc: Exception) -> int | None:
    """Extract HTTP status code from common exception shapes or message text."""
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value

    code_attr = getattr(exc, "code", None)
    if isinstance(code_attr, int) and 100 <= code_attr <= 599:
        return code_attr

    message = str(exc)
    patterns = [
        r"Error code:\s*(\d{3})",
        r"status(?:\s*code)?\s*[:=]\s*(\d{3})",
    ]
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

    return None
