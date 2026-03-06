"""LLM client setup and inference — wraps the openai SDK.

Supports any OpenAI-compatible backend: LM Studio (local) or OpenRouter
(cloud).  The ``create_client`` factory handles API-key resolution and
injects the extra headers required by OpenRouter when the base URL matches.

The public interface is ``LMStudioClient.complete(prompt)`` returning an
object with ``.text`` and ``.usage`` attributes, keeping all call sites and
mocks stable.
"""

import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

import openai as _openai

from summarizer.models import Config, LLMError

logger = logging.getLogger(__name__)

_MAX_TRANSIENT_RETRIES = 2

# ---------------------------------------------------------------------------
# Cost & usage data structures
# ---------------------------------------------------------------------------


@dataclass
class ModelPricing:
    """USD cost per token / per request (0.0 = free or unknown)."""

    prompt: float = 0.0            # per input token
    completion: float = 0.0        # per output token
    reasoning: float = 0.0         # per reasoning token
    request: float = 0.0           # flat per-request fee
    context_length: int = 0        # max context in tokens (informational)


@dataclass
class UsageStats:
    """Token counts returned by one completion call."""

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0


class CostAccumulator:
    """Thread-safe running total of tokens and USD cost across all calls."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_cost: float = 0.0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_reasoning_tokens: int = 0

    def add(self, usage: UsageStats, cost: float) -> None:
        with self._lock:
            self.total_cost += cost
            self.total_input_tokens += usage.input_tokens
            self.total_output_tokens += usage.output_tokens
            self.total_reasoning_tokens += usage.reasoning_tokens


# ---------------------------------------------------------------------------
# Client wrapper
# ---------------------------------------------------------------------------


class _CompletionResponse:
    """Thin wrapper presenting an openai chat response as ``response.text``."""

    __slots__ = ("text", "usage")

    def __init__(self, text: str, usage: "UsageStats | None" = None) -> None:
        self.text = text
        self.usage = usage


class LMStudioClient:
    """OpenAI-compatible client for LM Studio or OpenRouter.

    Wraps ``openai.OpenAI`` so that the model name is stored at construction
    time and call sites use ``client.complete(prompt)``.

    Attributes:
        model:   The model identifier passed to every completion request.
        pricing: USD cost rates for this model (zero by default).
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "lm-studio",
        extra_headers: dict | None = None,
        timeout_s: int = 120,
        max_output_tokens: int | None = None,
        pricing: ModelPricing | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.timeout_s = timeout_s
        self.max_output_tokens = max_output_tokens
        self.pricing = pricing or ModelPricing()
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
        usage = _extract_usage(response)
        return _CompletionResponse(
            text=response.choices[0].message.content,
            usage=usage,
        )


# ---------------------------------------------------------------------------
# Pricing helpers
# ---------------------------------------------------------------------------


def fetch_model_pricing(model_id: str, api_key: str, base_url: str) -> ModelPricing:
    """Fetch pricing for ``model_id`` from the OpenRouter Models API.

    Returns a zero ``ModelPricing`` (and logs a WARNING) if the request fails
    or the model is not found in the response.

    Args:
        model_id: Model identifier, e.g. ``"openai/gpt-4o"``.
        api_key:  OpenRouter API key for the Authorization header.
        base_url: Base URL of the API, e.g. ``"https://openrouter.ai/api/v1"``.
    """
    # Derive models endpoint from base_url (strip trailing path components)
    parsed = urllib.parse.urlparse(base_url)
    models_url = f"{parsed.scheme}://{parsed.netloc}/api/v1/models"

    req = urllib.request.Request(
        models_url,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
    except Exception as exc:
        logger.warning("Failed to fetch model pricing from %s: %s — using $0.00", models_url, exc)
        return ModelPricing()

    data = body.get("data", [])
    model_info = next((m for m in data if m.get("id") == model_id), None)
    if model_info is None:
        logger.warning(
            "Model %r not found in OpenRouter models list — using $0.00 pricing", model_id
        )
        return ModelPricing()

    p = model_info.get("pricing", {})
    context_length = model_info.get("context_length", 0)

    def _f(key: str) -> float:
        val = p.get(key, "0")
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    pricing = ModelPricing(
        prompt=_f("prompt"),
        completion=_f("completion"),
        reasoning=_f("internal_reasoning"),
        request=_f("request"),
        context_length=int(context_length) if context_length else 0,
    )
    logger.info(
        "Model pricing fetched: %s  in=$%.2e  out=$%.2e  reason=$%.2e  ctx=%d",
        model_id,
        pricing.prompt,
        pricing.completion,
        pricing.reasoning,
        pricing.context_length,
    )
    return pricing


def _extract_usage(response) -> "UsageStats | None":
    """Extract token counts from an OpenAI SDK response object.

    Returns ``None`` if ``response.usage`` is absent or ``None``.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    input_tokens: int = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens: int = getattr(usage, "completion_tokens", 0) or 0

    reasoning_tokens = 0
    details = getattr(usage, "completion_tokens_details", None)
    if details is not None:
        raw = getattr(details, "reasoning_tokens", None)
        if raw is not None:
            reasoning_tokens = int(raw)

    return UsageStats(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def _calculate_cost(usage: "UsageStats | None", pricing: ModelPricing) -> float:
    """Compute USD cost from token counts and pricing rates.

    When ``usage`` is ``None``, only the flat ``request`` fee is applied.
    """
    cost = pricing.request
    if usage is not None:
        cost += (
            usage.input_tokens * pricing.prompt
            + usage.output_tokens * pricing.completion
            + usage.reasoning_tokens * pricing.reasoning
        )
    return cost


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
    contains ``"openrouter.ai"``.  Pricing is fetched from the OpenRouter
    Models API for remote backends; local backends use zero pricing.
    """
    api_key = config.api_key or os.environ.get("LLM_API_KEY") or "lm-studio"

    extra_headers: dict = {}
    pricing: ModelPricing | None = None

    if "openrouter.ai" in config.base_url:
        extra_headers = {
            "HTTP-Referer": "https://github.com/agent-paper",
            "X-Title": "agent-paper",
        }
        pricing = fetch_model_pricing(config.model, api_key, config.base_url)

    return LMStudioClient(
        model=config.model,
        base_url=config.base_url,
        api_key=api_key,
        extra_headers=extra_headers,
        timeout_s=config.timeout_s,
        max_output_tokens=config.max_output_tokens,
        pricing=pricing,
    )


def call_llm(
    client: LMStudioClient,
    prompt: str,
    accumulator: "CostAccumulator | None" = None,
) -> dict:
    """Send a prompt to the LLM and return the parsed JSON response.

    Logs per-call token counts and USD cost.  When ``accumulator`` is
    provided, updates the running totals.

    Raises:
        LLMError: if the LLM call fails or the response is not valid JSON.
    """
    logger.info("Calling LLM  model=%s  backend=%s", client.model, client.base_url)
    logger.info("Awaiting response...")
    t0 = time.monotonic()
    completion = _complete_with_retries(client, prompt)
    elapsed = time.monotonic() - t0

    usage = completion.usage
    cost = _calculate_cost(usage, client.pricing)

    if usage is not None:
        logger.info(
            "Response received (%.1fs, %s chars, in=%d out=%d reason=%d tokens, cost=$%.6f)",
            elapsed,
            f"{len(completion.text):,}",
            usage.input_tokens,
            usage.output_tokens,
            usage.reasoning_tokens,
            cost,
        )
    else:
        logger.info(
            "Response received (%.1fs, %s chars, in=0 out=0 reason=0 tokens, cost=$%.6f)",
            elapsed,
            f"{len(completion.text):,}",
            cost,
        )

    if accumulator is not None:
        if usage is not None:
            accumulator.add(usage, cost)
        else:
            # Only the flat request fee applies; use zero usage stats
            accumulator.add(UsageStats(), cost)

    try:
        return _extract_json(completion.text)
    except LLMError as parse_exc:
        logger.warning("Initial JSON parse failed; running one syntax-repair retry")
        try:
            repaired = _repair_json_once(client, completion.text, accumulator=accumulator)
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


def _repair_json_once(
    client: LMStudioClient,
    bad_text: str,
    accumulator: "CostAccumulator | None" = None,
) -> str:
    """Attempt one JSON syntax repair call and return repaired text.

    The model is instructed to preserve meaning and output valid JSON only.
    Tokens and cost are logged and added to ``accumulator`` when provided.
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
    usage = response.usage
    cost = _calculate_cost(usage, client.pricing)
    if usage is not None:
        logger.info(
            "JSON repair call (in=%d out=%d reason=%d tokens, cost=$%.6f)",
            usage.input_tokens,
            usage.output_tokens,
            usage.reasoning_tokens,
            cost,
        )
    if accumulator is not None:
        accumulator.add(usage if usage is not None else UsageStats(), cost)
    return response.text


def _complete_with_retries(client: LMStudioClient, prompt: str) -> _CompletionResponse:
    """Run one completion with retry/backoff on transient 429/5xx errors."""
    attempts = _MAX_TRANSIENT_RETRIES + 1
    for attempt in range(1, attempts + 1):
        try:
            return client.complete(prompt)
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
