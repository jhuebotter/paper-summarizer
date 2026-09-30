"""LLM client setup and inference — wraps the openai SDK.

Supports any OpenAI-compatible backend: LM Studio (local) or OpenRouter
(cloud).  The ``create_client`` factory handles API-key resolution and
injects the extra headers required by OpenRouter when the base URL matches.

The public interface is ``LLMClient.complete(prompt)`` returning an object
with ``.text`` and ``.usage`` attributes.

Retries live in exactly one place (``with_retries``); the SDK's own
retry loop is disabled so attempts don't multiply.  Exhausted quotas (daily
free-model cap, credits, key limits) raise ``QuotaExhausted`` instead, so runs
can stop cleanly.
"""

import json
import logging
import os
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

import openai as _openai

from summarizer.models import Config, LLMError, LLMResponse

logger = logging.getLogger(__name__)

_MAX_TRANSIENT_RETRIES = 3

# ---------------------------------------------------------------------------
# Cost & usage data structures
# ---------------------------------------------------------------------------


@dataclass
class ModelPricing:
    """USD cost per token / per request and model limits (0 / False = free or unknown)."""

    prompt: float = 0.0  # per input token
    completion: float = 0.0  # per output token
    reasoning: float = 0.0  # per reasoning token
    request: float = 0.0  # flat per-request fee
    context_length: int = 0  # max context in tokens


@dataclass
class UsageStats:
    """Token counts (and, from OpenRouter, the billed cost) of one completion."""

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cost: float | None = None  # USD actually charged, when the backend reports it


class CostAccumulator:
    """Thread-safe running totals of tokens, USD cost, completions and repairs.

    With a ``parent``, every update is also applied to it, so a per-paper
    accumulator can feed a run-wide total.
    """

    def __init__(self, parent: "CostAccumulator | None" = None) -> None:
        self._parent = parent
        self._lock = threading.Lock()
        self.total_cost: float = 0.0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_reasoning_tokens: int = 0
        self.calls: int = 0
        self.json_repairs: int = 0
        self.schema_repairs: int = 0

    def add(self, usage: UsageStats, cost: float) -> None:
        """Record one completion (including repair calls)."""
        with self._lock:
            self.calls += 1
            self.total_cost += cost
            self.total_input_tokens += usage.input_tokens
            self.total_output_tokens += usage.output_tokens
            self.total_reasoning_tokens += usage.reasoning_tokens
        if self._parent is not None:
            self._parent.add(usage, cost)

    def add_cost(self, cost: float) -> None:
        """Record spend that is not an LLM call (e.g. a decision model)."""
        with self._lock:
            self.total_cost += cost
        if self._parent is not None:
            self._parent.add_cost(cost)

    def note_json_repair(self) -> None:
        with self._lock:
            self.json_repairs += 1
        if self._parent is not None:
            self._parent.note_json_repair()

    def note_schema_repair(self) -> None:
        with self._lock:
            self.schema_repairs += 1
        if self._parent is not None:
            self._parent.note_schema_repair()


# ---------------------------------------------------------------------------
# Client wrapper
# ---------------------------------------------------------------------------


class QuotaExhausted(LLMError):
    """The backend refuses further calls for now (daily cap, credits, key limit)."""


class RejectedCompletion(LLMError):
    """A reply that was received (and billed) but is unusable; carries its usage."""

    def __init__(self, message: str, usage: "UsageStats | None") -> None:
        super().__init__(message)
        self.usage = usage


class ProviderError(RejectedCompletion):
    """The request was accepted but the provider failed; OpenRouter reports this
    as an ``error`` object in a 200 response (no choices, or ``finish_reason="error"``)."""

    def __init__(self, message: str, usage: "UsageStats | None", code: int | None) -> None:
        super().__init__(message, usage)
        self.code = code

    @property
    def retryable(self) -> bool:
        return self.code is None or self.code in (408, 429) or 500 <= self.code <= 599


def _provider_error(obj: object, fallback: str, usage: "UsageStats | None") -> ProviderError:
    extra = getattr(obj, "model_extra", None)
    error = extra.get("error") if isinstance(extra, dict) else None
    if not isinstance(error, dict):
        return ProviderError(fallback, usage, None)
    code = str(error.get("code"))
    code = int(code) if code.isdigit() else None
    return ProviderError(f"{fallback}: {error.get('message') or error} (code {code})", usage, code)


class CompletionResponse:
    """Thin wrapper presenting an openai chat response as ``response.text``."""

    __slots__ = ("text", "usage")

    def __init__(self, text: str, usage: "UsageStats | None" = None) -> None:
        self.text = text
        self.usage = usage


class LLMClient:
    """OpenAI-compatible client (OpenRouter, LM Studio, or any compatible server).

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
        response_format: dict | None = None,
        extra_body: dict | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.timeout_s = timeout_s
        self.max_output_tokens = max_output_tokens
        self.pricing = pricing or ModelPricing()
        self.response_format = response_format
        self.extra_body = extra_body
        self._client = _openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers=extra_headers or {},
            max_retries=0,  # retries are handled by with_retries
        )

    def complete(self, prompt: str) -> CompletionResponse:
        """Send a chat completion request and return the model's reply.

        Raises:
            ProviderError: if the provider failed (no choices, or a mid-response error).
            RejectedCompletion: if the reply is empty or was cut off by the
                output-token limit (``finish_reason == "length"``); a truncated
                JSON object cannot be repaired, so fail fast.
        """
        kwargs: dict = dict(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            timeout=self.timeout_s,
        )
        if self.max_output_tokens is not None:
            kwargs["max_tokens"] = self.max_output_tokens
        if self.response_format is not None:
            kwargs["response_format"] = self.response_format
        if self.extra_body is not None:
            kwargs["extra_body"] = self.extra_body
        response = self._client.chat.completions.create(**kwargs)
        usage = _extract_usage(response)
        if not getattr(response, "choices", None):
            raise _provider_error(response, "LLM response contains no choices", usage)
        choice = response.choices[0]
        if choice.finish_reason == "error":
            raise _provider_error(choice, "LLM generation failed mid-response", usage)
        if choice.finish_reason == "length":
            raise RejectedCompletion(
                "LLM output was truncated by the token limit (finish_reason=length); "
                "raise --max-output-tokens or use a model with a larger output budget",
                usage,
            )
        text = choice.message.content
        if text is None or not text.strip():
            raise RejectedCompletion(
                f"LLM returned no content (finish_reason={choice.finish_reason!r})", usage
            )
        return CompletionResponse(text=text, usage=usage)

    def decide(self, body: dict) -> dict:
        """POST a decision request in the System One wire format (``/systemone``,
        relative to the base URL) and return the JSON reply."""
        return self._client.post(
            "/systemone", body=body, cast_to=object, options={"timeout": self.timeout_s}
        )


# ---------------------------------------------------------------------------
# Pricing helpers
# ---------------------------------------------------------------------------


def is_openrouter(base_url: str) -> bool:
    host = urllib.parse.urlparse(base_url).hostname or ""
    return host == "openrouter.ai" or host.endswith(".openrouter.ai")


def fetch_openrouter_key_info(base_url: str, api_key: str) -> dict | None:
    """Return the key's usage/limit info from ``GET /api/v1/key``, or ``None``."""
    parsed = urllib.parse.urlparse(base_url)
    request = urllib.request.Request(
        f"{parsed.scheme}://{parsed.netloc}/api/v1/key",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as resp:
            data = json.loads(resp.read())["data"]
        return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.debug("Could not read OpenRouter key info: %s", exc)
        return None


#: OpenRouter routing shortcuts that are valid on any listed model but are not
#: themselves listed.  Other suffixes (notably ``:free``) are distinct model ids.
OPENROUTER_ROUTING_SUFFIXES = frozenset({"nitro", "floor", "online", "exacto"})


def openrouter_listed_id(model: str) -> str:
    """Return the id under which ``model`` appears in OpenRouter's models list."""
    base, sep, suffix = model.rpartition(":")
    return base if sep and suffix in OPENROUTER_ROUTING_SUFFIXES else model


def fetch_openrouter_models(base_url: str, api_key: str | None = None) -> list[dict] | None:
    """Return OpenRouter's model entries, or ``None`` if unavailable or malformed."""
    parsed = urllib.parse.urlparse(base_url)
    models_url = f"{parsed.scheme}://{parsed.netloc}/api/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        with urllib.request.urlopen(
            urllib.request.Request(models_url, headers=headers), timeout=10
        ) as resp:
            body = json.loads(resp.read())
        data = body["data"]
        if not isinstance(data, list):
            raise TypeError("'data' is not a list")
    except Exception as exc:
        logger.warning("Could not list models from %s: %s", models_url, exc)
        return None
    return [m for m in data if isinstance(m, dict) and isinstance(m.get("id"), str)]


def fetch_model_pricing(model_id: str, api_key: str, base_url: str) -> ModelPricing:
    """Fetch pricing for ``model_id`` from the OpenRouter Models API.

    Returns a zero ``ModelPricing`` (and logs a WARNING) if the request fails
    or the model is not found in the response.

    Args:
        model_id: Model identifier, e.g. ``"openai/gpt-4o"``.
        api_key:  OpenRouter API key for the Authorization header.
        base_url: Base URL of the API, e.g. ``"https://openrouter.ai/api/v1"``.
    """
    models = fetch_openrouter_models(base_url, api_key)
    if models is None:
        logger.warning("Model pricing unavailable — using $0.00")
        return ModelPricing()

    listed_id = openrouter_listed_id(model_id)
    model_info = next((m for m in models if m["id"] == listed_id), None)
    if model_info is None:
        logger.warning(
            "Model %r not found in OpenRouter models list — using $0.00 pricing", model_id
        )
        return ModelPricing()

    p = model_info.get("pricing")
    if not isinstance(p, dict):
        p = {}
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
        context_length=int(context_length) if isinstance(context_length, int) else 0,
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

    reasoning_tokens = _int_attr(
        getattr(usage, "completion_tokens_details", None), "reasoning_tokens"
    )
    cached_tokens = _int_attr(getattr(usage, "prompt_tokens_details", None), "cached_tokens")
    # OpenRouter adds the billed USD amount as an extra (untyped) field.
    cost = getattr(usage, "cost", None)
    if not isinstance(cost, int | float) or isinstance(cost, bool):
        cost = None

    return UsageStats(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        cached_tokens=cached_tokens,
        cost=cost,
    )


def _int_attr(obj, name: str) -> int:
    value = getattr(obj, name, None) if obj is not None else None
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _calculate_cost(usage: "UsageStats | None", pricing: ModelPricing) -> float:
    """Return the billed cost if the backend reported it, else estimate from list prices.

    When ``usage`` is ``None``, only the flat ``request`` fee is applied.
    """
    if usage is not None and usage.cost is not None:
        return float(usage.cost)
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


def create_client(config: Config) -> LLMClient:
    """Create a client from configuration, resolving API key and headers.

    API key resolution order:
        1. ``config.api_key`` (explicit)
        2. ``LLM_API_KEY`` environment variable
        3. ``"lm-studio"`` fallback (LM Studio ignores the value)

    For OpenRouter, attribution headers are added and pricing is fetched from
    its Models API; local backends use zero pricing.  With
    ``config.structured_output`` the JSON schema of ``LLMResponse`` is sent as
    ``response_format`` (and OpenRouter is told to route only to endpoints that
    support it).
    """
    api_key = config.api_key or os.environ.get("LLM_API_KEY") or "lm-studio"

    extra_headers: dict = {}
    pricing: ModelPricing | None = None
    extra_body: dict | None = None

    if is_openrouter(config.base_url):
        extra_headers = {
            "HTTP-Referer": "https://github.com/jhuebotter/paper-summarizer",
            "X-Title": "paper-summarizer",
        }
        pricing = fetch_model_pricing(config.model, api_key, config.base_url)
        if config.structured_output:
            extra_body = {"provider": {"require_parameters": True}}

    response_format = None
    if config.structured_output:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "paper_summary",
                "strict": False,
                "schema": portable_schema(LLMResponse.model_json_schema()),
            },
        }

    return LLMClient(
        model=config.model,
        base_url=config.base_url,
        api_key=api_key,
        extra_headers=extra_headers,
        timeout_s=config.timeout_s,
        max_output_tokens=config.max_output_tokens,
        pricing=pricing,
        response_format=response_format,
        extra_body=extra_body,
    )


def portable_schema(schema: dict) -> dict:
    """Inline ``$ref``s and use ``anyOf`` instead of ``oneOf``/``discriminator``.

    Many providers support only a JSON Schema subset without references or
    OpenAPI's ``discriminator`` keyword.
    """
    defs = schema.get("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].rsplit("/", 1)[-1]])
            return {
                ("anyOf" if key == "oneOf" else key): resolve(value)
                for key, value in node.items()
                if key not in ("$defs", "discriminator")
            }
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(schema)


def call_llm(
    client: LLMClient,
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
    try:
        completion = _complete_with_retries(client, prompt, accumulator)
    except RejectedCompletion as exc:
        _record(accumulator, exc.usage, _calculate_cost(exc.usage, client.pricing))
        raise
    elapsed = time.monotonic() - t0

    usage = completion.usage
    cost = _calculate_cost(usage, client.pricing)

    if usage is not None:
        logger.info(
            "Response received (%.1fs, %s chars, in=%d (cached %d) out=%d reason=%d tokens, "
            "cost=$%.6f)",
            elapsed,
            f"{len(completion.text):,}",
            usage.input_tokens,
            usage.cached_tokens,
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

    _record(accumulator, usage, cost)

    try:
        return _extract_json(completion.text)
    except LLMError as parse_exc:
        logger.warning("Initial JSON parse failed; running one syntax-repair retry")
        if accumulator is not None:
            accumulator.note_json_repair()
        logger.debug("Unparseable response (first 500 chars): %r", completion.text[:500])
        try:
            repaired = _repair_json_once(client, completion.text, accumulator=accumulator)
        except QuotaExhausted:
            raise
        except Exception:
            raise parse_exc from None
        try:
            return _extract_json(repaired)
        except LLMError:
            raise parse_exc from None


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
        return json.loads(json_str, strict=False)  # models emit raw tabs/newlines in strings
    except json.JSONDecodeError as e:
        raise LLMError(f"Failed to parse LLM response as JSON: {e}") from e


def _repair_json_once(
    client: LLMClient,
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
    try:
        response = _complete_with_retries(client, repair_prompt, accumulator)
    except RejectedCompletion as exc:
        _record(accumulator, exc.usage, _calculate_cost(exc.usage, client.pricing))
        raise
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
    _record(accumulator, usage, cost)
    return response.text


def _record(accumulator: "CostAccumulator | None", usage: "UsageStats | None", cost: float) -> None:
    if accumulator is not None:
        accumulator.add(usage if usage is not None else UsageStats(), cost)


def _complete_with_retries(
    client: LLMClient, prompt: str, accumulator: "CostAccumulator | None" = None
) -> CompletionResponse:
    return with_retries(lambda: client.complete(prompt), client, accumulator)


def with_retries(call, client: LLMClient, accumulator: "CostAccumulator | None" = None):
    """Run one backend request with retry/backoff on transient errors.

    Retried: HTTP 429 (per-minute limits), 5xx, timeouts, connection errors and
    transient provider errors reported inside a 200 response (whose usage, if
    billed, is recorded).  Exhausted quotas (daily free-model cap, credits, key
    limit) raise ``QuotaExhausted`` without retrying.
    """
    attempts = _MAX_TRANSIENT_RETRIES + 1
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except ProviderError as exc:
            if exc.code == 402:
                raise QuotaExhausted(f"Credits exhausted: {exc}") from exc
            if attempt >= attempts or not exc.retryable:
                raise
            if exc.usage is not None:
                _record(accumulator, exc.usage, _calculate_cost(exc.usage, client.pricing))
            error = exc
        except LLMError:
            raise
        except Exception as exc:
            quota = _quota_exhausted_message(exc)
            if quota:
                raise QuotaExhausted(quota) from exc
            if attempt >= attempts or not _is_retryable_status_error(exc):
                raise LLMError(f"LLM call failed: {exc}") from exc
            error = exc

        delay_s = _retry_delay_seconds(attempt, error)
        logger.warning(
            "Transient LLM error on attempt %d/%d (%s); retrying in %.1fs",
            attempt,
            attempts,
            error,
            delay_s,
        )
        time.sleep(delay_s)

    raise LLMError("LLM call failed after retries")


def _retry_delay_seconds(attempt: int, exc: Exception | None = None) -> float:
    """Wait for a short ``Retry-After`` or rate-limit reset, else jittered exponential backoff.

    OpenRouter's per-minute 429s carry the reset time (epoch ms) in the error body;
    other 429s (free models throttled upstream) back off from 5 s rather than 1 s.
    """
    response = getattr(exc, "response", None)
    try:
        retry_after = float(response.headers.get("retry-after"))
    except (AttributeError, TypeError, ValueError):
        retry_after = None
    if retry_after is None and exc is not None:
        try:
            retry_after = (
                float(_error_details(exc)[1].get("X-RateLimit-Reset")) / 1000 - time.time()
            )
        except (TypeError, ValueError):
            pass
    if retry_after is not None and 0 < retry_after <= 60:
        return retry_after + random.uniform(0, 1)  # spread workers waiting for the same reset
    base = 5 if exc is not None and _extract_status_code(exc) == 429 else 1
    return base * 2 ** (attempt - 1) * random.uniform(0.5, 1.5)


def _error_details(exc: Exception) -> tuple[str, dict]:
    """Return (error message, rate-limit headers) from an OpenAI SDK API error."""
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return str(exc), {}
    metadata = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
    headers = metadata.get("headers") if isinstance(metadata.get("headers"), dict) else {}
    return str(body.get("message") or exc), headers


def _quota_exhausted_message(exc: Exception) -> str | None:
    """Describe an exhausted quota: 402 (credits), 403 key limit, or a 429 that won't clear soon."""
    status = _extract_status_code(exc)
    if status not in (402, 403, 429):
        return None
    message, headers = _error_details(exc)
    if status == 402:
        return f"Credits exhausted: {message}"
    if status == 403:
        return f"Key limit exhausted: {message}" if "key limit" in message.lower() else None
    lowered = message.lower()
    reset_ms = headers.get("X-RateLimit-Reset")
    try:
        reset_in_s = float(reset_ms) / 1000 - time.time()
    except (TypeError, ValueError):
        reset_in_s = None
    daily = "per-day" in lowered or "per day" in lowered or "daily" in lowered
    drained = str(headers.get("X-RateLimit-Remaining")) == "0" and (reset_in_s or 0) > 600
    if not (daily or drained):
        return None
    when = ""
    if reset_in_s is not None and reset_in_s > 0:
        when = f"; resets at {time.strftime('%H:%M', time.localtime(time.time() + reset_in_s))}"
    return f"Rate limit exhausted ({message}){when}"


def _is_retryable_status_error(exc: Exception) -> bool:
    """Return True for transient API errors that should be retried."""
    # APITimeoutError is a subclass of APIConnectionError.
    if isinstance(exc, _openai.APIConnectionError):
        return True
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
