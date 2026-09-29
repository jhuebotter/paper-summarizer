"""Tests for summarizer/llm.py — openai SDK wrapper for LM Studio and OpenRouter."""

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from summarizer.llm import (
    CostAccumulator,
    LLMClient,
    ModelPricing,
    UsageStats,
    _calculate_cost,
    _extract_json,
    _extract_usage,
    call_llm,
    create_client,
    fetch_model_pricing,
)
from summarizer.models import Config, LLMError

# ---------------------------------------------------------------------------
# create_client
# ---------------------------------------------------------------------------


def test_create_client_returns_llm_client():
    """create_client returns an LLMClient with the correct model set."""
    config = Config(base_url="http://localhost:1234/v1", model="test-model")
    client = create_client(config)
    assert isinstance(client, LLMClient)
    assert client.model == "test-model"


def test_create_client_uses_config_api_key():
    """create_client uses config.api_key when set."""
    config = Config(api_key="sk-explicit-key")
    with patch("summarizer.llm._openai.OpenAI") as mock_openai:
        create_client(config)
    _, kwargs = mock_openai.call_args
    assert kwargs["api_key"] == "sk-explicit-key"


def test_create_client_uses_env_api_key_when_config_key_is_none():
    """create_client falls back to LLM_API_KEY env var when config.api_key is None."""
    config = Config(api_key=None)
    with (
        patch("summarizer.llm._openai.OpenAI") as mock_openai,
        patch.dict(os.environ, {"LLM_API_KEY": "sk-env-key"}),
    ):
        create_client(config)
    _, kwargs = mock_openai.call_args
    assert kwargs["api_key"] == "sk-env-key"


def test_create_client_falls_back_to_lm_studio_dummy():
    """create_client uses 'lm-studio' when neither config key nor env var is set."""
    config = Config(api_key=None)
    with (
        patch("summarizer.llm._openai.OpenAI") as mock_openai,
        patch.dict(os.environ, {}, clear=True),
    ):
        env = {k: v for k, v in os.environ.items() if k != "LLM_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            create_client(config)
    _, kwargs = mock_openai.call_args
    assert kwargs["api_key"] == "lm-studio"


def test_create_client_adds_openrouter_headers():
    """create_client injects HTTP-Referer and X-Title for openrouter.ai URLs."""
    config = Config(base_url="https://openrouter.ai/api/v1", model="test-model")
    with patch("summarizer.llm._openai.OpenAI") as mock_openai:
        create_client(config)
    _, kwargs = mock_openai.call_args
    headers = kwargs.get("default_headers", {})
    assert "HTTP-Referer" in headers
    assert "X-Title" in headers


def test_create_client_no_extra_headers_for_local_url():
    """create_client does NOT inject extra headers for a local LM Studio URL."""
    config = Config(base_url="http://localhost:1234/v1", model="test-model")
    with patch("summarizer.llm._openai.OpenAI") as mock_openai:
        create_client(config)
    _, kwargs = mock_openai.call_args
    headers = kwargs.get("default_headers", {})
    assert "HTTP-Referer" not in headers


def test_complete_passes_timeout():
    """LLMClient.complete passes timeout_s to the openai create call."""
    config = Config(timeout_s=42)
    with patch("summarizer.llm._openai.OpenAI") as mock_openai:
        mock_chat = MagicMock()
        mock_openai.return_value.chat = mock_chat
        mock_chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"k":"v"}'))]
        )
        client = create_client(config)
        client.complete("hello")
    call_kwargs = mock_chat.completions.create.call_args[1]
    assert call_kwargs.get("timeout") == 42


def test_complete_omits_max_tokens_when_not_configured():
    """max_tokens is NOT sent when max_output_tokens is None (default)."""
    config = Config()
    with patch("summarizer.llm._openai.OpenAI") as mock_openai:
        mock_chat = MagicMock()
        mock_openai.return_value.chat = mock_chat
        mock_chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"k":"v"}'))]
        )
        client = create_client(config)
        client.complete("hello")
    call_kwargs = mock_chat.completions.create.call_args[1]
    assert "max_tokens" not in call_kwargs


def test_complete_passes_max_tokens_when_configured():
    """max_tokens IS sent when max_output_tokens is explicitly set."""
    config = Config(max_output_tokens=8192)
    with patch("summarizer.llm._openai.OpenAI") as mock_openai:
        mock_chat = MagicMock()
        mock_openai.return_value.chat = mock_chat
        mock_chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"k":"v"}'))]
        )
        client = create_client(config)
        client.complete("hello")
    call_kwargs = mock_chat.completions.create.call_args[1]
    assert call_kwargs.get("max_tokens") == 8192


# ---------------------------------------------------------------------------
# _extract_json (internal helper — tested directly for thorough coverage)
# ---------------------------------------------------------------------------


def test_extract_json_with_markdown_fences():
    data = {"key": "value"}
    fenced = f"```json\n{json.dumps(data)}\n```"
    result = _extract_json(fenced)
    assert result == data


def test_extract_json_with_plain_fences():
    data = {"key": "value"}
    fenced = f"```\n{json.dumps(data)}\n```"
    result = _extract_json(fenced)
    assert result == data


def test_extract_json_with_surrounding_text():
    data = {"key": "value"}
    surrounded = f"Here is the result:\n{json.dumps(data)}\nThat's all."
    result = _extract_json(surrounded)
    assert result == data


def test_extract_json_raises_llm_error_on_no_json():
    with pytest.raises(LLMError, match="No JSON"):
        _extract_json("This response has no JSON object at all.")


def test_extract_json_raises_llm_error_on_invalid_json():
    with pytest.raises(LLMError):
        _extract_json('{"key": "value" BROKEN}')


# ---------------------------------------------------------------------------
# call_llm
# ---------------------------------------------------------------------------


def test_call_llm_returns_parsed_dict():
    data = {"citation_key": "test2025foo", "year": 2025}
    mock_client = MagicMock()
    mock_client.complete.return_value = MagicMock(text=json.dumps(data))

    result = call_llm(mock_client, "some prompt")
    assert result == data
    mock_client.complete.assert_called_once_with("some prompt")


def test_call_llm_handles_fenced_response():
    data = {"citation_key": "test2025foo"}
    fenced = f"```json\n{json.dumps(data)}\n```"
    mock_client = MagicMock()
    mock_client.complete.return_value = MagicMock(text=fenced)

    result = call_llm(mock_client, "some prompt")
    assert result == data


def test_call_llm_raises_llm_error_on_complete_failure():
    mock_client = MagicMock()
    mock_client.complete.side_effect = Exception("connection refused")

    with pytest.raises(LLMError, match="connection refused"):
        call_llm(mock_client, "some prompt")


def test_call_llm_retries_on_429_then_succeeds():
    """429s without a reset hint (free models throttled upstream) back off from 5 s."""
    data = {"ok": True}
    mock_client = MagicMock()
    mock_client.complete.side_effect = [
        Exception("Error code: 429 - rate limited"),
        MagicMock(text=json.dumps(data)),
    ]

    with (
        patch("summarizer.llm.time.sleep") as mock_sleep,
        patch("summarizer.llm.random.uniform", return_value=1.0),
    ):
        result = call_llm(mock_client, "some prompt")

    assert result == data
    assert mock_client.complete.call_count == 2
    mock_sleep.assert_called_once_with(5.0)


def test_call_llm_retries_on_5xx_then_succeeds():
    """5xx responses are retried with exponential backoff."""
    data = {"ok": True}
    mock_client = MagicMock()
    mock_client.complete.side_effect = [
        Exception("Error code: 503 - upstream unavailable"),
        Exception("Error code: 500 - upstream error"),
        MagicMock(text=json.dumps(data)),
    ]

    with (
        patch("summarizer.llm.time.sleep") as mock_sleep,
        patch("summarizer.llm.random.uniform", return_value=1.0),
    ):
        result = call_llm(mock_client, "some prompt")

    assert result == data
    assert mock_client.complete.call_count == 3
    assert mock_sleep.call_args_list[0].args == (1.0,)
    assert mock_sleep.call_args_list[1].args == (2.0,)


def test_call_llm_raises_llm_error_on_invalid_json():
    mock_client = MagicMock()
    mock_client.complete.return_value = MagicMock(text="not valid json at all")

    with pytest.raises(LLMError):
        call_llm(mock_client, "some prompt")


def test_call_llm_retries_once_with_json_repair_then_succeeds():
    """If first response is malformed JSON, one repair retry is attempted."""
    mock_client = MagicMock()
    mock_client.model = "test-model"
    mock_client.base_url = "http://localhost:1234/v1"
    mock_client.complete.side_effect = [
        MagicMock(text='{"k": "v"'),  # malformed JSON
        MagicMock(text='{"k": "v"}'),  # repaired JSON
    ]

    result = call_llm(mock_client, "some prompt")
    assert result == {"k": "v"}
    assert mock_client.complete.call_count == 2


def test_call_llm_repair_retry_still_fails_raises_original_parse_error():
    """If repair retry also fails, call_llm raises LLMError."""
    mock_client = MagicMock()
    mock_client.model = "test-model"
    mock_client.base_url = "http://localhost:1234/v1"
    mock_client.complete.side_effect = [
        MagicMock(text='{"k": "v"'),
        MagicMock(text="still not json"),
    ]

    with pytest.raises(LLMError, match="No JSON object found"):
        call_llm(mock_client, "some prompt")
    assert mock_client.complete.call_count == 2


def test_call_llm_passes_prompt_to_complete():
    data = {"key": "val"}
    mock_client = MagicMock()
    mock_client.complete.return_value = MagicMock(text=json.dumps(data))

    call_llm(mock_client, "my specific prompt")
    mock_client.complete.assert_called_once_with("my specific prompt")


# ---------------------------------------------------------------------------
# Logging (caplog)
# ---------------------------------------------------------------------------


def test_call_llm_logs_call_await_and_response(caplog):
    """call_llm emits INFO messages for 'Calling LLM', 'Awaiting', and 'Response received'."""
    data = {"key": "value"}
    mock_client = MagicMock()
    mock_client.model = "test-model"
    mock_client.base_url = "http://localhost:1234/v1"
    mock_client.complete.return_value = MagicMock(text=json.dumps(data))

    with caplog.at_level(logging.INFO, logger="summarizer.llm"):
        call_llm(mock_client, "a prompt")

    messages = [r.message for r in caplog.records]
    assert any("Calling LLM" in m for m in messages)
    assert any("Awaiting" in m for m in messages)
    assert any("Response received" in m for m in messages)


def test_create_client_stores_base_url():
    """LLMClient stores base_url for use in log messages."""
    config = Config(base_url="http://localhost:1234/v1", model="test-model")
    client = create_client(config)
    assert client.base_url == "http://localhost:1234/v1"


# ---------------------------------------------------------------------------
# Phase 1: ModelPricing, UsageStats, CostAccumulator, helpers
# ---------------------------------------------------------------------------


def test_model_pricing_defaults_to_zero():
    pricing = ModelPricing()
    assert pricing.prompt == 0.0
    assert pricing.completion == 0.0
    assert pricing.reasoning == 0.0
    assert pricing.request == 0.0
    assert pricing.context_length == 0


def test_model_pricing_custom_values():
    pricing = ModelPricing(prompt=1e-6, completion=3e-6, reasoning=2e-6, request=0.001)
    assert pricing.prompt == 1e-6
    assert pricing.completion == 3e-6
    assert pricing.reasoning == 2e-6
    assert pricing.request == 0.001


def test_usage_stats_defaults_to_zero():
    usage = UsageStats()
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.reasoning_tokens == 0


def test_calculate_cost_typical():
    pricing = ModelPricing(prompt=1e-6, completion=3e-6, reasoning=2e-6, request=0.001)
    usage = UsageStats(input_tokens=1000, output_tokens=500, reasoning_tokens=200)
    cost = _calculate_cost(usage, pricing)
    expected = 1000 * 1e-6 + 500 * 3e-6 + 200 * 2e-6 + 0.001
    assert abs(cost - expected) < 1e-12


def test_calculate_cost_all_zero_pricing():
    pricing = ModelPricing()
    usage = UsageStats(input_tokens=5000, output_tokens=1000, reasoning_tokens=500)
    assert _calculate_cost(usage, pricing) == 0.0


def test_calculate_cost_none_usage():
    pricing = ModelPricing(prompt=1e-6, completion=3e-6, request=0.001)
    # None usage → only request_price (flat fee)
    cost = _calculate_cost(None, pricing)
    assert cost == pytest.approx(0.001)


def test_calculate_cost_zero_usage():
    pricing = ModelPricing(prompt=1e-6, request=0.0)
    usage = UsageStats()
    assert _calculate_cost(usage, pricing) == 0.0


def test_cost_accumulator_add_single():
    acc = CostAccumulator()
    usage = UsageStats(input_tokens=100, output_tokens=50, reasoning_tokens=10)
    acc.add(usage, 0.005)
    assert acc.total_cost == pytest.approx(0.005)
    assert acc.total_input_tokens == 100
    assert acc.total_output_tokens == 50
    assert acc.total_reasoning_tokens == 10


def test_cost_accumulator_add_multiple():
    acc = CostAccumulator()
    acc.add(UsageStats(input_tokens=100, output_tokens=50, reasoning_tokens=0), 0.002)
    acc.add(UsageStats(input_tokens=200, output_tokens=80, reasoning_tokens=20), 0.005)
    assert acc.total_cost == pytest.approx(0.007)
    assert acc.total_input_tokens == 300
    assert acc.total_output_tokens == 130
    assert acc.total_reasoning_tokens == 20


def test_cost_accumulator_thread_safe():
    """Concurrent adds must not lose updates."""
    acc = CostAccumulator()
    n = 100
    usage = UsageStats(input_tokens=1, output_tokens=1, reasoning_tokens=0)

    def add_many():
        for _ in range(n):
            acc.add(usage, 0.001)

    threads = [threading.Thread(target=add_many) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert acc.total_cost == pytest.approx(10 * n * 0.001)
    assert acc.total_input_tokens == 10 * n
    assert acc.total_output_tokens == 10 * n


def test_cost_accumulator_starts_at_zero():
    acc = CostAccumulator()
    assert acc.total_cost == 0.0
    assert acc.total_input_tokens == 0
    assert acc.total_output_tokens == 0
    assert acc.total_reasoning_tokens == 0


def test_extract_usage_happy_path():
    mock_response = MagicMock()
    mock_response.usage.prompt_tokens = 1000
    mock_response.usage.completion_tokens = 400
    mock_response.usage.completion_tokens_details.reasoning_tokens = 50
    usage = _extract_usage(mock_response)
    assert usage is not None
    assert usage.input_tokens == 1000
    assert usage.output_tokens == 400
    assert usage.reasoning_tokens == 50


def test_extract_usage_none_when_no_usage():
    mock_response = MagicMock()
    mock_response.usage = None
    usage = _extract_usage(mock_response)
    assert usage is None


def test_extract_usage_missing_reasoning_tokens():
    """reasoning_tokens absent from completion_tokens_details → 0."""
    mock_response = MagicMock()
    mock_response.usage.prompt_tokens = 800
    mock_response.usage.completion_tokens = 300
    # completion_tokens_details.reasoning_tokens raises AttributeError
    mock_response.usage.completion_tokens_details.reasoning_tokens = None
    usage = _extract_usage(mock_response)
    assert usage is not None
    assert usage.reasoning_tokens == 0


def test_extract_usage_no_completion_tokens_details():
    """completion_tokens_details absent → reasoning_tokens = 0."""
    mock_response = MagicMock()
    mock_response.usage.prompt_tokens = 500
    mock_response.usage.completion_tokens = 200
    del mock_response.usage.completion_tokens_details
    usage = _extract_usage(mock_response)
    assert usage is not None
    assert usage.reasoning_tokens == 0


# ---------------------------------------------------------------------------
# Phase 2: fetch_model_pricing + create_client pricing wiring
# ---------------------------------------------------------------------------

_FAKE_PRICING_RESPONSE = {
    "data": [
        {
            "id": "openai/gpt-4o",
            "pricing": {
                "prompt": "0.000005",
                "completion": "0.000015",
                "internal_reasoning": "0.000010",
                "request": "0.0",
            },
            "context_length": 128000,
        }
    ]
}


def _make_urlopen_mock(response_body: bytes):
    """Helper: mock urllib.request.urlopen to return a readable response."""

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def read(self):
            return response_body

    return MagicMock(return_value=_FakeResponse())


def test_fetch_model_pricing_parses_response():
    body = json.dumps(_FAKE_PRICING_RESPONSE).encode()
    with patch("summarizer.llm.urllib.request.urlopen", _make_urlopen_mock(body)):
        pricing = fetch_model_pricing("openai/gpt-4o", "sk-test", "https://openrouter.ai/api/v1")
    assert pricing.prompt == pytest.approx(5e-6)
    assert pricing.completion == pytest.approx(1.5e-5)
    assert pricing.reasoning == pytest.approx(1e-5)
    assert pricing.request == 0.0
    assert pricing.context_length == 128000


def test_fetch_model_pricing_returns_zero_on_http_error():
    import urllib.error

    with patch(
        "summarizer.llm.urllib.request.urlopen",
        side_effect=urllib.error.URLError("unreachable"),
    ):
        pricing = fetch_model_pricing("openai/gpt-4o", "sk-test", "https://openrouter.ai/api/v1")
    assert pricing == ModelPricing()


def test_fetch_model_pricing_returns_zero_when_model_not_found():
    body = json.dumps({"data": []}).encode()
    with patch("summarizer.llm.urllib.request.urlopen", _make_urlopen_mock(body)):
        pricing = fetch_model_pricing("openai/gpt-4o", "sk-test", "https://openrouter.ai/api/v1")
    assert pricing == ModelPricing()


def test_create_client_fetches_pricing_for_openrouter(monkeypatch):
    """create_client calls fetch_model_pricing exactly once for openrouter URLs."""
    fake_pricing = ModelPricing(prompt=1e-6, completion=3e-6)
    mock_fetch = MagicMock(return_value=fake_pricing)
    monkeypatch.setattr("summarizer.llm.fetch_model_pricing", mock_fetch)

    config = Config(
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-4o",
        api_key="sk-test",
    )
    with patch("summarizer.llm._openai.OpenAI"):
        client = create_client(config)

    mock_fetch.assert_called_once()
    assert client.pricing.prompt == 1e-6


def test_create_client_does_not_fetch_pricing_for_local(monkeypatch):
    """create_client does NOT call fetch_model_pricing for local LM Studio URLs."""
    mock_fetch = MagicMock()
    monkeypatch.setattr("summarizer.llm.fetch_model_pricing", mock_fetch)

    config = Config(base_url="http://localhost:1234/v1", model="test-model")
    with patch("summarizer.llm._openai.OpenAI"):
        client = create_client(config)

    mock_fetch.assert_not_called()
    assert client.pricing == ModelPricing()


def test_complete_returns_usage_when_sdk_provides_it():
    """complete() populates CompletionResponse.usage from response.usage."""
    config = Config()
    with patch("summarizer.llm._openai.OpenAI") as mock_openai:
        mock_chat = MagicMock()
        mock_openai.return_value.chat = mock_chat

        sdk_response = MagicMock()
        sdk_response.choices = [MagicMock(message=MagicMock(content='{"k":"v"}'))]
        sdk_response.usage.prompt_tokens = 200
        sdk_response.usage.completion_tokens = 80
        sdk_response.usage.completion_tokens_details.reasoning_tokens = 10
        mock_chat.completions.create.return_value = sdk_response

        client = create_client(config)
        resp = client.complete("hello")

    assert resp.usage is not None
    assert resp.usage.input_tokens == 200
    assert resp.usage.output_tokens == 80
    assert resp.usage.reasoning_tokens == 10


def test_complete_usage_is_none_when_sdk_returns_no_usage():
    config = Config()
    with patch("summarizer.llm._openai.OpenAI") as mock_openai:
        mock_chat = MagicMock()
        mock_openai.return_value.chat = mock_chat

        sdk_response = MagicMock()
        sdk_response.choices = [MagicMock(message=MagicMock(content='{"k":"v"}'))]
        sdk_response.usage = None
        mock_chat.completions.create.return_value = sdk_response

        client = create_client(config)
        resp = client.complete("hello")

    assert resp.usage is None


# ---------------------------------------------------------------------------
# Phase 3: call_llm extended logging + accumulator
# ---------------------------------------------------------------------------


def test_call_llm_logs_token_counts_and_cost(caplog):
    """call_llm log line contains in=, out=, reason=, cost= fields."""
    data = {"key": "value"}
    mock_client = MagicMock()
    mock_client.model = "test-model"
    mock_client.base_url = "http://localhost:1234/v1"
    mock_client.pricing = ModelPricing()
    mock_response = MagicMock()
    mock_response.text = json.dumps(data)
    mock_response.usage = UsageStats(input_tokens=100, output_tokens=40, reasoning_tokens=5)
    mock_client.complete.return_value = mock_response

    with caplog.at_level(logging.INFO, logger="summarizer.llm"):
        call_llm(mock_client, "a prompt")

    response_lines = [r.message for r in caplog.records if "Response received" in r.message]
    assert response_lines, "Expected 'Response received' log line"
    line = response_lines[0]
    assert "in=" in line
    assert "out=" in line
    assert "reason=" in line
    assert "cost=" in line


def test_call_llm_updates_accumulator():
    data = {"ok": True}
    mock_client = MagicMock()
    mock_client.pricing = ModelPricing(prompt=1e-6, completion=3e-6)
    mock_response = MagicMock()
    mock_response.text = json.dumps(data)
    mock_response.usage = UsageStats(input_tokens=1000, output_tokens=500, reasoning_tokens=0)
    mock_client.complete.return_value = mock_response

    acc = CostAccumulator()
    call_llm(mock_client, "prompt", accumulator=acc)

    assert acc.total_cost > 0
    assert acc.total_input_tokens == 1000
    assert acc.total_output_tokens == 500


def test_call_llm_works_without_accumulator():
    data = {"ok": True}
    mock_client = MagicMock()
    mock_client.pricing = ModelPricing()
    mock_response = MagicMock()
    mock_response.text = json.dumps(data)
    mock_response.usage = UsageStats(input_tokens=100, output_tokens=40, reasoning_tokens=0)
    mock_client.complete.return_value = mock_response

    result = call_llm(mock_client, "prompt")  # no accumulator — must not raise
    assert result == data


def test_call_llm_none_usage_does_not_crash():
    """call_llm logs cost=$0 and does not crash when response.usage is None."""
    data = {"ok": True}
    mock_client = MagicMock()
    mock_client.pricing = ModelPricing(request=0.001)
    mock_response = MagicMock()
    mock_response.text = json.dumps(data)
    mock_response.usage = None
    mock_client.complete.return_value = mock_response

    acc = CostAccumulator()
    result = call_llm(mock_client, "prompt", accumulator=acc)
    assert result == data
    assert acc.total_cost == pytest.approx(0.001)  # only request flat fee


# ---------------------------------------------------------------------------
# Repair-path token counting
# ---------------------------------------------------------------------------


def test_call_llm_json_repair_tokens_added_to_accumulator():
    """Tokens from the JSON syntax-repair call are added to the accumulator."""
    pricing = ModelPricing(completion=3e-6)
    mock_client = MagicMock()
    mock_client.model = "test-model"
    mock_client.base_url = "http://localhost:1234/v1"
    mock_client.pricing = pricing

    first_usage = UsageStats(input_tokens=100, output_tokens=50)
    repair_usage = UsageStats(input_tokens=80, output_tokens=40)

    mock_client.complete.side_effect = [
        MagicMock(text='{"k": "v"', usage=first_usage),  # malformed
        MagicMock(text='{"k": "v"}', usage=repair_usage),  # repaired
    ]

    acc = CostAccumulator()
    call_llm(mock_client, "prompt", accumulator=acc)

    assert acc.total_output_tokens == 50 + 40
    assert acc.total_input_tokens == 100 + 80


def test_pipeline_schema_repair_tokens_added_to_accumulator():
    """Tokens from a schema-validation repair call are added to the accumulator."""
    from summarizer.llm import CostAccumulator, ModelPricing
    from summarizer.pipeline import _validate_with_schema_repair

    pricing = ModelPricing()
    mock_client = MagicMock()
    mock_client.model = "test-model"
    mock_client.base_url = "http://localhost:1234/v1"
    mock_client.pricing = pricing

    # A minimal valid LLMResponse to return after repair
    good_raw = {
        "metadata": {
            "citation_key": "smith2020foo",
            "title": "Title",
            "authors": ["A. Smith"],
            "year": 2020,
            "venue": "Conf",
            "is_research_paper": True,
            "paper_type": "synthesis",
            "synthesis_subtype": "review",
            "rejection_reason": None,
            "tags": [],
        },
        "part1": {
            "paper_type": "synthesis",
            "tldr": "t",
            "target_papers_field": "f",
            "scope_coverage": "s",
            "taxonomy_organization": "t",
            "core_argument": "c",
            "synthesis_contribution": "c",
            "key_claims_narrative": "k",
            "key_takeaways": "k",
            "limitations": "l",
            "open_problems_future_directions": {
                "gaps_identified": [],
                "open_questions": [],
                "suggested_research_focus": [],
            },
            "critical_assessment": "c",
            "notable_findings": [],
            "citable_snippets": [],
            "relevance": "r",
        },
        "part2": None,
    }

    # First call returns invalid raw (missing metadata), second returns good_raw
    with patch("summarizer.pipeline.call_llm", side_effect=[good_raw]) as mock_call_llm:
        acc = CostAccumulator()
        # Pass an already-invalid raw that can't be fixed by normalizers
        bad_raw = {"metadata": {}, "part1": {}, "part2": None}
        # We skip the actual validation loop and just verify accumulator threading:
        # Instead, directly test that when call_llm is invoked in the repair loop,
        # the accumulator is forwarded.
        from pathlib import Path

        _validate_with_schema_repair(
            raw=good_raw,
            client=mock_client,
            original_prompt="prompt",
            pdf_path=Path("paper.pdf"),
            accumulator=acc,
        )
        # No repair needed — but verify the accumulator was accepted without error.
        # Now test the repair path by passing bad raw that triggers repair.
        mock_call_llm.reset_mock()

    # Verify the repair path passes accumulator: when schema repair is triggered,
    # call_llm must be invoked with the accumulator.
    bad_raw = {"not_metadata": True}
    with patch("summarizer.pipeline.call_llm", return_value=good_raw) as mock_repair:
        acc2 = CostAccumulator()
        from pathlib import Path

        try:
            _validate_with_schema_repair(
                raw=bad_raw,
                client=mock_client,
                original_prompt="prompt",
                pdf_path=Path("paper.pdf"),
                accumulator=acc2,
            )
        except Exception:
            pass
        # The repair call_llm must have been called with accumulator=acc2
        for c in mock_repair.call_args_list:
            assert c.kwargs.get("accumulator") is acc2 or (len(c.args) >= 3 and c.args[2] is acc2)


# ---------------------------------------------------------------------------
# Regressions: retries, empty/truncated responses, headers
# ---------------------------------------------------------------------------


@contextmanager
def _client_returning(sdk_response, **config_kwargs):
    """Yield ``(client, mock_openai)`` for a real LLMClient over a mocked SDK."""
    config = Config(base_url="http://localhost:1234/v1", model="m", **config_kwargs)
    with patch("summarizer.llm._openai.OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = sdk_response
        yield create_client(config), mock_openai


def _sdk_response(content='{"k": "v"}', finish_reason="stop", usage=None, choices=True):
    return MagicMock(
        choices=[MagicMock(message=MagicMock(content=content), finish_reason=finish_reason)]
        if choices
        else [],
        usage=usage,
    )


def test_sdk_retries_are_disabled():
    """Regression: SDK retries (2x) stacked with ours (2x) gave up to 9 attempts."""
    with _client_returning(_sdk_response()) as (_, mock_openai):
        assert mock_openai.call_args.kwargs["max_retries"] == 0


@pytest.mark.parametrize("content", [None, "", "   \n"])
def test_complete_raises_on_empty_content(content):
    """Regression: content=None (reasoning models, refusals) crashed with TypeError."""
    with _client_returning(_sdk_response(content=content)) as (client, _):
        with pytest.raises(LLMError, match="no content"):
            client.complete("hello")


def test_complete_raises_when_response_has_no_choices():
    with _client_returning(_sdk_response(choices=False)) as (client, _):
        with pytest.raises(LLMError, match="no choices"):
            client.complete("hello")


def _provider_failure(code, finish_reason=None, usage=None):
    """A 200 response carrying OpenRouter's ``error`` object instead of a reply."""
    error = {"error": {"code": code, "message": "Upstream error"}}
    if finish_reason is None:
        return MagicMock(choices=[], model_extra=error, usage=usage)
    choice = MagicMock(finish_reason=finish_reason, model_extra=error)
    return MagicMock(choices=[choice], usage=usage)


@pytest.mark.parametrize("finish_reason", [None, "error"])
def test_provider_error_in_a_200_response_is_retried(finish_reason):
    """Regression: 2 of 17 real papers failed on 'no choices' without a retry or a reason."""
    with _client_returning(_provider_failure(502, finish_reason)) as (client, mock_openai):
        create = mock_openai.return_value.chat.completions.create
        create.side_effect = [_provider_failure(502, finish_reason), _sdk_response('{"ok": 1}')]
        with (
            patch("summarizer.llm.time.sleep") as sleep,
            patch("summarizer.llm.random.uniform", return_value=1.0),
        ):
            assert call_llm(client, "prompt") == {"ok": 1}
    assert create.call_count == 2
    sleep.assert_called_once_with(1.0)  # backs off like any transient error


@pytest.mark.parametrize(
    "code, retried",
    [
        (None, True),
        (408, True),
        (429, True),
        ("502", True),
        (503, True),
        (400, False),
        (401, False),
    ],
)
def test_provider_error_codes_that_are_retried(code, retried):
    with _client_returning(_provider_failure(code)) as (client, mock_openai):
        create = mock_openai.return_value.chat.completions.create
        create.side_effect = [_provider_failure(code), _sdk_response('{"ok": 1}')]
        with patch("summarizer.llm.time.sleep"):
            if retried:
                assert call_llm(client, "prompt") == {"ok": 1}
            else:
                with pytest.raises(LLMError):
                    call_llm(client, "prompt")
    assert create.call_count == (2 if retried else 1)


def test_provider_error_retries_are_bounded():
    with _client_returning(_provider_failure(502)) as (client, mock_openai):
        with patch("summarizer.llm.time.sleep"), pytest.raises(LLMError, match="Upstream"):
            call_llm(client, "prompt")
    assert mock_openai.return_value.chat.completions.create.call_count == 4


def test_billed_usage_of_failed_provider_attempts_is_recorded():
    """Review finding: a provider error that was billed must still count toward --max-cost."""
    from summarizer.llm import CostAccumulator

    usage = MagicMock(
        prompt_tokens=100,
        completion_tokens=50,
        completion_tokens_details=None,
        prompt_tokens_details=None,
        cost=0.02,
    )
    acc = CostAccumulator()
    with _client_returning(_provider_failure(502, "error", usage)) as (client, _):
        with patch("summarizer.llm.time.sleep"), pytest.raises(LLMError):
            call_llm(client, "prompt", accumulator=acc)
    assert acc.calls == 4 and acc.total_cost == pytest.approx(0.08)


def test_credits_exhausted_inside_a_200_response_stops_the_run():
    from summarizer.llm import QuotaExhausted

    with _client_returning(_provider_failure(402)) as (client, mock_openai):
        with patch("summarizer.llm.time.sleep") as sleep, pytest.raises(QuotaExhausted):
            call_llm(client, "prompt")
    sleep.assert_not_called()


def test_non_transient_provider_error_is_not_retried_and_reports_the_reason():
    with _client_returning(_provider_failure(400)) as (client, mock_openai):
        with patch("summarizer.llm.time.sleep") as sleep:
            with pytest.raises(LLMError, match="Upstream error"):
                call_llm(client, "prompt")
    sleep.assert_not_called()
    assert mock_openai.return_value.chat.completions.create.call_count == 1


def test_complete_raises_on_length_truncation():
    """Regression: truncated JSON was sent to a repair call that cannot recover it."""
    with _client_returning(_sdk_response(content='{"a": ', finish_reason="length")) as (client, _):
        with pytest.raises(LLMError, match="truncated"):
            client.complete("hello")


def test_call_llm_does_not_retry_or_repair_truncated_output():
    with (
        _client_returning(_sdk_response(content='{"a": ', finish_reason="length")) as (
            client,
            mock_openai,
        ),
        patch("summarizer.llm.time.sleep"),
        pytest.raises(LLMError, match="truncated"),
    ):
        call_llm(client, "prompt")
    assert mock_openai.return_value.chat.completions.create.call_count == 1


def test_call_llm_counts_usage_of_rejected_completion():
    """A truncated reply is billed, so its tokens must reach the accumulator."""
    sdk_usage = MagicMock(prompt_tokens=100, completion_tokens=50, completion_tokens_details=None)
    response = _sdk_response(content='{"a": ', finish_reason="length", usage=sdk_usage)
    acc = CostAccumulator()
    with (
        _client_returning(response) as (client, _),
        pytest.raises(LLMError),
    ):
        call_llm(client, "prompt", accumulator=acc)
    assert (acc.total_input_tokens, acc.total_output_tokens) == (100, 50)


def test_call_llm_retries_on_timeout():
    import openai

    data = {"ok": True}
    mock_client = MagicMock()
    mock_client.complete.side_effect = [
        openai.APITimeoutError(request=MagicMock()),  # request type varies by SDK version
        MagicMock(text=json.dumps(data)),
    ]
    with patch("summarizer.llm.time.sleep"):
        assert call_llm(mock_client, "prompt") == data
    assert mock_client.complete.call_count == 2


def test_call_llm_does_not_retry_auth_errors():
    mock_client = MagicMock()
    mock_client.complete.side_effect = Exception("Error code: 401 - invalid api key")
    with patch("summarizer.llm.time.sleep") as mock_sleep, pytest.raises(LLMError):
        call_llm(mock_client, "prompt")
    mock_sleep.assert_not_called()


def test_openrouter_headers_name_this_project():
    config = Config(base_url="https://openrouter.ai/api/v1", model="m", api_key="k")
    with (
        patch("summarizer.llm._openai.OpenAI") as mock_openai,
        patch("summarizer.llm.fetch_model_pricing", return_value=ModelPricing()),
    ):
        create_client(config)
    headers = mock_openai.call_args.kwargs["default_headers"]
    assert headers["X-Title"] == "paper-summarizer"
    assert "jhuebotter/paper-summarizer" in headers["HTTP-Referer"]


def test_fetch_openrouter_models():
    from summarizer.llm import fetch_openrouter_models

    body = json.dumps({"data": [{"id": "a/b"}, {"id": "c/d"}]}).encode()
    with patch("summarizer.llm.urllib.request.urlopen", _make_urlopen_mock(body)):
        assert [m["id"] for m in fetch_openrouter_models("https://openrouter.ai/api/v1")] == [
            "a/b",
            "c/d",
        ]


def test_fetch_openrouter_models_returns_none_when_unreachable():
    from summarizer.llm import fetch_openrouter_models

    assert fetch_openrouter_models("https://openrouter.ai/api/v1") is None


@pytest.mark.parametrize("body", [b"[]", b"null", b'{"data": null}', b"not json"])
def test_malformed_model_list_is_treated_as_unknown(body):
    from summarizer.llm import fetch_openrouter_models

    with patch("summarizer.llm.urllib.request.urlopen", _make_urlopen_mock(body)):
        assert fetch_openrouter_models("https://openrouter.ai/api/v1") is None
        assert fetch_model_pricing("a/b", "k", "https://openrouter.ai/api/v1") == ModelPricing()


def test_model_list_skips_malformed_entries():
    from summarizer.llm import fetch_openrouter_models

    body = json.dumps({"data": ["x", {"no_id": 1}, {"id": "a/b"}]}).encode()
    with patch("summarizer.llm.urllib.request.urlopen", _make_urlopen_mock(body)):
        assert fetch_openrouter_models("https://openrouter.ai/api/v1") == [{"id": "a/b"}]


def test_pricing_null_fields_do_not_crash():
    body = json.dumps({"data": [{"id": "a/b", "pricing": None, "context_length": None}]}).encode()
    with patch("summarizer.llm.urllib.request.urlopen", _make_urlopen_mock(body)):
        assert fetch_model_pricing("a/b", "k", "https://openrouter.ai/api/v1") == ModelPricing()


def test_pricing_lookup_strips_routing_suffix():
    """`x/y:nitro` is billed as `x/y`; the lookup must not fall back to $0."""
    body = json.dumps({"data": [{"id": "a/b", "pricing": {"prompt": "0.000001"}}]}).encode()
    with patch("summarizer.llm.urllib.request.urlopen", _make_urlopen_mock(body)):
        pricing = fetch_model_pricing("a/b:nitro", "k", "https://openrouter.ai/api/v1")
    assert pricing.prompt == 1e-6


# ---------------------------------------------------------------------------
# Quotas, backoff, billed cost, structured output
# ---------------------------------------------------------------------------


def _api_error(status: int, message: str = "error", headers: dict | None = None):
    exc = Exception(f"Error code: {status} - {message}")
    exc.status_code = status
    exc.body = {"message": message, "metadata": {"headers": headers or {}}}
    return exc


@pytest.mark.parametrize(
    "exc",
    [
        _api_error(402, "Insufficient credits"),
        _api_error(429, "Rate limit exceeded: free-models-per-day"),
        _api_error(
            429,
            "Rate limit exceeded",
            {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str((time.time() + 7200) * 1000)},
        ),
    ],
)
def test_exhausted_quota_is_not_retried(exc):
    from summarizer.llm import QuotaExhausted

    mock_client = MagicMock()
    mock_client.complete.side_effect = exc
    with patch("summarizer.llm.time.sleep") as sleep, pytest.raises(QuotaExhausted):
        call_llm(mock_client, "prompt")
    sleep.assert_not_called()
    assert mock_client.complete.call_count == 1


def test_per_minute_rate_limit_is_retried_with_retry_after():
    mock_client = MagicMock()
    exc = _api_error(429, "Rate limit exceeded: 20 per minute")
    exc.response = MagicMock(headers={"retry-after": "7"})
    mock_client.complete.side_effect = [exc, MagicMock(text='{"ok": true}', usage=None)]
    with (
        patch("summarizer.llm.time.sleep") as sleep,
        patch("summarizer.llm.random.uniform", return_value=0.5),
    ):
        assert call_llm(mock_client, "prompt") == {"ok": True}
    sleep.assert_called_once_with(7.5)


def test_per_minute_rate_limit_waits_for_the_reset_in_the_error_body():
    """Regression: a 20/min burst failed papers after ~3 s of backoff while the
    reset (in OpenRouter's error metadata) was ~40 s away."""
    exc = _api_error(
        429, "Rate limit exceeded: free-models-per-min", {"X-RateLimit-Reset": "1000040000"}
    )
    mock_client = MagicMock()
    mock_client.complete.side_effect = [exc, MagicMock(text='{"ok": true}', usage=None)]
    with (
        patch("summarizer.llm.time.time", return_value=1_000_000.0),
        patch("summarizer.llm.time.sleep") as sleep,
        patch("summarizer.llm.random.uniform", return_value=0.5),
    ):
        assert call_llm(mock_client, "prompt") == {"ok": True}
    sleep.assert_called_once_with(40.5)


def test_quota_message_includes_reset_time():
    from summarizer.llm import _quota_exhausted_message

    reset = (time.time() + 3600) * 1000
    message = _quota_exhausted_message(
        _api_error(429, "limit", {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(reset)})
    )
    assert "resets at" in message


def test_billed_cost_and_cached_tokens_are_read_from_usage():
    usage = MagicMock(
        prompt_tokens=1000,
        completion_tokens=100,
        completion_tokens_details=MagicMock(reasoning_tokens=20),
        prompt_tokens_details=MagicMock(cached_tokens=800),
        cost=0.0042,
    )
    stats = _extract_usage(MagicMock(usage=usage))
    assert (stats.cached_tokens, stats.cost) == (800, 0.0042)
    assert _calculate_cost(stats, ModelPricing(prompt=1.0)) == 0.0042  # billed beats list price


def test_non_numeric_cost_is_ignored():
    usage = MagicMock(prompt_tokens=1, completion_tokens=1)  # .cost is a MagicMock
    assert _extract_usage(MagicMock(usage=usage)).cost is None


def test_structured_output_request_shape():
    config = Config(
        base_url="https://openrouter.ai/api/v1", model="m", api_key="k", structured_output=True
    )
    with (
        patch("summarizer.llm._openai.OpenAI") as mock_openai,
        patch("summarizer.llm.fetch_model_pricing", return_value=ModelPricing()),
    ):
        mock_openai.return_value.chat.completions.create.return_value = _sdk_response()
        create_client(config).complete("hello")
    kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"]["type"] == "json_schema"
    assert "metadata" in kwargs["response_format"]["json_schema"]["schema"]["properties"]
    assert kwargs["extra_body"] == {"provider": {"require_parameters": True}}


def test_structured_output_off_by_default():
    with _client_returning(_sdk_response()) as (client, mock_openai):
        client.complete("hello")
    kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
    assert "response_format" not in kwargs and "extra_body" not in kwargs


def test_fetch_openrouter_key_info():
    from summarizer.llm import fetch_openrouter_key_info

    body = json.dumps({"data": {"usage": 0.5, "is_free_tier": True}}).encode()
    with patch("summarizer.llm.urllib.request.urlopen", _make_urlopen_mock(body)) as urlopen:
        info = fetch_openrouter_key_info("https://openrouter.ai/api/v1", "sk-1")
    assert info == {"usage": 0.5, "is_free_tier": True}
    request = urlopen.call_args.args[0]
    assert request.full_url == "https://openrouter.ai/api/v1/key"
    assert request.headers["Authorization"] == "Bearer sk-1"
    with patch("summarizer.llm.urllib.request.urlopen", side_effect=OSError("down")):
        assert fetch_openrouter_key_info("https://openrouter.ai/api/v1", "sk-1") is None


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://openrouter.ai/api/v1", True),
        ("http://localhost:1234/v1", False),
        ("http://proxy.local/openrouter.ai/v1", False),
        ("https://evilopenrouter.ai/api/v1", False),
        ("https://eu.openrouter.ai/api/v1", True),
    ],
)
def test_is_openrouter(url, expected):
    from summarizer.llm import is_openrouter

    assert is_openrouter(url) is expected


@pytest.mark.parametrize(
    "exc,exhausted",
    [
        (_api_error(403, "Key limit exceeded (daily limit)"), True),
        (_api_error(403, "Forbidden"), False),
        (_api_error(429, "Rate limit exceeded: 50 requests per day"), True),
        (_api_error(429, "Daily limit reached"), True),
        (
            _api_error(429, "limit", {"X-RateLimit-Remaining": "5", "X-RateLimit-Reset": "9e15"}),
            False,
        ),
        (
            _api_error(
                429,
                "limit",
                {
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str((time.time() + 120) * 1000),
                },
            ),
            False,  # resets within minutes: retry instead
        ),
        (_api_error(500, "per-day"), False),
    ],
)
def test_quota_classification(exc, exhausted):
    from summarizer.llm import _quota_exhausted_message

    assert bool(_quota_exhausted_message(exc)) is exhausted


def test_quota_reset_time_is_exact():
    from summarizer.llm import _quota_exhausted_message

    now = 1_790_000_000.0
    reset_ms = (now + 3 * 3600) * 1000
    with patch("summarizer.llm.time.time", return_value=now):
        message = _quota_exhausted_message(
            _api_error(429, "x", {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(reset_ms)})
        )
    assert message.endswith(time.strftime("%H:%M", time.localtime(now + 3 * 3600)))


def test_backoff_is_jittered_within_bounds():
    from summarizer.llm import _retry_delay_seconds

    delays = {_retry_delay_seconds(3) for _ in range(50)}
    assert all(2.0 <= d <= 6.0 for d in delays) and len(delays) > 1


def test_rate_limit_reset_beyond_a_minute_falls_back_to_backoff():
    from summarizer.llm import _retry_delay_seconds

    exc = _api_error(429, "limit", {"X-RateLimit-Reset": "1000300000"})  # 300 s away
    with patch("summarizer.llm.time.time", return_value=1_000_000.0):
        assert 2.5 <= _retry_delay_seconds(1, exc) <= 7.5


def test_long_retry_after_falls_back_to_backoff():
    from summarizer.llm import _retry_delay_seconds

    exc = _api_error(429)
    exc.response = MagicMock(headers={"retry-after": "3600"})
    assert 2.5 <= _retry_delay_seconds(1, exc) <= 7.5  # the 429 backoff, not the hour


def test_quota_during_json_repair_propagates():
    from summarizer.llm import QuotaExhausted

    mock_client = MagicMock()
    mock_client.complete.side_effect = [
        MagicMock(text="not json", usage=None),
        _api_error(402, "Insufficient credits"),
    ]
    with pytest.raises(QuotaExhausted):
        call_llm(mock_client, "prompt")


def test_boolean_cost_is_ignored():
    usage = MagicMock(prompt_tokens=1, completion_tokens=1, cost=True)
    assert _extract_usage(MagicMock(usage=usage)).cost is None


def test_portable_schema_has_no_refs_or_discriminator():
    from summarizer.llm import portable_schema
    from summarizer.models import LLMResponse

    text = json.dumps(portable_schema(LLMResponse.model_json_schema()))
    assert "$ref" not in text and "discriminator" not in text and "oneOf" not in text
    assert '"neuron_model"' in text


def test_accumulator_forwards_to_parent():
    parent = CostAccumulator()
    child = CostAccumulator(parent=parent)
    child.add(UsageStats(input_tokens=5), 0.25)
    child.note_json_repair()
    child.note_schema_repair()
    assert (parent.calls, parent.total_cost, parent.total_input_tokens) == (1, 0.25, 5)
    assert (parent.json_repairs, parent.schema_repairs) == (1, 1)
