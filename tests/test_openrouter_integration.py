"""Integration tests for OpenRouter cost tracking.

These tests make real HTTP calls to the OpenRouter API.
They are skipped automatically when ``LLM_API_KEY`` is not set.

The model used is the default OSS free-tier model from ``Config``.
"""

import json
import os
import pytest

from dotenv import load_dotenv

load_dotenv()


@pytest.fixture
def api_key():
    key = os.environ.get("LLM_API_KEY", "")
    if not key:
        pytest.skip("LLM_API_KEY not set")
    return key


@pytest.fixture
def model():
    return os.environ.get("LLM_MODEL", "openai/gpt-oss-120b:free")


@pytest.fixture
def base_url():
    return "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------------------
# Pricing fetch
# ---------------------------------------------------------------------------


def test_fetch_model_pricing_returns_real_data(api_key, model, base_url):
    """fetch_model_pricing returns non-zero context_length for a known model."""
    from summarizer.llm import ModelPricing, fetch_model_pricing

    pricing = fetch_model_pricing(model, api_key, base_url)
    assert isinstance(pricing, ModelPricing)
    # context_length should be a positive integer for any real model
    assert pricing.context_length > 0, f"Expected context_length > 0, got {pricing.context_length}"
    print(f"\nPricing for {model}:")
    print(f"  prompt=$%.2e  completion=$%.2e  reasoning=$%.2e  request=$%.2e  ctx={pricing.context_length}" % (
        pricing.prompt, pricing.completion, pricing.reasoning, pricing.request
    ))


# ---------------------------------------------------------------------------
# Real completion with usage tracking
# ---------------------------------------------------------------------------


def test_real_completion_returns_usage(api_key, model, base_url):
    """A real completion returns token counts in response.usage."""
    from summarizer.models import Config
    from summarizer.llm import create_client, CostAccumulator

    config = Config(base_url=base_url, model=model, api_key=api_key)
    client = create_client(config)

    acc = CostAccumulator()
    prompt = '{"greeting": "hello"}'
    response = client.complete(prompt)

    print(f"\nResponse text (first 200 chars): {response.text[:200]!r}")
    print(f"Usage: {response.usage}")

    assert response.text, "Expected non-empty response text"
    # Usage may be None for some free models — but if present, check it
    if response.usage is not None:
        assert response.usage.input_tokens > 0, "Expected input_tokens > 0"
        assert response.usage.output_tokens > 0, "Expected output_tokens > 0"
        assert response.usage.reasoning_tokens >= 0


def test_real_call_llm_logs_cost(api_key, model, base_url, caplog):
    """call_llm with a real OpenRouter client logs token counts and cost."""
    import logging
    from summarizer.models import Config
    from summarizer.llm import create_client, call_llm, CostAccumulator

    config = Config(base_url=base_url, model=model, api_key=api_key)
    client = create_client(config)
    acc = CostAccumulator()

    prompt = 'Reply with this exact JSON and nothing else: {"ok": true}'

    with caplog.at_level(logging.INFO, logger="summarizer.llm"):
        result = call_llm(client, prompt, accumulator=acc)

    response_lines = [r.message for r in caplog.records if "Response received" in r.message]
    assert response_lines, "Expected 'Response received' in log"
    line = response_lines[0]
    print(f"\nLog line: {line}")
    assert "in=" in line
    assert "out=" in line
    assert "cost=" in line

    print(f"Accumulator: total_cost=${acc.total_cost:.6f}")
    print(f"  input_tokens={acc.total_input_tokens}")
    print(f"  output_tokens={acc.total_output_tokens}")
    print(f"Result: {result}")


def test_real_batch_accumulates_cost(api_key, model, base_url):
    """Two consecutive call_llm calls accumulate cost in the CostAccumulator."""
    from summarizer.models import Config
    from summarizer.llm import create_client, call_llm, CostAccumulator

    config = Config(base_url=base_url, model=model, api_key=api_key)
    client = create_client(config)
    acc = CostAccumulator()

    for _ in range(2):
        call_llm(client, 'Reply with exactly: {"ok": true}', accumulator=acc)

    print(f"\nAfter 2 calls: total_cost=${acc.total_cost:.6f}")
    print(f"  input_tokens={acc.total_input_tokens}, output_tokens={acc.total_output_tokens}")

    assert acc.total_input_tokens > 0
    assert acc.total_output_tokens > 0
    # Costs might be 0 for free models — that's acceptable
    assert acc.total_cost >= 0
