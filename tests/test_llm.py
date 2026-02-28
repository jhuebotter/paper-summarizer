"""Tests for summarizer/llm.py — openai SDK wrapper for LM Studio and OpenRouter."""

import json
import logging
import os
import pytest
from unittest.mock import MagicMock, patch

from summarizer.models import Config, LLMError
from summarizer.llm import LMStudioClient, create_client, call_llm, _extract_json


# ---------------------------------------------------------------------------
# create_client
# ---------------------------------------------------------------------------


def test_create_client_returns_lm_studio_client():
    """create_client returns an LMStudioClient with the correct model set."""
    config = Config(base_url="http://localhost:1234/v1", model="test-model")
    client = create_client(config)
    assert isinstance(client, LMStudioClient)
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
    """LMStudioClient.complete passes timeout_s to the openai create call."""
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


def test_extract_json_plain_json():
    data = {"citation_key": "foo2025bar", "year": 2025}
    result = _extract_json(json.dumps(data))
    assert result == data


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
    """429 responses are retried with backoff, then succeed."""
    data = {"ok": True}
    mock_client = MagicMock()
    mock_client.complete.side_effect = [
        Exception("Error code: 429 - rate limited"),
        MagicMock(text=json.dumps(data)),
    ]

    with patch("summarizer.llm.time.sleep") as mock_sleep:
        result = call_llm(mock_client, "some prompt")

    assert result == data
    assert mock_client.complete.call_count == 2
    mock_sleep.assert_called_once_with(1.0)


def test_call_llm_retries_on_5xx_then_succeeds():
    """5xx responses are retried with exponential backoff."""
    data = {"ok": True}
    mock_client = MagicMock()
    mock_client.complete.side_effect = [
        Exception("Error code: 503 - upstream unavailable"),
        Exception("Error code: 500 - upstream error"),
        MagicMock(text=json.dumps(data)),
    ]

    with patch("summarizer.llm.time.sleep") as mock_sleep:
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
    """LMStudioClient stores base_url for use in log messages."""
    config = Config(base_url="http://localhost:1234/v1", model="test-model")
    client = create_client(config)
    assert client.base_url == "http://localhost:1234/v1"
