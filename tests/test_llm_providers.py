"""Unit tests for LLM provider adapters."""

from unittest.mock import MagicMock, patch

import pytest

from rag_core.llm_providers import (
    build_chat_provider,
    default_base_url_for_provider,
    default_model_for_provider,
    normalize_provider_name,
)


def test_normalize_provider_name_accepts_common_variants():
    """Provider labels from env/CLI should normalize predictably."""
    assert normalize_provider_name(None) == "openai-compatible"
    assert normalize_provider_name(" Google_AI_Studio ") == "google-ai-studio"
    assert normalize_provider_name("Acme AI") == "acme-ai"


def test_default_provider_metadata_for_known_aliases():
    """Known aliases should provide default endpoint/model metadata."""
    assert default_base_url_for_provider("groq") == "https://api.groq.com/openai/v1"
    assert default_model_for_provider("groq") == "llama-3.3-70b-versatile"
    assert default_base_url_for_provider("ollama") == "http://localhost:11434/v1"
    assert default_model_for_provider("ollama") == "llama3.2"


@patch("rag_core.llm_providers.OpenAI")
def test_build_chat_provider_uses_alias_default_base_url(mock_openai_cls):
    """Known provider aliases should configure the OpenAI-compatible client."""
    build_chat_provider(provider="groq", api_key="fake-key", base_url=None)

    mock_openai_cls.assert_called_once_with(
        api_key="fake-key",
        base_url="https://api.groq.com/openai/v1",
    )


@patch("rag_core.llm_providers.OpenAI")
def test_build_chat_provider_accepts_custom_openai_compatible_endpoint(mock_openai_cls):
    """Unknown provider names are valid when an explicit base_url is supplied."""
    build_chat_provider(
        provider="mi-proveedor",
        api_key="fake-key",
        base_url="https://llm.example.com/v1",
    )

    mock_openai_cls.assert_called_once_with(
        api_key="fake-key",
        base_url="https://llm.example.com/v1",
    )


def test_unknown_provider_without_base_url_raises_clear_error():
    """Unknown aliases without a base URL would silently hit the wrong provider."""
    with pytest.raises(ValueError, match="Proveedor LLM desconocido"):
        build_chat_provider(provider="mi-proveedor", api_key="fake-key", base_url=None)


@patch("rag_core.llm_providers.OpenAI")
def test_openai_compatible_provider_complete_returns_message_content(mock_openai_cls):
    """The adapter should unwrap OpenAI-compatible chat completion responses."""
    mock_message = MagicMock()
    mock_message.content = "respuesta"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_cls.return_value = mock_client

    provider = build_chat_provider(provider="openai", api_key="fake-key", base_url=None)
    result = provider.complete(model="gpt-test", messages=[{"role": "user", "content": "hola"}])

    assert result == "respuesta"
    mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-test",
        messages=[{"role": "user", "content": "hola"}],
    )
