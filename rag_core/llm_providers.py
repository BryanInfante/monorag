"""LLM provider adapters used by the bundled Generator.

The core contract is intentionally tiny: a provider receives a normalized chat
messages list and returns text. Providers with OpenAI-compatible APIs share the
same adapter; non-compatible providers can implement the same ``complete``
method and be injected into ``Generator``.
"""

from __future__ import annotations

from typing import Protocol

from openai import OpenAI


class ChatProvider(Protocol):
    """Protocol implemented by LLM chat completion providers."""

    def complete(self, *, model: str, messages: list[dict[str, str]]) -> str:
        """Return the generated answer text for a normalized chat request."""


_PROVIDER_BASE_URLS: dict[str, str | None] = {
    "openai-compatible": None,
    "openai": None,
    "groq": "https://api.groq.com/openai/v1",
    "google-ai-studio": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "ollama": "http://localhost:11434/v1",
    "lm-studio": "http://localhost:1234/v1",
    "lmstudio": "http://localhost:1234/v1",
}

_PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "openai-compatible": "gpt-4o-mini",
    "openai": "gpt-4o-mini",
    "groq": "llama-3.3-70b-versatile",
    "google-ai-studio": "gemini-2.0-flash",
    "google": "gemini-2.0-flash",
    "gemini": "gemini-2.0-flash",
    "ollama": "llama3.2",
    "lm-studio": "local-model",
    "lmstudio": "local-model",
}


def normalize_provider_name(provider: str | None) -> str:
    """Normalize a provider label into a supported built-in adapter alias."""
    if provider is None or not provider.strip():
        return "openai-compatible"
    return provider.strip().lower().replace("_", "-").replace(" ", "-")


def default_base_url_for_provider(provider: str | None) -> str | None:
    """Return the default OpenAI-compatible base URL for a known provider."""
    return _PROVIDER_BASE_URLS.get(normalize_provider_name(provider))


def default_model_for_provider(provider: str | None) -> str:
    """Return a practical default model for a known provider alias."""
    return _PROVIDER_DEFAULT_MODELS.get(
        normalize_provider_name(provider),
        _PROVIDER_DEFAULT_MODELS["openai-compatible"],
    )


def is_known_provider(provider: str | None) -> bool:
    """Return whether provider is one of MonoRAG's built-in aliases."""
    return normalize_provider_name(provider) in _PROVIDER_BASE_URLS


class OpenAICompatibleProvider:
    """LLM provider adapter for OpenAI-compatible chat completion APIs."""

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        """Initialize an OpenAI-compatible client.

        Args:
            api_key: Provider API key. Local providers may accept a dummy value.
            base_url: Optional API base URL. ``None`` keeps the OpenAI client
                default, useful for the official OpenAI API.
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, *, model: str, messages: list[dict[str, str]]) -> str:
        """Call the provider and return the first response message content."""
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return response.choices[0].message.content or ""


def build_chat_provider(
    *,
    provider: str | None,
    api_key: str,
    base_url: str | None,
) -> ChatProvider:
    """Build a bundled chat provider adapter.

    The built-in adapter is OpenAI-compatible. Provider aliases such as
    ``groq``, ``google-ai-studio``, ``ollama`` and ``lm-studio`` select a
    default ``base_url``. Custom OpenAI-compatible vendors are supported by
    passing ``base_url`` explicitly. If a vendor is not OpenAI-compatible,
    inject a custom object implementing ``ChatProvider`` into ``Generator``.
    """
    provider_name = normalize_provider_name(provider)
    resolved_base_url = base_url

    if resolved_base_url is None:
        if provider is not None and provider_name not in _PROVIDER_BASE_URLS:
            raise ValueError(
                "Proveedor LLM desconocido. Pasá LLM_BASE_URL/llm_base_url "
                "para endpoints OpenAI-compatible custom o inyectá un ChatProvider."
            )
        resolved_base_url = _PROVIDER_BASE_URLS.get(provider_name)

    return OpenAICompatibleProvider(api_key=api_key, base_url=resolved_base_url)
