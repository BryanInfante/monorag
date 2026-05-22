"""Secret storage helpers for MonoRAG.

The CLI persists non-secret configuration in ``config.json``. Sensitive values
such as LLM API keys should live in the operating system credential store when
the optional keyring backend is available. This module keeps keyring imports
lazy so MCP startup remains cheap and so headless environments can fall back
gracefully.
"""

from __future__ import annotations

import os
from typing import Any

KEYRING_REFERENCE = "__monorag_keyring__"
KEYRING_SERVICE = "monorag"
LLM_API_KEY_SECRET = "llm_api_key"


def is_keyring_reference(value: Any) -> bool:
    """Return whether a persisted value points to the OS keyring."""
    return value == KEYRING_REFERENCE


def _keyring_disabled() -> bool:
    return os.getenv("MONORAG_DISABLE_KEYRING", "").lower() in {"1", "true", "yes"}


def _load_keyring():
    """Load keyring lazily, returning None when unavailable or disabled."""
    if _keyring_disabled():
        return None
    try:
        import keyring  # type: ignore
    except Exception:
        return None
    return keyring


def set_secret(name: str, value: str) -> bool:
    """Store a secret in the OS keyring.

    Returns False when keyring is unavailable, disabled, or the active backend
    cannot store the secret.
    """
    keyring = _load_keyring()
    if keyring is None:
        return False
    try:
        keyring.set_password(KEYRING_SERVICE, name, value)
    except Exception:
        return False
    return True


def get_secret(name: str) -> str | None:
    """Read a secret from the OS keyring."""
    keyring = _load_keyring()
    if keyring is None:
        return None
    try:
        value = keyring.get_password(KEYRING_SERVICE, name)
    except Exception:
        return None
    return value or None


def delete_secret(name: str) -> bool:
    """Delete a secret from the OS keyring when present."""
    keyring = _load_keyring()
    if keyring is None:
        return False
    try:
        keyring.delete_password(KEYRING_SERVICE, name)
    except Exception:
        return False
    return True
