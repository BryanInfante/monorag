"""Storage configuration helpers for MonoRAG vector data.

The default bundled adapter is ChromaDB, but the public RAGModule boundary can
receive any retriever object with the expected methods. These helpers keep the
Chroma configuration import-light so MCP startup stays lazy.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse


def _default_monorag_data_root() -> Path:
    """Return the OS-appropriate user data root for MonoRAG."""
    if os.name == "nt":
        local_appdata = os.getenv("LOCALAPPDATA")
        if local_appdata:
            return Path(local_appdata) / "monorag"
        return Path.home() / "AppData" / "Local" / "monorag"

    if os.sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "monorag"

    xdg_data_home = os.getenv("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home) / "monorag"
    return Path.home() / ".local" / "share" / "monorag"


def default_chroma_db_path() -> str:
    """Return the default ChromaDB path used by CLI and MCP flows.

    When ``MONORAG_DB_PATH`` is set, preserve it exactly so callers can choose
    absolute paths, relative paths, or test doubles explicitly.

    Without the env var, resolve to an OS-appropriate user data directory so
    installed CLI/MCP tools do not persist Chroma data inside the package or
    working tree.
    """
    env_path = os.getenv("MONORAG_DB_PATH")
    if env_path:
        return env_path
    return str(_default_monorag_data_root() / "chroma_db")


def default_config_path() -> str:
    """Return the default persistent CLI configuration file path."""
    env_path = os.getenv("MONORAG_CONFIG_PATH")
    if env_path:
        return env_path
    return str(_default_monorag_data_root() / "config.json")


def default_chroma_url() -> str | None:
    """Return the configured remote Chroma URL, if any.

    ``MONORAG_CHROMA_URL`` is the preferred variable. ``CHROMA_URL`` is accepted
    as a generic fallback for hosted Chroma deployments and test environments.
    """
    return os.getenv("MONORAG_CHROMA_URL") or os.getenv("CHROMA_URL")


def default_chroma_api_key() -> str | None:
    """Return an API key for remote Chroma deployments, if configured."""
    return os.getenv("MONORAG_CHROMA_API_KEY") or os.getenv("CHROMA_API_KEY")


def default_chroma_tenant() -> str | None:
    """Return the configured Chroma tenant, if any."""
    return os.getenv("MONORAG_CHROMA_TENANT") or os.getenv("CHROMA_TENANT")


def default_chroma_database() -> str | None:
    """Return the configured Chroma database, if any."""
    return os.getenv("MONORAG_CHROMA_DATABASE") or os.getenv("CHROMA_DATABASE")


def parse_chroma_url(url: str) -> tuple[str, int, bool]:
    """Parse a Chroma HTTP(S) URL into host, port, and SSL flag.

    Args:
        url: URL such as ``http://localhost:8000`` or
            ``https://chroma.example.com``.

    Returns:
        Tuple of ``(host, port, ssl)`` suitable for ``chromadb.HttpClient``.

    Raises:
        ValueError: If the URL is not HTTP(S) or has no hostname.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(
            "MONORAG_CHROMA_URL debe ser una URL HTTP(S), por ejemplo "
            "http://localhost:8000."
        )

    ssl = parsed.scheme == "https"
    port = parsed.port or (443 if ssl else 80)
    return parsed.hostname, port, ssl
