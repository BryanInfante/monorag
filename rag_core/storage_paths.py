"""Storage path helpers for persistent MonoRAG data."""

from __future__ import annotations

import os
from pathlib import Path


def default_chroma_db_path() -> str:
    """Return the default ChromaDB path used by CLI and MCP flows.

    When ``MONORAG_DB_PATH`` is set, preserve it exactly so callers can choose
    absolute paths, relative paths, or test doubles explicitly.

    Without the env var, resolve to the repository/project ``chroma_db`` folder
    instead of ``./chroma_db`` relative to the current process. MCP clients such
    as Kiro may launch the server with a different cwd; using process cwd here
    makes Chroma look at the wrong database and can surface as missing
    ``default_tenant`` even when the real project DB is healthy.
    """
    env_path = os.getenv("MONORAG_DB_PATH")
    if env_path:
        return env_path
    return str(Path(__file__).resolve().parents[1] / "chroma_db")
