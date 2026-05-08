"""rag_core package."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_core.module import RAGModule

__all__ = ["RAGModule"]


def __getattr__(name: str):
    """Load heavy public exports lazily.

    Importing rag_core.mcp_server must stay lightweight for MCP startup. The
    legacy `from rag_core import RAGModule` API is preserved, but loading
    sentence-transformers/torch is deferred until RAGModule is actually used.
    """
    if name == "RAGModule":
        from rag_core.module import RAGModule

        return RAGModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
