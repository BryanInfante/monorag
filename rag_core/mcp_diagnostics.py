"""Shared MCP diagnostic breadcrumbs written to stderr.

This module intentionally has no heavy dependencies. It can be imported from
RAG internals that may run under the MCP STDIO transport, where stdout must
remain reserved for JSON-RPC protocol bytes.

Diagnostics are only emitted when MONORAG_MCP_DIAGNOSTICS=1 is set (the MCP
server sets this automatically). In CLI or programmatic usage, breadcrumbs
are silently discarded.
"""

from __future__ import annotations

import logging
import os
import sys

_DIAGNOSTICS_ENABLED = os.getenv("MONORAG_MCP_DIAGNOSTICS", "0") == "1"


class _DynamicStderrHandler(logging.StreamHandler):
    """Logging handler that follows the current sys.stderr capture target."""

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stderr
        super().emit(record)


logger = logging.getLogger("rag_core.mcp_diagnostics")
logger.propagate = False

if _DIAGNOSTICS_ENABLED:
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = _DynamicStderrHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s pid=%(process)d thread=%(threadName)s %(levelname)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )
        logger.addHandler(handler)
else:
    logger.setLevel(logging.CRITICAL + 1)  # Effectively disabled


def emit_mcp_breadcrumb(
    event: str,
    *,
    collection: str | None = None,
    detail: str | None = None,
) -> None:
    """Emit an exact MCP diagnostic breadcrumb to stderr."""
    parts = [event]
    if collection is not None:
        parts.append(f"collection={collection}")
    if detail:
        parts.append(f"detail={detail}")
    logger.info(" ".join(parts))
