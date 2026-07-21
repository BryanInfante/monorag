"""Inspector END — Web server that mounts on MonoRAG's MCP HTTP transport.

This is a thin layer that adds static file serving and a simple JSON API
on top of the MCP server. The MCP server handles all RAG operations;
this file just adds the web UI endpoints.

Run with: python server.py
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

load_dotenv()

# Import the MCP server instance
from rag_core.mcp_server import mcp, _get_or_create, _suppress_stdout

from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_PROVIDER,
    MONORAG_COLLECTION,
    MONORAG_DB_PATH,
    SUGGESTED_QUESTIONS,
)
from agent_generator import InspectorGenerator

STATIC_DIR = Path(__file__).parent / "static"

# --- RAG Module with custom generator ---

_rag = None
_rag_error = None


def _get_inspector_rag():
    """Get RAGModule instance with Inspector END personality."""
    global _rag, _rag_error
    if _rag is not None:
        return _rag
    if _rag_error is not None:
        return None

    if not LLM_API_KEY:
        _rag_error = "LLM_API_KEY is not configured."
        return None

    try:
        from rag_core import RAGModule

        generator = InspectorGenerator(
            api_key=LLM_API_KEY,
            model=LLM_MODEL,
            provider_name=LLM_PROVIDER,
            base_url=LLM_BASE_URL,
        )

        kwargs = {
            "collection": MONORAG_COLLECTION,
            "max_history": 10,
            "generator": generator,
        }
        if MONORAG_DB_PATH:
            kwargs["db_path"] = MONORAG_DB_PATH

        _rag = RAGModule(**kwargs)
        return _rag
    except Exception as e:
        _rag_error = str(e)
        return None


def _get_document_count() -> int:
    rag = _get_inspector_rag()
    if rag is None:
        return 0
    try:
        return rag.retriever._collection.count()
    except Exception:
        return 0


# --- Web endpoints ---


async def index(request: Request):
    return FileResponse(str(STATIC_DIR / "index.html"))


async def api_status(request: Request):
    rag = _get_inspector_rag()
    return JSONResponse({
        "connected": rag is not None,
        "document_count": _get_document_count(),
        "collection": MONORAG_COLLECTION,
        "model": LLM_MODEL,
        "provider": LLM_PROVIDER,
        "suggested_questions": SUGGESTED_QUESTIONS,
        "error": _rag_error,
    })


async def api_ask(request: Request):
    body = await request.json()
    query = body.get("query", "").strip()

    if not query:
        return JSONResponse({"error": "La pregunta no puede estar vacía."})

    rag = _get_inspector_rag()
    if rag is None:
        return JSONResponse({"error": _rag_error or "Service not available."})

    try:
        result = rag.ask(query, top_k=5)
        return JSONResponse({
            "answer": result["answer"],
            "sources": result["sources"],
        })
    except RuntimeError as e:
        return JSONResponse({
            "error": f"Error al generar la respuesta: {e}. Intenta de nuevo en un momento."
        })
    except Exception as e:
        return JSONResponse({"error": f"Error inesperado: {e}"})


# --- Starlette app with MCP mounted ---

mcp_app = mcp.http_app(path="/mcp")

app = Starlette(
    routes=[
        Route("/", index),
        Route("/api/status", api_status, methods=["GET"]),
        Route("/api/ask", api_ask, methods=["POST"]),
        Mount("/mcp", app=mcp_app),
        Mount("/static", app=StaticFiles(directory=str(STATIC_DIR)), name="static"),
    ],
    lifespan=mcp_app.lifespan,
)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
