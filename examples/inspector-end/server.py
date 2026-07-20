"""Inspector END — FastAPI server.

Serves the static HTML/CSS/JS frontend and provides API endpoints
for the chat functionality backed by MonoRAG's RAGModule.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()

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

app = FastAPI(title="Inspector END", docs_url=None, redoc_url=None)

# --- RAG Module Singleton ---

_rag = None
_rag_error = None


def get_rag():
    """Lazily initialize and return the RAGModule instance."""
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


def get_document_count() -> int:
    """Get document count from the active collection."""
    rag = get_rag()
    if rag is None:
        return 0
    try:
        return rag.retriever._collection.count()
    except Exception:
        return 0


# --- API Models ---


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str | None = None
    sources: list | None = None
    error: str | None = None


class StatusResponse(BaseModel):
    connected: bool
    document_count: int
    collection: str
    model: str
    provider: str
    suggested_questions: list[str]
    error: str | None = None


# --- API Endpoints ---


@app.get("/api/status")
def status() -> StatusResponse:
    """Return current connection status and configuration."""
    rag = get_rag()
    return StatusResponse(
        connected=rag is not None,
        document_count=get_document_count(),
        collection=MONORAG_COLLECTION,
        model=LLM_MODEL,
        provider=LLM_PROVIDER,
        suggested_questions=SUGGESTED_QUESTIONS,
        error=_rag_error,
    )


@app.post("/api/ask")
def ask(request: AskRequest) -> AskResponse:
    """Ask a question and return the answer with sources."""
    rag = get_rag()
    if rag is None:
        return AskResponse(error=_rag_error or "Service not available.")

    if not request.query.strip():
        return AskResponse(error="La pregunta no puede estar vacía.")

    try:
        result = rag.ask(request.query, top_k=5)
        return AskResponse(
            answer=result["answer"],
            sources=result["sources"],
        )
    except RuntimeError as e:
        return AskResponse(
            error=f"Error al generar la respuesta: {e}. Intenta de nuevo en un momento."
        )
    except Exception as e:
        return AskResponse(error=f"Error inesperado: {e}")


# --- Static Files ---

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    """Serve the main HTML page."""
    return FileResponse(str(STATIC_DIR / "index.html"))
