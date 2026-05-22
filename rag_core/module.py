"""RAGModule: main orchestrator for document indexing, search, and Q&A."""

import logging
import os
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from rag_core.mcp_diagnostics import emit_mcp_breadcrumb
from rag_core.secret_store import LLM_API_KEY_SECRET, get_secret, is_keyring_reference
from rag_core.storage_paths import default_config_path

logger = logging.getLogger(__name__)

# Keep this module cheap to import for MCP startup/get_or_create. These names
# intentionally exist for tests and monkeypatching, but the heavy implementations
# are imported lazily at the point of use. Importing ``rag_core.embedder`` pulls
# sentence-transformers; importing ``rag_core.retriever`` pulls ChromaDB.
Chunker = None
Embedder = None
Generator = None
Retriever = None
extract_pdf = None
extract_txt = None


def _load_persisted_runtime_config() -> dict[str, Any]:
    """Load persisted MonoRAG config.json used by pipx/CLI installs."""
    try:
        path = Path(default_config_path())
        if not path.exists():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(raw, dict) and is_keyring_reference(raw.get("llm_api_key")):
        raw["llm_api_key"] = get_secret(LLM_API_KEY_SECRET)
    return raw if isinstance(raw, dict) else {}


def _resolve_setting(
    explicit_value: str | None, env_var_name: str, persisted: dict[str, Any], persisted_key: str
) -> str | None:
    """Resolve a runtime setting with precedence: explicit > env > config.json."""
    if explicit_value not in (None, ""):
        return explicit_value
    env_value = os.getenv(env_var_name)
    if env_value not in (None, ""):
        return env_value
    persisted_value = persisted.get(persisted_key)
    if persisted_value in (None, ""):
        return None
    return str(persisted_value)


def _load_chunker_class():
    global Chunker
    if Chunker is None:
        emit_mcp_breadcrumb("RAGModule:init:before_import_chunker")
        from rag_core.chunker import Chunker as _Chunker

        Chunker = _Chunker
        emit_mcp_breadcrumb("RAGModule:init:after_import_chunker")
    return Chunker


def _load_embedder_class():
    global Embedder
    if Embedder is None:
        emit_mcp_breadcrumb("RAGModule:init:before_import_embedder")
        from rag_core.embedder import Embedder as _Embedder

        Embedder = _Embedder
        emit_mcp_breadcrumb("RAGModule:init:after_import_embedder")
    return Embedder


def _load_retriever_class():
    global Retriever
    if Retriever is None:
        emit_mcp_breadcrumb("RAGModule:init:before_import_retriever")
        from rag_core.retriever import Retriever as _Retriever

        Retriever = _Retriever
        emit_mcp_breadcrumb("RAGModule:init:after_import_retriever")
    return Retriever


def _load_generator_class():
    global Generator
    if Generator is None:
        emit_mcp_breadcrumb("RAGModule:init:before_import_generator")
        from rag_core.generator import Generator as _Generator

        Generator = _Generator
        emit_mcp_breadcrumb("RAGModule:init:after_import_generator")
    return Generator


def _load_extractors():
    global extract_pdf, extract_txt
    if extract_pdf is None or extract_txt is None:
        emit_mcp_breadcrumb("RAGModule:index:before_import_extractors")
        from rag_core.utils import extract_pdf as _extract_pdf, extract_txt as _extract_txt

        extract_pdf = _extract_pdf
        extract_txt = _extract_txt
        emit_mcp_breadcrumb("RAGModule:index:after_import_extractors")
    return extract_pdf, extract_txt


class RAGModule:
    """Main RAG module providing document indexing, search, and Q&A.

    Orchestrates Chunker, Embedder, Retriever, and Generator components
    to provide a unified interface for document processing and retrieval.
    """

    def __init__(
        self,
        collection: str,
        max_history: int = 10,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
        llm_model: str | None = None,
        llm_provider: str | None = None,
        db_path: str | None = None,
        db_url: str | None = None,
        db_api_key: str | None = None,
        db_tenant: str | None = None,
        db_database: str | None = None,
        generator: Any | None = None,
        retriever: Any | None = None,
        embedder: Any | None = None,
        chunker: Any | None = None,
    ) -> None:
        """Initialize with a named collection.

        Args:
            collection: Name of the collection to create or connect to.
            max_history: Maximum number of conversation history turns to send
                to the Generator. Defaults to 10. A value of 0 disables history.
            chunk_size: Maximum number of tokens per chunk. Configure this from
                callers such as the CLI, not from ``.env``.
            chunk_overlap: Number of overlapping tokens between consecutive
                chunks. Configure this from callers such as the CLI, not from
                ``.env``.
            llm_api_key: API key for the LLM provider. Falls back to LLM_API_KEY
                env var, then GROQ_API_KEY for backwards compatibility.
            llm_base_url: Base URL for OpenAI-compatible APIs (e.g. Ollama:
                `http://localhost:11434/v1`). Falls back to LLM_BASE_URL env var.
            llm_model: Model identifier. Falls back to LLM_MODEL env var.
            llm_provider: Built-in LLM provider alias. Falls back to
                LLM_PROVIDER env var. Examples: ``openai``, ``groq``,
                ``google-ai-studio``, ``ollama``, ``lm-studio``. Non-compatible
                providers can be passed through a custom ``generator``.
            db_path: Optional local ChromaDB persistence directory. When omitted,
                the bundled Chroma adapter uses ``MONORAG_DB_PATH`` or the
                project ``chroma_db`` folder.
            db_url: Optional remote Chroma HTTP(S) URL. When omitted, the bundled
                adapter checks ``MONORAG_CHROMA_URL``/``CHROMA_URL``.
            db_api_key: Optional API key for remote/hosted Chroma deployments.
            db_tenant: Optional Chroma tenant for hosted deployments.
            db_database: Optional Chroma database for hosted deployments.
            generator: Optional custom generator object implementing
                ``generate(query, context_chunks, history=None)``. Passing this
                makes RAGModule independent from any built-in LLM provider.
            retriever: Optional custom retriever object implementing the
                retriever methods used by RAGModule. Passing this makes RAGModule
                independent from ChromaDB.
            embedder: Optional custom embedder object implementing ``embed`` and
                ``embed_query``.
            chunker: Optional custom chunker object implementing ``chunk`` and
                ``chunk_pages``.

        Raises:
            ValueError: If collection name is not provided.
            ValueError: If max_history is negative.
            ValueError: If chunk parameters are invalid.
            RuntimeError: If no API key is found and no custom generator is
                provided.
        """
        emit_mcp_breadcrumb("RAGModule:init:start", collection=collection)
        if not collection:
            raise ValueError("Se requiere un nombre de colección.")

        if max_history < 0:
            raise ValueError("max_history debe ser mayor o igual a 0.")
        if chunk_size < 1:
            raise ValueError("chunk_size debe ser mayor o igual a 1.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap debe ser mayor o igual a 0.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap debe ser menor que chunk_size.")

        self._history: list[dict] = []
        self._max_history = max_history

        load_dotenv()
        persisted_config = _load_persisted_runtime_config()

        if chunker is None:
            ChunkerClass = _load_chunker_class()
            self.chunker = ChunkerClass(chunk_size=chunk_size, overlap=chunk_overlap)
        else:
            self.chunker = chunker

        if embedder is None:
            emit_mcp_breadcrumb("RAGModule:init:before_embedder", collection=collection)
            EmbedderClass = _load_embedder_class()
            self.embedder = EmbedderClass()
            emit_mcp_breadcrumb("RAGModule:init:after_embedder", collection=collection)
        else:
            self.embedder = embedder

        if retriever is None:
            emit_mcp_breadcrumb("RAGModule:init:before_retriever", collection=collection)
            RetrieverClass = _load_retriever_class()
            self.retriever = RetrieverClass(
                collection_name=collection,
                persist_dir=db_path,
                remote_url=db_url,
                api_key=db_api_key,
                tenant=db_tenant,
                database=db_database,
            )
            emit_mcp_breadcrumb("RAGModule:init:after_retriever", collection=collection)
        else:
            self.retriever = retriever

        if generator is None:
            provider_name = _resolve_setting(
                llm_provider, "LLM_PROVIDER", persisted_config, "llm_provider"
            )
            api_key = _resolve_setting(
                llm_api_key, "LLM_API_KEY", persisted_config, "llm_api_key"
            )
            if not api_key:
                legacy_groq_api_key = os.getenv("GROQ_API_KEY")
                if legacy_groq_api_key:
                    api_key = legacy_groq_api_key
                    provider_name = provider_name or "groq"
            if not api_key:
                raise RuntimeError(
                    "No se encontró una clave de API. Configure LLM_API_KEY en el archivo .env "
                    "o inyecte un generator personalizado."
                )
            base_url = _resolve_setting(
                llm_base_url, "LLM_BASE_URL", persisted_config, "llm_base_url"
            )
            model_name = _resolve_setting(
                llm_model, "LLM_MODEL", persisted_config, "llm_model"
            )

            kwargs = {
                "api_key": api_key,
                "base_url": base_url,
                "provider_name": provider_name,
            }
            if model_name:
                kwargs["model"] = model_name
            emit_mcp_breadcrumb("RAGModule:init:before_generator", collection=collection)
            GeneratorClass = _load_generator_class()
            self.generator = GeneratorClass(**kwargs)
            emit_mcp_breadcrumb("RAGModule:init:after_generator", collection=collection)
        else:
            self.generator = generator

        self._deleted = False
        emit_mcp_breadcrumb("RAGModule:init:return", collection=collection)

    def _check_deleted(self) -> None:
        """Check if the collection has been deleted and raise if so.

        Raises:
            RuntimeError: If the collection has been deleted.
        """
        if self._deleted:
            raise RuntimeError("La colección ha sido eliminada.")

    def add_documents(self, directory: str) -> int:
        """Index all PDF and TXT files from a directory recursively.

        Files whose filename already exists in the collection are skipped
        with a logged warning.

        Args:
            directory: Path to directory containing documents.

        Returns:
            Number of chunks indexed (excluding skipped duplicates).

        Raises:
            RuntimeError: If collection is deleted.
            FileNotFoundError: If directory does not exist.
            ValueError: If path is not a directory.
        """
        self._check_deleted()

        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {directory}")
        if not dir_path.is_dir():
            raise ValueError(f"La ruta no es un directorio: {directory}")

        # Recursively discover .pdf, .txt, and .md files
        files = sorted(
            p for p in dir_path.rglob("*")
            if p.is_file() and p.suffix.lower() in (".pdf", ".txt", ".md")
        )

        total_chunks = 0
        for file_path in files:
            count = self._index_file(file_path)
            total_chunks += count

        return total_chunks

    def _index_file(self, file_path: Path) -> int:
        """Index a single file into the collection.

        Handles duplicate detection, text extraction, chunking, embedding,
        and storage. Shared by add_documents and add_file.

        Args:
            file_path: Path object pointing to the file.

        Returns:
            Number of chunks indexed (0 if skipped as duplicate).
        """
        filename = file_path.name

        if self.retriever.has_source(filename):
            logger.warning(
                "Archivo '%s' ya existe en la colección, se omite.",
                filename,
            )
            return 0

        suffix = file_path.suffix.lower()
        extract_pdf_func, extract_txt_func = _load_extractors()
        if suffix == ".pdf":
            pages = extract_pdf_func(str(file_path))
            chunks = self.chunker.chunk_pages(pages, source=filename)
        else:
            text = extract_txt_func(str(file_path))
            chunks = self.chunker.chunk(text, source=filename, start_page=None)

        if not chunks:
            return 0

        ids = [f"{filename}_{c['metadata']['chunk_index']}" for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        embeddings = self.embedder.embed(documents)

        self.retriever.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(chunks)

    def add_file(self, file_path: str) -> int:
        """Index a single PDF or TXT file.

        If the file's name already exists in the collection, the file is
        skipped with a logged warning and 0 is returned.

        Args:
            file_path: Path to the file.

        Returns:
            Number of chunks indexed (0 if file was skipped as duplicate).

        Raises:
            RuntimeError: If collection is deleted.
            FileNotFoundError: If file does not exist.
            ValueError: If file type is not PDF or TXT.
        """
        self._check_deleted()

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        if path.suffix.lower() not in (".pdf", ".txt", ".md"):
            raise ValueError(
                f"Tipo de archivo no soportado: {path.suffix}. "
                "Solo se admiten archivos .pdf, .txt y .md."
            )

        return self._index_file(path)

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Semantic search over the collection.

        Args:
            query: Natural language query string.
            top_k: Number of results to return. Falls back to MONORAG_TOP_K
                env var, then 5.

        Returns:
            List of dicts with keys: text, metadata (source, page, chunk_index).

        Raises:
            RuntimeError: If collection is deleted.
            ValueError: If query is empty.
        """
        self._check_deleted()

        if not query or not query.strip():
            raise ValueError("La consulta debe ser una cadena no vacía.")

        resolved_top_k = top_k if top_k is not None else int(os.getenv("MONORAG_TOP_K", "5"))

        emit_mcp_breadcrumb("RAGModule:search:before_embed_query")
        query_embedding = self.embedder.embed_query(query)
        emit_mcp_breadcrumb("RAGModule:search:after_embed_query")
        emit_mcp_breadcrumb("RAGModule:search:before_retriever_query")
        if callable(getattr(type(self.retriever), "hybrid_query", None)):
            results = self.retriever.hybrid_query(query, query_embedding, top_k=resolved_top_k)
        else:
            results = self.retriever.query(query_embedding, top_k=resolved_top_k)
        emit_mcp_breadcrumb("RAGModule:search:after_retriever_query")
        emit_mcp_breadcrumb("RAGModule:search:return")
        return results

    def ask(self, query: str, top_k: int | None = None) -> dict:
        """Ask a question and get an LLM-generated answer with sources.

        Args:
            query: Natural language question.
            top_k: Number of context chunks to use. Falls back to MONORAG_TOP_K
                env var, then 5.

        Returns:
            Dict with keys: answer (str), sources (list of chunk dicts).

        Raises:
            RuntimeError: If collection is deleted or the LLM provider call fails.
            ValueError: If query is empty.
        """
        self._check_deleted()

        if not query or not query.strip():
            raise ValueError("La consulta debe ser una cadena no vacía.")

        resolved_top_k = top_k if top_k is not None else int(os.getenv("MONORAG_TOP_K", "5"))
        results = self.search(query, top_k=resolved_top_k)

        # Guard: return predefined message when no relevant documents are found
        if not results:
            return {
                "answer": "No se encontraron documentos relevantes en la colección. "
                          "Indexe documentos antes de hacer preguntas.",
                "sources": [],
            }

        # Slice conversation history for the Generator
        history_slice = self._history[-self._max_history:] if self._max_history > 0 else []

        answer = self.generator.generate(query, results, history=history_slice)

        # Append turn only after successful generation
        self._history.append({"query": query, "answer": answer})

        return {"answer": answer, "sources": results}

    def clear_history(self) -> None:
        """Clear all conversation history turns.

        This method works regardless of collection state (_deleted flag).
        """
        self._history = []

    def delete_collection(self) -> None:
        """Delete the active collection and all its data.

        Raises:
            RuntimeError: If collection is already deleted.
        """
        if self._deleted:
            raise RuntimeError("La colección ya ha sido eliminada.")

        self.retriever.delete_collection()
        self._deleted = True

    def list_collections(self) -> list[str]:
        """List all collection names in the persist directory.

        Returns:
            List of collection name strings.
        """
        return self.retriever.list_collections()
