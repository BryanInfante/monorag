"""RAGModule: main orchestrator for document indexing, search, and Q&A."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from rag_core.chunker import Chunker
from rag_core.embedder import Embedder
from rag_core.generator import Generator
from rag_core.retriever import Retriever
from rag_core.utils import extract_pdf, extract_txt

logger = logging.getLogger(__name__)


class RAGModule:
    """Main RAG module providing document indexing, search, and Q&A.

    Orchestrates Chunker, Embedder, Retriever, and Generator components
    to provide a unified interface for document processing and retrieval.
    """

    def __init__(self, collection: str, max_history: int = 10, llm_api_key: str | None = None, llm_base_url: str | None = None, llm_model: str | None = None) -> None:
        """Initialize with a named collection.

        Args:
            collection: Name of the ChromaDB collection to create or connect to.
            max_history: Maximum number of conversation history turns to send
                to the Generator. Defaults to 10. A value of 0 disables history.
            llm_api_key: API key for the LLM provider. Falls back to LLM_API_KEY
                env var, then GROQ_API_KEY for backwards compatibility.
            llm_base_url: Base URL for OpenAI-compatible APIs (e.g. Ollama:
                `http://localhost:11434/v1`). Falls back to LLM_BASE_URL env var.
            llm_model: Model identifier. Falls back to LLM_MODEL env var.

        Raises:
            ValueError: If collection name is not provided.
            ValueError: If max_history is negative.
            RuntimeError: If no API key is found.
        """
        if not collection:
            raise ValueError("Se requiere un nombre de colección.")

        if max_history < 0:
            raise ValueError("max_history debe ser mayor o igual a 0.")

        self._history: list[dict] = []
        self._max_history = max_history

        load_dotenv()
        api_key = llm_api_key or os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No se encontró una clave de API. Configure LLM_API_KEY en el archivo .env."
            )
        base_url = llm_base_url or os.getenv("LLM_BASE_URL")
        model_name = llm_model or os.getenv("LLM_MODEL")

        self.chunker = Chunker()
        self.embedder = Embedder()
        self.retriever = Retriever(collection_name=collection)
        kwargs = {"api_key": api_key, "base_url": base_url}
        if model_name:
            kwargs["model"] = model_name
        self.generator = Generator(**kwargs)
        self._deleted = False

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

        # Recursively discover .pdf and .txt files
        files = sorted(
            p for p in dir_path.rglob("*")
            if p.is_file() and p.suffix.lower() in (".pdf", ".txt")
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
        if suffix == ".pdf":
            pages = extract_pdf(str(file_path))
            chunks = self.chunker.chunk_pages(pages, source=filename)
        else:
            text = extract_txt(str(file_path))
            chunks = self.chunker.chunk(text, source=filename)

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
        if path.suffix.lower() not in (".pdf", ".txt"):
            raise ValueError(
                f"Tipo de archivo no soportado: {path.suffix}. "
                "Solo se admiten archivos .pdf y .txt."
            )

        return self._index_file(path)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Semantic search over the collection.

        Args:
            query: Natural language query string.
            top_k: Number of results to return (default 5).

        Returns:
            List of dicts with keys: text, metadata (source, page, chunk_index).

        Raises:
            RuntimeError: If collection is deleted.
            ValueError: If query is empty.
        """
        self._check_deleted()

        if not query or not query.strip():
            raise ValueError("La consulta debe ser una cadena no vacía.")

        query_embedding = self.embedder.embed_query(query)
        return self.retriever.query(query_embedding, top_k=top_k)

    def ask(self, query: str, top_k: int = 5) -> dict:
        """Ask a question and get an LLM-generated answer with sources.

        Args:
            query: Natural language question.
            top_k: Number of context chunks to use (default 5).

        Returns:
            Dict with keys: answer (str), sources (list of chunk dicts).

        Raises:
            RuntimeError: If collection is deleted or Groq API call fails.
            ValueError: If query is empty.
        """
        self._check_deleted()

        if not query or not query.strip():
            raise ValueError("La consulta debe ser una cadena no vacía.")

        results = self.search(query, top_k=top_k)

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
