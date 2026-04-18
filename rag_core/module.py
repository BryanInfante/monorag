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

    def __init__(self, collection: str) -> None:
        """Initialize with a named collection.

        Args:
            collection: Name of the ChromaDB collection to create or connect to.

        Raises:
            ValueError: If collection name is not provided.
            RuntimeError: If GROQ_API_KEY is not set.
        """
        if not collection:
            raise ValueError("Se requiere un nombre de colección.")

        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "La variable de entorno GROQ_API_KEY no está configurada. "
                "Cree un archivo .env con su clave de API de Groq."
            )

        self.chunker = Chunker()
        self.embedder = Embedder()
        self.retriever = Retriever(collection_name=collection)
        self.generator = Generator(api_key=api_key)
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
        answer = self.generator.generate(query, results)
        return {"answer": answer, "sources": results}

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
