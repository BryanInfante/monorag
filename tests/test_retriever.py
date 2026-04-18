"""Integration tests for rag_core.retriever.Retriever and full RAG pipelines.

Tests use real ChromaDB with temporary directories and mock Embedder/Generator
to avoid loading sentence-transformers models and calling the Groq API.
"""

import os
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from rag_core.retriever import Retriever


# ---------------------------------------------------------------------------
# 12.1 — Retriever integration test with real ChromaDB
# ---------------------------------------------------------------------------


class TestRetrieverIntegration:
    """Integration tests for Retriever with real ChromaDB (temp directory)."""

    def test_add_and_query_returns_correct_structure(self, tmp_path):
        """Add chunks to ChromaDB, query by embedding, verify result structure and metadata."""
        chroma_dir = str(tmp_path / "chroma_db")
        retriever = Retriever(collection_name="test-collection", persist_dir=chroma_dir)

        # Prepare sample chunks
        ids = ["doc.txt_0", "doc.txt_1", "doc.txt_2"]
        documents = [
            "First chunk about machine learning algorithms.",
            "Second chunk about natural language processing.",
            "Third chunk about computer vision techniques.",
        ]
        embeddings = [
            [1.0] + [0.0] * 383,
            [0.0] + [1.0] + [0.0] * 382,
            [0.0] * 2 + [1.0] + [0.0] * 381,
        ]
        metadatas = [
            {"source": "doc.txt", "page": 0, "chunk_index": 0},
            {"source": "doc.txt", "page": 0, "chunk_index": 1},
            {"source": "doc.txt", "page": 0, "chunk_index": 2},
        ]

        retriever.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

        # Query with an embedding close to the first chunk
        query_embedding = [1.0] + [0.0] * 383
        results = retriever.query(query_embedding, top_k=2)

        assert isinstance(results, list)
        assert len(results) == 2

        for result in results:
            assert "text" in result
            assert "metadata" in result
            assert isinstance(result["text"], str)
            assert len(result["text"]) > 0
            meta = result["metadata"]
            assert "source" in meta
            assert "page" in meta
            assert "chunk_index" in meta
            assert meta["source"] == "doc.txt"

        # The first result should be the closest match
        assert results[0]["text"] == "First chunk about machine learning algorithms."
        assert results[0]["metadata"]["chunk_index"] == 0

    def test_has_source_returns_true_for_existing(self, tmp_path):
        """has_source returns True for a source that has been added."""
        chroma_dir = str(tmp_path / "chroma_db")
        retriever = Retriever(collection_name="src-check", persist_dir=chroma_dir)

        retriever.add(
            ids=["file.txt_0"],
            documents=["Some text content."],
            embeddings=[[0.5] * 384],
            metadatas=[{"source": "file.txt", "page": 0, "chunk_index": 0}],
        )

        assert retriever.has_source("file.txt") is True
        assert retriever.has_source("other.txt") is False

    def test_delete_collection_removes_data(self, tmp_path):
        """delete_collection removes the collection from ChromaDB."""
        chroma_dir = str(tmp_path / "chroma_db")
        retriever = Retriever(collection_name="to-delete", persist_dir=chroma_dir)

        retriever.add(
            ids=["a_0"],
            documents=["text"],
            embeddings=[[0.1] * 384],
            metadatas=[{"source": "a.txt", "page": 0, "chunk_index": 0}],
        )

        retriever.delete_collection()

        # After deletion, the collection should not appear in list
        collections = retriever.list_collections()
        assert "to-delete" not in collections

    def test_query_empty_collection_returns_empty(self, tmp_path):
        """Querying an empty collection returns an empty list."""
        chroma_dir = str(tmp_path / "chroma_db")
        retriever = Retriever(collection_name="empty-col", persist_dir=chroma_dir)

        results = retriever.query([0.0] * 384, top_k=5)
        assert results == []

    def test_list_collections_includes_created(self, tmp_path):
        """list_collections returns names of all created collections."""
        chroma_dir = str(tmp_path / "chroma_db")
        r1 = Retriever(collection_name="col-alpha", persist_dir=chroma_dir)
        r2 = Retriever(collection_name="col-beta", persist_dir=chroma_dir)

        names = r1.list_collections()
        assert "col-alpha" in names
        assert "col-beta" in names


# ---------------------------------------------------------------------------
# Helper: build a RAGModule with mocked Embedder and Generator,
# but real Chunker and real Retriever (temp ChromaDB dir).
# ---------------------------------------------------------------------------

from rag_core.module import RAGModule
from rag_core.chunker import Chunker


EMBEDDING_DIM = 384
FAKE_EMBEDDING = [0.0] * EMBEDDING_DIM
CANNED_ANSWER = "Esta es una respuesta generada de prueba."


def _make_integration_module(collection_name, chroma_dir, monkeypatch):
    """Build a RAGModule with real Chunker + Retriever, mocked Embedder + Generator.

    Returns the module instance plus the mock objects for Embedder and Generator.
    """
    monkeypatch.setenv("GROQ_API_KEY", "test-fake-api-key-integration")

    with patch("rag_core.module.Embedder") as MockEmbedder, \
         patch("rag_core.module.Generator") as MockGenerator:

        # Configure mock Embedder
        mock_embedder = MagicMock()
        mock_embedder.embed.side_effect = lambda texts: [FAKE_EMBEDDING] * len(texts)
        mock_embedder.embed_query.return_value = FAKE_EMBEDDING
        MockEmbedder.return_value = mock_embedder

        # Configure mock Generator
        mock_generator = MagicMock()
        mock_generator.generate.return_value = CANNED_ANSWER
        MockGenerator.return_value = mock_generator

        # Patch Retriever to use the temp chroma dir
        with patch("rag_core.module.Retriever") as MockRetriever:
            real_retriever = Retriever(
                collection_name=collection_name, persist_dir=chroma_dir
            )
            MockRetriever.return_value = real_retriever

            module = RAGModule(collection_name)

    return module, mock_embedder, mock_generator


# ---------------------------------------------------------------------------
# 12.2 — Full indexing pipeline integration test
# ---------------------------------------------------------------------------


class TestIndexingPipelineIntegration:
    """Integration test: add a known TXT file via RAGModule, verify chunks in ChromaDB."""

    def test_add_file_indexes_txt_with_correct_metadata(self, tmp_path, monkeypatch):
        """Add a known TXT file, verify chunks stored in ChromaDB with correct metadata."""
        chroma_dir = str(tmp_path / "chroma_db")

        # Create a known TXT file with enough tokens to produce multiple chunks
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        words = [f"word{i}" for i in range(600)]
        txt_content = " ".join(words)
        txt_file = docs_dir / "sample.txt"
        txt_file.write_text(txt_content, encoding="utf-8")

        module, mock_embedder, _ = _make_integration_module(
            "indexing-test", chroma_dir, monkeypatch
        )

        chunk_count = module.add_file(str(txt_file))

        # With 600 tokens, chunk_size=500, overlap=50 → step=450
        # Chunks: [0:500], [450:600] → 2 chunks
        assert chunk_count == 2

        # Verify embedder was called with the chunk texts
        assert mock_embedder.embed.call_count == 1
        embedded_texts = mock_embedder.embed.call_args[0][0]
        assert len(embedded_texts) == 2

        # Verify chunks are stored in ChromaDB by querying
        results = module.search("word0")
        assert len(results) > 0

        for result in results:
            assert result["metadata"]["source"] == "sample.txt"
            assert result["metadata"]["page"] == 0
            assert isinstance(result["metadata"]["chunk_index"], int)


# ---------------------------------------------------------------------------
# 12.3 — Search pipeline integration test
# ---------------------------------------------------------------------------


class TestSearchPipelineIntegration:
    """Integration test: index known documents, search, verify relevant results."""

    def test_search_returns_relevant_results(self, tmp_path, monkeypatch):
        """Index known documents, search, verify results contain expected structure."""
        chroma_dir = str(tmp_path / "chroma_db")

        # Create two TXT files with distinct content
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        file_a = docs_dir / "alpha.txt"
        file_a.write_text("alpha " * 100, encoding="utf-8")

        file_b = docs_dir / "beta.txt"
        file_b.write_text("beta " * 100, encoding="utf-8")

        module, _, _ = _make_integration_module(
            "search-test", chroma_dir, monkeypatch
        )

        module.add_file(str(file_a))
        module.add_file(str(file_b))

        results = module.search("alpha content", top_k=3)

        assert isinstance(results, list)
        assert len(results) > 0

        for result in results:
            assert "text" in result
            assert "metadata" in result
            assert isinstance(result["text"], str)
            assert len(result["text"]) > 0
            meta = result["metadata"]
            assert "source" in meta
            assert "page" in meta
            assert "chunk_index" in meta
            assert meta["source"] in ("alpha.txt", "beta.txt")


# ---------------------------------------------------------------------------
# 12.4 — Ask pipeline integration test
# ---------------------------------------------------------------------------


class TestAskPipelineIntegration:
    """Integration test: index known documents, ask question (mock Groq), verify answer + sources."""

    def test_ask_returns_answer_and_sources(self, tmp_path, monkeypatch):
        """Index documents, ask a question, verify answer + sources structure."""
        chroma_dir = str(tmp_path / "chroma_db")

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        txt_file = docs_dir / "knowledge.txt"
        txt_file.write_text("knowledge " * 100, encoding="utf-8")

        module, _, mock_generator = _make_integration_module(
            "ask-test", chroma_dir, monkeypatch
        )

        module.add_file(str(txt_file))

        result = module.ask("What is the knowledge about?")

        assert isinstance(result, dict)
        assert "answer" in result
        assert "sources" in result
        assert result["answer"] == CANNED_ANSWER
        assert isinstance(result["sources"], list)
        assert len(result["sources"]) > 0

        for source in result["sources"]:
            assert "text" in source
            assert "metadata" in source
            meta = source["metadata"]
            assert meta["source"] == "knowledge.txt"
            assert "page" in meta
            assert "chunk_index" in meta

        # Verify generator was called with the query and context chunks
        mock_generator.generate.assert_called_once()
        call_args = mock_generator.generate.call_args
        assert call_args[0][0] == "What is the knowledge about?"
        assert isinstance(call_args[0][1], list)


# ---------------------------------------------------------------------------
# 12.5 — Collection lifecycle integration test
# ---------------------------------------------------------------------------


class TestCollectionLifecycleIntegration:
    """Integration test: create → index → search → delete → verify cleanup."""

    def test_full_collection_lifecycle(self, tmp_path, monkeypatch):
        """Create collection, index, search, delete, verify operations fail after delete."""
        chroma_dir = str(tmp_path / "chroma_db")

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        txt_file = docs_dir / "lifecycle.txt"
        txt_file.write_text("lifecycle test content " * 50, encoding="utf-8")

        module, _, _ = _make_integration_module(
            "lifecycle-test", chroma_dir, monkeypatch
        )

        # Index
        count = module.add_file(str(txt_file))
        assert count > 0

        # Search
        results = module.search("lifecycle")
        assert len(results) > 0

        # List collections — should include our collection
        collections = module.list_collections()
        assert "lifecycle-test" in collections

        # Delete
        module.delete_collection()

        # After deletion, operations should raise RuntimeError
        with pytest.raises(RuntimeError, match="La colección ha sido eliminada"):
            module.search("lifecycle")

        with pytest.raises(RuntimeError, match="La colección ha sido eliminada"):
            module.add_file(str(txt_file))

        with pytest.raises(RuntimeError, match="La colección ha sido eliminada"):
            module.ask("lifecycle question")


# ---------------------------------------------------------------------------
# 12.6 — Multi-collection isolation integration test
# ---------------------------------------------------------------------------


class TestMultiCollectionIsolationIntegration:
    """Integration test: create two collections, verify data isolation."""

    def test_collections_are_isolated(self, tmp_path, monkeypatch):
        """Data in one collection should not appear in another."""
        chroma_dir = str(tmp_path / "chroma_db")

        # Create distinct documents for each collection
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        file_col1 = docs_dir / "collection1_doc.txt"
        file_col1.write_text("unique_alpha " * 100, encoding="utf-8")

        file_col2 = docs_dir / "collection2_doc.txt"
        file_col2.write_text("unique_beta " * 100, encoding="utf-8")

        # Build two modules with separate collections but same chroma dir
        module1, _, _ = _make_integration_module(
            "isolation-col1", chroma_dir, monkeypatch
        )
        module2, _, _ = _make_integration_module(
            "isolation-col2", chroma_dir, monkeypatch
        )

        module1.add_file(str(file_col1))
        module2.add_file(str(file_col2))

        # Search in collection 1 — should only find collection1_doc
        results1 = module1.search("unique_alpha")
        assert len(results1) > 0
        for r in results1:
            assert r["metadata"]["source"] == "collection1_doc.txt"

        # Search in collection 2 — should only find collection2_doc
        results2 = module2.search("unique_beta")
        assert len(results2) > 0
        for r in results2:
            assert r["metadata"]["source"] == "collection2_doc.txt"

        # Cross-check: collection 1 should not have collection 2's source
        assert not module1.retriever.has_source("collection2_doc.txt")
        assert not module2.retriever.has_source("collection1_doc.txt")
