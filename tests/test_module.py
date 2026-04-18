"""Unit tests for rag_core.module.RAGModule.

Validates: Requirements 1.2, 1.5, 2.9, 2.10, 3.6, 3.7, 4.5, 2.8, 3.5, 6.2, 11.1, 11.2, 11.3
"""

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_core.module import RAGModule


def _build_rag_module(collection="test-col", monkeypatch=None, env_key="fake-key"):
    """Helper to build a RAGModule with all heavy deps mocked.

    Patches Embedder, Retriever, Generator, and load_dotenv so that
    no real model loading, ChromaDB, or Groq connections happen.
    """
    patches = {
        "embedder": patch("rag_core.module.Embedder"),
        "retriever": patch("rag_core.module.Retriever"),
        "generator": patch("rag_core.module.Generator"),
        "dotenv": patch("rag_core.module.load_dotenv"),
    }
    mocks = {k: p.start() for k, p in patches.items()}

    if monkeypatch:
        monkeypatch.setenv("GROQ_API_KEY", env_key)
    else:
        os.environ["GROQ_API_KEY"] = env_key

    module = RAGModule(collection)
    return module, mocks, patches


def _stop_patches(patches):
    """Stop all active patches."""
    for p in patches.values():
        p.stop()


class TestRAGModuleInit:
    """Tests for RAGModule initialization validation."""

    def test_missing_collection_name_raises_value_error(self, monkeypatch):
        """Empty collection name should raise ValueError with Spanish message."""
        with pytest.raises(ValueError, match="Se requiere un nombre de colección"):
            with patch("rag_core.module.load_dotenv"):
                RAGModule("")

    def test_none_collection_name_raises_value_error(self, monkeypatch):
        """None collection name should raise ValueError with Spanish message."""
        with pytest.raises(ValueError, match="Se requiere un nombre de colección"):
            with patch("rag_core.module.load_dotenv"):
                RAGModule(None)

    def test_missing_api_key_raises_runtime_error(self, monkeypatch):
        """Missing GROQ_API_KEY should raise RuntimeError with Spanish message."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with patch("rag_core.module.load_dotenv"):
            with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
                RAGModule("test-col")


class TestRAGModuleAddDocuments:
    """Tests for add_documents error handling."""

    def test_nonexistent_directory_raises_file_not_found(self, monkeypatch):
        """Non-existent directory should raise FileNotFoundError with Spanish message."""
        module, mocks, patches = _build_rag_module(monkeypatch=monkeypatch)
        try:
            with pytest.raises(FileNotFoundError, match="Directorio no encontrado"):
                module.add_documents("/nonexistent/path/to/nowhere")
        finally:
            _stop_patches(patches)

    def test_non_directory_path_raises_value_error(self, tmp_path, monkeypatch):
        """A file path (not a directory) should raise ValueError with Spanish message."""
        module, mocks, patches = _build_rag_module(monkeypatch=monkeypatch)
        try:
            regular_file = tmp_path / "not_a_dir.txt"
            regular_file.write_text("content")
            with pytest.raises(ValueError, match="La ruta no es un directorio"):
                module.add_documents(str(regular_file))
        finally:
            _stop_patches(patches)


class TestRAGModuleAddFile:
    """Tests for add_file error handling."""

    def test_unsupported_file_type_raises_value_error(self, tmp_path, monkeypatch):
        """Unsupported file extension should raise ValueError with Spanish message."""
        module, mocks, patches = _build_rag_module(monkeypatch=monkeypatch)
        try:
            csv_file = tmp_path / "data.csv"
            csv_file.write_text("a,b,c")
            with pytest.raises(ValueError, match="Tipo de archivo no soportado"):
                module.add_file(str(csv_file))
        finally:
            _stop_patches(patches)

    def test_nonexistent_file_raises_file_not_found(self, monkeypatch):
        """Non-existent file should raise FileNotFoundError with Spanish message."""
        module, mocks, patches = _build_rag_module(monkeypatch=monkeypatch)
        try:
            with pytest.raises(FileNotFoundError, match="Archivo no encontrado"):
                module.add_file("/nonexistent/file.pdf")
        finally:
            _stop_patches(patches)

    def test_duplicate_file_returns_zero_and_logs_warning(
        self, tmp_path, monkeypatch, caplog
    ):
        """Duplicate file should return 0 chunks and log a Spanish warning."""
        module, mocks, patches = _build_rag_module(monkeypatch=monkeypatch)
        try:
            # Configure retriever mock to report the file already exists
            module.retriever.has_source.return_value = True

            txt_file = tmp_path / "existing.txt"
            txt_file.write_text("some content")

            with caplog.at_level(logging.WARNING, logger="rag_core.module"):
                result = module.add_file(str(txt_file))

            assert result == 0
            assert "ya existe en la colección" in caplog.text
        finally:
            _stop_patches(patches)


class TestRAGModuleSearch:
    """Tests for search validation."""

    def test_empty_query_raises_value_error(self, monkeypatch):
        """Empty query string should raise ValueError with Spanish message."""
        module, mocks, patches = _build_rag_module(monkeypatch=monkeypatch)
        try:
            with pytest.raises(ValueError, match="La consulta debe ser una cadena no vacía"):
                module.search("")
        finally:
            _stop_patches(patches)

    def test_whitespace_query_raises_value_error(self, monkeypatch):
        """Whitespace-only query should raise ValueError with Spanish message."""
        module, mocks, patches = _build_rag_module(monkeypatch=monkeypatch)
        try:
            with pytest.raises(ValueError, match="La consulta debe ser una cadena no vacía"):
                module.search("   ")
        finally:
            _stop_patches(patches)


class TestRAGModuleDeletedCollection:
    """Tests for operations on a deleted collection."""

    def test_add_documents_on_deleted_raises_runtime_error(
        self, tmp_path, monkeypatch
    ):
        """add_documents after delete should raise RuntimeError with Spanish message."""
        module, mocks, patches = _build_rag_module(monkeypatch=monkeypatch)
        try:
            module.delete_collection()
            with pytest.raises(RuntimeError, match="La colección ha sido eliminada"):
                module.add_documents(str(tmp_path))
        finally:
            _stop_patches(patches)

    def test_add_file_on_deleted_raises_runtime_error(
        self, tmp_path, monkeypatch
    ):
        """add_file after delete should raise RuntimeError with Spanish message."""
        module, mocks, patches = _build_rag_module(monkeypatch=monkeypatch)
        try:
            txt_file = tmp_path / "test.txt"
            txt_file.write_text("content")
            module.delete_collection()
            with pytest.raises(RuntimeError, match="La colección ha sido eliminada"):
                module.add_file(str(txt_file))
        finally:
            _stop_patches(patches)

    def test_search_on_deleted_raises_runtime_error(self, monkeypatch):
        """search after delete should raise RuntimeError with Spanish message."""
        module, mocks, patches = _build_rag_module(monkeypatch=monkeypatch)
        try:
            module.delete_collection()
            with pytest.raises(RuntimeError, match="La colección ha sido eliminada"):
                module.search("test query")
        finally:
            _stop_patches(patches)

    def test_ask_on_deleted_raises_runtime_error(self, monkeypatch):
        """ask after delete should raise RuntimeError with Spanish message."""
        module, mocks, patches = _build_rag_module(monkeypatch=monkeypatch)
        try:
            module.delete_collection()
            with pytest.raises(RuntimeError, match="La colección ha sido eliminada"):
                module.ask("test query")
        finally:
            _stop_patches(patches)

    def test_double_delete_raises_runtime_error(self, monkeypatch):
        """Deleting an already-deleted collection should raise RuntimeError."""
        module, mocks, patches = _build_rag_module(monkeypatch=monkeypatch)
        try:
            module.delete_collection()
            with pytest.raises(RuntimeError, match="La colección ya ha sido eliminada"):
                module.delete_collection()
        finally:
            _stop_patches(patches)


# ---------------------------------------------------------------------------
# Property-Based Tests (Hypothesis)
# ---------------------------------------------------------------------------

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st


def _make_rag_module_with_patches():
    """Create a RAGModule with all heavy deps mocked, returning (module, patches).

    Caller must call _stop_all(patches) when done.
    """
    p = {
        "embedder": patch("rag_core.module.Embedder"),
        "retriever": patch("rag_core.module.Retriever"),
        "generator": patch("rag_core.module.Generator"),
        "dotenv": patch("rag_core.module.load_dotenv"),
    }
    mocks = {k: v.start() for k, v in p.items()}
    os.environ["GROQ_API_KEY"] = "fake-key-for-pbt"
    module = RAGModule("pbt-collection")
    return module, mocks, p


def _stop_all(patches):
    """Stop all active patches and clean up env."""
    for p in patches.values():
        p.stop()
    os.environ.pop("GROQ_API_KEY", None)


# Strategy: generate a list of filenames with mixed extensions
_extensions = st.sampled_from([".pdf", ".txt", ".csv", ".docx", ".py", ".md", ".jpg"])
_filename_st = st.tuples(
    st.from_regex(r"[a-z]{1,8}", fullmatch=True),
    _extensions,
).map(lambda t: t[0] + t[1])


class TestRAGModuleFileDiscoveryProperty:
    """Property-based test for file discovery."""

    # Feature: rag-core, Property 6: File discovery returns only PDF and TXT files
    @given(
        filenames=st.lists(_filename_st, min_size=1, max_size=30, unique=True),
    )
    @settings(max_examples=100)
    def test_add_documents_discovers_only_pdf_and_txt(self, filenames):
        """**Validates: Requirements 2.1**

        For any directory tree with mixed file types, add_documents discovers
        only .pdf and .txt files.
        """
        import tempfile

        module, mocks, patches = _make_rag_module_with_patches()
        try:
            # Configure retriever to say no file is a duplicate
            module.retriever.has_source.return_value = False

            # Mock extract_pdf and extract_txt to return dummy data
            with patch("rag_core.module.extract_pdf") as mock_pdf, \
                 patch("rag_core.module.extract_txt") as mock_txt:
                mock_pdf.return_value = [("dummy pdf text", 1)]
                mock_txt.return_value = "dummy txt text"

                # Mock embedder to return a valid embedding list
                module.embedder.embed.return_value = [[0.0] * 384]

                # Create files in a temp directory
                with tempfile.TemporaryDirectory() as tmpdir:
                    for fname in filenames:
                        filepath = os.path.join(tmpdir, fname)
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write("content")

                    module.add_documents(tmpdir)

                # Collect all source filenames that were checked via has_source
                checked_sources = [
                    call.args[0] for call in module.retriever.has_source.call_args_list
                ]

                expected = sorted(
                    f for f in filenames if f.endswith(".pdf") or f.endswith(".txt")
                )
                assert sorted(checked_sources) == expected, (
                    f"Expected {expected}, got {sorted(checked_sources)}"
                )
        finally:
            _stop_all(patches)


class TestRAGModuleDuplicateDetectionProperty:
    """Property-based test for duplicate file detection."""

    # Feature: rag-core, Property 7: Duplicate file detection skips already-indexed files
    @given(
        filename=st.from_regex(r"[a-z]{1,8}\.txt", fullmatch=True),
    )
    @settings(max_examples=100)
    def test_duplicate_file_returns_zero(self, filename):
        """**Validates: Requirements 2.8, 3.5**

        For any file already indexed, re-indexing via add_file skips it
        and returns 0.
        """
        import tempfile

        module, mocks, patches = _make_rag_module_with_patches()
        try:
            # Simulate the file already being indexed
            module.retriever.has_source.return_value = True

            with tempfile.TemporaryDirectory() as tmpdir:
                txt_file = os.path.join(tmpdir, filename)
                with open(txt_file, "w", encoding="utf-8") as f:
                    f.write("some content")

                result = module.add_file(txt_file)
                assert result == 0, f"Expected 0 chunks for duplicate, got {result}"

            # Verify that retriever.add was never called (no new chunks stored)
            module.retriever.add.assert_not_called()
        finally:
            _stop_all(patches)


class TestRAGModuleSearchResultStructureProperty:
    """Property-based test for search result structure."""

    # Feature: rag-core, Property 8: Search results contain text and complete metadata
    @given(
        query=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=1,
            max_size=50,
        ),
    )
    @settings(max_examples=100)
    def test_search_results_have_text_and_metadata(self, query):
        """**Validates: Requirements 4.4**

        For any non-empty query against a non-empty collection, every search
        result has `text` and `metadata` with required keys.
        """
        module, mocks, patches = _make_rag_module_with_patches()
        try:
            # Configure retriever.query to return properly structured results
            module.retriever.query.return_value = [
                {
                    "text": "Some relevant chunk text.",
                    "metadata": {"source": "doc.txt", "page": 0, "chunk_index": 0},
                },
                {
                    "text": "Another relevant chunk.",
                    "metadata": {"source": "doc.pdf", "page": 1, "chunk_index": 1},
                },
            ]
            module.embedder.embed_query.return_value = [0.0] * 384

            results = module.search(query)

            assert isinstance(results, list), "Search results must be a list"
            for r in results:
                assert "text" in r, "Result missing 'text' key"
                assert isinstance(r["text"], str), "'text' must be a string"
                assert len(r["text"]) > 0, "'text' must be non-empty"
                assert "metadata" in r, "Result missing 'metadata' key"
                meta = r["metadata"]
                assert "source" in meta, "Metadata missing 'source'"
                assert "page" in meta, "Metadata missing 'page'"
                assert "chunk_index" in meta, "Metadata missing 'chunk_index'"
        finally:
            _stop_all(patches)


class TestRAGModuleAskResultStructureProperty:
    """Property-based test for ask result structure."""

    # Feature: rag-core, Property 9: Ask results contain answer and sources
    @given(
        query=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=1,
            max_size=50,
        ),
    )
    @settings(max_examples=100)
    def test_ask_returns_answer_and_sources(self, query):
        """**Validates: Requirements 5.4**

        For any non-empty query, ask returns dict with `answer` (string)
        and `sources` (list).
        """
        module, mocks, patches = _make_rag_module_with_patches()
        try:
            # Configure retriever.query to return structured results
            module.retriever.query.return_value = [
                {
                    "text": "Context chunk.",
                    "metadata": {"source": "doc.txt", "page": 0, "chunk_index": 0},
                },
            ]
            module.embedder.embed_query.return_value = [0.0] * 384
            module.generator.generate.return_value = "Generated answer."

            result = module.ask(query)

            assert isinstance(result, dict), "Ask result must be a dict"
            assert "answer" in result, "Result missing 'answer' key"
            assert isinstance(result["answer"], str), "'answer' must be a string"
            assert "sources" in result, "Result missing 'sources' key"
            assert isinstance(result["sources"], list), "'sources' must be a list"

            for src in result["sources"]:
                assert "text" in src, "Source missing 'text'"
                assert "metadata" in src, "Source missing 'metadata'"
                meta = src["metadata"]
                assert "source" in meta, "Source metadata missing 'source'"
                assert "page" in meta, "Source metadata missing 'page'"
                assert "chunk_index" in meta, "Source metadata missing 'chunk_index'"
        finally:
            _stop_all(patches)


class TestRAGModuleDeletedCollectionProperty:
    """Property-based test for post-deletion behavior."""

    # Feature: rag-core, Property 10: Deleted collection rejects all operations
    @given(
        query=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=1,
            max_size=30,
        ),
    )
    @settings(max_examples=100)
    def test_deleted_collection_rejects_all_operations(self, query):
        """**Validates: Requirements 6.2**

        For any deleted collection, calling add_documents, add_file, search,
        or ask raises RuntimeError.
        """
        import tempfile

        module, mocks, patches = _make_rag_module_with_patches()
        try:
            module.delete_collection()

            with tempfile.TemporaryDirectory() as tmpdir:
                txt_file = os.path.join(tmpdir, "test.txt")
                with open(txt_file, "w", encoding="utf-8") as f:
                    f.write("content")

                with pytest.raises(RuntimeError):
                    module.add_documents(tmpdir)

                with pytest.raises(RuntimeError):
                    module.add_file(txt_file)

            with pytest.raises(RuntimeError):
                module.search(query)

            with pytest.raises(RuntimeError):
                module.ask(query)
        finally:
            _stop_all(patches)
