import asyncio
import json
import logging
import os
import sqlite3
import shutil
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, strategies as st

from rag_core import mcp_server
from rag_core.embedder import Embedder
from rag_core.retriever import Retriever
from rag_core.storage_paths import default_chroma_db_path


@pytest.fixture(autouse=True)
def clear_cache(monkeypatch):
    """Reset process-local MCP state between tests."""
    mcp_server._instances.clear()
    monkeypatch.delenv("MONORAG_DB_PATH", raising=False)
    monkeypatch.delenv("MONORAG_CHROMA_URL", raising=False)
    monkeypatch.delenv("CHROMA_URL", raising=False)
    monkeypatch.delenv("MONORAG_CHROMA_API_KEY", raising=False)
    monkeypatch.delenv("CHROMA_API_KEY", raising=False)
    monkeypatch.delenv("MONORAG_CHROMA_TENANT", raising=False)
    monkeypatch.delenv("CHROMA_TENANT", raising=False)
    monkeypatch.delenv("MONORAG_CHROMA_DATABASE", raising=False)
    monkeypatch.delenv("CHROMA_DATABASE", raising=False)


@pytest.fixture
def mock_rag_module():
    """Patch the lazy import boundary used by _get_or_create()."""
    with patch("rag_core.module.RAGModule") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance
        yield mock_cls, instance


TEST_ARTIFACTS = Path(".test-artifacts")


def clean_dir(path: Path) -> Path:
    """Create a clean workspace-local directory without relying on pytest tmp_path."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def create_chroma_catalog(base_dir: Path) -> Path:
    """Create the minimal Chroma SQLite schema used by list_collections()."""
    base_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = base_dir / "chroma.sqlite3"
    with sqlite3.connect(sqlite_path) as con:
        con.executescript(
            """
            CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT NOT NULL);
            CREATE TABLE segments (id TEXT PRIMARY KEY, collection TEXT, scope TEXT);
            CREATE TABLE embeddings (id TEXT PRIMARY KEY, segment_id TEXT);
            CREATE TABLE embedding_metadata (
                id TEXT,
                key TEXT,
                string_value TEXT
            );
            """
        )
    return sqlite_path


def test_tools_registered():
    tools = [tool.name for tool in asyncio.run(mcp_server.mcp.list_tools())]
    expected = {
        "search",
        "ask",
        "index_file",
        "index_directory",
        "list_collections",
        "create_collection",
        "delete_collection",
        "clear_history",
    }
    assert expected.issubset(set(tools))


def test_get_or_create_lazily_imports_and_caches_rag_module(mock_rag_module):
    mock_cls, instance = mock_rag_module

    first = mcp_server._get_or_create("manuales")
    second = mcp_server._get_or_create("manuales")

    assert first is instance
    assert second is instance
    mock_cls.assert_called_once_with(collection="manuales")
    assert mcp_server._instances == {"manuales": instance}
    assert not hasattr(mcp_server, "RAGModule")


def test_rag_module_import_does_not_load_heavy_embedding_or_chroma_modules():
    script = (
        "import json, sys; "
        "import rag_core.module; "
        "print(json.dumps({"
        "'sentence_transformers': 'sentence_transformers' in sys.modules, "
        "'chromadb': 'chromadb' in sys.modules, "
        "'openai': 'openai' in sys.modules"
        "}))"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )

    assert json.loads(result.stdout) == {
        "sentence_transformers": False,
        "chromadb": False,
        "openai": False,
    }


def test_mcp_server_import_does_not_load_heavy_runtime_modules():
    script = (
        "import json, sys; "
        "import rag_core.mcp_server; "
        "print(json.dumps({"
        "'sentence_transformers': 'sentence_transformers' in sys.modules, "
        "'chromadb': 'chromadb' in sys.modules, "
        "'openai': 'openai' in sys.modules"
        "}))"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )

    assert json.loads(result.stdout) == {
        "sentence_transformers": False,
        "chromadb": False,
        "openai": False,
    }


def test_mcp_safe_environment_is_configured():
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
    for logger_name in (
        "sentence_transformers",
        "transformers",
        "torch",
        "huggingface_hub",
    ):
        assert logging.getLogger(logger_name).level == logging.ERROR


def test_default_chroma_path_uses_localappdata_on_windows(monkeypatch):
    monkeypatch.delenv("MONORAG_DB_PATH", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\tester\AppData\Local")

    expected = r"C:\Users\tester\AppData\Local\monorag\chroma_db"

    assert default_chroma_db_path() == expected


def test_default_chroma_path_preserves_env_override(monkeypatch):
    monkeypatch.setenv("MONORAG_DB_PATH", "custom_chroma")

    assert default_chroma_db_path() == "custom_chroma"


def test_get_or_create_suppresses_stdout_noise_and_logs_breadcrumbs(capsys):
    class NoisyRAGModule:
        def __init__(self, collection):
            print("MODEL LOADING NOISE THAT MUST NOT HIT STDOUT")
            self.collection = collection

    with patch("rag_core.module.RAGModule", NoisyRAGModule):
        instance = mcp_server._get_or_create("noisy")

    captured = capsys.readouterr()
    assert isinstance(instance, NoisyRAGModule)
    assert "MODEL LOADING NOISE" not in captured.out
    assert "event=get_or_create.start" in captured.err
    assert "event=ragmodule.import.start" in captured.err
    assert "event=ragmodule.init.start" in captured.err


def test_run_with_timeout_returns_without_waiting_for_stuck_worker():
    started = time.monotonic()

    with pytest.raises(mcp_server.MCPOperationTimeout):
        mcp_server._run_with_timeout(
            "tool.search",
            0.05,
            lambda: time.sleep(5),
            collection="col",
        )

    assert time.monotonic() - started < 0.5


def test_search_timeout_returns_spanish_error(monkeypatch):
    instance = MagicMock()
    instance.search.side_effect = lambda *_args: time.sleep(5)
    mcp_server._instances["col"] = instance
    monkeypatch.setenv("MONORAG_MCP_SEARCH_TIMEOUT_SECONDS", "0.05")

    started = time.monotonic()
    result = mcp_server.search(query="q", collection="col")

    assert time.monotonic() - started < 0.5
    assert result.startswith("Error:")
    assert "search" in result
    assert "0.05 segundos" in result


def test_ask_timeout_returns_spanish_error(monkeypatch):
    instance = MagicMock()
    instance.ask.side_effect = lambda *_args: time.sleep(5)
    mcp_server._instances["col"] = instance
    monkeypatch.setenv("MONORAG_MCP_ASK_TIMEOUT_SECONDS", "0.05")

    started = time.monotonic()
    result = mcp_server.ask(question="q", collection="col")

    assert time.monotonic() - started < 0.5
    assert result.startswith("Error:")
    assert "ask" in result
    assert "0.05 segundos" in result


def test_instrumented_search_and_ask_emit_stage_breadcrumbs(capsys):
    class FakeEmbedder:
        def embed_query(self, query):
            print("EMBED STDOUT NOISE")
            return [1.0, 2.0]

    class FakeRetriever:
        def query(self, embedding, top_k=5):
            return [{"text": "hola", "metadata": {"source": "doc.pdf"}}]

    class FakeGenerator:
        def generate(self, query, chunks, history=None):
            return "respuesta"

    class FakeRAGModule:
        def __init__(self, collection):
            self.embedder = FakeEmbedder()
            self.retriever = FakeRetriever()
            self.generator = FakeGenerator()
            self._history = []

        def search(self, query, top_k=5):
            embedding = self.embedder.embed_query(query)
            return self.retriever.query(embedding, top_k=top_k)

        def ask(self, query, top_k=5):
            results = self.search(query, top_k=top_k)
            answer = self.generator.generate(query, results, history=self._history)
            return {"answer": answer, "sources": results}

    with patch("rag_core.module.RAGModule", FakeRAGModule):
        search_result = mcp_server.search(query="q", collection="diagnostic")
        ask_result = mcp_server.ask(question="q", collection="diagnostic")

    captured = capsys.readouterr()
    assert "EMBED STDOUT NOISE" not in captured.out
    assert json.loads(search_result) == [
        {"text": "hola", "metadata": {"source": "doc.pdf"}}
    ]
    assert "respuesta" in ask_result
    for event in (
        "search:start",
        "search:before_get_or_create",
        "search:after_get_or_create",
        "search:before_module_search",
        "search:after_module_search",
        "search:before_json_dumps",
        "search:after_json_dumps",
        "search:return",
        "event=tool.search.start",
        "event=search.core.start",
        "event=embed_query.start",
        "event=retriever.query.start",
        "event=tool.ask.start",
        "event=ask.generate.start",
    ):
        assert event in captured.err


def test_embedder_emits_exact_sentence_transformer_and_encode_breadcrumbs(capsys):
    with patch("rag_core.embedder.SentenceTransformer") as mock_model_cls:
        mock_model = MagicMock()
        mock_model.encode.return_value = [[1.0, 2.0]]
        mock_model_cls.return_value = mock_model

        embedder = Embedder()
        assert embedder.embed_query("hola") == [1.0, 2.0]

    captured = capsys.readouterr()
    for breadcrumb in (
        "Embedder:init:before_sentence_transformer",
        "Embedder:init:after_sentence_transformer",
        "embed_query:before_encode",
        "embed_query:after_encode",
    ):
        assert breadcrumb in captured.err


def test_retriever_emits_exact_persistent_client_and_query_breadcrumbs(capsys):
    with patch("rag_core.retriever.chromadb.PersistentClient") as mock_client_cls:
        collection = MagicMock()
        collection.count.return_value = 1
        collection.query.return_value = {
            "documents": [["texto"]],
            "metadatas": [[{"source": "doc.pdf"}]],
        }
        mock_client_cls.return_value.get_or_create_collection.return_value = collection

        retriever = Retriever(collection_name="manuales", persist_dir="db")
        assert retriever.query([1.0, 2.0], top_k=1) == [
            {"text": "texto", "metadata": {"source": "doc.pdf"}}
        ]

    captured = capsys.readouterr()
    for breadcrumb in (
        "Retriever:init:before_persistent_client",
        "Retriever:init:after_persistent_client",
        "Retriever:init:before_get_or_create_collection",
        "Retriever:init:after_get_or_create_collection",
        "retriever:before_collection_count",
        "retriever:after_collection_count",
        "retriever:before_collection_query",
        "retriever:after_collection_query",
        "retriever:before_output_build",
        "retriever:after_output_build",
        "retriever:return",
    ):
        assert breadcrumb in captured.err


def test_retriever_uses_user_data_chroma_path_by_default(monkeypatch):
    monkeypatch.delenv("MONORAG_DB_PATH", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\tester\AppData\Local")
    expected = r"C:\Users\tester\AppData\Local\monorag\chroma_db"

    with patch("rag_core.retriever.Path.mkdir") as mock_mkdir, patch(
        "rag_core.retriever.chromadb.PersistentClient"
    ) as mock_client_cls:
        Retriever(collection_name="manuales")

    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_client_cls.assert_called_once_with(path=expected)


def test_retriever_uses_http_client_for_remote_chroma_url(monkeypatch):
    monkeypatch.setenv("MONORAG_CHROMA_URL", "https://chroma.example.com")
    monkeypatch.setenv("MONORAG_CHROMA_API_KEY", "secret")

    with patch("rag_core.retriever.chromadb.HttpClient") as mock_client_cls:
        collection = MagicMock()
        collection.count.return_value = 0
        mock_client_cls.return_value.get_or_create_collection.return_value = collection

        Retriever(collection_name="manuales")

    mock_client_cls.assert_called_once_with(
        host="chroma.example.com",
        port=443,
        ssl=True,
        headers={"Authorization": "Bearer secret"},
    )


def test_search_success(mock_rag_module):
    _, instance = mock_rag_module
    instance.search.return_value = [{"text": "hola", "metadata": {"source": "doc.pdf"}}]

    result = mcp_server.search(query="test", collection="test_col")

    assert json.loads(result) == [{"text": "hola", "metadata": {"source": "doc.pdf"}}]
    instance.search.assert_called_once_with("test", 5)


def test_ask_success(mock_rag_module):
    _, instance = mock_rag_module
    instance.ask.return_value = {"answer": "Respuesta", "sources": [{"source": "doc.pdf"}]}

    result = mcp_server.ask(question="¿hola?", collection="test_col")

    assert "Respuesta" in result
    assert "doc.pdf" in result
    instance.ask.assert_called_once_with("¿hola?", 5)


def test_index_file_success(mock_rag_module):
    _, instance = mock_rag_module
    instance.add_file.return_value = 10

    result = mcp_server.index_file(path="doc.pdf", collection="test_col")

    assert "Fragmentos añadidos: 10" in result
    instance.add_file.assert_called_once_with("doc.pdf")


def test_index_directory_success(mock_rag_module):
    _, instance = mock_rag_module
    instance.add_documents.return_value = 25

    result = mcp_server.index_directory(path="docs", collection="test_col")

    assert "Fragmentos añadidos: 25" in result
    instance.add_documents.assert_called_once_with("docs")


def test_list_collections_reads_sqlite_catalog_readonly(monkeypatch):
    db_dir = clean_dir(TEST_ARTIFACTS / "chroma_db")
    create_chroma_catalog(db_dir)
    with sqlite3.connect(db_dir / "chroma.sqlite3") as con:
        con.executemany(
            "INSERT INTO collections (id, name) VALUES (?, ?)",
            [("c1", "API570"), ("c2", "AWS_D1_1")],
        )
        con.execute(
            "INSERT INTO segments (id, collection, scope) VALUES (?, ?, ?)",
            ("s1", "c1", "METADATA"),
        )
        con.executemany(
            "INSERT INTO embeddings (id, segment_id) VALUES (?, ?)",
            [("e1", "s1"), ("e2", "s1")],
        )
        con.executemany(
            "INSERT INTO embedding_metadata (id, key, string_value) VALUES (?, ?, ?)",
            [("e1", "source", "manual.pdf"), ("e2", "source", "manual.pdf")],
        )
    monkeypatch.setenv("MONORAG_DB_PATH", str(db_dir))

    with patch("rag_core.module.RAGModule") as mock_rag, patch(
        "chromadb.PersistentClient"
    ) as mock_client:
        result = mcp_server.list_collections()

    assert json.loads(result) == [
        {"name": "API570", "count": 1, "chunks": 2},
        {"name": "AWS_D1_1", "count": 0, "chunks": 0},
    ]
    mock_rag.assert_not_called()
    mock_client.assert_not_called()
    assert mcp_server._instances == {}


def test_list_collections_returns_empty_list_when_catalog_missing(monkeypatch):
    missing_dir = clean_dir(TEST_ARTIFACTS / "missing_chroma")
    shutil.rmtree(missing_dir)
    monkeypatch.setenv("MONORAG_DB_PATH", str(missing_dir))

    result = mcp_server.list_collections()

    assert json.loads(result) == []
    assert mcp_server._instances == {}


def test_create_collection_caches_instance(mock_rag_module):
    _, instance = mock_rag_module

    result = mcp_server.create_collection(name="nueva_col")

    assert "Colección 'nueva_col' creada correctamente." == result
    assert mcp_server._instances["nueva_col"] is instance


def test_delete_collection_cached_removes_from_cache():
    instance = MagicMock()
    mcp_server._instances["col"] = instance

    result = mcp_server.delete_collection(name="col")

    assert "Colección 'col' eliminada correctamente." == result
    assert "col" not in mcp_server._instances
    instance.delete_collection.assert_called_once()


def test_delete_collection_uncached_uses_chromadb_directly(monkeypatch):
    monkeypatch.setenv("MONORAG_DB_PATH", "custom_chroma")

    with patch("chromadb.PersistentClient") as mock_client_cls:
        result = mcp_server.delete_collection(name="col")

    assert "Colección 'col' eliminada correctamente." == result
    mock_client_cls.assert_called_once_with(path="custom_chroma")
    mock_client_cls.return_value.delete_collection.assert_called_once_with(name="col")


def test_clear_history_success_for_cached_collection():
    instance = MagicMock()
    mcp_server._instances["col"] = instance

    result = mcp_server.clear_history(collection="col")

    assert "Historial de la colección 'col' borrado correctamente." == result
    instance.clear_history.assert_called_once()


def test_clear_history_no_session_does_not_create_rag_module(mock_rag_module):
    mock_cls, _ = mock_rag_module

    result = mcp_server.clear_history(collection="no_existe")

    assert result == "Error: no existe una sesión activa para la colección 'no_existe'."
    mock_cls.assert_not_called()
    assert mcp_server._instances == {}


COLLECTION_NAMES = st.text(
    alphabet=st.sampled_from(
        list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    ),
    min_size=1,
    max_size=50,
)

WHITESPACE = st.text(
    alphabet=st.sampled_from([" ", "\t", "\n", "\r"]),
    min_size=0,
    max_size=20,
)


@settings(max_examples=100)
@given(value=WHITESPACE)
@pytest.mark.parametrize(
    ("tool", "kwargs"),
    [
        (mcp_server.search, {"query": None, "collection": "col"}),
        (mcp_server.search, {"query": "q", "collection": None}),
        (mcp_server.ask, {"question": None, "collection": "col"}),
        (mcp_server.ask, {"question": "q", "collection": None}),
        (mcp_server.index_file, {"path": None, "collection": "col"}),
        (mcp_server.index_file, {"path": "doc.pdf", "collection": None}),
        (mcp_server.index_directory, {"path": None, "collection": "col"}),
        (mcp_server.index_directory, {"path": "docs", "collection": None}),
        (mcp_server.create_collection, {"name": None}),
        (mcp_server.delete_collection, {"name": None}),
        (mcp_server.clear_history, {"collection": None}),
    ],
)
def test_empty_input_validation_property(tool, kwargs, value):
    mcp_server._instances.clear()
    call_kwargs = {key: (value if val is None else val) for key, val in kwargs.items()}

    with patch("rag_core.module.RAGModule") as mock_cls:
        result = tool(**call_kwargs)

    assert result.startswith("Error:")
    assert "Traceback" not in result
    mock_cls.assert_not_called()
    assert mcp_server._instances == {}


@settings(max_examples=100)
@given(message=st.text(min_size=1, max_size=100))
@pytest.mark.parametrize(
    ("method_name", "tool", "kwargs"),
    [
        ("search", mcp_server.search, {"query": "q", "collection": "col"}),
        ("ask", mcp_server.ask, {"question": "q", "collection": "col"}),
        ("add_file", mcp_server.index_file, {"path": "doc.pdf", "collection": "col"}),
        (
            "add_documents",
            mcp_server.index_directory,
            {"path": "docs", "collection": "col"},
        ),
    ],
)
def test_exception_containment_property(
    method_name, tool, kwargs, message
):
    mcp_server._instances.clear()
    with patch("rag_core.module.RAGModule") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance
        getattr(instance, method_name).side_effect = RuntimeError(message)

        result = tool(**kwargs)

    assert result.startswith("Error")
    assert message in result
    assert "Traceback" not in result


@settings(max_examples=100)
@given(name=COLLECTION_NAMES)
def test_cache_idempotence_property(name):
    mcp_server._instances.clear()
    with patch("rag_core.module.RAGModule") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance

        assert mcp_server._get_or_create(name) is instance
        assert mcp_server._get_or_create(name) is instance
        assert list(mcp_server._instances) == [name]
        mock_cls.assert_called_once_with(collection=name)


@settings(max_examples=100)
@given(name=COLLECTION_NAMES)
def test_cache_cleanup_on_delete_property(name):
    mcp_server._instances.clear()
    instance = MagicMock()
    mcp_server._instances[name] = instance

    result = mcp_server.delete_collection(name)

    assert "eliminada correctamente" in result
    assert name not in mcp_server._instances
    instance.delete_collection.assert_called_once()


@settings(max_examples=100)
@given(name=COLLECTION_NAMES)
def test_clear_history_requires_active_session_property(name):
    mcp_server._instances.clear()
    with patch("rag_core.module.RAGModule") as mock_cls:
        result = mcp_server.clear_history(name)

        assert "no existe una sesión activa" in result
        assert name not in mcp_server._instances
        mock_cls.assert_not_called()
