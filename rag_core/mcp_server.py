import contextlib
import functools
import json
import logging
import os
import queue
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, TypeVar

# Keep MCP STDIO protocol clean before any optional ML dependency can be imported.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Pre-import sentence_transformers in the main thread to avoid import-lock
# deadlocks when _get_or_create() runs in a worker thread. This is the heaviest
# import (~10-30s) but must happen in the main thread where the import lock is free.
import warnings as _warnings
_warnings.filterwarnings("ignore")
with contextlib.redirect_stdout(open(os.devnull, "w")):
    import sentence_transformers  # noqa: F401

from fastmcp import FastMCP
from rag_core.mcp_diagnostics import emit_mcp_breadcrumb
from rag_core.storage_paths import default_chroma_db_path

if TYPE_CHECKING:
    from rag_core.module import RAGModule

T = TypeVar("T")
DIAGNOSTICS_VERSION = "mcp-server-hang-v2"

_NOISY_LOGGERS = ("sentence_transformers", "transformers", "torch", "huggingface_hub")


def _configure_noisy_loggers() -> None:
    for logger_name in _NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.ERROR)


_configure_noisy_loggers()


class _DynamicStderrHandler(logging.StreamHandler):
    """Logging handler that follows the current sys.stderr for pytest/capture."""

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stderr
        super().emit(record)


class _ThreadLocalStdoutProxy:
    """Forward stdout except for threads marked as MCP risky sections.

    ``contextlib.redirect_stdout(os.devnull)`` is process-global. If a worker
    hangs while redirected, JSON-RPC responses can be swallowed too. This proxy
    makes suppression thread-local: noisy ML code in a worker is discarded,
    while the MCP transport can still write protocol bytes from other threads.
    """

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped

    def write(self, text: str) -> int:
        if getattr(_stdout_state, "suppressed", False):
            return len(text)
        return self._wrapped.write(text)

    def flush(self) -> None:
        return self._wrapped.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


_stdout_state = threading.local()
_stdout_proxy_lock = threading.Lock()
_instances_lock = threading.RLock()

# MCP server configuration
mcp = FastMCP("monorag")
_instances: dict[str, "RAGModule"] = {}
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _handler = _DynamicStderrHandler()
    _handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s pid=%(process)d thread=%(threadName)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    logger.addHandler(_handler)


class MCPOperationTimeout(TimeoutError):
    """Raised when a bounded MCP operation exceeds its configured timeout."""

    def __init__(self, operation: str, timeout_seconds: float) -> None:
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"la operación '{operation}' superó el límite configurado de "
            f"{timeout_seconds:g} segundos"
        )


def _ensure_stdout_proxy() -> None:
    """Install a thread-local stdout proxy around the current stdout object."""
    if isinstance(sys.stdout, _ThreadLocalStdoutProxy):
        return
    with _stdout_proxy_lock:
        if not isinstance(sys.stdout, _ThreadLocalStdoutProxy):
            sys.stdout = _ThreadLocalStdoutProxy(sys.stdout)


@contextlib.contextmanager
def _suppress_stdout():
    """Suppress stdout only for the current thread.

    This keeps JSON-RPC stdout usable even if a daemon worker times out while a
    third-party library is still running.
    """
    _ensure_stdout_proxy()
    previous = getattr(_stdout_state, "suppressed", False)
    _stdout_state.suppressed = True
    try:
        yield
    finally:
        _stdout_state.suppressed = previous


def _log_event(
    event: str,
    *,
    collection: str | None = None,
    elapsed: float | None = None,
    detail: str | None = None,
    level: int = logging.INFO,
) -> None:
    parts = [f"event={event}"]
    if collection is not None:
        parts.append(f"collection={collection}")
    if elapsed is not None:
        parts.append(f"elapsed={elapsed:.3f}s")
    if detail:
        parts.append(f"detail={detail}")
    logger.log(level, " ".join(parts))


@contextlib.contextmanager
def _stage(stage: str, *, collection: str | None = None):
    started = time.monotonic()
    _log_event(f"{stage}.start", collection=collection)
    try:
        yield
    except Exception as exc:
        _log_event(
            f"{stage}.error",
            collection=collection,
            elapsed=time.monotonic() - started,
            detail=str(exc),
            level=logging.ERROR,
        )
        raise
    else:
        _log_event(
            f"{stage}.end",
            collection=collection,
            elapsed=time.monotonic() - started,
        )


def _timeout_from_env(env_name: str, default: float) -> float:
    raw = os.getenv(env_name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Valor inválido para %s=%r; usando %.0fs", env_name, raw, default)
        return default
    if value <= 0:
        logger.warning("Valor no positivo para %s=%r; usando %.0fs", env_name, raw, default)
        return default
    return value


def _load_timeout_seconds() -> float:
    return _timeout_from_env("MONORAG_MCP_LOAD_TIMEOUT_SECONDS", 180.0)


def _search_timeout_seconds() -> float:
    return _timeout_from_env("MONORAG_MCP_SEARCH_TIMEOUT_SECONDS", 60.0)


def _ask_timeout_seconds() -> float:
    return _timeout_from_env("MONORAG_MCP_ASK_TIMEOUT_SECONDS", 120.0)


def _index_timeout_seconds() -> float:
    return _timeout_from_env("MONORAG_MCP_INDEX_TIMEOUT_SECONDS", 300.0)


def _log_configured_limits() -> None:
    _log_event(
        "mcp.limits.configured",
        detail=(
            f"load={_load_timeout_seconds():g}s "
            f"search={_search_timeout_seconds():g}s "
            f"ask={_ask_timeout_seconds():g}s "
            f"index={_index_timeout_seconds():g}s"
        ),
    )


def _run_with_timeout(
    operation: str,
    timeout_seconds: float,
    func: Callable[[], T],
    *,
    collection: str | None = None,
) -> T:
    """Run ``func`` in a daemon worker and return/raise within the timeout.

    This intentionally avoids ``with ThreadPoolExecutor(...).result(timeout=...)``.
    A context-managed executor waits during shutdown, so a stuck native worker can
    recreate the MCP hang. The tradeoff of this daemon-thread approach is that a
    timed-out worker cannot be killed; it is abandoned, while the request path
    returns a controlled JSON-RPC result.
    """
    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def runner() -> None:
        try:
            result_queue.put_nowait(("ok", func()))
        except BaseException as exc:  # noqa: BLE001 - propagate to request path.
            result_queue.put_nowait(("error", exc))

    started = time.monotonic()
    _log_event(f"{operation}.start", collection=collection)
    worker = threading.Thread(
        target=runner,
        name=f"mcp-{operation.replace('.', '-')}",
        daemon=True,
    )
    worker.start()

    try:
        status, payload = result_queue.get(timeout=timeout_seconds)
    except queue.Empty as exc:
        elapsed = time.monotonic() - started
        _log_event(
            f"{operation}.timeout",
            collection=collection,
            elapsed=elapsed,
            detail=f"limit={timeout_seconds:g}s",
            level=logging.ERROR,
        )
        raise MCPOperationTimeout(operation, timeout_seconds) from exc

    elapsed = time.monotonic() - started
    if status == "error":
        _log_event(
            f"{operation}.error",
            collection=collection,
            elapsed=elapsed,
            detail=str(payload),
            level=logging.ERROR,
        )
        raise payload

    _log_event(f"{operation}.end", collection=collection, elapsed=elapsed)
    return payload


def _call_stage(
    stage: str,
    collection: str,
    func: Callable[..., T],
    *args: Any,
    suppress_stdout: bool = False,
    **kwargs: Any,
) -> T:
    with _stage(stage, collection=collection):
        if suppress_stdout:
            with _suppress_stdout():
                return func(*args, **kwargs)
        return func(*args, **kwargs)


def _wrap_method(
    owner: Any,
    method_name: str,
    stage: str,
    collection: str,
    *,
    suppress_stdout: bool = False,
) -> None:
    method = getattr(owner, method_name, None)
    if method is None or getattr(method, "_mcp_instrumented", False):
        return

    @functools.wraps(method)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        return _call_stage(
            stage,
            collection,
            method,
            *args,
            suppress_stdout=suppress_stdout,
            **kwargs,
        )

    wrapped._mcp_instrumented = True  # type: ignore[attr-defined]
    setattr(owner, method_name, wrapped)


def _instrument_rag_module(instance: "RAGModule", collection: str) -> None:
    """Add diagnostic breadcrumbs to an actual RAGModule instance.

    Tests often use MagicMock instances; only wrap concrete attributes that were
    placed in the instance __dict__ by RAGModule.__init__().
    """
    if getattr(instance, "_mcp_instrumented", False):
        return

    attrs = vars(instance)
    if "embedder" in attrs:
        _wrap_method(attrs["embedder"], "embed_query", "embed_query", collection, suppress_stdout=True)
        _wrap_method(attrs["embedder"], "embed", "embed", collection, suppress_stdout=True)
    if "retriever" in attrs:
        _wrap_method(attrs["retriever"], "query", "retriever.query", collection)
    if "generator" in attrs:
        _wrap_method(attrs["generator"], "generate", "ask.generate", collection)

    _wrap_method(instance, "search", "search.core", collection, suppress_stdout=True)
    instance._mcp_instrumented = True  # type: ignore[attr-defined]


def _create_rag_module(collection: str) -> "RAGModule":
    with _stage("ragmodule.import", collection=collection):
        with _suppress_stdout():
            from rag_core.module import RAGModule
    _configure_noisy_loggers()

    with _stage("ragmodule.init", collection=collection):
        with _suppress_stdout():
            instance = RAGModule(collection=collection)
    _configure_noisy_loggers()
    return instance


def _get_or_create(collection: str) -> "RAGModule":
    """Get a cached RAGModule instance or create and cache a new one.

    RAGModule is imported lazily so the MCP server can complete the stdio
    handshake quickly. Loading sentence-transformers/torch belongs to the first
    tool call that needs embeddings, not to server startup.
    """
    if collection in _instances:
        return _instances[collection]

    with _instances_lock:
        if collection in _instances:
            return _instances[collection]
        instance = _run_with_timeout(
            "get_or_create",
            _load_timeout_seconds(),
            lambda: _create_rag_module(collection),
            collection=collection,
        )
        _instrument_rag_module(instance, collection)
        _instances[collection] = instance
        return instance


def _timeout_error_message(public_operation: str, timeout_seconds: float) -> str:
    return (
        f"Error: la operación '{public_operation}' superó el límite configurado "
        f"de {timeout_seconds:g} segundos. Revisá los logs stderr del MCP para "
        "identificar la última etapa completada."
    )


@mcp.tool
def search(query: str, collection: str, top_k: int = 5) -> str:
    """Search for relevant document fragments in a collection."""
    emit_mcp_breadcrumb("search:start", collection=collection)
    if not query.strip() or not collection.strip():
        return "Error: se requieren parámetros 'query' y 'collection' no vacíos."
    timeout = _search_timeout_seconds()
    try:
        # Ensure model is loaded before applying the search timeout.
        # _get_or_create has its own (longer) timeout for first-time model loading.
        emit_mcp_breadcrumb("search:before_get_or_create", collection=collection)
        module = _get_or_create(collection)
        emit_mcp_breadcrumb("search:after_get_or_create", collection=collection)

        def run() -> str:
            emit_mcp_breadcrumb("search:before_module_search", collection=collection)
            results = module.search(query, top_k)
            emit_mcp_breadcrumb("search:after_module_search", collection=collection)
            response = json.dumps(results, ensure_ascii=False)
            return response

        return _run_with_timeout("tool.search", timeout, run, collection=collection)
    except MCPOperationTimeout:
        return _timeout_error_message("search", timeout)
    except Exception as e:
        logger.error("Error en search: %s", e)
        return f"Error al realizar la búsqueda: {e}"


@mcp.tool
def ask(question: str, collection: str, top_k: int = 5) -> str:
    """Ask a question and get an LLM-generated answer with source references."""
    if not question.strip() or not collection.strip():
        return "Error: se requieren parámetros 'question' y 'collection' no vacíos."
    timeout = _ask_timeout_seconds()
    try:
        # Ensure model is loaded before applying the ask timeout.
        module = _get_or_create(collection)

        def run() -> str:
            answer = module.ask(question, top_k)
            return str(answer)

        return _run_with_timeout("tool.ask", timeout, run, collection=collection)
    except MCPOperationTimeout:
        return _timeout_error_message("ask", timeout)
    except Exception as e:
        logger.error("Error en ask: %s", e)
        return f"Error al responder la pregunta: {e}"


@mcp.tool
def index_file(path: str, collection: str) -> str:
    """Index a single PDF, TXT, or MD file into a collection."""
    if not path.strip() or not collection.strip():
        return "Error: se requieren parámetros 'path' y 'collection' no vacíos."
    timeout = _index_timeout_seconds()
    try:
        # Ensure model is loaded before applying the index timeout.
        module = _get_or_create(collection)

        def run() -> str:
            with _suppress_stdout():
                count = module.add_file(path)
            return f"Archivo indexado correctamente. Fragmentos añadidos: {count}"

        return _run_with_timeout("tool.index_file", timeout, run, collection=collection)
    except MCPOperationTimeout:
        return _timeout_error_message("index_file", timeout)
    except Exception as e:
        logger.error("Error en index_file: %s", e)
        return f"Error al indexar el archivo: {e}"


@mcp.tool
def index_directory(path: str, collection: str) -> str:
    """Index all PDF, TXT, and MD files from a directory into a collection."""
    if not path.strip() or not collection.strip():
        return "Error: se requieren parámetros 'path' y 'collection' no vacíos."
    timeout = _index_timeout_seconds()
    try:
        # Ensure model is loaded before applying the index timeout.
        module = _get_or_create(collection)

        def run() -> str:
            with _suppress_stdout():
                count = module.add_documents(path)
            return f"Directorio indexado correctamente. Fragmentos añadidos: {count}"

        return _run_with_timeout("tool.index_directory", timeout, run, collection=collection)
    except MCPOperationTimeout:
        return _timeout_error_message("index_directory", timeout)
    except Exception as e:
        logger.error("Error en index_directory: %s", e)
        return f"Error al indexar el directorio: {e}"


@mcp.tool
def list_collections() -> str:
    """List all available collections without booting ChromaDB.

    The collections panel only needs names for a dropdown. Using
    chromadb.PersistentClient for this path can block on migrations/locks and
    may even try to write to the database. Read the Chroma SQLite catalog in
    read-only mode instead, so the UI can load quickly while heavier RAG tools
    remain lazy-loaded.
    """
    try:
        import sqlite3
        from pathlib import Path

        db_dir = Path(default_chroma_db_path())
        sqlite_path = db_dir / "chroma.sqlite3"
        if not sqlite_path.exists():
            return json.dumps([], ensure_ascii=False)

        uri = f"file:{sqlite_path.resolve().as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=2.0) as con:
            rows = con.execute(
                """
                SELECT
                    c.name,
                    COUNT(DISTINCT CASE
                        WHEN em.key = 'source' THEN em.string_value
                    END) AS document_count,
                    COUNT(DISTINCT e.id) AS chunk_count
                FROM collections c
                LEFT JOIN segments s
                    ON s.collection = c.id AND s.scope = 'METADATA'
                LEFT JOIN embeddings e
                    ON e.segment_id = s.id
                LEFT JOIN embedding_metadata em
                    ON em.id = e.id
                GROUP BY c.name
                ORDER BY lower(c.name)
                """
            ).fetchall()

        return json.dumps(
            [
                {"name": name, "count": int(document_count), "chunks": int(chunk_count)}
                for name, document_count, chunk_count in rows
            ],
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error("Error en list_collections: %s", e)
        return f"Error al listar colecciones: {e}"


@mcp.tool
def create_collection(name: str) -> str:
    """Create a new collection and cache its RAGModule instance."""
    if not name.strip():
        return "Error: se requiere el parámetro 'name' no vacío."
    timeout = _load_timeout_seconds()
    try:
        _run_with_timeout("tool.create_collection", timeout, lambda: _get_or_create(name), collection=name)
        return f"Colección '{name}' creada correctamente."
    except MCPOperationTimeout:
        return _timeout_error_message("create_collection", timeout)
    except Exception as e:
        logger.error("Error en create_collection: %s", e)
        return f"Error al crear la colección: {e}"


@mcp.tool
def delete_collection(name: str) -> str:
    """Delete a collection and all its data."""
    if not name.strip():
        return "Error: se requiere el parámetro 'name' no vacío."
    try:
        if name in _instances:
            _instances[name].delete_collection()
            del _instances[name]
        else:
            import chromadb

            path = default_chroma_db_path()
            client = chromadb.PersistentClient(path=path)
            client.delete_collection(name=name)
        return f"Colección '{name}' eliminada correctamente."
    except Exception as e:
        logger.error("Error en delete_collection: %s", e)
        return f"Error al eliminar la colección: {e}"


@mcp.tool
def clear_history(collection: str) -> str:
    """Clear conversation history for a cached collection."""
    if not collection.strip():
        return "Error: se requiere el parámetro 'collection' no vacío."
    try:
        if collection in _instances:
            _instances[collection].clear_history()
            return f"Historial de la colección '{collection}' borrado correctamente."
        return f"Error: no existe una sesión activa para la colección '{collection}'."
    except Exception as e:
        logger.error("Error en clear_history: %s", e)
        return f"Error al borrar el historial: {e}"


def main():
    """Start the MCP server using STDIO transport."""
    emit_mcp_breadcrumb("mcp_server:startup", detail=f"diagnostics={DIAGNOSTICS_VERSION}")
    _log_configured_limits()
    mcp.run()


if __name__ == "__main__":
    main()
