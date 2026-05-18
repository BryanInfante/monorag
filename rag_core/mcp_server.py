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
os.environ["MONORAG_MCP_DIAGNOSTICS"] = "1"  # Enable breadcrumbs for MCP transport

from fastmcp import FastMCP
from rag_core.mcp_diagnostics import emit_mcp_breadcrumb
from rag_core.storage_paths import (
    default_chroma_api_key,
    default_chroma_db_path,
    default_chroma_url,
    parse_chroma_url,
)

if TYPE_CHECKING:
    from rag_core.module import RAGModule

T = TypeVar("T")
DIAGNOSTICS_VERSION = "mcp-server-single-flight-v3"

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

# ---------------------------------------------------------------------------
# Single-flight registry for _get_or_create
# ---------------------------------------------------------------------------
# Each entry tracks the load state for one collection so that concurrent
# callers never launch a second RAGModule loader for the same collection.
#
# States:
#   LOADING  – a worker thread is running; other callers wait on the Event.
#   READY    – instance is available; callers read it directly.
#   FAILED   – the load failed; callers receive the stored exception.
#
# The entry is created (LOADING) under _instances_lock, so only one thread
# ever transitions a collection from "absent" to LOADING.

_LOADING = "LOADING"
_READY = "READY"
_FAILED = "FAILED"


class _LoadEntry:
    """State machine for a single collection load."""

    __slots__ = ("state", "event", "instance", "error")

    def __init__(self) -> None:
        self.state: str = _LOADING
        self.event: threading.Event = threading.Event()
        self.instance: "RAGModule | None" = None
        self.error: BaseException | None = None

    def set_ready(self, instance: "RAGModule") -> None:
        self.instance = instance
        self.state = _READY
        self.event.set()

    def set_failed(self, error: BaseException) -> None:
        self.error = error
        self.state = _FAILED
        self.event.set()


_load_entries: dict[str, _LoadEntry] = {}
_warmup_error: BaseException | None = None  # set if the startup warmup thread fails

# Sentinel key used by the warmup thread in _load_entries so that _get_or_create
# can wait for the import to finish before attempting to load any collection.
_WARMUP_KEY = "__warmup__"

# ---------------------------------------------------------------------------
# MCP server lifespan: warm up sentence_transformers in a real OS thread
# ---------------------------------------------------------------------------
# FastMCP runs tool calls in AnyIO worker threads. Importing torch /
# sentence_transformers for the first time inside such a thread can hang
# because torch initialises global state (CUDA, multiprocessing) that is
# unsafe to do from a secondary thread.
#
# The fix: kick off the import in a plain threading.Thread at server startup
# (before AnyIO's thread pool is involved). The single-flight _LoadEntry
# machinery in _get_or_create ensures that any query arriving while the
# warmup is still running will wait on the same Event instead of launching
# a duplicate import.


@contextlib.asynccontextmanager
async def _warmup_lifespan(server: Any):
    """Import sentence_transformers in a real OS thread at server startup.

    Registers a _LoadEntry under _WARMUP_KEY so that _get_or_create blocks on
    the same Event instead of attempting the import in an AnyIO worker thread.
    Yields immediately so the MCP handshake completes while the model loads.
    """
    entry = _LoadEntry()
    _load_entries[_WARMUP_KEY] = entry

    def _do_warmup() -> None:
        global _warmup_error
        try:
            with _stage("warmup.sentence_transformers"):
                with _suppress_stdout():
                    from rag_core.embedder import _load_sentence_transformer_class
                    _load_sentence_transformer_class()
            _configure_noisy_loggers()
            entry.set_ready(None)  # type: ignore[arg-type]
        except BaseException as exc:  # noqa: BLE001
            _warmup_error = exc
            entry.set_failed(exc)
            _log_event(
                "warmup.sentence_transformers.error",
                detail=str(exc),
                level=logging.ERROR,
            )

    warmup_thread = threading.Thread(
        target=_do_warmup,
        name="mcp-warmup",
        daemon=True,
    )
    warmup_thread.start()
    _log_event("warmup.launched", detail="sentence_transformers loading in background")
    try:
        yield
    finally:
        warmup_thread.join(timeout=5)


# MCP server configuration
mcp = FastMCP("monorag", lifespan=_warmup_lifespan)
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


def _preload_embedding_stack_main_thread() -> None:
    """Load heavy embedding imports in main thread before mcp.run()."""
    with _stage("startup.preload.sentence_transformers"):
        emit_mcp_breadcrumb("before import sentence_transformers")
        with _suppress_stdout():
            from rag_core.embedder import _load_sentence_transformer_class

            _load_sentence_transformer_class()
        emit_mcp_breadcrumb("after import sentence_transformers")

    with _stage("startup.preload.transformers"):
        emit_mcp_breadcrumb("before import transformers")
        with _suppress_stdout():
            import transformers  # noqa: F401
        emit_mcp_breadcrumb("after import transformers")

    with _stage("startup.preload.numpy"):
        emit_mcp_breadcrumb("before import numpy")
        with _suppress_stdout():
            import numpy  # noqa: F401
        emit_mcp_breadcrumb("after import numpy")

    _configure_noisy_loggers()


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
        _wrap_method(attrs["retriever"], "hybrid_query", "retriever.hybrid_query", collection)
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

    Single-flight guarantee: if a load is already in progress for ``collection``,
    the caller waits for that load to finish and reuses its result.  A failed
    load cleans up its entry so a subsequent call can retry from a clean slate.

    Blocks on the warmup _LoadEntry Event if sentence_transformers is still
    being imported in the background thread, preventing the AnyIO worker from
    attempting the same import concurrently (which would hang).
    """
    # Wait for the warmup thread to finish importing sentence_transformers before
    # attempting anything else.  This is the critical gate: the AnyIO worker
    # blocks here on the Event instead of racing to do the import itself.
    warmup_entry = _load_entries.get(_WARMUP_KEY)
    if warmup_entry is not None and not warmup_entry.event.is_set():
        timeout = _load_timeout_seconds()
        _log_event("get_or_create.waiting_for_warmup", collection=collection)
        if not warmup_entry.event.wait(timeout=timeout):
            raise MCPOperationTimeout("warmup.sentence_transformers", timeout)
        if warmup_entry.state == _FAILED:
            raise RuntimeError(
                f"El servidor no pudo inicializar sentence_transformers al arrancar: {warmup_entry.error}"
            ) from warmup_entry.error

    # If the startup warmup failed, refuse to attempt the import in an AnyIO
    # worker thread — it would hang.  Surface the original error instead.
    if _warmup_error is not None:
        raise RuntimeError(
            f"El servidor no pudo inicializar sentence_transformers al arrancar: {_warmup_error}"
        ) from _warmup_error

    # Fast path: already READY (no lock needed for a dict read in CPython).
    instance = _instances.get(collection)
    if instance is not None:
        return instance

    # If the startup warmup failed, refuse to attempt the import in an AnyIO
    # worker thread — it would hang.  Surface the original error instead.
    if _warmup_error is not None:
        raise RuntimeError(
            f"El servidor no pudo inicializar sentence_transformers al arrancar: {_warmup_error}"
        ) from _warmup_error

    with _instances_lock:
        # Re-check under lock in case another thread just finished loading.
        instance = _instances.get(collection)
        if instance is not None:
            return instance

        entry = _load_entries.get(collection)
        if entry is None:
            # This thread wins the race — create the entry and start loading.
            entry = _LoadEntry()
            _load_entries[collection] = entry
            should_load = True
        else:
            should_load = False

    _log_event("get_or_create.start", collection=collection)

    if should_load:
        # We own this load.  Run it synchronously in the current thread so the
        # caller's timeout (from _run_with_timeout at the tool layer) applies
        # naturally.  No nested daemon thread needed here.
        try:
            new_instance = _create_rag_module(collection)
            _instrument_rag_module(new_instance, collection)
            with _instances_lock:
                _instances[collection] = new_instance
            entry.set_ready(new_instance)
            _log_event("get_or_create.end", collection=collection)
            return new_instance
        except BaseException as exc:
            with _instances_lock:
                _load_entries.pop(collection, None)
                _instances.pop(collection, None)
            entry.set_failed(exc)
            _log_event("get_or_create.error", collection=collection, detail=str(exc), level=logging.ERROR)
            raise

    # Another thread is loading (or has finished).  Wait for it.
    timeout = _load_timeout_seconds()
    signalled = entry.event.wait(timeout=timeout)
    if not signalled:
        raise MCPOperationTimeout("get_or_create", timeout)

    if entry.state == _READY:
        return entry.instance  # type: ignore[return-value]

    # FAILED — re-raise the original exception so the caller gets a real error.
    raise entry.error  # type: ignore[misc]


def _timeout_error_message(public_operation: str, timeout_seconds: float) -> str:
    return (
        f"Error: la operación '{public_operation}' superó el límite configurado "
        f"de {timeout_seconds:g} segundos. Revisá los logs stderr del MCP para "
        "identificar la última etapa completada."
    )


@mcp.tool
def search(query: str, collection: str, top_k: int | None = None) -> str:
    """Search for relevant document fragments in a collection."""
    emit_mcp_breadcrumb("search:start", collection=collection)
    if not query.strip() or not collection.strip():
        return "Error: se requieren parámetros 'query' y 'collection' no vacíos."
    resolved_top_k = 5 if top_k is None else top_k
    timeout = _search_timeout_seconds()
    try:
        # _get_or_create has its own timeout for first-time lazy model loading.
        emit_mcp_breadcrumb("search:before_get_or_create", collection=collection)
        module = _get_or_create(collection)
        emit_mcp_breadcrumb("search:after_get_or_create", collection=collection)

        def run() -> str:
            emit_mcp_breadcrumb("search:before_module_search", collection=collection)
            results = module.search(query, resolved_top_k)
            emit_mcp_breadcrumb("search:after_module_search", collection=collection)
            emit_mcp_breadcrumb("search:before_json_dumps", collection=collection)
            response = json.dumps(results, ensure_ascii=False)
            emit_mcp_breadcrumb("search:after_json_dumps", collection=collection)
            emit_mcp_breadcrumb("search:return", collection=collection)
            return response

        return _run_with_timeout("tool.search", timeout, run, collection=collection)
    except MCPOperationTimeout:
        return _timeout_error_message("search", timeout)
    except Exception as e:
        logger.error("Error en search: %s", e)
        return f"Error al realizar la búsqueda: {e}"


@mcp.tool
def ask(question: str, collection: str, top_k: int | None = None) -> str:
    """Ask a question and get an LLM-generated answer with source references."""
    if not question.strip() or not collection.strip():
        return "Error: se requieren parámetros 'question' y 'collection' no vacíos."
    resolved_top_k = 5 if top_k is None else top_k
    timeout = _ask_timeout_seconds()
    try:
        module = _get_or_create(collection)

        def run() -> str:
            answer = module.ask(question, resolved_top_k)
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

        remote_url = default_chroma_url()
        if remote_url:
            import chromadb

            host, port, ssl = parse_chroma_url(remote_url)
            api_key = default_chroma_api_key()
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
            client = chromadb.HttpClient(host=host, port=port, ssl=ssl, headers=headers)
            collections = client.list_collections()
            return json.dumps(
                [
                    {"name": col.name if hasattr(col, "name") else str(col)}
                    for col in collections
                ],
                ensure_ascii=False,
            )

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
    try:
        _get_or_create(name)
        return f"Colección '{name}' creada correctamente."
    except MCPOperationTimeout:
        return _timeout_error_message("create_collection", _load_timeout_seconds())
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

            remote_url = default_chroma_url()
            if remote_url:
                host, port, ssl = parse_chroma_url(remote_url)
                api_key = default_chroma_api_key()
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
                client = chromadb.HttpClient(host=host, port=port, ssl=ssl, headers=headers)
            else:
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
    _preload_embedding_stack_main_thread()
    mcp.run()


if __name__ == "__main__":
    main()
