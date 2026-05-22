"""MonoRAG — REPL-style CLI for the rag_core module.

Run with: python cli.py

Provides a command-based interactive shell to manage collections, configure
chunking/storage for the current CLI session, index documents, search, and ask
questions using the RAG pipeline. All user-facing text is in Spanish.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
import json
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

# Suppress HuggingFace and tokenizer noise before optional ML dependencies load.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from rag_core import RAGModule
from rag_core.llm_providers import (
    default_base_url_for_provider,
    default_model_for_provider,
    normalize_provider_name,
)
from rag_core.secret_store import (
    KEYRING_REFERENCE,
    LLM_API_KEY_SECRET,
    delete_secret,
    get_secret,
    is_keyring_reference,
    set_secret,
)
from rag_core.storage_paths import (
    default_chroma_api_key,
    default_chroma_db_path,
    default_chroma_url,
    default_config_path,
    parse_chroma_url,
)

console = Console()

PACKAGE_NAME = "monorag"
DISPLAY_NAME = "MonoRAG"
VERSION_FLAGS = ("--version", "-V", "-v")
MISSING_METADATA_VALUE = (None, "", "N/A")
NON_PAGINATED_SUFFIXES = (".txt", ".md")

LLM_PROVIDER_MENU = [
    ("openai", "OpenAI oficial"),
    ("groq", "Groq"),
    ("google-ai-studio", "Google AI Studio / Gemini"),
    ("ollama", "Ollama local"),
    ("lm-studio", "LM Studio local"),
    ("custom", "Otro endpoint OpenAI-compatible"),
]
LOCAL_LLM_PROVIDERS = {"ollama", "lm-studio", "lmstudio"}

BANNER = r"""
 __  __                   ____      _    ____
|  \/  | ___  _ __   ___ |  _ \    / \  / ___|
| |\/| |/ _ \| '_ \ / _ \| |_) |  / _ \| |  _
| |  | | (_) | | | | (_) |  _ <  / ___ \ |_| |
|_|  |_|\___/|_| |_|\___/|_| \_\/_/   \_\____|
"""


@dataclass
class CliConfig:
    """Configuration applied when the CLI creates RAGModule instances."""

    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    db_path: str | None = None
    db_url: str | None = None
    llm_provider: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_api_key: str | None = None


PERSISTED_CONFIG_FIELDS = frozenset(CliConfig.__annotations__)


def _coerce_optional_string(value) -> str | None:
    """Normalize persisted nullable string values."""
    if value in (None, ""):
        return None
    return str(value)


def _coerce_int(value, default: int) -> int:
    """Normalize persisted integer values, falling back on invalid data."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_persisted_llm_api_key(value) -> str | None:
    """Load the persisted LLM API key from keyring or legacy JSON."""
    if is_keyring_reference(value):
        return get_secret(LLM_API_KEY_SECRET)
    return _coerce_optional_string(value)


def load_cli_config() -> CliConfig:
    """Load persistent CLI configuration from the user config file."""
    path = Path(default_config_path())
    if not path.exists():
        return CliConfig()

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        console.print(
            f"[yellow]No se pudo leer la configuración persistente en {path}: {exc}[/yellow]"
        )
        return CliConfig()

    if not isinstance(raw, dict):
        console.print(f"[yellow]Configuración persistente inválida en {path}; se ignora.[/yellow]")
        return CliConfig()

    defaults = CliConfig()
    return CliConfig(
        chunk_size=_coerce_int(raw.get("chunk_size"), defaults.chunk_size),
        chunk_overlap=_coerce_int(raw.get("chunk_overlap"), defaults.chunk_overlap),
        top_k=_coerce_int(raw.get("top_k"), defaults.top_k),
        db_path=_coerce_optional_string(raw.get("db_path")),
        db_url=_coerce_optional_string(raw.get("db_url")),
        llm_provider=_coerce_optional_string(raw.get("llm_provider")),
        llm_base_url=_coerce_optional_string(raw.get("llm_base_url")),
        llm_model=_coerce_optional_string(raw.get("llm_model")),
        llm_api_key=_load_persisted_llm_api_key(raw.get("llm_api_key")),
    )


def _read_existing_config_value(path: Path, key: str):
    """Read one persisted config value without failing the save flow."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    return raw.get(key)


def save_cli_config(config: CliConfig, *, clear_llm_secret: bool = False) -> None:
    """Persist CLI configuration to the user config file."""
    path = Path(default_config_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        key: value
        for key, value in asdict(config).items()
        if key in PERSISTED_CONFIG_FIELDS
    }
    llm_api_key = data.get("llm_api_key")
    if isinstance(llm_api_key, str) and llm_api_key:
        if set_secret(LLM_API_KEY_SECRET, llm_api_key):
            data["llm_api_key"] = KEYRING_REFERENCE
    elif clear_llm_secret:
        delete_secret(LLM_API_KEY_SECRET)
        data["llm_api_key"] = None
    elif is_keyring_reference(_read_existing_config_value(path, "llm_api_key")):
        data["llm_api_key"] = KEYRING_REFERENCE
    else:
        data["llm_api_key"] = None
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def persist_cli_config_or_warn(config: CliConfig, *, clear_llm_secret: bool = False) -> None:
    """Persist CLI configuration and keep the session alive if saving fails."""
    try:
        save_cli_config(config, clear_llm_secret=clear_llm_secret)
    except OSError as exc:
        console.print(f"[yellow]Configuración aplicada solo a esta sesión; no se pudo guardar: {exc}[/yellow]")


def strip_quotes(s: str) -> str:
    """Remove surrounding single/double quotes from a string."""
    return s.strip('"').strip("'").strip()


def mask_secret(value: str | None) -> str:
    """Mask API keys in CLI output."""
    if not value:
        return "—"
    if len(value) <= 8:
        return "********"
    return f"{value[:4]}...{value[-4:]}"


def format_source_reference(metadata: dict, *, include_chunk: bool = False) -> str:
    """Format source metadata for CLI output without assuming page exists.

    TXT/MD documents are not paginated, so their chunks intentionally omit the
    ``page`` key. The CLI must treat page metadata as optional instead of
    crashing while printing sources.
    """
    source = metadata.get("source", "desconocido")
    details: list[str] = []

    page = metadata.get("page")
    has_fake_text_page = (
        isinstance(source, str)
        and source.lower().endswith(NON_PAGINATED_SUFFIXES)
        and page in (0, "0")
    )
    if page not in MISSING_METADATA_VALUE and not has_fake_text_page:
        details.append(f"pág. {page}")

    chunk_index = metadata.get("chunk_index")
    if include_chunk and chunk_index not in MISSING_METADATA_VALUE:
        details.append(f"fragmento {chunk_index}")

    if details:
        return f"[dim]{source}[/dim] — {', '.join(details)}"
    return f"[dim]{source}[/dim]"


def prompt_text(label: str, default: str | None = None) -> str:
    """Prompt for a value, returning the default when the user presses Enter."""
    suffix = f" [{default}]" if default else ""
    console.print(f"{label}{suffix}: ", end="")
    value = strip_quotes(input().strip())
    return value or (default or "")


def prompt_required(label: str, default: str | None = None) -> str:
    """Prompt until a non-empty value is provided or a default exists."""
    while True:
        value = prompt_text(label, default)
        if value:
            return value
        console.print("[red]Este valor es obligatorio.[/red]")


def prompt_secret(label: str, current: str | None = None) -> str:
    """Prompt for a secret without echoing the current value as a default."""
    if current:
        console.print(f"{label} (Enter para mantener la actual): ", end="")
        value = strip_quotes(input().strip())
        return value or current
    return prompt_required(label)


def resolve_current_llm_api_key(config: CliConfig) -> str | None:
    """Return the configured or environment API key without exposing it."""
    return config.llm_api_key or os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")


def choose_llm_provider(config: CliConfig) -> str:
    """Run the provider selection step of the LLM wizard."""
    current = normalize_provider_name(
        config.llm_provider or os.getenv("LLM_PROVIDER") or "openai-compatible"
    )
    table = Table(title="Proveedor LLM", show_header=True, header_style="bold cyan")
    table.add_column("#", style="bold green")
    table.add_column("Proveedor")
    table.add_column("Descripción")
    for idx, (name, description) in enumerate(LLM_PROVIDER_MENU, start=1):
        table.add_row(str(idx), name, description)
    console.print(table)
    console.print(
        f"Elegí un proveedor por número o escribí un nombre custom [actual: {current}]: ",
        end="",
    )
    raw = strip_quotes(input().strip())
    if not raw:
        return current

    if raw.isdigit():
        index = int(raw) - 1
        if 0 <= index < len(LLM_PROVIDER_MENU):
            selected = LLM_PROVIDER_MENU[index][0]
            if selected == "custom":
                return normalize_provider_name(prompt_required("Nombre del proveedor custom"))
            return selected
        console.print("[yellow]Opción inválida; se mantiene el proveedor actual.[/yellow]")
        return current

    return normalize_provider_name(raw)


def cmd_config_llm_wizard(config: CliConfig) -> CliConfig:
    """Configure LLM provider settings through a guided CLI flow."""
    console.print(Panel("Configuración guiada del proveedor LLM", style="bold cyan"))

    previous_provider = normalize_provider_name(
        config.llm_provider or os.getenv("LLM_PROVIDER") or "openai-compatible"
    )
    provider = choose_llm_provider(config)
    default_base_url = default_base_url_for_provider(provider)
    same_provider = provider == previous_provider
    current_base_url = (
        (config.llm_base_url or os.getenv("LLM_BASE_URL"))
        if same_provider
        else None
    )

    if default_base_url:
        console.print(f"Base URL por defecto para {provider}: [dim]{default_base_url}[/dim]")
        base_url = prompt_text(
            "Base URL custom (Enter para usar el default del proveedor)",
            current_base_url,
        )
        if current_base_url and base_url == current_base_url:
            console.print("[dim]Se mantiene la Base URL custom actual.[/dim]")
        if base_url == default_base_url:
            base_url = ""
    elif provider in {"openai", "openai-compatible"}:
        base_url = prompt_text(
            "Base URL custom (Enter para usar el default del cliente OpenAI)",
            current_base_url,
        )
    else:
        base_url = prompt_required("Base URL OpenAI-compatible", current_base_url)

    current_api_key = resolve_current_llm_api_key(config)
    provider_for_local = normalize_provider_name(provider)
    if not current_api_key and provider_for_local in LOCAL_LLM_PROVIDERS:
        current_api_key = provider_for_local

    if current_api_key:
        console.print(f"API key actual: [dim]{mask_secret(current_api_key)}[/dim]")
        api_key = prompt_secret("API key", current_api_key)
    else:
        api_key = prompt_required("API key")

    current_model = config.llm_model or os.getenv("LLM_MODEL") or default_model_for_provider(provider)
    model = prompt_text("Modelo", current_model)

    config.llm_provider = provider
    config.llm_base_url = base_url or None
    config.llm_api_key = api_key
    config.llm_model = model or None

    persist_cli_config_or_warn(config)
    console.print("[green]Proveedor LLM configurado y guardado.[/green]")
    show_config(config)
    return config


def build_rag(collection: str, config: CliConfig) -> RAGModule:
    """Create a RAGModule using the current CLI configuration."""
    return RAGModule(
        collection=collection,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        db_path=config.db_path,
        db_url=config.db_url,
        llm_provider=config.llm_provider,
        llm_base_url=config.llm_base_url,
        llm_model=config.llm_model,
        llm_api_key=config.llm_api_key,
    )


def show_banner() -> None:
    console.print(Text(BANNER, style="bold cyan"))
    console.print(
        Panel(
            "[bold]MonoRAG[/bold]\n"
            "Capa de conocimiento RAG local-first\n"
            "Escribe [bold]help[/bold] para ver los comandos disponibles.",
            style="bold magenta",
            expand=False,
        ),
        justify="center",
    )
    console.print()


def show_help() -> None:
    table = Table(
        title="Comandos de MonoRAG",
        show_header=True,
        header_style="bold cyan",
        expand=False,
    )
    table.add_column("Comando", style="bold green", min_width=28)
    table.add_column("Descripción")
    table.add_row("create <name>", "Crear y seleccionar una colección")
    table.add_row("use <name>", "Seleccionar una colección existente")
    table.add_row("index <path>", "Indexar un archivo o directorio")
    table.add_row("chat", "Entrar en modo chat — escribe preguntas directamente")
    table.add_row("ask <question>", "Hacer una pregunta puntual")
    table.add_row("search <query>", "Buscar fragmentos relevantes")
    table.add_row("clear", "Limpiar todos los documentos de la colección activa")
    table.add_row("list", "Listar todas las colecciones")
    table.add_row("delete", "Eliminar la colección activa")
    table.add_row("config", "Ver configuración actual")
    table.add_row(
        Text("config chunk <size> <overlap> [top_k]"),
        "Configurar chunk_size, chunk_overlap y top_k opcional",
    )
    table.add_row("config db path <path>", "Usar y guardar ChromaDB local en cualquier carpeta")
    table.add_row("config db url <url>", "Usar y guardar ChromaDB remoto por HTTP(S)")
    table.add_row("config db default", "Volver al storage por defecto y guardar")
    table.add_row("config llm", "Abrir asistente guiado y guardar proveedor LLM")
    table.add_row("config llm default", "Volver al proveedor LLM definido por .env/default y guardar")
    table.add_row("exit / quit", "Salir del CLI")
    console.print(table)


def show_config(config: CliConfig) -> None:
    table = Table(title="Configuración actual de MonoRAG", show_header=True, header_style="bold cyan")
    table.add_column("Clave", style="bold green")
    table.add_column("Valor")
    table.add_row("chunk_size", str(config.chunk_size))
    table.add_row("chunk_overlap", str(config.chunk_overlap))
    table.add_row("top_k", str(config.top_k))
    table.add_row("db_path", config.db_path or default_chroma_db_path())
    table.add_row("db_url", config.db_url or default_chroma_url() or "—")
    table.add_row("llm_provider", config.llm_provider or os.getenv("LLM_PROVIDER") or "openai-compatible")
    table.add_row("llm_base_url", config.llm_base_url or os.getenv("LLM_BASE_URL") or "—")
    table.add_row("llm_model", config.llm_model or os.getenv("LLM_MODEL") or "provider default")
    table.add_row(
        "llm_api_key",
        mask_secret(config.llm_api_key or os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")),
    )
    table.add_row("config_file", default_config_path())
    console.print(table)


def cmd_config(config: CliConfig, args: str) -> CliConfig:
    args = args.strip()
    if not args:
        show_config(config)
        return config

    parts = args.split()
    if parts[0] == "chunk":
        if len(parts) not in (3, 4):
            console.print("Uso: config chunk <size> <overlap> [top_k]", style="red", markup=False)
            return config
        try:
            chunk_size = int(parts[1])
            chunk_overlap = int(parts[2])
            top_k = int(parts[3]) if len(parts) == 4 else config.top_k
            if chunk_size < 1:
                raise ValueError("chunk_size debe ser mayor o igual a 1")
            if chunk_overlap < 0:
                raise ValueError("chunk_overlap debe ser mayor o igual a 0")
            if chunk_overlap >= chunk_size:
                raise ValueError("chunk_overlap debe ser menor que chunk_size")
            if top_k < 1:
                raise ValueError("top_k debe ser mayor o igual a 1")
        except ValueError as exc:
            console.print(f"[red]Configuración inválida: {exc}[/red]")
            return config

        config.chunk_size = chunk_size
        config.chunk_overlap = chunk_overlap
        config.top_k = top_k
        persist_cli_config_or_warn(config)
        console.print(
            "[green]Chunking actualizado y guardado (incluye top_k si se indicó). Se aplicará a nuevas consultas.[/green]"
        )
        return config

    if parts[0] == "llm":
        if len(parts) == 1:
            return cmd_config_llm_wizard(config)

        if parts[1] == "default":
            config.llm_provider = None
            config.llm_base_url = None
            config.llm_model = None
            config.llm_api_key = None
            persist_cli_config_or_warn(config, clear_llm_secret=True)
            console.print("[green]Proveedor LLM restaurado al valor de .env/default y guardado.[/green]")
            return config

        # Shortcuts avanzados: útiles para scripts/tests, pero el camino humano
        # recomendado es `config llm`, que abre el asistente guiado.
        if len(parts) < 3:
            console.print("[red]Uso recomendado: config llm[/red]")
            return config

        value = strip_quotes(" ".join(parts[2:]))
        if parts[1] == "provider":
            config.llm_provider = normalize_provider_name(value)
            persist_cli_config_or_warn(config)
            console.print(f"[green]Proveedor LLM configurado y guardado: {config.llm_provider}[/green]")
            return config

        if parts[1] == "base-url":
            config.llm_base_url = value
            persist_cli_config_or_warn(config)
            console.print(f"[green]Endpoint LLM configurado y guardado: {value}[/green]")
            return config

        if parts[1] == "model":
            config.llm_model = value
            persist_cli_config_or_warn(config)
            console.print(f"[green]Modelo LLM configurado y guardado: {value}[/green]")
            return config

        if parts[1] == "api-key":
            config.llm_api_key = value
            persist_cli_config_or_warn(config)
            console.print("[green]API key LLM configurada y guardada.[/green]")
            return config

    if parts[0] == "db":
        if len(parts) < 2:
            console.print("[red]Uso: config db path <path> | config db url <url> | config db default[/red]")
            return config

        if parts[1] == "default":
            config.db_path = None
            config.db_url = None
            persist_cli_config_or_warn(config)
            console.print("[green]Storage restaurado al valor por defecto y guardado.[/green]")
            return config

        if len(parts) < 3:
            console.print("[red]Falta el valor de configuración de base de datos.[/red]")
            return config

        value = strip_quotes(" ".join(parts[2:]))
        if parts[1] == "path":
            config.db_path = value
            config.db_url = None
            persist_cli_config_or_warn(config)
            console.print(f"[green]Base local configurada y guardada en: {value}[/green]")
            return config

        if parts[1] == "url":
            try:
                parse_chroma_url(value)
            except ValueError as exc:
                console.print(f"[red]{exc}[/red]")
                return config
            config.db_url = value
            config.db_path = None
            persist_cli_config_or_warn(config)
            console.print(f"[green]Base remota configurada y guardada en: {value}[/green]")
            return config

    console.print("[red]Comando de configuración desconocido. Escribe 'help'.[/red]")
    return config


def cmd_create(name: str, config: CliConfig) -> tuple[str | None, RAGModule | None]:
    if not name:
        console.print("[red]Uso: create <name>[/red]")
        return None, None
    try:
        with console.status("Creando colección..."):
            rag = build_rag(name, config)
        console.print(f"[green]Colección '[bold]{name}[/bold]' creada y seleccionada.[/green]")
        return name, rag
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return None, None


def cmd_use(name: str, config: CliConfig) -> tuple[str | None, RAGModule | None]:
    if not name:
        console.print("[red]Uso: use <name>[/red]")
        return None, None
    try:
        with console.status("Conectando a la colección..."):
            rag = build_rag(name, config)
        console.print(f"[green]Colección '[bold]{name}[/bold]' seleccionada.[/green]")
        return name, rag
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return None, None


def cmd_index(rag: RAGModule, path_str: str) -> None:
    if not path_str:
        console.print("[red]Uso: index <path>[/red]")
        return

    path = Path(strip_quotes(path_str))

    try:
        if path.is_dir():
            with console.status("Indexando directorio..."):
                count = rag.add_documents(str(path))
            console.print(f"[green]Directorio indexado: {count} fragmentos añadidos.[/green]")
        elif path.is_file():
            with console.status("Indexando archivo..."):
                count = rag.add_file(str(path))
            if count == 0:
                console.print("[yellow]El archivo ya existe en la colección, se omitió.[/yellow]")
            else:
                console.print(f"[green]Archivo indexado: {count} fragmentos añadidos.[/green]")
        else:
            console.print(f"[red]La ruta no existe: {path}[/red]")
    except Exception as e:
        console.print(f"[red]Error al indexar: {e}[/red]")


def cmd_ask(rag: RAGModule, query: str, *, top_k: int = 5) -> None:
    if not query:
        console.print("[red]Uso: ask <question>[/red]")
        return
    try:
        with console.status("Generando respuesta..."):
            try:
                result = rag.ask(query, top_k=top_k)
            except TypeError:
                result = rag.ask(query)
        console.print("\n[bold cyan]Respuesta:[/bold cyan]")
        console.print(result["answer"])
        if result["sources"]:
            console.print("\n[bold]Fuentes:[/bold]")
            for i, src in enumerate(result["sources"], 1):
                meta = src["metadata"]
                console.print(f"  {i}. {format_source_reference(meta)}")
        console.print()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def cmd_chat(rag: RAGModule, collection_name: str, *, top_k: int = 5) -> None:
    """Chat mode: each line is a question. Type 'salir' to return."""
    console.print(
        Panel(
            "Modo chat activo. Escribe tu pregunta y presiona Enter.\n"
            "Escribe [bold]salir[/bold] para volver al menú principal.",
            style="bold cyan",
            expand=False,
        )
    )
    while True:
        try:
            console.print(f"[bold cyan]chat[/bold cyan] [dim]({collection_name})[/dim] > ", end="")
            query = input().strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not query:
            continue
        if query.lower() in ("salir", "exit", "quit"):
            console.print("[dim]Saliendo del modo chat...[/dim]\n")
            break
        cmd_ask(rag, query, top_k=top_k)


def cmd_search(rag: RAGModule, query: str, *, top_k: int = 5) -> None:
    if not query:
        console.print("[red]Uso: search <query>[/red]")
        return
    try:
        with console.status("Buscando fragmentos..."):
            try:
                results = rag.search(query, top_k=top_k)
            except TypeError:
                results = rag.search(query)
        if not results:
            console.print("[yellow]No se encontraron resultados.[/yellow]")
            return
        console.print(f"\n[bold]{len(results)} resultado(s):[/bold]\n")
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            console.print(
                f"[bold cyan]{i}.[/bold cyan] "
                f"{format_source_reference(meta, include_chunk=True)}"
            )
            preview = r["text"][:500] + ("..." if len(r["text"]) > 500 else "")
            console.print(f"   {preview}\n")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def cmd_clear(rag: RAGModule, collection_name: str, config: CliConfig) -> tuple[str, RAGModule]:
    """Delete all documents in the current collection and recreate it empty."""
    console.print(
        f"[bold red]¿Limpiar todos los documentos de '[bold]{collection_name}[/bold]'?[/bold red] (s/n): ",
        end="",
    )
    answer = input().strip().lower()
    if answer not in ("s", "si", "sí", "y", "yes"):
        console.print("[dim]Operación cancelada.[/dim]")
        return collection_name, rag
    try:
        with console.status("Limpiando colección..."):
            rag.delete_collection()
            new_rag = build_rag(collection_name, config)
        console.print(f"[green]Colección '[bold]{collection_name}[/bold]' limpiada correctamente.[/green]")
        return collection_name, new_rag
    except Exception as e:
        console.print(f"[red]Error al limpiar: {e}[/red]")
        return collection_name, rag


def _make_collection_client(config: CliConfig):
    """Create a Chroma client for listing/deleting without RAGModule."""
    import chromadb

    remote_url = config.db_url or default_chroma_url()
    if remote_url:
        host, port, ssl = parse_chroma_url(remote_url)
        api_key = default_chroma_api_key()
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        return chromadb.HttpClient(host=host, port=port, ssl=ssl, headers=headers)

    return chromadb.PersistentClient(path=config.db_path or default_chroma_db_path())


def cmd_list(config: CliConfig) -> None:
    """List collections using the configured storage adapter."""
    try:
        client = _make_collection_client(config)
        collections = [c.name if hasattr(c, "name") else str(c) for c in client.list_collections()]
        if not collections:
            console.print("[yellow]No hay colecciones disponibles.[/yellow]")
        else:
            console.print("[bold]Colecciones disponibles:[/bold]")
            for name in collections:
                console.print(f"  • {name}")
    except Exception as e:
        console.print(f"[red]Error al listar colecciones: {e}[/red]")


def cmd_delete(rag: RAGModule, collection_name: str) -> tuple[None, None] | tuple[str, RAGModule]:
    console.print(
        f"[bold red]¿Eliminar la colección '[bold]{collection_name}[/bold]'?[/bold red] (s/n): ",
        end="",
    )
    answer = input().strip().lower()
    if answer not in ("s", "si", "sí", "y", "yes"):
        console.print("[dim]Operación cancelada.[/dim]")
        return collection_name, rag
    try:
        rag.delete_collection()
        console.print(f"[green]Colección '[bold]{collection_name}[/bold]' eliminada correctamente.[/green]")
        return None, None
    except Exception as e:
        console.print(f"[red]Error al eliminar: {e}[/red]")
        return collection_name, rag


def get_prompt(collection_name: str | None) -> str:
    if collection_name:
        return f"[bold cyan]{DISPLAY_NAME}[/bold cyan] [dim]({collection_name})[/dim] > "
    return f"[bold cyan]{DISPLAY_NAME}[/bold cyan] > "


def get_installed_version() -> str:
    """Return the installed MonoRAG package version."""
    try:
        return package_version(PACKAGE_NAME)
    except PackageNotFoundError:
        return "desconocida"


def handle_global_args(argv: list[str]) -> bool:
    """Handle non-interactive CLI flags before opening the REPL."""
    if len(argv) == 1 and argv[0] in VERSION_FLAGS:
        console.print(f"{DISPLAY_NAME} {get_installed_version()}")
        return True
    return False


def main(argv: list[str] | None = None) -> None:
    resolved_argv = sys.argv[1:] if argv is None else argv
    if handle_global_args(resolved_argv):
        return

    show_banner()

    config = load_cli_config()
    collection_name: str | None = None
    rag: RAGModule | None = None

    while True:
        try:
            console.print(get_prompt(collection_name), end="")
            raw = input().strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold cyan]¡Hasta luego![/bold cyan]\n")
            sys.exit(0)

        if not raw:
            continue

        parts = raw.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if command in ("exit", "quit", "salir"):
            console.print("[bold cyan]¡Hasta luego![/bold cyan]\n")
            sys.exit(0)

        if command == "help":
            show_help()
        elif command == "config":
            config = cmd_config(config, args)
        elif command == "create":
            new_name, new_rag = cmd_create(args.strip(), config)
            if new_name and new_rag:
                collection_name, rag = new_name, new_rag
        elif command == "use":
            new_name, new_rag = cmd_use(args.strip(), config)
            if new_name and new_rag:
                collection_name, rag = new_name, new_rag
        elif command == "index":
            if rag is None:
                console.print("[red]Primero selecciona una colección con 'create' o 'use'.[/red]")
            else:
                cmd_index(rag, args)
        elif command == "chat":
            if rag is None:
                console.print("[red]Primero selecciona una colección con 'create' o 'use'.[/red]")
            else:
                cmd_chat(rag, collection_name or "", top_k=config.top_k)
        elif command == "ask":
            if rag is None:
                console.print("[red]Primero selecciona una colección con 'create' o 'use'.[/red]")
            else:
                cmd_ask(rag, args.strip(), top_k=config.top_k)
        elif command == "search":
            if rag is None:
                console.print("[red]Primero selecciona una colección con 'create' o 'use'.[/red]")
            else:
                cmd_search(rag, args.strip(), top_k=config.top_k)
        elif command == "clear":
            if rag is None or collection_name is None:
                console.print("[red]No hay colección activa para limpiar.[/red]")
            else:
                collection_name, rag = cmd_clear(rag, collection_name, config)
        elif command == "list":
            cmd_list(config)
        elif command == "delete":
            if rag is None or collection_name is None:
                console.print("[red]No hay colección activa para eliminar.[/red]")
            else:
                collection_name, rag = cmd_delete(rag, collection_name)
        else:
            console.print(
                f"[red]Comando desconocido: '{command}'. "
                "Escribe [bold]help[/bold] para ver los comandos disponibles.[/red]"
            )


if __name__ == "__main__":
    main()
