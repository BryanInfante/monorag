"""MONORAG — REPL-style CLI for the rag_core module.

Run with: python cli.py

Provides a command-based interactive shell to manage collections,
index documents, perform semantic search, and ask questions using
the RAG pipeline. All user-facing text is in Spanish.
"""

import logging
import os
import sys
import warnings

# Suprimir warnings de HuggingFace y sentence-transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from rag_core import RAGModule

console = Console()

BANNER = r"""
 ███╗   ███╗ ██████╗ ███╗   ██╗ ██████╗ ██████╗  █████╗  ██████╗
 ████╗ ████║██╔═══██╗████╗  ██║██╔═══██╗██╔══██╗██╔══██╗██╔════╝
 ██╔████╔██║██║   ██║██╔██╗ ██║██║   ██║██████╔╝███████║██║  ███╗
 ██║╚██╔╝██║██║   ██║██║╚██╗██║██║   ██║██╔══██╗██╔══██║██║   ██║
 ██║ ╚═╝ ██║╚██████╔╝██║ ╚████║╚██████╔╝██║  ██║██║  ██║╚██████╔╝
 ╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝
"""


def strip_quotes(s: str) -> str:
    """Elimina comillas simples y dobles al inicio y fin de una cadena."""
    return s.strip('"').strip("'").strip()


def show_banner() -> None:
    console.print(Text(BANNER, style="bold cyan"))
    console.print(
        Panel(
            "Sistema de Recuperación Aumentada por Generación\n"
            "Escribe [bold]help[/bold] para ver los comandos disponibles.",
            style="bold magenta",
            expand=False,
        ),
        justify="center",
    )
    console.print()


def show_help() -> None:
    table = Table(
        title="Comandos disponibles",
        show_header=True,
        header_style="bold cyan",
        expand=False,
    )
    table.add_column("Comando", style="bold green", min_width=24)
    table.add_column("Descripción")
    table.add_row("create <nombre>", "Crear y seleccionar una colección")
    table.add_row("use <nombre>", "Seleccionar una colección existente")
    table.add_row("index <ruta>", "Indexar un archivo o directorio (auto-detecta)")
    table.add_row("chat", "Entrar en modo chat — escribe preguntas directamente")
    table.add_row("ask <pregunta>", "Hacer una pregunta puntual")
    table.add_row("search <consulta>", "Buscar fragmentos relevantes")
    table.add_row("clear", "Limpiar todos los documentos de la colección activa")
    table.add_row("list", "Listar todas las colecciones")
    table.add_row("delete", "Eliminar la colección activa (pide confirmación)")
    table.add_row("exit / quit", "Salir del CLI")
    console.print(table)


def cmd_create(name: str) -> tuple:
    if not name:
        console.print("[red]Uso: create <nombre>[/red]")
        return None, None
    try:
        with console.status("Creando colección..."):
            rag = RAGModule(collection=name)
        console.print(f"[green]Colección '[bold]{name}[/bold]' creada y seleccionada.[/green]")
        return name, rag
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return None, None


def cmd_use(name: str) -> tuple:
    if not name:
        console.print("[red]Uso: use <nombre>[/red]")
        return None, None
    try:
        with console.status("Conectando a la colección..."):
            rag = RAGModule(collection=name)
        console.print(f"[green]Colección '[bold]{name}[/bold]' seleccionada.[/green]")
        return name, rag
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return None, None


def cmd_index(rag: RAGModule, path_str: str) -> None:
    if not path_str:
        console.print("[red]Uso: index <ruta>[/red]")
        return

    from pathlib import Path
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


def cmd_ask(rag: RAGModule, query: str) -> None:
    if not query:
        console.print("[red]Uso: ask <pregunta>[/red]")
        return
    try:
        with console.status("Generando respuesta..."):
            result = rag.ask(query)
        console.print("\n[bold cyan]Respuesta:[/bold cyan]")
        console.print(result["answer"])
        if result["sources"]:
            console.print("\n[bold]Fuentes:[/bold]")
            for i, src in enumerate(result["sources"], 1):
                meta = src["metadata"]
                console.print(f"  {i}. [dim]{meta['source']}[/dim] — pág. {meta['page']}")
        console.print()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def cmd_chat(rag: RAGModule, collection_name: str) -> None:
    """Modo chat: cada línea es una pregunta. Escribe 'salir' para volver."""
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
        cmd_ask(rag, query)


def cmd_search(rag: RAGModule, query: str) -> None:
    if not query:
        console.print("[red]Uso: search <consulta>[/red]")
        return
    try:
        with console.status("Buscando fragmentos..."):
            results = rag.search(query, top_k=5)
        if not results:
            console.print("[yellow]No se encontraron resultados.[/yellow]")
            return
        console.print(f"\n[bold]{len(results)} resultado(s):[/bold]\n")
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            console.print(
                f"[bold cyan]{i}.[/bold cyan] [dim]{meta['source']}[/dim] "
                f"— pág. {meta['page']}, fragmento {meta['chunk_index']}"
            )
            preview = r["text"][:200] + ("..." if len(r["text"]) > 200 else "")
            console.print(f"   {preview}\n")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def cmd_clear(rag: RAGModule, collection_name: str) -> tuple:
    """Elimina todos los documentos de la colección y crea una nueva vacía."""
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
            new_rag = RAGModule(collection=collection_name)
        console.print(f"[green]Colección '[bold]{collection_name}[/bold]' limpiada correctamente.[/green]")
        return collection_name, new_rag
    except Exception as e:
        console.print(f"[red]Error al limpiar: {e}[/red]")
        return collection_name, rag


def cmd_list() -> None:
    """Lista colecciones usando chromadb directamente — sin instanciar RAGModule."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./chroma_db")
        collections = [c.name for c in client.list_collections()]
        if not collections:
            console.print("[yellow]No hay colecciones disponibles.[/yellow]")
        else:
            console.print("[bold]Colecciones disponibles:[/bold]")
            for name in collections:
                console.print(f"  • {name}")
    except Exception as e:
        console.print(f"[red]Error al listar colecciones: {e}[/red]")


def cmd_delete(rag: RAGModule, collection_name: str) -> tuple:
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


def get_prompt(collection_name) -> str:
    if collection_name:
        return f"[bold cyan]monorag[/bold cyan] [dim]({collection_name})[/dim] > "
    return "[bold cyan]monorag[/bold cyan] > "


def main() -> None:
    show_banner()

    collection_name = None
    rag = None

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

        elif command == "help":
            show_help()

        elif command == "create":
            new_name, new_rag = cmd_create(args.strip())
            if new_name and new_rag:
                collection_name, rag = new_name, new_rag

        elif command == "use":
            new_name, new_rag = cmd_use(args.strip())
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
                cmd_chat(rag, collection_name)

        elif command == "ask":
            if rag is None:
                console.print("[red]Primero selecciona una colección con 'create' o 'use'.[/red]")
            else:
                cmd_ask(rag, args.strip())

        elif command == "search":
            if rag is None:
                console.print("[red]Primero selecciona una colección con 'create' o 'use'.[/red]")
            else:
                cmd_search(rag, args.strip())

        elif command == "clear":
            if rag is None or collection_name is None:
                console.print("[red]No hay colección activa para limpiar.[/red]")
            else:
                collection_name, rag = cmd_clear(rag, collection_name)

        elif command == "list":
            cmd_list()

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