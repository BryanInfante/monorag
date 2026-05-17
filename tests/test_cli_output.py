"""Unit tests for CLI result rendering."""

from io import StringIO

from rich.console import Console

import cli


def _capture_cli_console(monkeypatch) -> StringIO:
    buffer = StringIO()
    monkeypatch.setattr(
        cli,
        "console",
        Console(file=buffer, force_terminal=False, color_system=None, width=120),
    )
    return buffer


def test_cmd_ask_prints_markdown_source_without_page(monkeypatch):
    """MD sources intentionally have no page metadata and must not crash output."""

    class FakeRAG:
        def ask(self, query):
            return {
                "answer": "Respuesta fake",
                "sources": [
                    {
                        "text": "contexto",
                        "metadata": {"source": "manual.md", "chunk_index": 0},
                    }
                ],
            }

    buffer = _capture_cli_console(monkeypatch)

    cli.cmd_ask(FakeRAG(), "¿qué dice?")

    output = buffer.getvalue()
    assert "Respuesta fake" in output
    assert "manual.md" in output
    assert "Error: 'page'" not in output
    assert "pág." not in output


def test_cmd_ask_hides_legacy_page_zero_for_markdown(monkeypatch):
    """Old MD chunks may still have page=0; the CLI should not show fake pages."""

    class FakeRAG:
        def ask(self, query):
            return {
                "answer": "Respuesta fake",
                "sources": [
                    {
                        "text": "contexto",
                        "metadata": {"source": "manual.md", "page": 0, "chunk_index": 0},
                    }
                ],
            }

    buffer = _capture_cli_console(monkeypatch)

    cli.cmd_ask(FakeRAG(), "¿qué dice?")

    output = buffer.getvalue()
    assert "manual.md" in output
    assert "pág. 0" not in output


def test_cmd_search_prints_markdown_source_without_page(monkeypatch):
    """Search output should keep chunk info while omitting fake pages for MD."""

    class FakeRAG:
        def search(self, query):
            return [
                {
                    "text": "fragmento de contexto",
                    "metadata": {"source": "manual.md", "chunk_index": 2},
                }
            ]

    buffer = _capture_cli_console(monkeypatch)

    cli.cmd_search(FakeRAG(), "cracks")

    output = buffer.getvalue()
    assert "manual.md" in output
    assert "fragmento 2" in output
    assert "Error: 'page'" not in output
    assert "pág." not in output
