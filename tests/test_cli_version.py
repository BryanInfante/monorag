"""Unit tests for CLI global version flags."""

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


def test_main_prints_version_for_long_flag(monkeypatch):
    """`monorag --version` should print version and not open the REPL."""
    buffer = _capture_cli_console(monkeypatch)
    monkeypatch.setattr(cli, "package_version", lambda package_name: "9.8.7")

    cli.main(["--version"])

    assert buffer.getvalue().strip() == "MonoRAG 9.8.7"


def test_main_prints_version_for_short_flag(monkeypatch):
    """`monorag -V` should print version and not open the REPL."""
    buffer = _capture_cli_console(monkeypatch)
    monkeypatch.setattr(cli, "package_version", lambda package_name: "9.8.7")

    cli.main(["-V"])

    assert buffer.getvalue().strip() == "MonoRAG 9.8.7"


def test_main_prints_version_for_lowercase_short_flag(monkeypatch):
    """`monorag -v` should match common CLI version flag expectations."""
    buffer = _capture_cli_console(monkeypatch)
    monkeypatch.setattr(cli, "package_version", lambda package_name: "9.8.7")

    cli.main(["-v"])

    assert buffer.getvalue().strip() == "MonoRAG 9.8.7"
