"""Unit tests for CLI configuration helpers."""

import builtins
import json
from pathlib import Path

import pytest

import cli
from cli import CliConfig, cmd_config, load_cli_config
from rag_core.secret_store import KEYRING_REFERENCE, LLM_API_KEY_SECRET


@pytest.fixture(autouse=True)
def _isolated_cli_config(monkeypatch, tmp_path):
    """Keep persistent CLI config tests away from the real user profile."""
    monkeypatch.setenv("MONORAG_CONFIG_PATH", str(tmp_path / "config.json"))
    monkeypatch.setenv("MONORAG_DISABLE_KEYRING", "1")


def _feed_input(monkeypatch, values):
    """Patch input() to return values sequentially."""
    iterator = iter(values)
    monkeypatch.setattr(builtins, "input", lambda: next(iterator))


def _clear_llm_env(monkeypatch):
    for key in ("LLM_PROVIDER", "LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(key, raising=False)


class _ConsoleProbe:
    def __init__(self):
        self.output = ""

    def print(self, *values, **kwargs):
        self.output += " ".join(str(value) for value in values)


def test_config_llm_opens_guided_wizard_for_known_provider(monkeypatch):
    """`config llm` should guide provider setup in one consecutive flow."""
    _clear_llm_env(monkeypatch)
    _feed_input(monkeypatch, ["2", "", "gsk-test-key", ""])

    config = cmd_config(CliConfig(), "llm")

    assert config.llm_provider == "groq"
    assert config.llm_base_url is None  # built-in alias supplies Groq base URL
    assert config.llm_api_key == "gsk-test-key"
    assert config.llm_model == "llama-3.3-70b-versatile"

    persisted = load_cli_config()
    assert persisted.llm_provider == "groq"
    assert persisted.llm_base_url is None
    assert persisted.llm_api_key == "gsk-test-key"
    assert persisted.llm_model == "llama-3.3-70b-versatile"


def test_config_llm_wizard_supports_custom_openai_compatible_provider(monkeypatch):
    """Custom providers should collect provider name, base URL, key, and model."""
    _clear_llm_env(monkeypatch)
    _feed_input(
        monkeypatch,
        [
            "6",
            "Acme AI",
            "https://llm.acme.test/v1",
            "acme-key",
            "acme-model",
        ],
    )

    config = cmd_config(CliConfig(), "llm")

    assert config.llm_provider == "acme-ai"
    assert config.llm_base_url == "https://llm.acme.test/v1"
    assert config.llm_api_key == "acme-key"
    assert config.llm_model == "acme-model"


def test_config_llm_default_resets_persisted_values():
    """`config llm default` should clear persisted LLM overrides."""
    config = CliConfig(
        llm_provider="groq",
        llm_base_url="https://api.groq.com/openai/v1",
        llm_model="llama-3.3-70b-versatile",
        llm_api_key="gsk-test-key",
    )

    result = cmd_config(config, "llm default")

    assert result.llm_provider is None
    assert result.llm_base_url is None
    assert result.llm_model is None
    assert result.llm_api_key is None

    persisted = load_cli_config()
    assert persisted.llm_provider is None
    assert persisted.llm_base_url is None
    assert persisted.llm_model is None
    assert persisted.llm_api_key is None


def test_config_db_path_is_persisted():
    """Custom DB path should survive a new CLI process."""
    result = cmd_config(CliConfig(), r"db path C:\monorag-db")

    assert result.db_path == r"C:\monorag-db"
    assert result.db_url is None

    persisted = load_cli_config()
    assert persisted.db_path == r"C:\monorag-db"
    assert persisted.db_url is None


def test_llm_api_key_uses_keyring_when_available(monkeypatch):
    """LLM API keys should move to keyring instead of staying in config JSON."""
    fake_keyring: dict[str, str] = {}

    def fake_set_secret(name: str, value: str) -> bool:
        fake_keyring[name] = value
        return True

    monkeypatch.setattr(cli, "set_secret", fake_set_secret)
    monkeypatch.setattr(cli, "get_secret", lambda name: fake_keyring.get(name))
    monkeypatch.setattr(cli, "delete_secret", lambda name: fake_keyring.pop(name, None) is not None)

    cli.save_cli_config(CliConfig(llm_provider="groq", llm_api_key="gsk-secret"))

    raw = json.loads(Path(cli.default_config_path()).read_text(encoding="utf-8"))
    assert raw["llm_api_key"] == KEYRING_REFERENCE
    assert fake_keyring[LLM_API_KEY_SECRET] == "gsk-secret"

    persisted = load_cli_config()
    assert persisted.llm_api_key == "gsk-secret"


def test_keyring_marker_is_preserved_when_secret_cannot_be_read():
    """Editing unrelated config should not erase a keyring-backed secret marker."""
    config_path = Path(cli.default_config_path())
    config_path.write_text(
        json.dumps({"llm_api_key": KEYRING_REFERENCE}),
        encoding="utf-8",
    )

    config = load_cli_config()
    config.chunk_size = 900
    cli.save_cli_config(config)

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    assert raw["chunk_size"] == 900
    assert raw["llm_api_key"] == KEYRING_REFERENCE


def test_config_llm_default_clears_keyring_marker():
    """Explicitly resetting LLM config should clear the persisted secret marker."""
    config_path = Path(cli.default_config_path())
    config_path.write_text(
        json.dumps({"llm_api_key": KEYRING_REFERENCE}),
        encoding="utf-8",
    )

    cmd_config(CliConfig(), "llm default")

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    assert raw["llm_api_key"] is None


def test_prompt_secret_keeps_current_value_without_printing_it(monkeypatch):
    """The secret prompt should not render the current API key as a default."""
    probe = _ConsoleProbe()
    monkeypatch.setattr(cli, "console", probe)
    _feed_input(monkeypatch, [""])

    result = cli.prompt_secret("API key", "gsk-secret")

    assert result == "gsk-secret"
    assert "gsk-secret" not in probe.output


def test_runtime_config_resolves_keyring_marker(monkeypatch):
    """RAGModule runtime config should read CLI secrets stored in keyring."""
    from rag_core import module as rag_module

    config_path = Path(cli.default_config_path())
    config_path.write_text(
        json.dumps({"llm_api_key": KEYRING_REFERENCE}),
        encoding="utf-8",
    )
    monkeypatch.setattr(rag_module, "get_secret", lambda name: "runtime-secret")

    config = rag_module._load_persisted_runtime_config()

    assert config["llm_api_key"] == "runtime-secret"
