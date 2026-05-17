"""Unit tests for CLI configuration helpers."""

import builtins

import pytest

from cli import CliConfig, cmd_config, load_cli_config


@pytest.fixture(autouse=True)
def _isolated_cli_config(monkeypatch, tmp_path):
    """Keep persistent CLI config tests away from the real user profile."""
    monkeypatch.setenv("MONORAG_CONFIG_PATH", str(tmp_path / "config.json"))


def _feed_input(monkeypatch, values):
    """Patch input() to return values sequentially."""
    iterator = iter(values)
    monkeypatch.setattr(builtins, "input", lambda: next(iterator))


def _clear_llm_env(monkeypatch):
    for key in ("LLM_PROVIDER", "LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(key, raising=False)


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
