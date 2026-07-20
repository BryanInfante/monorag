"""Unit tests for InspectorGenerator."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent_generator import InspectorGenerator, DEFAULT_SYSTEM_PROMPT


class TestAgentsMarkdownLoading:
    """Tests for agents.md loading and fallback behavior."""

    def test_loads_agents_md_from_path(self, tmp_path):
        """Given agents.md exists, When Generator is created, Then it loads the content."""
        agents_file = tmp_path / "agents.md"
        agents_file.write_text("# My Agent\n\nYou are a test agent.", encoding="utf-8")

        mock_provider = MagicMock()
        gen = InspectorGenerator(
            api_key="fake-key",
            agents_path=str(agents_file),
            provider=mock_provider,
        )

        assert "You are a test agent" in gen.system_prompt

    def test_uses_fallback_when_agents_md_missing(self):
        """Given agents.md does not exist, When Generator is created, Then fallback prompt used."""
        mock_provider = MagicMock()
        gen = InspectorGenerator(
            api_key="fake-key",
            agents_path="/nonexistent/agents.md",
            provider=mock_provider,
        )

        assert gen.system_prompt == DEFAULT_SYSTEM_PROMPT

    def test_uses_fallback_when_agents_md_empty(self, tmp_path):
        """Given agents.md is empty, When Generator is created, Then fallback prompt used."""
        agents_file = tmp_path / "agents.md"
        agents_file.write_text("", encoding="utf-8")

        mock_provider = MagicMock()
        gen = InspectorGenerator(
            api_key="fake-key",
            agents_path=str(agents_file),
            provider=mock_provider,
        )

        assert gen.system_prompt == DEFAULT_SYSTEM_PROMPT


class TestGenerate:
    """Tests for the generate method."""

    def test_generate_returns_provider_response(self):
        """Given a query and chunks, When generate is called, Then returns provider text."""
        mock_provider = MagicMock()
        mock_provider.complete.return_value = "The acceptance criteria is 3mm max."

        gen = InspectorGenerator(
            api_key="fake-key",
            agents_path="/nonexistent/agents.md",
            provider=mock_provider,
        )

        chunks = [
            {"text": "Cracks longer than 3mm are rejectable.", "metadata": {"source": "ndt.pdf", "page": 5}},
        ]
        result = gen.generate("What is the max crack length?", chunks)

        assert result == "The acceptance criteria is 3mm max."
        mock_provider.complete.assert_called_once()

    def test_generate_builds_correct_messages(self):
        """Given a query, When generate is called, Then messages include system + context + query."""
        mock_provider = MagicMock()
        mock_provider.complete.return_value = "Answer"

        gen = InspectorGenerator(
            api_key="fake-key",
            agents_path="/nonexistent/agents.md",
            provider=mock_provider,
        )

        chunks = [
            {"text": "Some NDT content.", "metadata": {"source": "doc.pdf", "page": 1}},
        ]
        gen.generate("My question?", chunks)

        call_args = mock_provider.complete.call_args
        messages = call_args.kwargs["messages"]

        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == DEFAULT_SYSTEM_PROMPT
        assert messages[-1]["role"] == "user"
        assert "My question?" in messages[-1]["content"]
        assert "Some NDT content." in messages[-1]["content"]

    def test_generate_includes_history(self):
        """Given conversation history, When generate is called, Then history is in messages."""
        mock_provider = MagicMock()
        mock_provider.complete.return_value = "Follow-up answer"

        gen = InspectorGenerator(
            api_key="fake-key",
            agents_path="/nonexistent/agents.md",
            provider=mock_provider,
        )

        history = [
            {"query": "First question?", "answer": "First answer."},
        ]
        chunks = [{"text": "Context.", "metadata": {"source": "x.pdf", "page": 1}}]
        gen.generate("Second question?", chunks, history=history)

        call_args = mock_provider.complete.call_args
        messages = call_args.kwargs["messages"]

        # system + history_user + history_assistant + current_user
        assert len(messages) == 4
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "First question?"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "First answer."

    def test_generate_wraps_provider_errors(self):
        """Given provider raises, When generate is called, Then RuntimeError raised."""
        mock_provider = MagicMock()
        mock_provider.complete.side_effect = Exception("Rate limited")

        gen = InspectorGenerator(
            api_key="fake-key",
            agents_path="/nonexistent/agents.md",
            provider=mock_provider,
        )

        chunks = [{"text": "Content.", "metadata": {"source": "x.pdf", "page": 1}}]

        with pytest.raises(RuntimeError, match="Rate limited"):
            gen.generate("Question?", chunks)
