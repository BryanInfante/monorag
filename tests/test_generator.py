"""Unit tests for rag_core.generator.Generator.

Validates: Requirements 5.3, 5.5
"""

from unittest.mock import MagicMock, patch

import pytest

from rag_core.generator import SYSTEM_PROMPT, Generator


class TestSystemPrompt:
    """Verify the system prompt content."""

    def test_system_prompt_is_in_spanish(self):
        """SYSTEM_PROMPT should contain expected Spanish instructions."""
        assert "Responde" in SYSTEM_PROMPT
        assert "español" in SYSTEM_PROMPT
        assert "contexto" in SYSTEM_PROMPT

    def test_system_prompt_mentions_normative_documents(self):
        """SYSTEM_PROMPT should reference technical normative documents."""
        assert "normativos" in SYSTEM_PROMPT

    def test_system_prompt_instructs_source_citation(self):
        """SYSTEM_PROMPT should instruct the model to cite sources."""
        assert "fuentes" in SYSTEM_PROMPT


class TestGeneratorGenerate:
    """Unit tests for Generator.generate method."""

    @patch("rag_core.generator.Groq")
    def test_generate_returns_answer(self, mock_groq_cls):
        """generate() should return the content from the Groq response."""
        mock_message = MagicMock()
        mock_message.content = "Respuesta generada por el modelo."

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_groq_cls.return_value = mock_client

        gen = Generator(api_key="fake-key")
        chunks = [
            {
                "text": "Some context text.",
                "metadata": {"source": "doc.pdf", "page": 1, "chunk_index": 0},
            }
        ]
        result = gen.generate("What is this?", chunks)

        assert result == "Respuesta generada por el modelo."
        mock_client.chat.completions.create.assert_called_once()

    @patch("rag_core.generator.Groq")
    def test_groq_api_error_raises_runtime_error(self, mock_groq_cls):
        """Groq API failure should propagate as RuntimeError with Spanish message."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception(
            "API rate limit exceeded"
        )
        mock_groq_cls.return_value = mock_client

        gen = Generator(api_key="fake-key")
        chunks = [
            {
                "text": "Context.",
                "metadata": {"source": "doc.txt", "page": 0, "chunk_index": 0},
            }
        ]

        with pytest.raises(RuntimeError, match="Error de la API de Groq"):
            gen.generate("test query", chunks)
