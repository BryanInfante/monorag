"""Unit tests for rag_core.generator.Generator.

Validates: Requirements 5.3, 5.5
"""

from unittest.mock import MagicMock

import pytest

from rag_core.generator import SYSTEM_PROMPT, Generator, _format_source_reference


class TestSystemPrompt:
    """Verify the system prompt content."""

    def test_system_prompt_instructs_same_language_as_question(self):
        """SYSTEM_PROMPT should instruct the model to answer in the question language."""
        assert "same language" in SYSTEM_PROMPT
        assert "user's question" in SYSTEM_PROMPT
        assert "provided context" in SYSTEM_PROMPT

    def test_system_prompt_mentions_normative_documents(self):
        """SYSTEM_PROMPT should reference technical normative documents."""
        assert "normative documents" in SYSTEM_PROMPT

    def test_system_prompt_instructs_source_citation(self):
        """SYSTEM_PROMPT should instruct the model to cite sources."""
        assert "sources" in SYSTEM_PROMPT


class TestGeneratorGenerate:
    """Unit tests for Generator.generate method."""

    def test_generate_returns_answer_from_injected_provider(self):
        """generate() should return the content from any injected provider."""
        provider = MagicMock()
        provider.complete.return_value = "Respuesta generada por el modelo."

        gen = Generator(api_key="fake-key", model="test-model", provider=provider)
        chunks = [
            {
                "text": "Some context text.",
                "metadata": {"source": "doc.pdf", "page": 1, "chunk_index": 0},
            }
        ]
        result = gen.generate("What is this?", chunks)

        assert result == "Respuesta generada por el modelo."
        provider.complete.assert_called_once()
        kwargs = provider.complete.call_args.kwargs
        assert kwargs["model"] == "test-model"
        messages = kwargs["messages"]
        assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert messages[-1]["role"] == "user"
        assert "Context:" in messages[-1]["content"]
        assert "source: doc.pdf, page: 1" in messages[-1]["content"]

    def test_provider_error_raises_runtime_error(self):
        """Provider failures should propagate as RuntimeError with Spanish message."""
        provider = MagicMock()
        provider.complete.side_effect = Exception("API rate limit exceeded")

        gen = Generator(api_key="fake-key", provider=provider)
        chunks = [
            {
                "text": "Context.",
                "metadata": {"source": "doc.txt", "page": 0, "chunk_index": 0},
            }
        ]

        with pytest.raises(RuntimeError, match="Error al llamar al LLM"):
            gen.generate("test query", chunks)

    def test_provider_alias_sets_default_model(self):
        """Provider aliases should select a sensible default model."""
        provider = MagicMock()
        provider.complete.return_value = "ok"

        gen = Generator(api_key="fake-key", provider_name="groq", provider=provider)
        gen.generate("pregunta", [])

        assert provider.complete.call_args.kwargs["model"] == "llama-3.3-70b-versatile"


class TestSourceFormatting:
    """Unit tests for human-friendly source references."""

    def test_txt_with_page_zero_omits_fake_page(self):
        """TXT/MD sources should not display page 0 as if it were a real page."""
        assert _format_source_reference({"source": "archivo.txt", "page": 0}) == "source: archivo.txt"
        assert _format_source_reference({"source": "notas.md", "page": 0}) == "source: notas.md"

    def test_pdf_keeps_real_page_numbers(self):
        """Paginated formats should preserve the page label."""
        assert _format_source_reference({"source": "manual.pdf", "page": 3}) == "source: manual.pdf, page: 3"
