"""Shared pytest fixtures for rag_core tests."""

import os
import tempfile

import pytest


@pytest.fixture
def tmp_chroma_dir(tmp_path):
    """Provide a temporary directory for ChromaDB persistence."""
    chroma_dir = tmp_path / "chroma_db"
    chroma_dir.mkdir()
    return str(chroma_dir)


@pytest.fixture
def tmp_docs_dir(tmp_path):
    """Provide a temporary directory for test document files."""
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    return docs_dir


@pytest.fixture
def sample_text():
    """Return a short sample text for chunking tests."""
    return "This is a simple sample text used for testing the chunker module."


@pytest.fixture
def sample_long_text():
    """Return a text with more than 500 whitespace-delimited tokens."""
    words = [f"word{i}" for i in range(600)]
    return " ".join(words)


@pytest.fixture
def sample_chunks():
    """Return pre-built sample chunk dicts for tests that need them."""
    return [
        {
            "text": "First chunk of text content.",
            "metadata": {"source": "test.txt", "page": 0, "chunk_index": 0},
        },
        {
            "text": "Second chunk of text content.",
            "metadata": {"source": "test.txt", "page": 0, "chunk_index": 1},
        },
    ]


@pytest.fixture
def mock_groq_response():
    """Return a mock object mimicking a Groq chat completion response."""
    from unittest.mock import MagicMock

    message = MagicMock()
    message.content = "Esta es una respuesta generada."

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.fixture
def mock_groq_client(mock_groq_response):
    """Return a mock Groq client whose chat.completions.create returns a canned response."""
    from unittest.mock import MagicMock

    client = MagicMock()
    client.chat.completions.create.return_value = mock_groq_response
    return client


@pytest.fixture
def set_groq_api_key(monkeypatch):
    """Set a fake GROQ_API_KEY environment variable for tests."""
    monkeypatch.setenv("GROQ_API_KEY", "test-fake-api-key-12345")
