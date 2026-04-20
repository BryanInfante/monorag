"""Property-based tests for rag_core.embedder.Embedder.

Validates: Requirements 2.6
"""

from unittest.mock import patch, MagicMock

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from rag_core.embedder import Embedder


class TestEmbedderProperties:
    """Property-based tests for the Embedder class."""

    # Feature: rag-core, Property 4: Embeddings are 384-dimensional float vectors
    @given(
        texts=st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N", "Z")),
                min_size=1,
                max_size=50,
            ),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=100)
    def test_embeddings_are_384_dim_float_vectors(self, texts):
        """**Validates: Requirements 2.6**

        For any non-empty list of text strings, Embedder produces one
        384-dim float vector per input.
        """
        n = len(texts)
        fake_embeddings = np.random.rand(n, 384).astype(np.float32)

        with patch("rag_core.embedder.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = fake_embeddings
            mock_st.return_value = mock_model

            embedder = Embedder()
            result = embedder.embed(texts)

        assert len(result) == n, (
            f"Expected {n} embeddings, got {len(result)}"
        )
        for i, vec in enumerate(result):
            assert len(vec) == 384, (
                f"Embedding {i} has {len(vec)} dims, expected 384"
            )
            assert all(isinstance(v, float) for v in vec), (
                f"Embedding {i} contains non-float values"
            )


# ---------------------------------------------------------------------------
# Unit tests for edge cases and integration (Task 4.1)
# ---------------------------------------------------------------------------

import logging
import math


class TestEmbedderDefaults:
    """Unit tests for Embedder constructor defaults."""

    def test_default_batch_size_is_256(self):
        """Default batch_size should be 256.

        **Validates: Requirement 1.1**
        """
        with patch("rag_core.embedder.SentenceTransformer") as mock_st:
            mock_st.return_value = MagicMock()
            embedder = Embedder()

        assert embedder.batch_size == 256


class TestEmbedderEmptyInput:
    """Unit tests for empty-input edge case."""

    def test_empty_list_returns_empty_without_encode(self):
        """embed([]) should return [] without calling encode().

        **Validates: Requirement 2.4**
        """
        with patch("rag_core.embedder.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model

            embedder = Embedder()
            result = embedder.embed([])

        assert result == []
        mock_model.encode.assert_not_called()


class TestEmbedderLogging:
    """Unit tests for Spanish log messages."""

    def test_log_messages_are_in_spanish(self, caplog):
        """Batch progress log messages must be in Spanish.

        **Validates: Requirement 7.2**
        """
        texts = ["text"] * 10
        batch_size = 3
        total_batches = math.ceil(len(texts) / batch_size)

        with patch("rag_core.embedder.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            # Return a fake embedding array for each encode() call
            mock_model.encode.side_effect = lambda batch: np.zeros(
                (len(batch), 384), dtype=np.float32
            )
            mock_st.return_value = mock_model

            embedder = Embedder(batch_size=batch_size)

            with caplog.at_level(logging.INFO, logger="rag_core.embedder"):
                embedder.embed(texts)

        # All INFO messages should contain the Spanish keyword "Procesando lote"
        info_messages = [
            r.message for r in caplog.records if r.levelno == logging.INFO
        ]
        assert len(info_messages) == total_batches
        for msg in info_messages:
            assert "Procesando lote" in msg


class TestRAGModuleIntegration:
    """Integration tests verifying RAGModule constructs Embedder unchanged."""

    def test_ragmodule_constructs_embedder_with_defaults(self, set_groq_api_key):
        """RAGModule should construct Embedder() without extra arguments.

        **Validates: Requirements 4.1, 4.2**
        """
        with (
            patch("rag_core.module.Embedder") as mock_embedder_cls,
            patch("rag_core.module.Retriever"),
            patch("rag_core.module.Generator"),
        ):
            mock_embedder_cls.return_value = MagicMock()

            from rag_core.module import RAGModule

            RAGModule(collection="test-collection")

        # Embedder() called with no arguments (defaults apply)
        mock_embedder_cls.assert_called_once_with()


class TestRetrieverReceivesCompleteEmbeddings:
    """Integration test: Retriever.add() receives the full embedding list."""

    def test_retriever_add_receives_complete_embeddings_after_batching(
        self, set_groq_api_key, tmp_path
    ):
        """After batching, Retriever.add() must receive the complete list.

        **Validates: Requirement 4.3**
        """
        num_chunks = 10
        dim = 384

        # Create a real temporary .txt file so Path.exists() passes
        dummy_file = tmp_path / "dummy.txt"
        dummy_file.write_text("dummy content")

        fake_chunks = [
            {
                "text": f"chunk {i}",
                "metadata": {"source": "dummy.txt", "page": 0, "chunk_index": i},
            }
            for i in range(num_chunks)
        ]

        # Build the expected full embedding list
        all_embeddings: list[list[float]] = [
            [float(i)] * dim for i in range(num_chunks)
        ]

        with (
            patch("rag_core.module.Embedder") as mock_embedder_cls,
            patch("rag_core.module.Retriever") as mock_retriever_cls,
            patch("rag_core.module.Generator"),
            patch("rag_core.module.Chunker") as mock_chunker_cls,
            patch("rag_core.module.extract_txt", return_value="dummy text"),
        ):
            mock_embedder = MagicMock()
            mock_embedder.embed.return_value = all_embeddings
            mock_embedder_cls.return_value = mock_embedder

            mock_retriever = MagicMock()
            mock_retriever.has_source.return_value = False
            mock_retriever_cls.return_value = mock_retriever

            mock_chunker = MagicMock()
            mock_chunker.chunk.return_value = fake_chunks
            mock_chunker_cls.return_value = mock_chunker

            from rag_core.module import RAGModule

            rag = RAGModule(collection="test-int")
            rag.add_file(str(dummy_file))

        # Retriever.add() should have been called once with the full embeddings
        mock_retriever.add.assert_called_once()
        _, kwargs = mock_retriever.add.call_args
        received_embeddings = kwargs["embeddings"]

        assert len(received_embeddings) == num_chunks
        assert received_embeddings == all_embeddings
