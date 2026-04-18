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
