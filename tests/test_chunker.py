"""Unit tests for rag_core.chunker.Chunker.

Validates: Requirements 7.1, 7.2, 7.3
"""

from rag_core.chunker import Chunker


class TestChunkerUnit:
    """Example-based unit tests for the Chunker class."""

    def test_empty_string_returns_empty_list(self):
        """Empty string input should produce no chunks."""
        chunker = Chunker()
        result = chunker.chunk("", source="test.txt")
        assert result == []

    def test_whitespace_only_returns_empty_list(self):
        """Whitespace-only input should produce no chunks."""
        chunker = Chunker()
        result = chunker.chunk("   \n\t  ", source="test.txt")
        assert result == []

    def test_single_chunk_for_text_within_chunk_size(self):
        """Text with fewer than 500 tokens should produce exactly one chunk."""
        chunker = Chunker()
        text = " ".join(f"token{i}" for i in range(100))
        result = chunker.chunk(text, source="small.txt")

        assert len(result) == 1
        assert result[0]["text"] == text
        assert result[0]["metadata"]["source"] == "small.txt"
        assert result[0]["metadata"]["chunk_index"] == 0

    def test_exactly_500_tokens_produces_two_chunks_due_to_overlap(self):
        """Text with exactly 500 tokens produces 2 chunks because step=450 leaves a tail."""
        chunker = Chunker()
        text = " ".join(f"w{i}" for i in range(500))
        result = chunker.chunk(text, source="exact.txt")

        # Step is 500-50=450, so position 450 still has 50 tokens left -> 2 chunks
        assert len(result) == 2
        assert len(result[0]["text"].split()) == 500
        assert len(result[1]["text"].split()) == 50

    def test_correct_chunk_index_sequencing(self):
        """Chunk indices should be sequential starting from 0."""
        chunker = Chunker()
        text = " ".join(f"w{i}" for i in range(1200))
        result = chunker.chunk(text, source="multi.txt")

        assert len(result) > 1
        for i, chunk in enumerate(result):
            assert chunk["metadata"]["chunk_index"] == i

    def test_chunk_metadata_contains_required_keys(self):
        """Every chunk must have source, page, and chunk_index in metadata."""
        chunker = Chunker()
        text = " ".join(f"w{i}" for i in range(100))
        result = chunker.chunk(text, source="meta.txt", start_page=3)

        for chunk in result:
            meta = chunk["metadata"]
            assert "source" in meta
            assert "page" in meta
            assert "chunk_index" in meta
            assert meta["source"] == "meta.txt"
            assert meta["page"] == 3

    def test_default_page_is_zero_for_txt(self):
        """Default start_page should be 0 (used for TXT files)."""
        chunker = Chunker()
        result = chunker.chunk("hello world", source="file.txt")
        assert result[0]["metadata"]["page"] == 0

    def test_chunk_pages_preserves_page_numbers(self):
        """chunk_pages should assign each chunk the page of its first token."""
        chunker = Chunker(chunk_size=5, overlap=0)
        pages = [
            ("a b c d e", 1),
            ("f g h i j", 2),
        ]
        result = chunker.chunk_pages(pages, source="doc.pdf")

        assert result[0]["metadata"]["page"] == 1
        assert result[1]["metadata"]["page"] == 2

    def test_chunk_pages_empty_pages_returns_empty(self):
        """Empty pages list should return no chunks."""
        chunker = Chunker()
        result = chunker.chunk_pages([], source="empty.pdf")
        assert result == []


# ---------------------------------------------------------------------------
# Property-Based Tests (Hypothesis)
# ---------------------------------------------------------------------------

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

# Strategy: generate a random word (fast, no regex)
_word_st = st.text(
    alphabet=st.characters(whitelist_categories=("Ll",)),
    min_size=1,
    max_size=10,
)


class TestChunkerProperties:
    """Property-based tests for the Chunker class."""

    # Feature: rag-core, Property 1: Non-final chunks are exactly 500 tokens
    @given(
        words=st.lists(_word_st, min_size=501, max_size=5000),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_non_final_chunks_have_exactly_500_tokens(self, words):
        """**Validates: Requirements 2.4, 7.1**

        For any text with >500 tokens, every non-final chunk has exactly
        500 tokens.
        """
        text = " ".join(words)
        chunker = Chunker()
        chunks = chunker.chunk(text, source="prop.txt")

        assert len(chunks) >= 2, "Expected at least 2 chunks for >500 tokens"
        for chunk in chunks[:-1]:
            token_count = len(chunk["text"].split())
            assert token_count == 500, (
                f"Non-final chunk has {token_count} tokens, expected 500"
            )

    # Feature: rag-core, Property 2: Consecutive chunks overlap by exactly 50 tokens
    @given(
        words=st.lists(_word_st, min_size=501, max_size=5000),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_consecutive_chunks_overlap_by_50_tokens(self, words):
        """**Validates: Requirements 7.2**

        For any text producing >=2 chunks, last 50 tokens of chunk N equal
        first 50 tokens of chunk N+1.
        """
        text = " ".join(words)
        chunker = Chunker()
        chunks = chunker.chunk(text, source="prop.txt")

        assert len(chunks) >= 2, "Expected at least 2 chunks"
        for i in range(len(chunks) - 1):
            tail = chunks[i]["text"].split()[-50:]
            head = chunks[i + 1]["text"].split()[:50]
            assert tail == head, (
                f"Overlap mismatch between chunk {i} and {i + 1}"
            )

    # Feature: rag-core, Property 3: Every chunk carries complete metadata
    @given(
        words=st.lists(_word_st, min_size=1, max_size=2000),
        source=st.from_regex(r"[a-z]{1,8}\.(txt|pdf)", fullmatch=True),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_every_chunk_carries_complete_metadata(self, words, source):
        """**Validates: Requirements 2.5**

        For any text and source filename, every chunk has metadata with
        `source`, `page` (>=0), and sequential `chunk_index`.
        """
        text = " ".join(words)
        chunker = Chunker()
        chunks = chunker.chunk(text, source=source)

        assert len(chunks) >= 1, "Expected at least 1 chunk"
        for i, chunk in enumerate(chunks):
            meta = chunk["metadata"]
            assert "source" in meta, "Missing 'source' in metadata"
            assert "page" in meta, "Missing 'page' in metadata"
            assert "chunk_index" in meta, "Missing 'chunk_index' in metadata"
            assert meta["source"] == source, (
                f"Expected source '{source}', got '{meta['source']}'"
            )
            assert meta["page"] >= 0, f"Page must be >= 0, got {meta['page']}"
            assert meta["chunk_index"] == i, (
                f"Expected chunk_index {i}, got {meta['chunk_index']}"
            )
