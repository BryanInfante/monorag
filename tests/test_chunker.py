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

    def test_exactly_500_tokens_single_paragraph_produces_one_chunk(self):
        """Text with exactly 500 tokens and no paragraph breaks fits in one chunk.

        With paragraph-aware chunking, text without \\n\\n is a single
        paragraph.  500 tokens == chunk_size, so it fits in one chunk.
        """
        chunker = Chunker()
        text = " ".join(f"w{i}" for i in range(500))
        result = chunker.chunk(text, source="exact.txt")

        assert len(result) == 1
        assert len(result[0]["text"].split()) == 500
        assert result[0]["metadata"]["chunk_index"] == 0

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


# ---------------------------------------------------------------------------
# Edge-Case Unit Tests for Smart Chunking
# ---------------------------------------------------------------------------

import inspect


class TestSmartChunkingEdgeCases:
    """Example-based edge-case tests for paragraph-aware chunking."""

    # 1. Requirement 8.1
    def test_empty_string_returns_empty(self):
        """chunk('', source='t.txt') returns []."""
        chunker = Chunker()
        assert chunker.chunk("", source="t.txt") == []

    # 2. Requirement 8.2
    def test_whitespace_only_returns_empty(self):
        """chunk('   \\n\\t  ', source='t.txt') returns []."""
        chunker = Chunker()
        assert chunker.chunk("   \n\t  ", source="t.txt") == []

    # 3. Requirement 8.3
    def test_single_small_paragraph_returns_one_chunk(self):
        """Text with < chunk_size tokens and no \\n\\n returns exactly 1 chunk."""
        chunker = Chunker(chunk_size=20, overlap=0)
        text = "hello world foo bar"
        result = chunker.chunk(text, source="t.txt")
        assert len(result) == 1
        assert result[0]["text"] == text

    # 4. Requirement 1.4
    def test_no_paragraph_breaks_treated_as_single_paragraph(self):
        """Text without \\n\\n is treated as one paragraph."""
        chunker = Chunker(chunk_size=50, overlap=0)
        text = "one two three four five six seven eight nine ten"
        result = chunker.chunk(text, source="t.txt")
        assert len(result) == 1
        assert result[0]["text"] == text

    # 5. Requirement 3.2
    def test_large_paragraph_flushes_accumulated_small_paragraphs(self):
        """When a large paragraph follows small ones, the small ones are flushed first."""
        chunker = Chunker(chunk_size=10, overlap=0)
        large_para = " ".join(f"w{i}" for i in range(15))
        text = "a b c\n\n" + large_para
        result = chunker.chunk(text, source="t.txt")

        # First chunk must contain the small paragraph
        assert "a b c" in result[0]["text"]
        # Subsequent chunks contain the large paragraph tokens
        large_tokens = large_para.split()
        remaining_tokens = []
        for ch in result[1:]:
            remaining_tokens.extend(ch["text"].split())
        for token in large_tokens:
            assert token in remaining_tokens

    # 6. Requirement 8.5
    def test_empty_pages_returns_empty(self):
        """chunk_pages([], source='t.txt') returns []."""
        chunker = Chunker()
        assert chunker.chunk_pages([], source="t.txt") == []

    # 7. Requirement 7.1
    def test_constructor_defaults_unchanged(self):
        """Chunker() has chunk_size=500 and overlap=50."""
        chunker = Chunker()
        assert chunker.chunk_size == 500
        assert chunker.overlap == 50

    # 8. Requirement 7.2
    def test_chunk_signature_unchanged(self):
        """chunk() accepts (text, source, start_page) params."""
        sig = inspect.signature(Chunker.chunk)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "text" in params
        assert "source" in params
        assert "start_page" in params

    # 9. Requirement 7.3
    def test_chunk_pages_signature_unchanged(self):
        """chunk_pages() accepts (pages, source) params."""
        sig = inspect.signature(Chunker.chunk_pages)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "pages" in params
        assert "source" in params

    # 10. Requirement 8.4
    def test_all_large_paragraphs_use_token_fallback(self):
        """Text where every paragraph exceeds chunk_size uses token windowing."""
        chunker = Chunker(chunk_size=5, overlap=0)
        para1 = " ".join(f"a{i}" for i in range(10))
        para2 = " ".join(f"b{i}" for i in range(10))
        text = para1 + "\n\n" + para2
        result = chunker.chunk(text, source="t.txt")

        for ch in result:
            assert len(ch["text"].split()) <= 5

    # 11. Requirement 4.3
    def test_overlap_zero_no_shared_tokens(self):
        """With overlap=0, consecutive chunks share no tokens."""
        chunker = Chunker(chunk_size=5, overlap=0)
        text = " ".join(f"t{i}" for i in range(20))
        result = chunker.chunk(text, source="t.txt")

        assert len(result) >= 2
        for i in range(len(result) - 1):
            current_tokens = set(result[i]["text"].split())
            next_tokens = set(result[i + 1]["text"].split())
            assert current_tokens & next_tokens == set()

    # 12. Requirement 7.5
    def test_output_dict_format(self):
        """Output has {"text": str, "metadata": {"source": str, "page": int, "chunk_index": int}}."""
        chunker = Chunker(chunk_size=50, overlap=0)
        result = chunker.chunk("hello world", source="t.txt")

        assert len(result) == 1
        chunk = result[0]
        assert isinstance(chunk["text"], str)
        meta = chunk["metadata"]
        assert isinstance(meta["source"], str)
        assert isinstance(meta["page"], int)
        assert isinstance(meta["chunk_index"], int)
