"""Unit tests for rag_core.utils (PDF and TXT extraction).

Validates: Requirements 8.3, 8.4, 9.3
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from rag_core.utils import extract_pdf, extract_txt


class TestExtractPdf:
    """Unit tests for extract_pdf using mocked pdfplumber."""

    @patch("rag_core.utils.pdfplumber")
    def test_extracts_pages_with_text(self, mock_pdfplumber):
        """Should return (text, page_number) tuples for pages with text."""
        page1 = MagicMock()
        page1.extract_text.return_value = "Page one content"
        page2 = MagicMock()
        page2.extract_text.return_value = "Page two content"

        mock_pdf = MagicMock()
        mock_pdf.pages = [page1, page2]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdfplumber.open.return_value = mock_pdf

        result = extract_pdf("fake.pdf")

        assert len(result) == 2
        assert result[0] == ("Page one content", 1)
        assert result[1] == ("Page two content", 2)

    @patch("rag_core.utils.pdfplumber")
    def test_skips_pages_with_no_text(self, mock_pdfplumber):
        """Pages returning None or empty text should be skipped."""
        page1 = MagicMock()
        page1.extract_text.return_value = None
        page2 = MagicMock()
        page2.extract_text.return_value = ""
        page3 = MagicMock()
        page3.extract_text.return_value = "   "
        page4 = MagicMock()
        page4.extract_text.return_value = "Actual content"

        mock_pdf = MagicMock()
        mock_pdf.pages = [page1, page2, page3, page4]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdfplumber.open.return_value = mock_pdf

        result = extract_pdf("fake.pdf")

        assert len(result) == 1
        assert result[0] == ("Actual content", 4)

    @patch("rag_core.utils.pdfplumber")
    def test_corrupted_pdf_raises_runtime_error(self, mock_pdfplumber):
        """A corrupted PDF should raise RuntimeError with Spanish message."""
        mock_pdfplumber.open.side_effect = Exception("corrupt file")

        with pytest.raises(RuntimeError, match="No se pudo analizar el PDF"):
            extract_pdf("corrupted.pdf")


class TestExtractTxt:
    """Unit tests for extract_txt."""

    def test_reads_known_content(self, tmp_path):
        """Should return the exact content of a UTF-8 text file."""
        content = "Contenido de prueba con acentos: ñ, ü, é."
        txt_file = tmp_path / "sample.txt"
        txt_file.write_text(content, encoding="utf-8")

        result = extract_txt(str(txt_file))
        assert result == content

    def test_reads_empty_file(self, tmp_path):
        """An empty file should return an empty string."""
        txt_file = tmp_path / "empty.txt"
        txt_file.write_text("", encoding="utf-8")

        result = extract_txt(str(txt_file))
        assert result == ""

    def test_encoding_error_raises_runtime_error(self, tmp_path):
        """A file with invalid encoding should raise RuntimeError with Spanish message."""
        bad_file = tmp_path / "bad_encoding.txt"
        bad_file.write_bytes(b"\x80\x81\x82\x83\xff\xfe")

        with pytest.raises(RuntimeError, match="No se pudo leer"):
            extract_txt(str(bad_file))


# ---------------------------------------------------------------------------
# Property-Based Tests (Hypothesis)
# ---------------------------------------------------------------------------

from hypothesis import given, settings
from hypothesis import strategies as st


class TestExtractTxtProperties:
    """Property-based tests for TXT extraction."""

    # Feature: rag-core, Property 5: TXT extraction round-trip preserves content
    @given(content=st.text(alphabet=st.characters(blacklist_characters="\r")))
    @settings(max_examples=100)
    def test_txt_round_trip_preserves_content(self, content):
        """**Validates: Requirements 2.3, 3.2, 9.1, 9.2**

        For any valid UTF-8 string, write to TXT then extract_txt returns
        the original.
        """
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=False
        ) as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            result = extract_txt(path)
            assert result == content, "Round-trip content mismatch"
        finally:
            os.unlink(path)
