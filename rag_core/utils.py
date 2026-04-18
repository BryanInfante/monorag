"""Document extraction utilities for PDF and TXT files."""

import pdfplumber


def extract_pdf(file_path: str) -> list[tuple[str, int]]:
    """Extract text from a PDF file, page by page.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of (page_text, page_number) tuples. Pages with no text are skipped.

    Raises:
        RuntimeError: If the PDF cannot be parsed.
    """
    try:
        pages: list[tuple[str, int]] = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text is not None and text.strip():
                    pages.append((text, i))
        return pages
    except Exception as exc:
        raise RuntimeError(
            f"No se pudo analizar el PDF {file_path}: {exc}"
        ) from exc


def extract_txt(file_path: str) -> str:
    """Read the full text content of a TXT file.

    Args:
        file_path: Path to the TXT file.

    Returns:
        The file's text content.

    Raises:
        RuntimeError: If the file cannot be read (e.g., encoding error).
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as exc:
        raise RuntimeError(
            f"No se pudo leer {file_path}: {exc}"
        ) from exc
