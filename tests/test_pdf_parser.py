"""Tests for extract_text_from_pdf in app.py."""
import io
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app import extract_text_from_pdf


def make_mock_reader(page_texts: list[str]):
    """Build a mock PdfReader whose .pages list returns pages with extract_text()."""
    pages = []
    for text in page_texts:
        page = MagicMock()
        page.extract_text.return_value = text
        pages.append(page)
    reader = MagicMock()
    reader.pages = pages
    return reader


# ---------------------------------------------------------------------------
# Property 3: Page extraction limit
# Feature: smart-study-buddy, Property 3: Page extraction limit
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(st.integers(min_value=0, max_value=20))
def test_page_extraction_limit(n):
    """For any PDF with N pages, extract_text_from_pdf uses exactly min(N, 10) pages,
    resulting in min(N, 10) - 1 separator tokens when N > 0.

    Validates: Requirements 1.4, 1.5, 2.1, 2.3
    """
    # Each page gets unique non-empty text so separators are always emitted
    page_texts = [f"Content of page {i}" for i in range(1, n + 1)]
    mock_reader = make_mock_reader(page_texts)

    with patch("app.PyPDF2.PdfReader", return_value=mock_reader):
        if n == 0:
            with pytest.raises(ValueError, match="No readable text found in the PDF."):
                extract_text_from_pdf(io.BytesIO(b"fake"))
        else:
            result = extract_text_from_pdf(io.BytesIO(b"fake"))
            expected_pages = min(n, 10)
            separator = "--- Page "
            separator_count = result.count(separator)
            assert separator_count == expected_pages - 1, (
                f"Expected {expected_pages - 1} separators for {n} pages "
                f"(capped at {expected_pages}), got {separator_count}"
            )


# ---------------------------------------------------------------------------
# Unit tests for edge cases
# ---------------------------------------------------------------------------

def test_zero_pages_raises_value_error():
    """PDF with 0 extractable pages raises ValueError."""
    mock_reader = make_mock_reader([])
    with patch("app.PyPDF2.PdfReader", return_value=mock_reader):
        with pytest.raises(ValueError, match="No readable text found in the PDF."):
            extract_text_from_pdf(io.BytesIO(b"fake"))


def test_exactly_10_pages_returns_9_separators():
    """PDF with exactly 10 pages returns text with exactly 9 separator tokens."""
    page_texts = [f"Page {i} text" for i in range(1, 11)]
    mock_reader = make_mock_reader(page_texts)
    with patch("app.PyPDF2.PdfReader", return_value=mock_reader):
        result = extract_text_from_pdf(io.BytesIO(b"fake"))
    assert result.count("--- Page ") == 9


def test_15_pages_returns_9_separators():
    """PDF with 15 pages is capped at 10, resulting in exactly 9 separator tokens."""
    page_texts = [f"Page {i} text" for i in range(1, 16)]
    mock_reader = make_mock_reader(page_texts)
    with patch("app.PyPDF2.PdfReader", return_value=mock_reader):
        result = extract_text_from_pdf(io.BytesIO(b"fake"))
    assert result.count("--- Page ") == 9
