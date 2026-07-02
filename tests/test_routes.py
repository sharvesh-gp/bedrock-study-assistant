"""Tests for Flask routes: property tests (Properties 1, 2) and unit tests for edge cases."""

import io
import json
from unittest.mock import patch, MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app import app

# ---------------------------------------------------------------------------
# Pytest fixture: Flask test client
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pdf_upload(data: bytes = b"%PDF-fake", filename: str = "notes.pdf",
                     content_type: str = "application/pdf"):
    """Return a FileStorage-compatible tuple for use with test client data."""
    return (io.BytesIO(data), filename, content_type)


def _post_process(client, file_tuple=None, mode="summarize", question=""):
    data = {"mode": mode, "question": question}
    if file_tuple is not None:
        data["file"] = file_tuple
    return client.post(
        "/process",
        data=data,
        content_type="multipart/form-data",
    )


# ---------------------------------------------------------------------------
# Property 1: File type rejection
# Feature: smart-study-buddy, Property 1: File type rejection
# ---------------------------------------------------------------------------

# Non-PDF extensions to generate
NON_PDF_EXTENSIONS = [".txt", ".docx", ".png", ".jpg", ".csv", ".html", ".xml", ".zip"]

non_pdf_extension = st.sampled_from(NON_PDF_EXTENSIONS)
non_pdf_mime = st.sampled_from([
    "text/plain", "image/png", "image/jpeg",
    "application/msword", "text/html", "application/zip",
])


@settings(max_examples=100)
@given(ext=non_pdf_extension, mime=non_pdf_mime)
def test_property1_file_type_rejection(ext, mime):
    """For any file whose extension or MIME type is not PDF, /process returns
    success: false with a non-empty error.

    Validates: Requirements 1.1, 1.3
    """
    # Feature: smart-study-buddy, Property 1: File type rejection
    with app.test_client() as client:
        filename = f"notes{ext}"
        file_tuple = (io.BytesIO(b"some content"), filename, mime)
        resp = _post_process(client, file_tuple=file_tuple)

    assert resp.status_code == 200
    body = json.loads(resp.data)
    assert body["success"] is False
    assert body.get("error", "") != ""


# ---------------------------------------------------------------------------
# Property 2: File size rejection
# Feature: smart-study-buddy, Property 2: File size rejection
# ---------------------------------------------------------------------------

MAX_SIZE = 5 * 1024 * 1024  # 5 MB

# Generate a small header byte so Hypothesis varies the input; the large payload
# is constructed in the test body to avoid slow/oversized strategy generation.
@settings(max_examples=25)
@given(header=st.binary(min_size=1, max_size=16))
def test_property2_file_size_rejection(header):
    """For any file exceeding 5 MB, /process returns success: false with a size-related error.

    Validates: Requirements 1.2
    """
    # Feature: smart-study-buddy, Property 2: File size rejection
    # Build a payload guaranteed to exceed 5 MB
    payload = header + b"\x00" * (MAX_SIZE + 1)
    with app.test_client() as client:
        file_tuple = (io.BytesIO(payload), "notes.pdf", "application/pdf")
        resp = _post_process(client, file_tuple=file_tuple)

    assert resp.status_code == 200
    body = json.loads(resp.data)
    assert body["success"] is False
    assert "5MB" in body.get("error", "") or "size" in body.get("error", "").lower()


# ---------------------------------------------------------------------------
# Unit tests: route edge cases (task 6.4)
# ---------------------------------------------------------------------------

def test_no_file_in_request_returns_error(client):
    """No file in request → success: false."""
    resp = client.post(
        "/process",
        data={"mode": "summarize"},
        content_type="multipart/form-data",
    )
    body = json.loads(resp.data)
    assert body["success"] is False
    assert body["error"] != ""


def test_valid_pdf_no_mode_returns_error(client):
    """Valid PDF but no mode → success: false."""
    file_tuple = _make_pdf_upload()
    resp = _post_process(client, file_tuple=file_tuple, mode="")
    body = json.loads(resp.data)
    assert body["success"] is False
    assert body["error"] != ""


def test_ask_mode_empty_question_returns_error(client):
    """Ask mode with empty question → success: false."""
    file_tuple = _make_pdf_upload()
    resp = _post_process(client, file_tuple=file_tuple, mode="ask", question="")
    body = json.loads(resp.data)
    assert body["success"] is False
    assert "question" in body["error"].lower()


def test_valid_pdf_summarize_with_mocked_bedrock(client):
    """Valid PDF + summarize mode + mocked Bedrock → success: true, non-empty result."""
    small_pdf = b"%PDF-fake content"
    file_tuple = _make_pdf_upload(data=small_pdf)

    mock_text = "Mocked summary output from Bedrock."

    with patch("app.extract_text_from_pdf", return_value="Some extracted text"), \
         patch("app.invoke_bedrock", return_value=mock_text):
        resp = _post_process(client, file_tuple=file_tuple, mode="summarize")

    body = json.loads(resp.data)
    assert body["success"] is True
    assert body["result"] == mock_text
