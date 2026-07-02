"""Tests for invoke_bedrock: property tests (Properties 5 & 6) and unit tests for error handling."""

import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import botocore.exceptions
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app import invoke_bedrock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_client(output_text: str) -> MagicMock:
    """Return a mock boto3 bedrock-runtime client that returns output_text."""
    mock_client = MagicMock()
    response_dict = {
        "output": {
            "message": {
                "content": [{"text": output_text}]
            }
        }
    }
    mock_response = {"body": BytesIO(json.dumps(response_dict).encode())}
    mock_client.invoke_model.return_value = mock_response
    return mock_client


# ---------------------------------------------------------------------------
# Property 5: Bedrock request body correctness
# Feature: smart-study-buddy, Property 5: Bedrock request body correctness
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(st.text(min_size=1))
def test_bedrock_request_body_correctness(prompt: str):
    # Feature: smart-study-buddy, Property 5: Bedrock request body correctness
    # Validates: Requirements 4.1, 4.2
    mock_client = _make_mock_client("some output")

    with patch("app.boto3.client", return_value=mock_client):
        invoke_bedrock(prompt)

    mock_client.invoke_model.assert_called_once()
    call_kwargs = mock_client.invoke_model.call_args

    # Verify model ID
    assert call_kwargs.kwargs.get("modelId") == "amazon.nova-lite-v1:0"

    # Decode and verify body
    body = json.loads(call_kwargs.kwargs.get("body", "{}"))
    # Prompt must appear as the user message text
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"][0]["text"] == prompt

    config = body["inferenceConfig"]
    assert config["maxTokens"] == 2048
    assert config["temperature"] == 0.7
    assert config["topP"] == 0.9


# ---------------------------------------------------------------------------
# Property 6: Response parsing round trip
# Feature: smart-study-buddy, Property 6: Response parsing round trip
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(st.text())
def test_response_parsing_round_trip(output_text: str):
    # Feature: smart-study-buddy, Property 6: Response parsing round trip
    # Validates: Requirements 4.3
    mock_client = _make_mock_client(output_text)

    with patch("app.boto3.client", return_value=mock_client):
        result = invoke_bedrock("any prompt")

    assert result == output_text


# ---------------------------------------------------------------------------
# Unit tests: error handling (task 4.4)
# ---------------------------------------------------------------------------

def test_client_error_raises_runtime_error():
    """ClientError from invoke_model raises RuntimeError with a user-friendly message."""
    mock_client = MagicMock()
    error_response = {"Error": {"Code": "ValidationException", "Message": "bad request"}}
    mock_client.invoke_model.side_effect = botocore.exceptions.ClientError(
        error_response, "InvokeModel"
    )

    with patch("app.boto3.client", return_value=mock_client):
        with pytest.raises(RuntimeError) as exc_info:
            invoke_bedrock("test prompt")

    # Message should be user-friendly, not a raw boto3 traceback token
    assert "AI service error" in str(exc_info.value)
    assert "ClientError" not in str(exc_info.value)


def test_no_credentials_error_raises_runtime_error():
    """NoCredentialsError raises RuntimeError mentioning credentials."""
    mock_client = MagicMock()
    mock_client.invoke_model.side_effect = botocore.exceptions.NoCredentialsError()

    with patch("app.boto3.client", return_value=mock_client):
        with pytest.raises(RuntimeError) as exc_info:
            invoke_bedrock("test prompt")

    assert "credentials" in str(exc_info.value).lower()
