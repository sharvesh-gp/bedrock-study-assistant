"""Property and unit tests for build_prompt."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypothesis import given, settings, strategies as st
import pytest

from app import build_prompt

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Non-empty text: at least one non-whitespace character
non_empty_text = st.text(min_size=1).filter(lambda s: s.strip())

VALID_MODES = ["summarize", "quiz", "ask"]

mode_strategy = st.sampled_from(VALID_MODES)


# ---------------------------------------------------------------------------
# Property 4: Mode-specific prompt construction
# Feature: smart-study-buddy, Property 4: Mode-specific prompt construction
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(text=non_empty_text, mode=mode_strategy, question=non_empty_text)
def test_property4_mode_specific_prompt_construction(text, mode, question):
    """For any extracted text and any valid study mode, build_prompt returns a string
    that contains the extracted text and mode-specific instruction keywords.

    Validates: Requirements 3.2, 3.3, 3.5
    """
    # Feature: smart-study-buddy, Property 4: Mode-specific prompt construction
    prompt = build_prompt(mode, text, question=question)

    # The extracted text must always appear in the prompt
    assert text in prompt, f"Extracted text not found in prompt for mode={mode!r}"

    if mode == "summarize":
        lower = prompt.lower()
        assert "summary" in lower or "summarize" in lower, (
            f"summarize mode prompt missing 'summary'/'summarize' keyword"
        )
    elif mode == "quiz":
        lower = prompt.lower()
        assert "5" in prompt, "quiz mode prompt missing '5'"
        assert "multiple" in lower or "multiple-choice" in lower or "multiple choice" in lower, (
            "quiz mode prompt missing 'multiple choice' keyword"
        )
    elif mode == "ask":
        assert question in prompt, (
            f"ask mode prompt missing the user question"
        )
