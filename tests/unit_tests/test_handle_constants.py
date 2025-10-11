"""Tests for documentation constant substitution helpers."""

import logging
from pathlib import Path

import pytest

from pipeline.preprocessors import constants_map
from pipeline.preprocessors.handle_constants import replace_constants
from pipeline.preprocessors.markdown_preprocessor import preprocess_markdown


def test_replace_constants_substitutes_known_constant(
    monkeypatch,
) -> None:
    """Verify that known constants are replaced with mapped values."""
    monkeypatch.setitem(constants_map.CONSTANTS_MAP, "example-constant", "example-value")

    result = replace_constants("Use $[example-constant] here.", "test.mdx")

    assert result == "Use example-value here."


def test_replace_constants_logs_when_constant_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Ensure missing constants are left untouched and logged."""
    caplog.set_level(logging.INFO, logger="pipeline.preprocessors.handle_constants")

    result = replace_constants("Unknown $[missing-constant] value.", "missing.mdx")

    assert result == "Unknown $[missing-constant] value."
    assert "missing.mdx: Constant 'missing-constant' not found" in caplog.text


def test_replace_constants_respects_escaped_tokens(monkeypatch) -> None:
    """Ensure escaped constant tokens render literally without substitution."""
    monkeypatch.setitem(constants_map.CONSTANTS_MAP, "example", "value")

    result = replace_constants(r"Literal \$[example] token.", "literal.mdx")

    assert result == "Literal $[example] token."


def test_preprocess_markdown_applies_constant_substitution(monkeypatch) -> None:
    """Integration smoke test that preprocess_markdown applies substitutions."""
    monkeypatch.setitem(constants_map.CONSTANTS_MAP, "integration-constant", "final-value")

    result = preprocess_markdown(
        "Result: $[integration-constant]",
        Path("integration.mdx"),
        target_language="python",
    )

    assert result == "Result: final-value"
