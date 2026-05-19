"""Property-based fuzz tests for scripts/generate_image.py.

Uses Hypothesis @given decorators so that OpenSSF Scorecard's Fuzzing check
recognises this repository as having fuzzing integration. Tests target pure
helper functions in the CLI script; no network calls, no file I/O, no API
keys are required.

Run deterministically in CI with:
    pytest tests/test_fuzz.py --hypothesis-seed=0 -v
"""

from __future__ import annotations

import argparse

from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------

# Text strategy that avoids surrogate codepoints so that the strings can be
# encoded as valid UTF-8 (surrogate pairs are not valid in UTF-8 and cause
# codec errors in paths that eventually write to files or log handlers).
_text_no_surrogates = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    min_size=0,
    max_size=512,
)

# Arbitrary byte sequences up to 16 bytes (magic-byte detection only reads
# the first 12 bytes; larger inputs add no new coverage and slow the suite).
_bytes_strategy = st.binary(min_size=0, max_size=16)


# ---------------------------------------------------------------------------
# Fuzz target 1: detect_image_format(data: bytes) -> str
# ---------------------------------------------------------------------------


class TestFuzzDetectImageFormat:
    """detect_image_format() must never raise on arbitrary byte input."""

    @given(_bytes_strategy)
    @settings(max_examples=50, deadline=None)
    def test_never_raises_on_arbitrary_bytes(self, data: bytes) -> None:
        """Arbitrary byte sequences return a dot-prefixed extension string."""
        from scripts.generate_image import detect_image_format

        result = detect_image_format(data)
        assert isinstance(result, str)
        assert result.startswith(".")

    @given(_bytes_strategy)
    @settings(max_examples=50, deadline=None)
    def test_result_is_known_extension(self, data: bytes) -> None:
        """Return value is always one of the four known extensions."""
        from scripts.generate_image import detect_image_format

        result = detect_image_format(data)
        assert result in {".png", ".jpg", ".gif", ".webp"}


# ---------------------------------------------------------------------------
# Fuzz target 2: get_extension_for_mime(mime_type: str) -> str
# ---------------------------------------------------------------------------


class TestFuzzGetExtensionForMime:
    """get_extension_for_mime() must never raise on arbitrary MIME strings."""

    @given(_text_no_surrogates)
    @settings(max_examples=50, deadline=None)
    def test_never_raises_on_arbitrary_mime_string(self, mime_type: str) -> None:
        """Arbitrary MIME strings return a dot-prefixed extension string."""
        from scripts.generate_image import get_extension_for_mime

        result = get_extension_for_mime(mime_type)
        assert isinstance(result, str)
        assert result.startswith(".")

    @given(_text_no_surrogates)
    @settings(max_examples=50, deadline=None)
    def test_result_is_known_extension(self, mime_type: str) -> None:
        """Return value is always one of the four known extensions."""
        from scripts.generate_image import get_extension_for_mime

        result = get_extension_for_mime(mime_type)
        assert result in {".png", ".jpg", ".gif", ".webp"}


# ---------------------------------------------------------------------------
# Fuzz target 3: CLI argument parser (prompt handling surface)
# ---------------------------------------------------------------------------


class TestFuzzArgumentParser:
    """The CLI argparser must handle arbitrary prompt strings without crashing.

    ``argparse`` raises ``SystemExit`` for genuinely invalid arguments (e.g.
    unrecognised flags). Arbitrary positional prompt strings are always valid
    and must never cause an unhandled exception other than ``SystemExit``.
    """

    @given(_text_no_surrogates)
    @settings(max_examples=50, deadline=None)
    def test_prompt_positional_parsed_or_system_exit(self, prompt: str) -> None:
        """Arbitrary prompt strings are accepted by the parser or cause SystemExit."""
        from scripts.generate_image import (
            ASPECT_RATIOS,
            DEFAULT_MODEL,
            IMAGE_SIZES,
            MODELS,
            THINKING_LEVELS,
        )

        # Reconstruct the prompt-handling surface of the _run() parser.
        # ``add_help=False`` prevents argparse from writing to stderr during
        # fuzzing runs that generate '-h' or '--help' as prompt text.
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("prompt", nargs="?")
        parser.add_argument(
            "-m", "--model", choices=list(MODELS.keys()), default=DEFAULT_MODEL
        )
        parser.add_argument("--aspect", choices=ASPECT_RATIOS)
        parser.add_argument("--size", choices=IMAGE_SIZES)
        parser.add_argument("--thinking", choices=THINKING_LEVELS)
        parser.add_argument("--search", action="store_true")
        parser.add_argument("--verbose", "-v", action="store_true")
        parser.add_argument("--list-models", action="store_true")

        try:
            args = parser.parse_args([prompt] if prompt else [])
            # When the parser succeeds the prompt attribute is the string or None.
            assert args.prompt == (prompt if prompt else None) or args.prompt is None
        except SystemExit:
            # argparse raises SystemExit for unrecognised options that start
            # with '-'. This is the documented argparse error path; not a bug.
            pass
