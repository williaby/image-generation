"""Shared pytest fixtures for the image-generation test suite.

``scripts.generate_image`` no longer configures structlog at import time (that
import-time side effect was removed so importing the module is side-effect
free). The production entry point ``main()`` configures logging before doing
any work; tests, however, call internal functions directly without going
through ``main()``. This autouse fixture reproduces that configuration for
every test so that ``capsys``-based assertions on stderr log output keep
working, mirroring the previous import-time behavior.
"""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def configure_logging():
    """Configure structlog for stderr capture before each test.

    Imported lazily inside the fixture to honor the deferred-import pattern
    documented in tests/CLAUDE.md (avoid importing the module at collection
    time).
    """
    from scripts.generate_image import _configure_logging

    _configure_logging(verbose=False)


@pytest.fixture(autouse=True)
def isolate_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Point the module-level .env path at a nonexistent file for every test.

    The developer's real repo-root .env must never leak into the suite. A real
    GEMINI_API_KEY/TOPAZ_API_KEY there would otherwise override test
    expectations, and a real .env read inside a ``patch("builtins.open")`` block
    consumes a mocked side effect and feeds dotenv bytes (a string-vs-bytes
    TypeError). Tests that need a .env override this by repointing
    ``scripts.generate_image._ENV_FILE`` at their own temp file.

    Effective only because ``get_settings()`` resolves ``_ENV_FILE`` at call
    time; see that function in scripts/generate_image.py.
    """
    from scripts import generate_image as mod

    monkeypatch.setattr(mod, "_ENV_FILE", tmp_path / ".env-does-not-exist")
