"""Shared pytest fixtures for the image-generation test suite.

``scripts.generate_image`` no longer configures structlog at import time (that
import-time side effect was removed so importing the module is side-effect
free). The production entry point ``main()`` configures logging before doing
any work; tests, however, call internal functions directly without going
through ``main()``. This autouse fixture reproduces that configuration for
every test so that ``capsys``-based assertions on stderr log output keep
working, mirroring the previous import-time behavior.
"""

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
