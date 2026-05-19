# GitHub Copilot Review Instructions

Focus reviews on:
- Business logic correctness and edge cases (especially CLI argument validation, prompt sanitization, file I/O paths in scripts/generate_image.py).
- Error handling: bare `except Exception` is a code smell; flag silent fallbacks.
- Concurrency: this is a sync CLI today, but flag any future async/threading code for race conditions.
- Security: API key handling, prompt injection in user inputs, environment variable loading order, untrusted file paths.
- Test coverage: any new public function in scripts/ or src/ should have unit tests; flag missing assertions.

Skip:
- Stylistic nits already enforced by ruff or basedpyright.
- Documentation typos (handled by markdownlint).
- Em-dash usage (handled by the no-em-dash pre-commit hook).
