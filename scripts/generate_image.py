#!/usr/bin/env python3
"""
Generate images using Google Gemini's image generation models, with optional
Topaz Labs post-processing for professional-grade enhancement and upscaling.

Supports:
- Nano Banana (gemini-2.5-flash-image) - Legacy fast model
- Nano Banana 2 (gemini-3.1-flash-image-preview) - Default; Pro-quality at Flash speed,
  14 aspect ratios, 512/1K/2K/4K, configurable thinking, Search grounding
- Nano Banana Pro (gemini-3-pro-image-preview) - Highest quality, best text rendering
- Topaz Labs API - Post-generation enhancement: upscaling, denoising, sharpening, face enhancement

Environment Variables:
    GEMINI_API_KEY: Required. Your Google AI API key.
    TOPAZ_API_KEY:  Required for Topaz features. Get from developer.topazlabs.com.

Usage:
    # Basic text-to-image generation
    python generate_image.py "A futuristic city at sunset"

    # With output filename
    python generate_image.py "A futuristic city at sunset" -o city.png

    # Using Nano Banana Pro model (Gemini 3 Pro, highest quality)
    python generate_image.py "A futuristic city at sunset" --model pro

    # Using Nano Banana 2 (default; Pro-quality at Flash speed)
    python generate_image.py "A futuristic city at sunset" --model flash-2

    # Trade latency for quality on Nano Banana 2
    python generate_image.py "A complex diagram" --model flash-2 --thinking high

    # With reference image for editing/style
    python generate_image.py "Make this building taller" -r reference.png

    # With aspect ratio and resolution
    python generate_image.py "A landscape" --model flash-2 --aspect 21:9 --size 4K
    python generate_image.py "A landscape" --model pro --aspect 16:9 --size 4K

    # Show thinking process and save thought images (pro / flash-2)
    python generate_image.py "Complex blueprint" --save-thoughts --verbose

    # Multi-part story generation
    python generate_image.py "A 3-part story about data governance" --story-parts 3 -o story

    # List available models
    python generate_image.py --list-models

    # Enhance an existing image with Topaz (standalone post-processing)
    python generate_image.py --enhance photo.png
    python generate_image.py --enhance photo.png --topaz-model "High Fidelity V2"
    python generate_image.py --enhance photo.png --topaz-denoise 0.5 --topaz-sharpen 0.3
    python generate_image.py --enhance photo.png --topaz-face-enhance -o portrait_enhanced.png

    # Generate then immediately enhance with Topaz
    python generate_image.py "A network diagram" --topaz
    python generate_image.py "A network diagram" --topaz --topaz-model "Text Refine"

    # Use Topaz to finalize a draft (instead of re-generating via Gemini)
    python generate_image.py --finalize draft.png --topaz -o final.png
"""

import argparse
import base64
import logging
import re
import secrets
import stat
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import structlog
from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from scripts import _config as _config_module
from scripts import _images as _images_module

# Compatibility shim for the ``datetime.UTC`` constant. Python 3.11 added
# ``datetime.UTC`` as an alias for ``datetime.timezone.utc``; we target
# 3.10+, so use the older spelling and expose ``UTC`` here for the rest of
# the module to import.
UTC = timezone.utc

# Hard cap on reference / enhancement input image size. 4K PNGs are ~10 MiB;
# 100 MiB is generous and protects against pathological local inputs that would
# otherwise be fully loaded into memory. Peak memory is roughly 2.33x the input
# size because both the raw bytes and the base64-encoded copy (~1.33x expansion)
# are held simultaneously before the API call.
MAX_INPUT_IMAGE_BYTES = 100 * 1024 * 1024

# Topaz Labs download host allowlist. SSRF guard for the image fetch step in
# topaz_enhance_image: Topaz returns a signed download URL; we restrict the
# host to a known allowlist before issuing the GET. Kept at module scope so
# the security boundary is visible at audit time, not buried mid-function.
TOPAZ_DOWNLOAD_HOSTS: frozenset[str] = frozenset(
    {"api.topazlabs.com", "cdn.topazlabs.com"}
)

# ``google-genai`` is a hard dependency declared in pyproject.toml, but tests
# in CI environments sometimes install it lazily. The try/except preserves
# the legacy ``GENAI_AVAILABLE`` runtime flag (used by tests that simulate
# the "module missing" path) without forcing the runtime import to be
# conditional in the type checker's view.
genai: Any
types: Any
try:
    from google import genai as _genai_runtime
    from google.genai import types as _types_runtime

    genai = _genai_runtime
    types = _types_runtime
    _genai_imported = True
except ImportError:
    genai = None
    types = None
    _genai_imported = False

GENAI_AVAILABLE: bool = _genai_imported

# httpx is a hard dependency; this flag remains for tests that exercise the
# "library missing" guard path inside ``topaz_enhance_image``. Setting the
# flag to False at runtime via monkeypatch triggers the same code path that a
# missing import would.
HTTPX_AVAILABLE: bool = True

# Backwards-compatible alias retained so that downstream code or tests written
# against the legacy ``REQUESTS_AVAILABLE`` flag continue to work. ``requests``
# itself is no longer imported; ``httpx`` is the canonical HTTP client.
REQUESTS_AVAILABLE: bool = HTTPX_AVAILABLE


class _StderrLogger:
    """structlog logger that always writes to the current ``sys.stderr``.

    ``structlog.PrintLogger`` and ``WriteLogger`` capture their file handle at
    construction. Pytest's ``capsys`` replaces ``sys.stderr`` after import,
    which would leave a captured handle pointing at the original (unwatched)
    stream. Resolving the stream at call time is the simplest way to keep
    structured logging without breaking ``capsys`` based assertions.

    ``exception`` is implemented separately from ``msg`` so that calling
    ``log.exception(...)`` outside structlog's processor chain still emits a
    traceback. Inside structlog's processor chain, ConsoleRenderer renders
    ``exc_info`` into the message string before this method runs, so the
    ``traceback.print_exc()`` fallback below is a no-op in that path; it
    becomes load-bearing only when callers bypass structlog (uncommon but
    not impossible during teardown / monkeypatch scenarios).
    """

    def msg(self, message: str) -> None:
        print(message, file=sys.stderr)

    def exception(self, message: str) -> None:
        print(message, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    log = msg
    debug = msg
    info = msg
    warning = msg
    warn = msg
    error = msg
    err = msg
    critical = msg
    fatal = msg


class _StderrLoggerFactory:
    """Factory returning the shared :class:`_StderrLogger` instance."""

    _logger = _StderrLogger()

    def __call__(self, *args: object, **kwargs: object) -> _StderrLogger:
        return self._logger


def _configure_logging(verbose: bool = False) -> None:
    """Configure structlog for human-readable stderr output.

    Rendered with ``ConsoleRenderer`` (colors off so tests parsing
    ``capsys.readouterr().err`` see plain text). The stream is resolved at
    call time via :class:`_StderrLogger`, so test harnesses that redirect
    ``sys.stderr`` after import still observe log output.
    """
    level = logging.DEBUG if verbose else logging.INFO
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=_StderrLoggerFactory(),
        cache_logger_on_first_use=False,
    )


# #CRITICAL: Module-import side effect. ``structlog.configure(...)`` mutates
# global processor state, so any test that imports this module observes the
# configured pipeline regardless of fixture isolation. The deferred-import
# pattern in tests/CLAUDE.md helps but does not eliminate this -- once a test
# body imports ``scripts.generate_image``, all subsequent ``structlog.get_logger``
# calls in the test process use this config.
# #VERIFY: tests/test_generate_image.py exercises the _configure_logging path
# at least once (currently via _StderrLogger capsys interaction); add a direct
# assertion that re-calling _configure_logging is idempotent.
# ``_run`` re-configures after parsing ``--verbose``.
_configure_logging(verbose=False)
log = structlog.get_logger(__name__)


# Model configurations
# Each entry carries its own aspect_ratios / image_sizes because the three
# Gemini image models accept different sets. Heterogeneous values are typed
# as ``Any`` so the static checker does not treat ``MODELS[key]["foo"]`` as a
# concrete union; per-call site checks (e.g. ``if not supports_image_config``)
# operate on the values directly.
MODELS: dict[str, dict[str, Any]] = {
    "flash": {
        "id": "gemini-2.5-flash-image",
        "name": "Nano Banana (Gemini 2.5 Flash)",
        "description": "Legacy fast image generation model (no aspect/size control)",
        "supports_image_config": False,
        "supports_thinking_config": False,
        "aspect_ratios": [],
        "image_sizes": [],
    },
    "flash-2": {
        "id": "gemini-3.1-flash-image-preview",
        "name": "Nano Banana 2 (Gemini 3.1 Flash Image)",
        "description": "Pro-quality reasoning at Flash speed; 512/1K/2K/4K, 14 aspect ratios, configurable thinking, Search grounding",
        "supports_image_config": True,
        "supports_thinking_config": True,
        "aspect_ratios": [
            "1:1",
            "1:4",
            "1:8",
            "2:3",
            "3:2",
            "3:4",
            "4:1",
            "4:3",
            "4:5",
            "5:4",
            "8:1",
            "9:16",
            "16:9",
            "21:9",
        ],
        "image_sizes": ["512", "1K", "2K", "4K"],
    },
    "pro": {
        "id": "gemini-3-pro-image-preview",
        "name": "Nano Banana Pro (Gemini 3 Pro)",
        "description": "Highest quality, best text rendering, Google Search grounding, thinking mode",
        "supports_image_config": True,
        "supports_thinking_config": False,
        "aspect_ratios": ["1:1", "3:4", "4:3", "9:16", "16:9"],
        "image_sizes": ["1K", "2K", "4K"],
    },
}

DEFAULT_MODEL = "flash-2"  # Google's new default; Pro-quality at Flash speed.

# Union of valid values across all models (used for argparse choices). Derived
# from MODELS so a future model edit cannot silently desync the CLI from the
# actual capabilities. Per-model validation still happens inside
# generate_image() against MODELS[key] for accurate error messages.
# dict.fromkeys preserves first-seen order while deduplicating.
ASPECT_RATIOS = list(
    dict.fromkeys(r for m in MODELS.values() for r in m.get("aspect_ratios", []))
)
IMAGE_SIZES = list(
    dict.fromkeys(s for m in MODELS.values() for s in m.get("image_sizes", []))
)

# Valid thinking levels for models that support thinking_config.
THINKING_LEVELS = ["minimal", "high"]

# Topaz Labs API base URL
TOPAZ_BASE_URL = "https://api.topazlabs.com/image/v1"

# Topaz endpoint paths, named once so the registry below and the dispatch in
# list_topaz_models() share a single literal.
_ENDPOINT_ENHANCE = "enhance/async"
_ENDPOINT_ENHANCE_GEN = "enhance-gen/async"

# Topaz model registry: "enhance" -> /enhance/async; "enhance-gen" -> /enhance-gen/async
# Generative models (Wonder, Bloom) cost ~6-12x more credits than precision models.
TOPAZ_MODELS = {
    # Gigapixel precision upscaling (24 MP per credit)
    "Standard V2": {
        "endpoint": _ENDPOINT_ENHANCE,
        "description": "Precision upscaling, best for most images",
    },
    "High Fidelity V2": {
        "endpoint": _ENDPOINT_ENHANCE,
        "description": "Highest quality, preserves fine detail",
    },
    "Low Resolution V2": {
        "endpoint": _ENDPOINT_ENHANCE,
        "description": "Optimized for very low-resolution sources",
    },
    "CGI": {
        "endpoint": _ENDPOINT_ENHANCE,
        "description": "Optimized for CGI and rendered imagery",
    },
    "Text Refine": {
        "endpoint": _ENDPOINT_ENHANCE,
        "description": "Preserves and sharpens text in diagrams",
    },
    "Detail Faces": {
        "endpoint": _ENDPOINT_ENHANCE,
        "description": "Enhances facial clarity",
    },
    "Recover Faces": {
        "endpoint": _ENDPOINT_ENHANCE,
        "description": "Restores damaged or degraded faces",
    },
    "Transparency Upscale": {
        "endpoint": _ENDPOINT_ENHANCE,
        "description": "Upscales images with alpha transparency",
    },
    # Generative upscaling (4 MP per credit; significantly more expensive)
    "Wonder": {
        "endpoint": _ENDPOINT_ENHANCE_GEN,
        "description": "Generative upscaling, adds intelligent detail",
    },
    "Wonder 2": {
        "endpoint": _ENDPOINT_ENHANCE_GEN,
        "description": "Improved generative upscaling",
    },
    "Standard Max": {
        "endpoint": _ENDPOINT_ENHANCE_GEN,
        "description": "Maximum quality generative upscaling",
    },
    "Recover 3": {
        "endpoint": _ENDPOINT_ENHANCE_GEN,
        "description": "Advanced recovery with generation",
    },
    "Redefine": {
        "endpoint": _ENDPOINT_ENHANCE_GEN,
        "description": "Creative reinterpretation with upscaling",
    },
    "Bloom": {
        "endpoint": _ENDPOINT_ENHANCE_GEN,
        "description": "Creative upscaling for AI-generated art",
    },
}

DEFAULT_TOPAZ_MODEL = "Standard V2"


# Repo-root .env path used by ``Settings``. Resolved at module import; the
# pydantic-settings loader reads the file lazily when ``Settings()`` is
# instantiated, not at import time.
_ENV_FILE = Path(__file__).parent.parent / ".env"


class Settings(BaseSettings):
    """Project settings loaded from environment variables and ``.env``.

    Environment variables always take precedence over values in the .env file.
    Both keys are optional at construction time so that tests, CLI subcommands
    that do not need either key (``--list-models``), and standalone Topaz
    enhancement (which needs only ``TOPAZ_API_KEY``) work without forcing the
    other variable to be present. The CLI entry-point still raises
    ``ConfigError`` via ``get_api_key()`` when the Gemini key is required but
    unset, preserving the historical exit-on-missing-key UX.
    """

    GEMINI_API_KEY: str | None = None
    TOPAZ_API_KEY: str | None = None

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )


def get_settings() -> Settings:
    """Return a fresh ``Settings`` instance.

    A fresh instance is returned on every call so that tests that mutate the
    process environment with ``monkeypatch.setenv`` / ``monkeypatch.delenv``
    observe their changes without having to invalidate a cache.

    A malformed or unreadable ``.env`` file raises ``ConfigError`` rather than
    propagating raw ``pydantic.ValidationError`` / ``OSError``, so the user
    sees a single typed error class instead of an "Unexpected error" blob.
    The legacy behavior (silently fall back to environment variables when the
    ``.env`` file is unreadable) is preserved for ``OSError`` to avoid breaking
    users who set keys via the process environment but happen to have an
    unreadable stub file.
    """
    try:
        return Settings()
    except OSError as exc:
        log.warning(
            f"Could not read .env file at {_ENV_FILE}: {exc}; "
            "falling back to process environment variables only."
        )
        return Settings(_env_file=None)  # type: ignore[call-arg]
    except UnicodeDecodeError as exc:
        # python-dotenv reads the .env stream as UTF-8 before pydantic ever
        # sees it; non-UTF-8 bytes raise here, outside ValidationError. Map
        # to ConfigError so the user sees a single typed error class.
        raise ConfigError(
            f"Cannot decode .env file at {_ENV_FILE} as UTF-8 "
            f"(byte {exc.start}: {exc.reason}). "
            "Re-save the file as UTF-8 or delete it to use process env "
            "variables only."
        ) from exc
    except ValidationError as exc:
        raise ConfigError(
            f"Invalid configuration in {_ENV_FILE}: {exc}. "
            "Check that the file is valid `KEY=value` lines and that values "
            "match the expected types in the Settings model."
        ) from exc


# ---------------------------------------------------------------------------
# Typed exception hierarchy
# ---------------------------------------------------------------------------


class AppError(Exception):
    """Base for all expected, user-visible failures in this CLI.

    ``main()`` catches ``AppError`` and produces a clean stderr message + an
    exit code of 1, instead of a Python traceback. Any subclass implies the
    failure is part of normal operation (missing config, upstream API error,
    bad file I/O); programmer bugs should raise ordinary exceptions and
    surface as tracebacks via the outermost ``except Exception`` safety net.
    """


class ConfigError(AppError):
    """Configuration is missing or malformed (e.g. unset GEMINI_API_KEY)."""


class GeminiAPIError(AppError):
    """The Gemini API returned an error or unparseable response."""


class TopazAPIError(AppError):
    """The Topaz Labs API returned an error or unparseable response."""


class FileIOError(AppError):
    """An expected file write or read failed (e.g. output directory unwritable)."""


# Re-export contract for consumers and tests: these names in
# scripts.generate_image resolve to the same objects from scripts._config
# and scripts._images.
MAX_INPUT_IMAGE_BYTES = _config_module.MAX_INPUT_IMAGE_BYTES  # pyright: ignore[reportConstantRedefinition]
TOPAZ_DOWNLOAD_HOSTS = _config_module.TOPAZ_DOWNLOAD_HOSTS  # pyright: ignore[reportConstantRedefinition]
MODELS = _config_module.MODELS  # pyright: ignore[reportConstantRedefinition]
DEFAULT_MODEL = _config_module.DEFAULT_MODEL  # pyright: ignore[reportConstantRedefinition]
ASPECT_RATIOS = _config_module.ASPECT_RATIOS  # pyright: ignore[reportConstantRedefinition]
IMAGE_SIZES = _config_module.IMAGE_SIZES  # pyright: ignore[reportConstantRedefinition]
THINKING_LEVELS = _config_module.THINKING_LEVELS  # pyright: ignore[reportConstantRedefinition]
TOPAZ_BASE_URL = _config_module.TOPAZ_BASE_URL  # pyright: ignore[reportConstantRedefinition]
TOPAZ_MODELS = _config_module.TOPAZ_MODELS  # pyright: ignore[reportConstantRedefinition]
DEFAULT_TOPAZ_MODEL = _config_module.DEFAULT_TOPAZ_MODEL  # pyright: ignore[reportConstantRedefinition]
AppError = _config_module.AppError  # pyright: ignore[reportAssignmentType]
ConfigError = _config_module.ConfigError  # noqa: F811  # pyright: ignore[reportAssignmentType]
GeminiAPIError = _config_module.GeminiAPIError  # noqa: F811  # pyright: ignore[reportAssignmentType]
TopazAPIError = _config_module.TopazAPIError  # noqa: F811  # pyright: ignore[reportAssignmentType]
FileIOError = _config_module.FileIOError  # noqa: F811  # pyright: ignore[reportAssignmentType]


def _load_api_key(env_var: str) -> str | None:
    """Load an API key from the environment or the repo-root .env file.

    Thin wrapper around :class:`Settings` retained for backwards compatibility
    with callers and tests that patch ``scripts.generate_image._load_api_key``.
    Only ``GEMINI_API_KEY`` and ``TOPAZ_API_KEY`` are recognised; other names
    return ``None``.
    """
    settings = get_settings()
    value = getattr(settings, env_var, None)
    return value or None


def get_api_key() -> str:
    """Get the Gemini API key from environment or .env.

    Raises ``ConfigError`` if the key is missing; ``main()`` catches this and
    exits with code 1 after printing the message to stderr. The RAD
    ``#CRITICAL`` tag in CLAUDE.md asserts this exit-on-missing-key behavior
    is preserved.
    """
    api_key = _load_api_key("GEMINI_API_KEY")
    if not api_key:
        raise ConfigError(
            "GEMINI_API_KEY environment variable not set.\n"
            "Set it with: export GEMINI_API_KEY='your-api-key'\n"
            "Or create a .env file in the repository root."
        )
    return api_key


def get_topaz_api_key() -> str | None:
    """Get the Topaz Labs API key from environment or .env file."""
    api_key = _load_api_key("TOPAZ_API_KEY")
    if not api_key:
        log.error("Error: TOPAZ_API_KEY not set.")
        log.error("Set it with: export TOPAZ_API_KEY='your-api-key'")
        log.error("Or add it to the .env file in the repository root.")
        log.error("Get a key at: https://developer.topazlabs.com")
    return api_key


def _validate_topaz_input_file(input_path: Path) -> None:
    """Validate the Topaz input file exists, is regular, and is within size limits."""
    # stat() also covers the "missing file" case via FileNotFoundError,
    # which is a subclass of OSError. The explicit exists() check was
    # redundant and made the FileNotFoundError branch unreachable.
    # Follow symlinks so a symlink-to-regular-file is still accepted, but
    # reject character devices (/dev/zero), FIFOs, and other non-regular
    # files where st_size is meaningless and would bypass the size cap.
    try:
        resolved_stat = input_path.resolve().stat()
    except FileNotFoundError as exc:
        raise FileIOError(f"Input image not found: {input_path}") from exc
    except OSError as exc:
        raise FileIOError(f"Cannot stat input image {input_path}: {exc}") from exc
    if not stat.S_ISREG(resolved_stat.st_mode):
        raise FileIOError(f"Input image {input_path} is not a regular file.")
    input_size = resolved_stat.st_size
    if input_size > MAX_INPUT_IMAGE_BYTES:
        raise FileIOError(
            f"Input image {input_path} is {input_size} bytes; "
            f"exceeds limit of {MAX_INPUT_IMAGE_BYTES} bytes."
        )


def _validate_topaz_params(
    model: str,
    output_format: str,
    sharpen: float | None,
    denoise: float | None,
    face_enhancement: bool,
    face_enhancement_strength: float | None,
) -> dict:
    """Validate Topaz model/format/strength arguments; return the model config."""
    model_config = TOPAZ_MODELS.get(model)
    if model_config is None:
        raise ConfigError(
            f"Unknown Topaz model '{model}'. Available: {', '.join(TOPAZ_MODELS)}"
        )

    valid_formats = ("png", "jpg", "jpeg", "webp")
    if output_format not in valid_formats:
        raise ConfigError(
            f"output_format must be one of {valid_formats}, got '{output_format}'"
        )

    def _check_strength(name: str, value: float | None) -> None:
        if value is not None and not (0.0 <= value <= 1.0):
            raise ConfigError(f"{name} must be between 0.0 and 1.0, got {value}")

    _check_strength("sharpen", sharpen)
    _check_strength("denoise", denoise)
    _check_strength("face_enhancement_strength", face_enhancement_strength)

    if face_enhancement_strength is not None and not face_enhancement:
        raise ConfigError(
            "--topaz-face-strength requires --topaz-face-enhance to be set."
        )
    return model_config


def _build_topaz_form_data(
    model: str,
    output_format: str,
    sharpen: float | None,
    denoise: float | None,
    face_enhancement: bool,
    face_enhancement_strength: float | None,
) -> dict:
    """Build the multipart form fields for a Topaz enhancement request."""
    data: dict = {"model": model, "output_format": output_format}
    if sharpen is not None:
        data["sharpen"] = sharpen
    if denoise is not None:
        data["denoise"] = denoise
    if face_enhancement:
        data["face_enhancement"] = "true"
        if face_enhancement_strength is not None:
            data["face_enhancement_strength"] = face_enhancement_strength
    return data


# Topaz request tuning. Defined ahead of the functions that use them (the submit /
# poll / download helpers below) so the whole retry-and-timeout envelope reads in
# one place.
_TOPAZ_POLL_ITERATIONS = 25
_TOPAZ_POLL_INITIAL_WAIT = 2.0
_TOPAZ_POLL_429_MULTIPLIER = 2.0
_TOPAZ_POLL_429_MAX_WAIT = 30.0
_TOPAZ_POLL_DEFAULT_MULTIPLIER = 1.5
_TOPAZ_POLL_DEFAULT_MAX_WAIT = 15.0

# Per-request HTTP timeouts (seconds). Submit uploads the source image so it gets
# a longer budget than the lightweight status/download-URL calls; the final image
# download is the largest transfer and gets the longest.
_TOPAZ_SUBMIT_TIMEOUT = 30
_TOPAZ_STATUS_TIMEOUT = 15
_TOPAZ_DOWNLOAD_URL_TIMEOUT = 15
_TOPAZ_IMAGE_DOWNLOAD_TIMEOUT = 120

# Max characters of an upstream response body to echo into an error/log message.
# Bounds log size while keeping enough of the payload to debug a bad response.
_LOG_BODY_TRUNCATION = 200


def _topaz_submit_job(
    endpoint_url: str,
    headers: dict,
    data: dict,
    input_path: Path,
    verbose: bool,
) -> str:
    """Submit the async Topaz job and return its process_id."""
    # Submit async job. Split into two try blocks:
    #   1. Network + file-read errors (httpx.HTTPError covers timeouts, HTTP
    #      status errors, transport failures; OSError covers input file read
    #      issues).
    #   2. JSON parse errors (ValueError from `resp.json()` on a 200 response
    #      with a non-JSON body, the c780c57 fix surface).
    # Splitting means `resp` is provably bound at the JSON-parse site (no
    # exception escaped the first block), removing the need for `assert resp
    # is not None` narrowing -- which bandit flags as B101 in production code.
    # Other exceptions (KeyboardInterrupt, MemoryError, programmer bugs)
    # propagate to main()'s outer handler.
    try:
        with open(input_path, "rb") as f:
            resp = httpx.post(
                endpoint_url,
                headers=headers,
                data=data,
                files={"image": f},
                timeout=_TOPAZ_SUBMIT_TIMEOUT,
            )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise TopazAPIError(f"Error submitting Topaz job (transport): {exc}") from exc
    except OSError as exc:
        raise FileIOError(f"Error reading Topaz input file: {exc}") from exc

    try:
        submit_payload = resp.json()
    except ValueError as exc:
        raise TopazAPIError(
            f"Topaz submit returned non-JSON body "
            f"(status={resp.status_code}): {exc}; "
            f"body[:{_LOG_BODY_TRUNCATION}]={resp.text[:_LOG_BODY_TRUNCATION]!r}"
        ) from exc

    if not isinstance(submit_payload, dict):
        # Defend against a top-level JSON value that is not a JSON object
        # (array, scalar, `null`); `.get(...)` would raise `AttributeError`
        # which is not in any catch clause above.
        raise TopazAPIError(
            f"Topaz submit response was not a JSON object "
            f"(got {type(submit_payload).__name__}): "
            f"body[:{_LOG_BODY_TRUNCATION}]={resp.text[:_LOG_BODY_TRUNCATION]!r}"
        )

    process_id = submit_payload.get("process_id")
    if not process_id:
        raise TopazAPIError(
            f"Topaz API returned unexpected response "
            f"(missing process_id, keys={sorted(submit_payload)[:10]}): "
            f"{resp.text[:_LOG_BODY_TRUNCATION]}"
        )
    if verbose:
        print(f"  Job submitted: {process_id}")
    return process_id


def _topaz_poll_job(process_id: str, headers: dict, verbose: bool) -> None:
    """Poll the Topaz job until it completes, fails, or the limit is reached."""
    # #ASSUME: job completes within 25 poll iterations. Happy-path wall time is
    # ~5 minutes (1.5x backoff capped at 15 s per iteration). Under sustained
    # HTTP 429 backoff the 2x backoff caps at 30 s per iteration, so the
    # worst-case wall time is ~12 minutes before the loop falls through to
    # the "did not complete" branch.
    # #VERIFY -- Confirm polling cap and the 30 s/15 s caps against Topaz
    # SLA docs before changing the iteration count or the backoff factors.
    # #EDGE: sustained 429 responses exhaust iterations and are reported as
    # "did not complete".
    # #VERIFY -- Test with mock that returns 429 indefinitely; confirm
    # timeout message emitted (no separate exception type today).
    wait = _TOPAZ_POLL_INITIAL_WAIT
    for _ in range(_TOPAZ_POLL_ITERATIONS):
        time.sleep(wait)
        try:
            status_resp = httpx.get(
                f"{TOPAZ_BASE_URL}/status/{process_id}",
                headers=headers,
                timeout=_TOPAZ_STATUS_TIMEOUT,
            )
            if status_resp.status_code == 429:
                wait = min(wait * _TOPAZ_POLL_429_MULTIPLIER, _TOPAZ_POLL_429_MAX_WAIT)
                continue
            status_resp.raise_for_status()
            status_payload = status_resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise TopazAPIError(
                f"Error polling Topaz status for job {process_id}: {exc}"
            ) from exc
        if not isinstance(status_payload, dict):
            raise TopazAPIError(
                f"Topaz status response for job {process_id} was "
                f"not a JSON object (got {type(status_payload).__name__})"
            )
        status = status_payload.get("status", "")
        if verbose:
            print(f"  Status: {status}")
        if status == "Completed":
            break
        if status in ("Failed", "Error"):
            raise TopazAPIError(f"Topaz job {process_id} failed (status: {status})")
        wait = min(
            wait * _TOPAZ_POLL_DEFAULT_MULTIPLIER,
            _TOPAZ_POLL_DEFAULT_MAX_WAIT,
        )
    else:
        raise TopazAPIError(
            f"Topaz job {process_id} did not complete within the polling limit."
        )


def _poll_topaz_status(
    process_id: str, headers: dict[str, str], *, verbose: bool
) -> None:
    """Compatibility wrapper exposing the Topaz polling helper."""
    _topaz_poll_job(process_id, headers, verbose)


def _topaz_get_download_url(process_id: str, headers: dict) -> str:
    """Fetch and validate (HTTPS + host allowlist) the Topaz result download URL."""
    # Get download URL -- same split-try pattern used in _topaz_submit_job:
    # network errors first, then JSON parse errors with dl_resp provably bound.
    try:
        dl_resp = httpx.get(
            f"{TOPAZ_BASE_URL}/download/{process_id}",
            headers=headers,
            timeout=_TOPAZ_DOWNLOAD_URL_TIMEOUT,
        )
        dl_resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise TopazAPIError(
            f"Error getting Topaz download URL for job {process_id} (transport): {exc}"
        ) from exc

    try:
        dl_payload = dl_resp.json()
    except ValueError as exc:
        raise TopazAPIError(
            f"Topaz download URL response was non-JSON for job "
            f"{process_id} (status={dl_resp.status_code}): {exc}; "
            f"body[:{_LOG_BODY_TRUNCATION}]={dl_resp.text[:_LOG_BODY_TRUNCATION]!r}"
        ) from exc

    if not isinstance(dl_payload, dict):
        raise TopazAPIError(
            f"Topaz download URL response for job {process_id} was "
            f"not a JSON object (got {type(dl_payload).__name__}): "
            f"body[:{_LOG_BODY_TRUNCATION}]={dl_resp.text[:_LOG_BODY_TRUNCATION]!r}"
        )

    download_url = dl_payload.get("url")
    if not download_url:
        raise TopazAPIError(f"Topaz download response missing URL for job {process_id}")

    # #ASSUME: download_url is a valid HTTPS URL served from api.topazlabs.com or its CDN.
    # #VERIFY   -- Check Topaz CDN policy if infrastructure change announced.
    _parsed = urlparse(download_url)
    if _parsed.scheme != "https" or _parsed.hostname not in TOPAZ_DOWNLOAD_HOSTS:
        raise TopazAPIError(f"Topaz returned unexpected download URL: {download_url!r}")
    return download_url


def _topaz_download_image(download_url: str, process_id: str) -> bytes:
    """Download the enhanced image bytes from the validated Topaz URL."""
    # Download the enhanced image
    try:
        img_resp = httpx.get(
            download_url,
            timeout=_TOPAZ_IMAGE_DOWNLOAD_TIMEOUT,
            follow_redirects=False,
        )
        img_resp.raise_for_status()
        image_data = img_resp.content
    except httpx.HTTPError as exc:
        raise TopazAPIError(
            f"Error downloading Topaz result for job {process_id}: {exc}"
        ) from exc

    if not image_data:
        raise TopazAPIError(
            f"Topaz download returned empty response for job {process_id}"
        )
    return image_data


def _finalize_topaz_output(
    output_path: Path | None, input_path: Path, image_data: bytes
) -> Path:
    """Resolve the output path (correcting extension) and write the image."""
    # Resolve output path
    if output_path is None:
        detected_ext = detect_image_format(image_data)
        output_path = input_path.parent / f"{input_path.stem}_topaz{detected_ext}"
    else:
        # Correct extension if needed
        detected_ext = detect_image_format(image_data)
        user_ext = output_path.suffix.lower()
        if user_ext not in (".png", ".jpg", ".jpeg", ".webp"):
            output_path = output_path.with_suffix(detected_ext)
        elif user_ext != detected_ext:
            log.warning(
                f"Warning: specified extension {user_ext!r} does not match "
                f"Topaz result {detected_ext!r}; correcting."
            )
            output_path = output_path.with_suffix(detected_ext)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(image_data)
    except OSError as exc:
        raise FileIOError(
            f"Error writing Topaz output file {output_path}: {exc}"
        ) from exc
    return output_path


def topaz_enhance_image(
    input_path: Path,
    output_path: Path | None = None,
    model: str = DEFAULT_TOPAZ_MODEL,
    output_format: str = "png",
    sharpen: float | None = None,
    denoise: float | None = None,
    face_enhancement: bool = False,
    face_enhancement_strength: float | None = None,
    verbose: bool = False,
) -> Path:
    """Enhance an image using the Topaz Labs API (async job with polling).

    Precision models (Gigapixel family): 24 MP per credit.
    Generative models (Wonder, Bloom): 2-4 MP per credit, significantly more expensive.

    Raises:
        ConfigError: invalid input arguments (unknown model, bad output
            format, strength out of range, face-strength without face flag).
        FileIOError: input file missing / not a regular file / over size cap,
            or output write failure.
        TopazAPIError: Topaz API transport, HTTP, JSON, or status failure;
            also covers SSRF allowlist rejection and missing httpx runtime.
    """
    if not HTTPX_AVAILABLE:
        raise TopazAPIError(
            "'httpx' package is required for Topaz enhancement. "
            "Install with: pip install httpx"
        )

    api_key = get_topaz_api_key()
    if not api_key:
        raise ConfigError(
            "TOPAZ_API_KEY is not set. Set it via the environment or .env file."
        )
    _validate_topaz_input_file(input_path)
    model_config = _validate_topaz_params(
        model,
        output_format,
        sharpen,
        denoise,
        face_enhancement,
        face_enhancement_strength,
    )
    # #CRITICAL: API key sent as X-API-KEY header; never log headers containing this value.
    # #VERIFY   -- Audit log handler config before enabling debug logging.
    # #ASSUME: Topaz API at TOPAZ_BASE_URL accepts multipart/form-data for all model endpoints.
    # #VERIFY   -- Check developer.topazlabs.com changelog before upgrading httpx.
    # SSRF guard for the download step uses the module-level
    # `TOPAZ_DOWNLOAD_HOSTS` allowlist.
    headers = {"X-API-KEY": api_key}
    endpoint_url = f"{TOPAZ_BASE_URL}/{model_config['endpoint']}"

    if verbose:
        print(f"Topaz enhance: {input_path.name} [{model}]")

    data = _build_topaz_form_data(
        model,
        output_format,
        sharpen,
        denoise,
        face_enhancement,
        face_enhancement_strength,
    )
    process_id = _topaz_submit_job(endpoint_url, headers, data, input_path, verbose)
    _poll_topaz_status(process_id, headers, verbose=verbose)
    download_url = _topaz_get_download_url(process_id, headers)
    image_data = _topaz_download_image(download_url, process_id)
    output_path = _finalize_topaz_output(output_path, input_path, image_data)
    print(f"Topaz result saved to: {output_path}")
    return output_path


def list_topaz_models() -> None:
    """Print available Topaz enhancement models."""
    print("Available Topaz models:\n")
    print("  Precision upscaling (24 MP/credit):")
    for name, cfg in TOPAZ_MODELS.items():
        if cfg["endpoint"] == _ENDPOINT_ENHANCE:
            print(f"    {name:<22}  {cfg['description']}")
    print("\n  Generative upscaling (2-4 MP/credit, more expensive):")
    for name, cfg in TOPAZ_MODELS.items():
        if cfg["endpoint"] == _ENDPOINT_ENHANCE_GEN:
            print(f"    {name:<22}  {cfg['description']}")
    print()


# Image-format helpers live in scripts._images (pure, side-effect free). They are
# re-exported here so both ``scripts.generate_image.detect_image_format`` and
# ``scripts._images.detect_image_format`` resolve to the same object and the test
# suite's patch targets keep working.
detect_image_format = _images_module.detect_image_format
get_extension_for_mime = _images_module.get_extension_for_mime
get_mime_for_extension = _images_module.get_mime_for_extension
EXT_TO_MIME = _images_module.EXT_TO_MIME


def load_image_as_base64(image_path: Path) -> tuple[str, str]:
    """Load an image file and return base64 data and mime type.

    Detects actual image format from file contents (magic bytes),
    not from file extension, to avoid MIME type mismatches.

    Enforces MAX_INPUT_IMAGE_BYTES to bound peak memory use when a caller
    points at an unexpectedly large file. Rejects non-regular files (FIFOs,
    character devices like /dev/zero) where st_size is meaningless and
    would bypass the size cap. Missing files raise FileNotFoundError (from
    stat) and oversize / non-regular / unreadable files raise FileIOError;
    both are caught by the call site in generate_image().
    """
    raw_data, mime_type = load_image_bytes(image_path)
    data = base64.standard_b64encode(raw_data).decode("utf-8")
    return data, mime_type


def load_image_bytes(image_path: Path) -> tuple[bytes, str]:
    """Load an image file and return raw bytes plus detected MIME type.

    Raises :class:`FileIOError` (an :class:`AppError` subclass) when the path is
    not a regular file, exceeds ``MAX_INPUT_IMAGE_BYTES``, or cannot be read, so
    callers get the CLI's typed-error flow rather than a bare ``ValueError`` /
    ``OSError``. ``FileNotFoundError`` from ``stat`` propagates unchanged for the
    missing-file contract exercised by the test suite.
    """
    # #EDGE: a path swap or file growth between stat() and read() could bypass
    # the size / regular-file checks. Resolve once and stat+open the *same*
    # resolved path, bound the read to MAX_INPUT_IMAGE_BYTES + 1, then re-check
    # the length -- closing the TOCTOU window and keeping peak memory bounded.
    # #VERIFY: tests/test_security_hardening.py exercises the size-cap path.
    resolved_path = image_path.resolve()
    resolved_stat = resolved_path.stat()
    if not stat.S_ISREG(resolved_stat.st_mode):
        raise FileIOError(f"Reference image {image_path} is not a regular file.")
    size = resolved_stat.st_size
    if size > MAX_INPUT_IMAGE_BYTES:
        raise FileIOError(
            f"Reference image {image_path} is {size} bytes; "
            f"exceeds limit of {MAX_INPUT_IMAGE_BYTES} bytes."
        )
    try:
        with open(resolved_path, "rb") as f:
            raw_data = f.read(MAX_INPUT_IMAGE_BYTES + 1)
    except OSError as exc:
        raise FileIOError(f"Error reading reference image {image_path}: {exc}") from exc
    if len(raw_data) > MAX_INPUT_IMAGE_BYTES:
        raise FileIOError(
            f"Reference image {image_path} exceeds limit of "
            f"{MAX_INPUT_IMAGE_BYTES} bytes (grew after validation)."
        )

    # Detect actual format from magic bytes
    detected_ext = detect_image_format(raw_data)
    mime_type = get_mime_for_extension(detected_ext)

    # Warn if extension doesn't match actual format
    suffix = image_path.suffix.lower()
    if suffix != detected_ext and suffix in EXT_TO_MIME:
        print(
            f"Warning: File {image_path.name} has extension {suffix} but contains {detected_ext} data"
        )
        print(f"  Using detected MIME type: {mime_type}")

    return raw_data, mime_type


def _build_detailed_entry(
    image_path: Path,
    model_key: str,
    model_name: str,
    timestamp: str,
    prompt: str,
    params_str: str,
    rel_path: object,
    reference_images: list[Path] | None,
    is_draft: bool,
    is_final: bool,
    purpose: str | None,
) -> str:
    """Build the detailed PROMPTS.md entry block for one generated image."""
    detailed_entry = f"""
### {image_path.name}

- **Model**: {model_name} ({MODELS[model_key]["id"]})
- **Date Generated**: {timestamp}
"""

    if reference_images:
        detailed_entry += "- **Attachments**:\n"
        for ref in reference_images:
            detailed_entry += f"  - {ref.name}\n"

    # Render the prompt as an indented code block. The detailed entry sits
    # inside a Markdown list item ('- **Prompt**:'), whose content indent is
    # column 2. Per CommonMark, an indented code block within a list item
    # requires content-indent + 4 spaces = 6 spaces total. Using only 4 here
    # would render as paragraph continuation, leaving triple-backtick fences
    # in the prompt free to escape into rendered Markdown.
    lines = prompt.splitlines() if prompt else []
    indented_prompt = (
        "\n".join("      " + line for line in lines) if lines else "      "
    )
    detailed_entry += f"""- **Prompt**:

{indented_prompt}

- **Parameters**: {params_str}
"""

    if purpose:
        detailed_entry += f"- **Purpose**: {purpose}\n"

    # Only document final images location (drafts are temporary)
    if is_final:
        detailed_entry += f"- **Location**: `{rel_path}`\n"

    if is_draft:
        detailed_entry += (
            "- **Type**: Draft (temporary, will be removed after finalization)\n"
        )
    elif is_final:
        detailed_entry += "- **Type**: Final (production ready)\n"

    detailed_entry += "\n---\n"
    return detailed_entry


def document_image_prompt(
    image_path: Path,
    prompt: str,
    model_key: str,
    aspect_ratio: str | None,
    image_size: str | None,
    reference_images: list[Path] | None = None,
    is_draft: bool = False,
    is_final: bool = False,
    purpose: str | None = None,
) -> None:
    """
    Document image generation in PROMPTS.md registry.
    Matches existing format from web UI documentation.
    """
    # Prompts file is in examples directory
    prompts_file = Path(__file__).parent.parent / "examples" / "PROMPTS.md"

    # Read existing content
    if prompts_file.exists():
        with open(prompts_file, encoding="utf-8") as f:
            content = f.read()
    else:
        content = """# AI-Generated Image Registry

This folder contains images generated by AI models. Each entry documents the model, prompt, and generation parameters.

## Image Registry

| Filename | Model | Date | Prompt | Parameters |
|----------|-------|------|--------|------------|

## Detailed Prompts

"""

    # Skip if already documented
    if f"### {image_path.name}" in content:
        return

    # Prepare data
    timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    model_name = MODELS[model_key]["name"]
    params_str = f"--aspect {aspect_ratio or '16:9'} --size {image_size or '2K'}"

    # Relative path from output root
    try:
        rel_path = image_path.relative_to(Path(__file__).parent.parent / "output")
    except ValueError:
        rel_path = image_path.name

    # Short description for table; escape '|' so a prompt containing a pipe
    # cannot break out of the Markdown table column structure, and strip
    # newlines that would terminate the row.
    raw_short = prompt[:50] + "..." if len(prompt) > 50 else prompt
    prompt_short = raw_short.replace("\\", "\\\\").replace("|", "\\|")
    prompt_short = re.sub(r"[\r\n]+", " ", prompt_short)

    # Add to registry table
    table_end = content.find("\n## Detailed Prompts")
    if table_end > 0:
        new_row = f"| {rel_path} | {model_name} | {timestamp} | {prompt_short} | {params_str} |\n"
        content = content[:table_end] + new_row + content[table_end:]

    detailed_entry = _build_detailed_entry(
        image_path=image_path,
        model_key=model_key,
        model_name=model_name,
        timestamp=timestamp,
        prompt=prompt,
        params_str=params_str,
        rel_path=rel_path,
        reference_images=reference_images,
        is_draft=is_draft,
        is_final=is_final,
        purpose=purpose,
    )

    # Append
    content += detailed_entry

    # Write
    with open(prompts_file, "w", encoding="utf-8") as f:
        f.write(content)

    if is_final:
        print("✓ Documented in PROMPTS.md")


def _build_reference_contents(
    reference_images: list[Path] | None,
) -> list[Any] | None:
    """Build the content parts from reference images; None signals a load failure."""
    contents: list[Any] = []
    if reference_images:
        for img_path in reference_images:
            if not img_path.exists():
                print(f"Warning: Reference image not found: {img_path}")
                continue

            print(f"Including reference image: {img_path}")
            try:
                img_data, mime_type = load_image_as_base64(img_path)
            except (FileIOError, OSError) as exc:
                # FileIOError covers the size cap / non-regular-file / read
                # failures raised by load_image_bytes; OSError remains as a
                # belt-and-suspenders catch for the resolve()/stat() call that
                # runs before load_image_bytes's own try block. A bad reference
                # image is non-fatal here: skip the whole request by signalling
                # None to generate_image, which stops cleanly.
                log.error(f"Error: cannot load reference image {img_path}: {exc}")
                return None
            contents.append(
                types.Part.from_bytes(
                    data=base64.standard_b64decode(img_data),
                    mime_type=mime_type,
                )
            )
    return contents


def _apply_image_config(
    model_config: dict,
    aspect_ratio: str | None,
    image_size: str | None,
    use_search: bool,
    config_kwargs: dict,
) -> None:
    """Add image_config and search grounding to config_kwargs for capable models."""
    # Add image config for models that support it (pro, flash-2)
    if model_config.get("supports_image_config"):
        model_aspects = model_config.get("aspect_ratios", [])
        model_sizes = model_config.get("image_sizes", [])
        image_config_kwargs = {}
        if aspect_ratio:
            if aspect_ratio not in model_aspects:
                print(
                    f"Warning: Aspect ratio '{aspect_ratio}' not supported by {model_config['name']}."
                    f" Valid for this model: {model_aspects}"
                )
            else:
                image_config_kwargs["aspect_ratio"] = aspect_ratio
                print(f"Aspect ratio: {aspect_ratio}")
        if image_size:
            if image_size not in model_sizes:
                print(
                    f"Warning: Image size '{image_size}' not supported by {model_config['name']}."
                    f" Valid for this model: {model_sizes}"
                )
            else:
                image_config_kwargs["image_size"] = image_size
                print(f"Image size: {image_size}")

        if image_config_kwargs:
            config_kwargs["image_config"] = types.ImageConfig(**image_config_kwargs)

        # Add Google Search grounding if requested
        if use_search:
            config_kwargs["tools"] = [{"google_search": {}}]
            print("Google Search grounding: enabled")


def _apply_thinking_config(
    model_config: dict, thinking_level: str | None, config_kwargs: dict
) -> None:
    """Add thinking_config to config_kwargs for models that expose thinking_level."""
    # Add thinking config for models that expose thinking_level (flash-2 only).
    # #ASSUME -- google-genai >=2.2.0 accepts thinking_level as a string ("minimal"
    #            or "high"); the SDK normalizes to types.ThinkingLevel.HIGH /
    #            ThinkingLevel.MINIMAL internally via field aliases. The flash-2
    #            model (gemini-3.1-flash-image-preview) is the only image model
    #            that exposes this control; pro and flash silently ignore it.
    # #VERIFY -- Re-check types.ThinkingConfig signature after any google-genai
    #            version bump: `python -c "from google.genai import types;
    #            import inspect; print(inspect.signature(types.ThinkingConfig))"`
    #            and re-check the Nano Banana 2 docs for breaking changes to
    #            thinking_level accepted values.
    if thinking_level:
        if not model_config.get("supports_thinking_config"):
            print(
                f"Warning: --thinking has no effect on {model_config['name']};"
                f" only flash-2 exposes thinking_level."
            )
        elif thinking_level not in THINKING_LEVELS:
            # #EDGE -- Unreachable from CLI (argparse rejects via choices=THINKING_LEVELS)
            #          but reachable when generate_image() is called programmatically
            #          with a bogus value. Kept as a defensive guard rather than
            #          silently passing through to the SDK.
            print(
                f"Warning: Invalid thinking level '{thinking_level}'. Valid: {THINKING_LEVELS}"
            )
        else:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=thinking_level
            )
            print(f"Thinking level: {thinking_level}")


def _build_generate_config(
    model_config: dict,
    aspect_ratio: str | None,
    image_size: str | None,
    use_search: bool,
    thinking_level: str | None,
) -> Any:
    """Assemble the GenerateContentConfig from the requested options."""
    config_kwargs: dict[str, Any] = {
        "response_modalities": ["IMAGE", "TEXT"],
    }

    # Warn when aspect/size are passed to a model that cannot honor them
    # (e.g. legacy 'flash'). Without this, argparse's union choices accept the
    # value but the API call silently drops it.
    if not model_config.get("supports_image_config") and (aspect_ratio or image_size):
        print(
            f"Warning: {model_config['name']} does not support --aspect or --size;"
            f" these flags will be ignored. Use --model flash-2 or pro for image_config control."
        )

    _apply_image_config(
        model_config, aspect_ratio, image_size, use_search, config_kwargs
    )
    _apply_thinking_config(model_config, thinking_level, config_kwargs)
    return types.GenerateContentConfig(**config_kwargs)


def _validate_response_candidates(response: Any) -> Any:
    """Return the first candidate content, raising GeminiAPIError on empty output."""
    if not response.candidates:
        feedback = getattr(response, "prompt_feedback", None)
        raise GeminiAPIError(
            f"Gemini returned no response candidates (prompt_feedback={feedback!r})."
        )

    candidate_content = response.candidates[0].content
    if candidate_content is None:
        feedback = getattr(response, "prompt_feedback", None)
        raise GeminiAPIError(
            "Gemini returned a candidate with no content "
            "(typically a safety-filter refusal); "
            f"prompt_feedback={feedback!r}."
        )
    return candidate_content


def _save_thought_part(
    part: Any,
    thought_count: int,
    save_thoughts: bool,
    output_path: Path | None,
    verbose: bool,
) -> None:
    """Print thought reasoning and, when requested, save the thought image."""
    if verbose:
        print(f"\n[Thought {thought_count}]")

    # Handle thought text
    if part.text is not None and verbose:
        print(f"Reasoning: {part.text}")

    # Handle thought image
    if part.inline_data is not None and save_thoughts:
        thought_data = part.inline_data.data
        thought_mime = part.inline_data.mime_type
        thought_ext = ".png" if "png" in thought_mime else ".jpg"

        # Save thought image
        if output_path:
            thought_path = (
                output_path.parent
                / f"{output_path.stem}_thought{thought_count}{thought_ext}"
            )
        else:
            # Anchor under {repo}/output rather than CWD so
            # repeated runs (and the test suite) do not litter
            # whichever directory the script was invoked from.
            timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
            token = secrets.token_hex(16)
            thought_path = (
                Path(__file__).parent.parent
                / "output"
                / f"thought{thought_count}_{timestamp}_{token}{thought_ext}"
            )

        thought_path.parent.mkdir(parents=True, exist_ok=True)
        with open(thought_path, "wb") as f:
            f.write(thought_data)

        print(f"Thought image {thought_count} saved to: {thought_path}")


def _collect_response_parts(
    candidate_content: Any,
    save_thoughts: bool,
    output_path: Path | None,
    verbose: bool,
) -> tuple:
    """Walk response parts; return (image_data, mime_type, signature, thought_count)."""
    thought_count = 0
    final_image_data = None
    final_mime_type = None
    final_signature = None

    for part in candidate_content.parts:
        is_thought = hasattr(part, "thought") and part.thought

        if is_thought:
            thought_count += 1
            _save_thought_part(part, thought_count, save_thoughts, output_path, verbose)
        elif part.inline_data is not None:
            final_image_data = part.inline_data.data
            final_mime_type = part.inline_data.mime_type
            _sig = getattr(part, "thought_signature", None)
            if _sig:
                final_signature = _sig
                if verbose:
                    print(f"\n[Thought Signature]: {final_signature[:100]}...")
        elif part.text is not None:
            print(f"\nModel response: {part.text}")
            _sig = getattr(part, "thought_signature", None)
            if _sig:
                final_signature = _sig
                if verbose:
                    print(f"[Thought Signature]: {final_signature[:100]}...")

    return final_image_data, final_mime_type, final_signature, thought_count


def _reanchor_output_path(output_path: Path, is_draft: bool, script_dir: Path) -> Path:
    """Re-anchor a user output path that resolves outside the repo output/ tree.

    Security boundary: a user-supplied path like ``output/../../../etc/cron.d/x``
    is textually rooted at ``output`` but resolves outside the tree. Resolving
    both sides and comparing with ``is_relative_to`` (rather than a string
    ``startswith`` check) is what closes that path-traversal gap; any path that
    escapes ``script_dir/output`` is re-anchored back inside it by basename.
    """
    allowed_root = (script_dir / "output").resolve()
    try:
        resolved_out = output_path.resolve()
    except OSError as exc:
        # resolve() can fail on a path with a broken symlink loop or a name that
        # is invalid for the filesystem. Treat as "outside the allowed root" so
        # the path is re-anchored safely below; record why for debugging.
        log.debug(f"Could not resolve output path {output_path}: {exc}")
        resolved_out = None
    if resolved_out is None or not resolved_out.is_relative_to(allowed_root):
        if is_draft:
            output_path = script_dir / "output/drafts" / output_path.name
        # Check if this looks like a final (contains "final" in name)
        elif "final" in output_path.stem.lower():
            output_path = script_dir / "output/finals" / output_path.name
        else:
            output_path = script_dir / "output" / output_path.name
    return output_path


def _resolve_generated_output_path(
    output_path: Path | None, detected_ext: str, is_draft: bool, script_dir: Path
) -> Path:
    """Determine the final output path, organizing drafts/finals and fixing the extension.

    For a user-supplied path this also enforces the path-traversal boundary via
    :func:`_reanchor_output_path`; the ``output_path is None`` branch builds a
    safe path under ``output/`` directly and needs no re-anchoring.
    """
    if output_path is None:
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        # Random suffix makes default filenames non-guessable when the
        # output/ directory is later served or shared. Avoids reliance
        # on second-resolution timestamps for uniqueness.
        token = secrets.token_hex(16)
        ext = detected_ext  # Use detected format, not MIME type

        # Auto-organize: drafts go to drafts/, finals would go to root
        if is_draft:
            output_path = script_dir / f"output/drafts/draft_{timestamp}_{token}{ext}"
        else:
            output_path = script_dir / f"output/generated_{timestamp}_{token}{ext}"
    else:
        # If user specified path, check if extension matches actual format
        user_ext = output_path.suffix.lower()
        if user_ext and user_ext != detected_ext:
            # User specified different extension - correct it
            print(
                f"Warning: Specified extension {user_ext} doesn't match actual format {detected_ext}"
            )
            output_path = output_path.with_suffix(detected_ext)
            print(f"  Saving as: {output_path.name}")
        elif not user_ext:
            # No extension specified - add the detected one
            output_path = output_path.with_suffix(detected_ext)
        output_path = _reanchor_output_path(output_path, is_draft, script_dir)
    return output_path


def _write_generated_image(output_path: Path, final_image_data: bytes) -> None:
    """Write the generated image bytes, raising FileIOError on disk failure."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(final_image_data)
    except OSError as exc:
        raise FileIOError(f"Failed to write image to {output_path}: {exc}") from exc


def _document_generated_prompt(
    output_path: Path,
    prompt: str,
    model_key: str,
    aspect_ratio: str | None,
    image_size: str | None,
    reference_images: list[Path] | None,
    is_draft: bool,
) -> None:
    """Record the generation in PROMPTS.md, tolerating a registry write failure."""
    is_final_image = (
        "final" in output_path.stem.lower() or output_path.parent.name == "finals"
    )
    try:
        document_image_prompt(
            image_path=output_path,
            prompt=prompt,
            model_key=model_key,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            reference_images=reference_images,
            is_draft=is_draft,
            is_final=is_final_image,
        )
    except OSError as exc:
        log.warning(
            f"Warning: image saved to {output_path} but PROMPTS.md update failed: {exc}"
        )


def _save_final_image(
    final_image_data: bytes,
    final_mime_type: str | None,
    final_signature: object,
    thought_count: int,
    output_path: Path | None,
    is_draft: bool,
    document_prompt: bool,
    prompt: str,
    model_key: str,
    aspect_ratio: str | None,
    image_size: str | None,
    reference_images: list[Path] | None,
    verbose: bool,
) -> Path:
    """Write the final image, optional signature sidecar, and registry entry."""
    # Detect actual image format from magic bytes (more reliable than MIME type)
    detected_ext = detect_image_format(final_image_data)

    # Also check MIME type for comparison
    mime_ext = get_extension_for_mime(final_mime_type) if final_mime_type else ".png"
    if detected_ext != mime_ext:
        print(
            f"Note: API returned MIME type for {mime_ext}, but data is {detected_ext}"
        )
        print(f"  Using detected format: {detected_ext}")

    script_dir = Path(__file__).parent.parent
    output_path = _resolve_generated_output_path(
        output_path, detected_ext, is_draft, script_dir
    )
    _write_generated_image(output_path, final_image_data)

    if thought_count > 0:
        print(f"\nProcessed {thought_count} thought step(s)")
    print(f"Final image saved to: {output_path}")

    if final_signature and verbose:
        sig_path = output_path.with_suffix(".signature.bin")
        # The signature sidecar is a verbose-only diagnostic; a failure writing
        # it must not turn a successful image save into a fatal GeminiAPIError.
        # Log a non-fatal warning and continue.
        try:
            with open(sig_path, "wb") as f:
                # Signature is binary data
                if isinstance(final_signature, bytes):
                    f.write(final_signature)
                else:
                    f.write(str(final_signature).encode())
            print(f"Thought signature saved to: {sig_path}")
        except OSError as exc:
            log.warning(f"Could not write thought signature to {sig_path}: {exc}")

    if document_prompt:
        _document_generated_prompt(
            output_path,
            prompt,
            model_key,
            aspect_ratio,
            image_size,
            reference_images,
            is_draft,
        )

    return output_path


def generate_image(
    prompt: str,
    model_key: str = DEFAULT_MODEL,
    reference_images: list[Path] | None = None,
    output_path: Path | None = None,
    aspect_ratio: str | None = None,
    image_size: str | None = None,
    use_search: bool = False,
    save_thoughts: bool = False,
    verbose: bool = False,
    is_draft: bool = False,
    document_prompt: bool = True,
    thinking_level: str | None = None,
) -> Path | None:
    """
    Generate an image using Gemini.

    Args:
        prompt: Text description of the image to generate
        model_key: Model to use ('flash', 'flash-2', or 'pro')
        reference_images: Optional list of reference images for editing/style
        output_path: Optional output file path
        aspect_ratio: Aspect ratio (model-dependent; see MODELS[key]['aspect_ratios'])
        image_size: Image size (model-dependent; see MODELS[key]['image_sizes'])
        use_search: Enable Google Search grounding (pro/flash-2)
        save_thoughts: Save intermediate thought images (pro/flash-2)
        verbose: Show detailed thinking process and thought signatures
        thinking_level: 'minimal' or 'high' (flash-2 only)

    Returns:
        Path to the generated image, or None on failure
    """
    api_key = get_api_key()

    if model_key not in MODELS:
        raise ConfigError(
            f"Unknown model '{model_key}'. Use --list-models to see options."
        )

    model_config = MODELS[model_key]
    model_id = model_config["id"]

    print(f"Using model: {model_config['name']}")
    print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

    # Initialize client
    # #CRITICAL -- api_key is passed directly to genai.Client; never log
    #              client objects or request headers in debug mode.
    # #VERIFY   -- Confirm GEMINI_API_KEY is not echoed in any log handler.
    client = genai.Client(api_key=api_key)

    # Build the content parts (reference images + text prompt)
    contents = _build_reference_contents(reference_images)
    if contents is None:
        return None
    contents.append(prompt)

    generate_config = _build_generate_config(
        model_config, aspect_ratio, image_size, use_search, thinking_level
    )

    try:
        print("Generating image...")
        # #ASSUME -- response.candidates[0].content.parts is an iterable of Part
        #            objects; candidates[0].content itself may be None (model
        #            refused). Verify against google-genai SDK changelog before
        #            upgrading google-genai.
        # #EDGE   -- Empty candidates list is guarded below (returns None). Remaining
        #            risk: candidates[0].content is None raises AttributeError on
        #            .parts; caught by the outer except Exception but produces a
        #            generic error indistinguishable from network failures.
        # #VERIFY -- Run: python -c "from google import genai; from google.genai
        #            import types; help(types.GenerateContentResponse)" after any
        #            google-genai version bump to confirm response schema.
        response = client.models.generate_content(
            model=model_id,
            contents=contents,
            config=generate_config,
        )

        # Process response. Two empty-output cases surface as
        # ``GeminiAPIError`` so callers see a single, typed error rather than
        # an opaque ``AttributeError`` from a ``None`` content body:
        #   1. ``response.candidates`` is empty (rare; usually means safety
        #      filter rejected the prompt before any candidate was scored).
        #   2. ``candidates[0].content`` is ``None`` (model refused or
        #      returned no parts; ``prompt_feedback`` typically carries the
        #      reason).
        # In both cases ``prompt_feedback`` is included in the exception
        # message when present so the user can act on the refusal reason.
        candidate_content = _validate_response_candidates(response)
        final_image_data, final_mime_type, final_signature, thought_count = (
            _collect_response_parts(
                candidate_content, save_thoughts, output_path, verbose
            )
        )

        # Save final image
        if final_image_data is not None:
            return _save_final_image(
                final_image_data,
                final_mime_type,
                final_signature,
                thought_count,
                output_path,
                is_draft,
                document_prompt,
                prompt,
                model_key,
                aspect_ratio,
                image_size,
                reference_images,
                verbose,
            )

        raise GeminiAPIError(
            "Gemini response contained no inline image data; "
            "the candidate yielded only thought / text parts."
        )

    except AppError:
        # Already a typed application error; propagate so ``main()`` produces
        # the canonical stderr message + nonzero exit.
        raise
    except Exception as e:
        # Wrap any SDK-side or response-parsing failure as ``GeminiAPIError``
        # so the CLI presents a single, consistent error class. The original
        # exception is chained for debugging via ``__cause__``.
        msg = f"Error generating image: {e}"
        if "API_KEY" in str(e).upper():
            msg = f"{msg}\nCheck that your GEMINI_API_KEY is valid."
        raise GeminiAPIError(msg) from e


def generate_story_sequence(
    base_prompt: str,
    num_parts: int,
    model_key: str = DEFAULT_MODEL,
    output_prefix: Path | None = None,
    aspect_ratio: str | None = None,
    image_size: str | None = None,
    verbose: bool = False,
    thinking_level: str | None = None,
) -> list[Path]:
    """
    Generate a multi-part story sequence using conversational refinement.

    Args:
        base_prompt: Base story description
        num_parts: Number of story parts to generate
        model_key: Model to use
        output_prefix: Prefix for output files (e.g., "story" -> story_part1.png, story_part2.png)
        aspect_ratio: Aspect ratio for all images
        image_size: Image size for all images
        verbose: Show detailed process
        thinking_level: 'minimal' or 'high' (flash-2 only) applied to every part

    Returns:
        List of paths to generated images
    """
    if num_parts < 1:
        raise ConfigError("Number of story parts must be at least 1.")

    if output_prefix is None:
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        token = secrets.token_hex(16)
        output_prefix = Path(f"story_{timestamp}_{token}")

    get_api_key()  # Validate API key exists
    if model_key not in MODELS:
        raise ConfigError(
            f"Unknown model '{model_key}'. Use --list-models to see options."
        )

    generated_images = []
    previous_image_path = None

    print(f"Generating {num_parts}-part story sequence...")
    print(f"Base prompt: {base_prompt}\n")

    for part_num in range(1, num_parts + 1):
        print(f"\n{'=' * 60}")
        print(f"PART {part_num}/{num_parts}")
        print(f"{'=' * 60}")

        # Build prompt for this part
        if part_num == 1:
            prompt = f"{base_prompt}\n\nThis is part 1 of {num_parts}. Create the opening scene that establishes the context and visual style for the entire sequence."
        elif part_num == num_parts:
            prompt = f"This is part {part_num} of {num_parts}, the final scene. Building on the previous image, create a concluding scene that resolves the narrative. Maintain visual consistency with the established style."
        else:
            prompt = f"This is part {part_num} of {num_parts}. Building on the previous image, advance the narrative while maintaining visual consistency with the established style."

        # Build output path
        output_path = output_prefix.parent / f"{output_prefix.stem}_part{part_num}.png"

        # Build reference images list
        reference_images = [previous_image_path] if previous_image_path else None

        print(f"Prompt: {prompt[:100]}...")

        # Generate this part. ``generate_image`` now raises typed AppError
        # subclasses (GeminiAPIError, FileIOError, ConfigError) on failure
        # rather than returning None. Catch the API and file errors here so
        # the story sequence can stop cleanly and still emit the partial
        # completion summary below; let ConfigError propagate because a
        # configuration failure is fatal for the whole sequence.
        try:
            result = generate_image(
                prompt=prompt,
                model_key=model_key,
                reference_images=reference_images,
                output_path=output_path,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                use_search=False,
                save_thoughts=False,
                verbose=verbose,
                thinking_level=thinking_level,
            )
        except (GeminiAPIError, FileIOError) as exc:
            log.error(f"Failed to generate part {part_num}: {exc}")
            break

        if result:
            generated_images.append(result)
            previous_image_path = result
            print(f"Part {part_num} complete: {result}")
        else:
            # Defensive: generate_image() should now always raise on failure,
            # but ``None`` could still appear from a non-error empty-response
            # path. Stop the sequence rather than recurse on a missing image.
            log.error(f"Part {part_num} produced no image; stopping sequence.")
            break

    print(f"\n{'=' * 60}")
    print(
        f"Story sequence complete: {len(generated_images)}/{num_parts} parts generated"
    )
    print(f"{'=' * 60}\n")

    for i, path in enumerate(generated_images, 1):
        print(f"  Part {i}: {path}")

    return generated_images


def list_models():
    """Print available models."""
    print("Available models:\n")
    for key, config in MODELS.items():
        print(f"  {key}:")
        print(f"    Name: {config['name']}")
        print(f"    ID: {config['id']}")
        print(f"    Description: {config['description']}")
        print()


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate images using Google Gemini (Nano Banana / Nano Banana 2 / Nano Banana Pro)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image generation
  %(prog)s "A serene mountain landscape at dawn"
  %(prog)s "A data governance diagram" -o governance.png

  # Draft-then-finalize workflow (cost-effective iteration)
  %(prog)s "A technical blueprint" --draft-mode -o draft.png
  %(prog)s "Adjust colors" -r draft.png --draft-mode -o draft_v2.png
  %(prog)s --finalize draft_v2.png --size 2K -o final.png

  # Image editing with reference
  %(prog)s "Make the building taller" -r blueprint.png
  %(prog)s "Refine this architectural drawing" -r img1.png -r img2.png

  # Advanced options
  %(prog)s "A landscape" --aspect 16:9 --size 4K
  %(prog)s "Current weather in Tokyo" --search
  %(prog)s "Complex blueprint design" --save-thoughts --verbose

  # Multi-part story generation (automatic continuity)
  %(prog)s "A 3-part journey through data governance" --story-parts 3 -o journey
  %(prog)s "Evolution of a data platform" --story-parts 4 --aspect 16:9 --size 2K -o evolution
        """,
    )

    parser.add_argument(
        "prompt",
        nargs="?",
        help="Text prompt describing the image to generate",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file path (default: generated_TIMESTAMP.png)",
    )

    parser.add_argument(
        "-m",
        "--model",
        choices=list(MODELS.keys()),
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )

    parser.add_argument(
        "-r",
        "--reference",
        type=Path,
        action="append",
        dest="references",
        help="Reference image(s) for editing or style (can be used multiple times)",
    )

    parser.add_argument(
        "--aspect",
        choices=ASPECT_RATIOS,
        help=(
            "Aspect ratio (model-dependent). pro: 1:1, 3:4, 4:3, 9:16, 16:9. "
            "flash-2 also supports: 1:4, 1:8, 2:3, 3:2, 4:1, 4:5, 5:4, 8:1, 21:9."
        ),
    )

    parser.add_argument(
        "--size",
        choices=IMAGE_SIZES,
        help="Image size (model-dependent). flash-2: 512, 1K, 2K, 4K. pro: 1K, 2K, 4K.",
    )

    parser.add_argument(
        "--thinking",
        choices=THINKING_LEVELS,
        help="Thinking level for flash-2 (minimal=faster, high=better quality). Ignored on other models.",
    )

    parser.add_argument(
        "--search",
        action="store_true",
        help="Enable Google Search grounding for real-time data (pro / flash-2)",
    )

    parser.add_argument(
        "--save-thoughts",
        action="store_true",
        help="Save intermediate thought images (pro / flash-2)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed thinking process and save thought signatures",
    )

    parser.add_argument(
        "--story-parts",
        type=int,
        metavar="N",
        help="Generate a multi-part story with N parts (uses previous image as reference for continuity)",
    )

    parser.add_argument(
        "--draft-mode",
        action="store_true",
        help="Generate at 1K resolution for faster, lower-cost iteration (upgrades to 2K/4K when finalized)",
    )

    parser.add_argument(
        "--finalize",
        type=Path,
        metavar="DRAFT_IMAGE",
        help="Finalize a draft image by regenerating at higher resolution (2K by default, or use --size)",
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available Gemini models and exit",
    )

    parser.add_argument(
        "--list-topaz-models",
        action="store_true",
        help="List available Topaz enhancement models and exit",
    )

    # --- Topaz post-processing ---
    topaz_group = parser.add_argument_group(
        "Topaz Labs post-processing (requires TOPAZ_API_KEY)"
    )

    topaz_group.add_argument(
        "--enhance",
        type=Path,
        metavar="IMAGE",
        help="Enhance an existing image with Topaz (standalone mode, no generation)",
    )

    topaz_group.add_argument(
        "--topaz",
        action="store_true",
        help="Pipe generated (or finalized) image through Topaz after generation",
    )

    topaz_group.add_argument(
        "--topaz-model",
        default=DEFAULT_TOPAZ_MODEL,
        metavar="MODEL",
        help=f"Topaz model to use (default: {DEFAULT_TOPAZ_MODEL}). See --list-topaz-models.",
    )

    topaz_group.add_argument(
        "--topaz-sharpen",
        type=float,
        metavar="STRENGTH",
        help="Sharpening strength 0.0-1.0 applied during Topaz enhancement",
    )

    topaz_group.add_argument(
        "--topaz-denoise",
        type=float,
        metavar="STRENGTH",
        help="Denoising strength 0.0-1.0 applied during Topaz enhancement",
    )

    topaz_group.add_argument(
        "--topaz-face-enhance",
        action="store_true",
        help="Enable Topaz face enhancement during post-processing",
    )

    topaz_group.add_argument(
        "--topaz-face-strength",
        type=float,
        metavar="STRENGTH",
        help="Face enhancement strength 0.0-1.0 (used with --topaz-face-enhance)",
    )

    return parser


def _handle_list_flags(args: argparse.Namespace) -> bool:
    """Handle --list-models / --list-topaz-models. Return True if handled."""
    if args.list_models:
        list_models()
        return True
    if args.list_topaz_models:
        list_topaz_models()
        return True
    return False


def _print_genai_missing() -> None:
    """Print install guidance when the google-genai package is unavailable."""
    print("Error: google-genai package not installed.")
    print()
    print("To install, create a virtual environment:")
    print("  python3 -m venv .venv")
    print("  source .venv/bin/activate")
    print("  pip install google-genai")
    print()
    print("Or use uv:")
    print("  uv venv && source .venv/bin/activate && uv pip install google-genai")


def _topaz_from_args(
    args: argparse.Namespace,
    input_path: Path,
    output_path: Path | None = None,
) -> Path:
    """Call topaz_enhance_image with the shared --topaz-* options read from args."""
    return topaz_enhance_image(
        input_path=input_path,
        output_path=output_path,
        model=args.topaz_model,
        sharpen=args.topaz_sharpen,
        denoise=args.topaz_denoise,
        face_enhancement=args.topaz_face_enhance,
        face_enhancement_strength=args.topaz_face_strength,
        verbose=args.verbose,
    )


def _handle_enhance(args: argparse.Namespace) -> None:
    """Standalone Topaz enhancement mode (--enhance); exits the process."""
    _topaz_from_args(args, args.enhance, args.output)
    sys.exit(0)


def _resolve_finalize_output(args: argparse.Namespace) -> Path:
    """Return the output path for finalize mode, defaulting next to the draft."""
    if args.output:
        return args.output
    return args.finalize.parent / f"{args.finalize.stem}_final.png"


def _finalize_via_topaz(args: argparse.Namespace, output_path: Path) -> None:
    """Finalize a draft through Topaz (--finalize --topaz); exits the process."""
    print(f"Finalizing draft image via Topaz: {args.finalize}")
    print(f"Topaz model: {args.topaz_model}")
    result = _topaz_from_args(args, args.finalize, output_path)

    print(f"\n{'=' * 60}")
    print("Topaz finalization complete!")
    print(f"Draft: {args.finalize}")
    print(f"Final: {result}")
    print(f"{'=' * 60}")

    sys.exit(0)


def _finalize_via_gemini(args: argparse.Namespace, output_path: Path) -> None:
    """Finalize a draft by regenerating at higher resolution; exits the process."""
    final_size = args.size or "2K"
    final_aspect = args.aspect or "16:9"

    print(f"Finalizing draft image: {args.finalize}")
    print(f"Target resolution: {final_size} ({final_aspect})")

    prompts_file = Path(__file__).parent.parent / "examples" / "PROMPTS.md"
    if prompts_file.exists():
        with open(prompts_file, encoding="utf-8") as f:
            content = f.read()
            image_name = args.finalize.stem
            if image_name in content:
                print("Found original prompt in PROMPTS.md")
                if not args.prompt:
                    print(
                        "\nNote: Use the same prompt as the draft, or provide a refinement prompt."
                    )
                    print(
                        'Example: python scripts/generate_image.py --finalize draft.png "Same as draft"'
                    )
                    print("\nProceeding with reference-based upscaling...")

    prompt = (
        args.prompt
        or "Recreate this image at higher resolution with the same composition, style, and details"
    )

    result = generate_image(
        prompt=prompt,
        model_key=args.model,
        reference_images=[args.finalize],
        output_path=output_path,
        aspect_ratio=final_aspect,
        image_size=final_size,
        use_search=False,
        save_thoughts=args.save_thoughts,
        verbose=args.verbose,
        thinking_level=args.thinking,
    )

    if result:
        print(f"\n{'=' * 60}")
        print("Finalization complete!")
        print(f"Draft (1K): {args.finalize}")
        print(f"Final ({final_size}): {result}")
        print(f"{'=' * 60}")

    sys.exit(0 if result else 1)


def _handle_finalize(args: argparse.Namespace) -> None:
    """Finalize mode (--finalize): upscale a draft to final resolution."""
    if not args.finalize.exists():
        print(f"Error: Draft image not found: {args.finalize}")
        sys.exit(1)

    output_path = _resolve_finalize_output(args)

    if args.topaz:
        _finalize_via_topaz(args, output_path)

    _finalize_via_gemini(args, output_path)


def _enhance_story_results(args: argparse.Namespace, results: list) -> list:
    """Apply Topaz to each story image, recovering per image; return new paths."""
    print(f"\nApplying Topaz enhancement to {len(results)} image(s)...")
    enhanced = []
    # Catch per-image so a single failure does not abort the batch.
    # ConfigError propagates because misconfiguration applies to
    # every image in the batch -- recovering would just produce
    # N identical errors.
    for path in results:
        try:
            enhanced.append(_topaz_from_args(args, path))
        except (TopazAPIError, FileIOError) as exc:  # noqa: PERF203 - per-image recovery is the point of the loop
            log.error(f"Topaz enhancement failed for {path}: {exc}")
    return enhanced


def _handle_story(args: argparse.Namespace) -> None:
    """Story sequence mode (--story-parts); exits the process."""
    if args.story_parts < 2:
        print("Error: Story must have at least 2 parts")
        sys.exit(1)

    output_prefix = args.output or Path("story")

    results = generate_story_sequence(
        base_prompt=args.prompt,
        num_parts=args.story_parts,
        model_key=args.model,
        output_prefix=output_prefix,
        aspect_ratio=args.aspect,
        image_size=args.size,
        verbose=args.verbose,
        thinking_level=args.thinking,
    )

    if args.topaz and results:
        results = _enhance_story_results(args, results)

    sys.exit(0 if len(results) == args.story_parts else 1)


def _resolve_effective_size(args: argparse.Namespace) -> str | None:
    """Pick the image size, applying draft-mode defaults read from MODELS."""
    if not (args.draft_mode and args.size is None):
        return args.size

    # Draft mode picks the smallest tier the active model supports so
    # iteration is fast and cheap. Capability is read from MODELS rather
    # than hardcoded so a future model entry that adds 512 (or removes
    # 1K) Just Works. Legacy 'flash' has no size control at all - we
    # return None so generate_image() does not receive a bogus value
    # (which would also trigger its "does not support --aspect or --size"
    # warning misleadingly). A user-specified --size always wins above.
    supported_sizes = MODELS[args.model].get("image_sizes", [])
    if "512" in supported_sizes:
        return "512"
    if "1K" in supported_sizes:
        return "1K"
    return None


def _print_draft_banner(args: argparse.Namespace, effective_size: str | None) -> None:
    """Print the draft-mode banner before a single-image draft generation."""
    if effective_size:
        print(
            f"Draft mode: Generating at {effective_size} resolution for fast iteration"
        )
    else:
        print(
            f"Draft mode: {MODELS[args.model]['name']} has no size control;"
            " generating at the model's default resolution"
        )
    print("Drafts are stored in output/drafts/")
    print("Use --finalize <draft_image.png> to upscale to final resolution\n")


def _handle_single(args: argparse.Namespace) -> None:
    """Single image mode (the default); exits the process."""
    effective_size = _resolve_effective_size(args)

    if args.draft_mode:
        _print_draft_banner(args, effective_size)

    result = generate_image(
        prompt=args.prompt,
        model_key=args.model,
        reference_images=args.references,
        output_path=args.output,
        aspect_ratio=args.aspect,
        image_size=effective_size,
        use_search=args.search,
        save_thoughts=args.save_thoughts,
        verbose=args.verbose,
        is_draft=args.draft_mode,
        document_prompt=True,
        thinking_level=args.thinking,
    )

    if result and args.topaz:
        # Single-image path: typed AppError propagates to main(); on
        # success ``topaz_enhance_image`` returns a Path. The ``result and
        # args.topaz`` guard already short-circuits when generate_image
        # returned None for a non-error empty-response path.
        result = _topaz_from_args(args, result)

    if result and args.draft_mode:
        print(f"\n{'=' * 60}")
        print("Draft complete! To finalize at higher resolution:")
        print(f"  python scripts/generate_image.py --finalize {result} --size 2K")
        print(f"  # Or with Topaz: --finalize {result} --topaz")
        print(f"{'=' * 60}")

    sys.exit(0 if result else 1)


def _reject_nul_byte_in_path_args(args: argparse.Namespace) -> None:
    """Reject NUL bytes in path-typed CLI arguments at the entry boundary.

    Defense in depth only. This does not resolve or constrain paths, so it never
    changes which file a valid path refers to; it surfaces an embedded NUL as a
    typed :class:`FileIOError` (rather than a raw ``ValueError`` deep in an
    ``open()`` call). The broader Snyk Code path-traversal findings on these
    sinks are accepted under this project's threat model (see SECURITY.md): a
    local single-user CLI has no trust boundary to cross.
    """
    candidates: list[Path] = [
        p for p in (args.output, args.finalize, args.enhance) if p is not None
    ]
    candidates.extend(args.references or ())
    for path in candidates:
        if "\x00" in str(path):
            raise FileIOError(f"Invalid path (contains NUL byte): {path!r}")


def _run() -> None:
    """CLI body. Raises ``AppError`` for expected failures and ``SystemExit``
    for terminal cases; :func:`main` is the user-facing entry point that
    translates ``AppError`` to a clean stderr message + exit code 1.
    """
    parser = _build_argument_parser()
    args = parser.parse_args()
    _reject_nul_byte_in_path_args(args)

    # Reconfigure structlog now that we know the verbosity flag.
    _configure_logging(verbose=bool(args.verbose))

    if _handle_list_flags(args):
        return

    if not args.prompt and not args.finalize and not args.enhance:
        parser.print_help()
        sys.exit(1)

    # --- Standalone Topaz enhancement mode ---
    if args.enhance:
        _handle_enhance(args)

    # Check for google-genai package
    if not GENAI_AVAILABLE:
        _print_genai_missing()
        sys.exit(1)

    if args.finalize:
        _handle_finalize(args)

    if args.story_parts:
        _handle_story(args)

    _handle_single(args)


def main() -> None:
    """User-facing CLI entry point.

    Catches :class:`AppError` (configuration, Gemini, Topaz, file I/O) and
    surfaces a clean stderr message plus exit code 1, instead of a Python
    traceback. The outermost ``except Exception`` is the safety net for
    truly unexpected bugs: it still prints + exits, but with a different
    prefix so users can distinguish "expected failure" from "this is a
    bug, please report it".
    """
    try:
        _run()
    except AppError as exc:
        log.error(f"Error: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        log.error("Interrupted.")
        sys.exit(130)
    except Exception:
        # ``log.exception`` is the idiomatic preserve-traceback path: it
        # captures ``sys.exc_info()`` automatically and either renders the
        # traceback via structlog's ConsoleRenderer (the configured path) or
        # via the ``_StderrLogger.exception`` fallback that calls
        # ``traceback.print_exc``. Users need the full traceback so the "this
        # is a bug, please report it" instruction is actionable.
        log.exception(
            "Unexpected error (please report as a bug, include the traceback below)"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
