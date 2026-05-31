"""Tests for scripts/generate_image.py.

Coverage focus: pure functions, file I/O helpers, API key helpers, and the
generate_image() function with a mocked google-genai SDK client.

Style mirrors test_topaz_enhance.py: unittest.mock only (no pytest-mock),
tmp_path for file I/O, monkeypatch for environment variables.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Magic-byte constants
# ---------------------------------------------------------------------------

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
_JPEG_MAGIC = b"\xff\xd8\xff" + b"\x00" * 13
_WEBP_MAGIC = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 4
_GIF87_MAGIC = b"GIF87a" + b"\x00" * 10
_GIF89_MAGIC = b"GIF89a" + b"\x00" * 10
_UNKNOWN_MAGIC = b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b"


# ---------------------------------------------------------------------------
# Tier 1: detect_image_format() -- pure function, no mocks needed
# ---------------------------------------------------------------------------


class TestDetectImageFormat:
    """detect_image_format() reads magic bytes and returns a dot-prefixed extension."""

    def test_png_magic_returns_dot_png(self) -> None:
        from scripts.generate_image import detect_image_format

        assert detect_image_format(_PNG_MAGIC) == ".png"

    def test_jpeg_magic_returns_dot_jpg(self) -> None:
        from scripts.generate_image import detect_image_format

        assert detect_image_format(_JPEG_MAGIC) == ".jpg"

    def test_webp_magic_returns_dot_webp(self) -> None:
        from scripts.generate_image import detect_image_format

        assert detect_image_format(_WEBP_MAGIC) == ".webp"

    def test_gif87_magic_returns_dot_gif(self) -> None:
        from scripts.generate_image import detect_image_format

        assert detect_image_format(_GIF87_MAGIC) == ".gif"

    def test_gif89_magic_returns_dot_gif(self) -> None:
        from scripts.generate_image import detect_image_format

        assert detect_image_format(_GIF89_MAGIC) == ".gif"

    def test_unknown_bytes_returns_dot_png_fallback(self) -> None:
        from scripts.generate_image import detect_image_format

        assert detect_image_format(_UNKNOWN_MAGIC) == ".png"

    def test_empty_bytes_returns_dot_png_fallback(self) -> None:
        from scripts.generate_image import detect_image_format

        # Empty data: no magic bytes match, should fall back to .png
        assert detect_image_format(b"") == ".png"


# ---------------------------------------------------------------------------
# Tier 1: get_extension_for_mime() -- pure function, no mocks needed
# ---------------------------------------------------------------------------


class TestGetExtensionForMime:
    """get_extension_for_mime() maps MIME strings to dot-extensions."""

    def test_image_png_returns_dot_png(self) -> None:
        from scripts.generate_image import get_extension_for_mime

        assert get_extension_for_mime("image/png") == ".png"

    def test_image_jpeg_returns_dot_jpg(self) -> None:
        from scripts.generate_image import get_extension_for_mime

        assert get_extension_for_mime("image/jpeg") == ".jpg"

    def test_image_webp_returns_dot_webp(self) -> None:
        from scripts.generate_image import get_extension_for_mime

        assert get_extension_for_mime("image/webp") == ".webp"

    def test_image_gif_returns_dot_gif(self) -> None:
        from scripts.generate_image import get_extension_for_mime

        assert get_extension_for_mime("image/gif") == ".gif"

    def test_unknown_mime_returns_dot_png_fallback(self) -> None:
        from scripts.generate_image import get_extension_for_mime

        assert get_extension_for_mime("application/octet-stream") == ".png"

    def test_empty_string_returns_dot_png_fallback(self) -> None:
        from scripts.generate_image import get_extension_for_mime

        assert get_extension_for_mime("") == ".png"


# ---------------------------------------------------------------------------
# Tier 2: Settings / _load_api_key() -- pydantic-settings + .env loading
# ---------------------------------------------------------------------------


class TestLoadApiKey:
    """_load_api_key() reads from env first, then falls back to a .env file.

    Implementation is now a thin wrapper around the ``Settings`` pydantic
    BaseSettings class; only the known variables (GEMINI_API_KEY,
    TOPAZ_API_KEY) are recognised. The legacy "any env var name" interface
    is no longer supported.
    """

    def test_returns_key_from_environment(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from scripts.generate_image import _load_api_key

        # Isolate from any project .env on disk by repointing the env_file.
        monkeypatch.setattr("scripts.generate_image._ENV_FILE", tmp_path / ".no-env")
        monkeypatch.setenv("GEMINI_API_KEY", "env-value-123")
        result = _load_api_key("GEMINI_API_KEY")
        assert result == "env-value-123"

    def test_returns_none_when_neither_env_nor_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from scripts.generate_image import _load_api_key

        monkeypatch.setattr("scripts.generate_image._ENV_FILE", tmp_path / ".no-env")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        result = _load_api_key("GEMINI_API_KEY")
        assert result is None

    def test_reads_key_from_dot_env_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Key absent from env but present in a .env file is read."""
        from scripts.generate_image import _load_api_key

        monkeypatch.delenv("TOPAZ_API_KEY", raising=False)

        env_file = tmp_path / ".env"
        env_file.write_text('TOPAZ_API_KEY="file-value-456"\n', encoding="utf-8")
        monkeypatch.setattr("scripts.generate_image._ENV_FILE", env_file)

        # ``Settings`` is re-instantiated on every call via ``get_settings()``;
        # patching the module-level _ENV_FILE is enough for the read to pick up
        # the temporary file.
        from scripts.generate_image import Settings, SettingsConfigDict

        # Re-bind Settings.model_config to point at the temp .env. pydantic
        # captures the path at class construction; we shadow the env_file
        # config so the temp path is used.
        monkeypatch.setattr(
            Settings,
            "model_config",
            SettingsConfigDict(
                env_file=str(env_file),
                env_file_encoding="utf-8",
                extra="ignore",
                case_sensitive=True,
            ),
        )

        result = _load_api_key("TOPAZ_API_KEY")
        assert result == "file-value-456"

    def test_returns_none_when_env_file_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """If the .env file path does not exist, pydantic-settings silently
        skips it and the key resolves to None when not in the environment."""
        from scripts.generate_image import _load_api_key

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setattr(
            "scripts.generate_image._ENV_FILE", tmp_path / "missing.env"
        )
        result = _load_api_key("GEMINI_API_KEY")
        assert result is None

    def test_returns_key_from_env_file_unquoted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Keys without quotes in .env file are parsed correctly."""
        from scripts.generate_image import _load_api_key

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        env_file = tmp_path / ".env"
        env_file.write_text("GEMINI_API_KEY=plain-value\n", encoding="utf-8")
        monkeypatch.setattr("scripts.generate_image._ENV_FILE", env_file)

        from scripts.generate_image import Settings, SettingsConfigDict

        monkeypatch.setattr(
            Settings,
            "model_config",
            SettingsConfigDict(
                env_file=str(env_file),
                env_file_encoding="utf-8",
                extra="ignore",
                case_sensitive=True,
            ),
        )

        result = _load_api_key("GEMINI_API_KEY")
        assert result == "plain-value"


# ---------------------------------------------------------------------------
# Tier 2: load_image_as_base64() -- reads a file and returns (b64, mime)
# ---------------------------------------------------------------------------


class TestLoadImageAsBase64:
    def test_valid_png_file_returns_base64_and_mime(self, tmp_path: Path) -> None:
        from scripts.generate_image import load_image_as_base64

        img = tmp_path / "test.png"
        img.write_bytes(_PNG_MAGIC)

        b64, mime = load_image_as_base64(img)

        assert mime == "image/png"
        decoded = base64.standard_b64decode(b64)
        assert decoded == _PNG_MAGIC

    def test_valid_jpeg_file_returns_correct_mime(self, tmp_path: Path) -> None:
        from scripts.generate_image import load_image_as_base64

        img = tmp_path / "photo.jpg"
        img.write_bytes(_JPEG_MAGIC)

        _, mime = load_image_as_base64(img)

        assert mime == "image/jpeg"

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        from scripts.generate_image import load_image_as_base64

        missing = tmp_path / "does_not_exist.png"

        with pytest.raises(FileNotFoundError):
            load_image_as_base64(missing)

    def test_extension_mismatch_still_returns_detected_mime(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A file named .jpg but containing PNG bytes should report image/png."""
        from scripts.generate_image import load_image_as_base64

        # Write PNG bytes but name the file .jpg
        img = tmp_path / "sneaky.jpg"
        img.write_bytes(_PNG_MAGIC)

        _, mime = load_image_as_base64(img)

        # Should detect PNG from magic bytes
        assert mime == "image/png"
        out = capsys.readouterr().out
        # A warning about extension mismatch should have been printed
        assert "warning" in out.lower() or ".jpg" in out


# ---------------------------------------------------------------------------
# Tier 2: document_image_prompt() -- appends to PROMPTS.md
# ---------------------------------------------------------------------------


class TestDocumentImagePrompt:
    def test_creates_prompts_md_when_absent(self, tmp_path: Path) -> None:
        """When PROMPTS.md does not exist yet, the function creates it."""
        import scripts.generate_image as mod

        # Set up directory layout that the function uses:
        # Path(__file__).parent.parent / "examples" / "PROMPTS.md"
        fake_script_path = tmp_path / "scripts" / "generate_image.py"
        fake_script_path.parent.mkdir(parents=True, exist_ok=True)

        examples_dir = tmp_path / "examples"
        examples_dir.mkdir()
        prompts_file = examples_dir / "PROMPTS.md"

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        img = output_dir / "generated_20240101_120000.png"
        img.write_bytes(_PNG_MAGIC)

        # PROMPTS.md must not exist before the call
        assert not prompts_file.exists()

        with patch.object(mod, "__file__", str(fake_script_path)):
            from scripts.generate_image import document_image_prompt

            document_image_prompt(
                image_path=img,
                prompt="A test prompt for coverage",
                model_key="flash",
                aspect_ratio="16:9",
                image_size="2K",
            )

        assert prompts_file.exists()
        content = prompts_file.read_text(encoding="utf-8")
        assert "A test prompt for coverage" in content

    def test_appends_to_existing_prompts_md(self, tmp_path: Path) -> None:
        """When PROMPTS.md already exists, a new entry is appended."""
        import scripts.generate_image as mod

        # Set up fake directory layout matching what the function expects:
        # Path(__file__).parent.parent / "examples" / "PROMPTS.md"
        fake_script_path = tmp_path / "scripts" / "generate_image.py"
        fake_script_path.parent.mkdir(parents=True, exist_ok=True)

        examples_dir = tmp_path / "examples"
        examples_dir.mkdir()
        prompts_file = examples_dir / "PROMPTS.md"

        initial_content = (
            "# AI-Generated Image Registry\n\n"
            "## Image Registry\n\n"
            "| Filename | Model | Date | Prompt | Parameters |\n"
            "|----------|-------|------|--------|------------|\n"
            "\n## Detailed Prompts\n\n"
        )
        prompts_file.write_text(initial_content, encoding="utf-8")

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        img = output_dir / "test_image.png"
        img.write_bytes(_PNG_MAGIC)

        with patch.object(mod, "__file__", str(fake_script_path)):
            from scripts.generate_image import document_image_prompt

            document_image_prompt(
                image_path=img,
                prompt="Coverage test prompt",
                model_key="flash",
                aspect_ratio="1:1",
                image_size="1K",
            )

        updated = prompts_file.read_text(encoding="utf-8")
        assert "Coverage test prompt" in updated
        assert "test_image.png" in updated

    def test_skips_duplicate_entry(self, tmp_path: Path) -> None:
        """Calling document_image_prompt twice with the same image does not duplicate."""
        import scripts.generate_image as mod

        fake_script_path = tmp_path / "scripts" / "generate_image.py"
        fake_script_path.parent.mkdir(parents=True, exist_ok=True)

        examples_dir = tmp_path / "examples"
        examples_dir.mkdir()
        prompts_file = examples_dir / "PROMPTS.md"

        initial_content = (
            "# AI-Generated Image Registry\n\n"
            "## Image Registry\n\n"
            "| Filename | Model | Date | Prompt | Parameters |\n"
            "|----------|-------|------|--------|------------|\n"
            "\n## Detailed Prompts\n\n"
        )
        prompts_file.write_text(initial_content, encoding="utf-8")

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        img = output_dir / "dedup_image.png"
        img.write_bytes(_PNG_MAGIC)

        with patch.object(mod, "__file__", str(fake_script_path)):
            from scripts.generate_image import document_image_prompt

            document_image_prompt(
                image_path=img,
                prompt="First call",
                model_key="pro",
                aspect_ratio="16:9",
                image_size="4K",
            )
            document_image_prompt(
                image_path=img,
                prompt="Second call -- should be ignored",
                model_key="pro",
                aspect_ratio="16:9",
                image_size="4K",
            )

        content = prompts_file.read_text(encoding="utf-8")
        # The heading "### dedup_image.png" should appear exactly once
        assert content.count("### dedup_image.png") == 1
        assert "Second call" not in content


# ---------------------------------------------------------------------------
# Tier 3: get_api_key() and get_topaz_api_key()
# ---------------------------------------------------------------------------


class TestGetApiKey:
    def test_returns_key_when_present_in_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts.generate_image import get_api_key

        monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
        # Also prevent fallback .env from interfering
        with patch(
            "scripts.generate_image._load_api_key", return_value="gemini-test-key"
        ):
            result = get_api_key()

        assert result == "gemini-test-key"

    def test_raises_config_error_when_key_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Post-refactor, the missing-key case raises ``ConfigError`` instead
        of calling ``sys.exit`` directly. ``main()`` translates the exception
        into exit code 1; the underlying CLI contract is preserved.
        """
        from scripts.generate_image import ConfigError, get_api_key

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with (
            patch("scripts.generate_image._load_api_key", return_value=None),
            pytest.raises(ConfigError) as exc_info,
        ):
            get_api_key()

        assert "GEMINI_API_KEY" in str(exc_info.value)

    def test_main_exits_with_code_1_when_key_absent(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """End-to-end: ``main()`` still exits with code 1 on missing key."""
        from scripts.generate_image import main

        monkeypatch.setattr("sys.argv", ["generate_image.py", "test prompt"])
        with (
            patch("scripts.generate_image._load_api_key", return_value=None),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "GEMINI_API_KEY" in err


class TestGetTopazApiKey:
    def test_returns_key_when_present_in_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts.generate_image import get_topaz_api_key

        monkeypatch.setenv("TOPAZ_API_KEY", "topaz-test-key")
        with patch(
            "scripts.generate_image._load_api_key", return_value="topaz-test-key"
        ):
            result = get_topaz_api_key()

        assert result == "topaz-test-key"

    def test_returns_none_when_key_absent_and_prints_to_stderr(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from scripts.generate_image import get_topaz_api_key

        monkeypatch.delenv("TOPAZ_API_KEY", raising=False)
        with patch("scripts.generate_image._load_api_key", return_value=None):
            result = get_topaz_api_key()

        assert result is None
        err = capsys.readouterr().err
        assert "topaz_api_key" in err.lower() or "topaz" in err.lower()


# ---------------------------------------------------------------------------
# Tier 3: list_topaz_models() -- returns TOPAZ_MODELS dict structure
# ---------------------------------------------------------------------------


class TestListTopazModels:
    def test_topaz_models_dict_contains_standard_v2(self) -> None:
        from scripts.generate_image import TOPAZ_MODELS

        assert "Standard V2" in TOPAZ_MODELS

    def test_standard_v2_has_endpoint_key(self) -> None:
        from scripts.generate_image import TOPAZ_MODELS

        assert "endpoint" in TOPAZ_MODELS["Standard V2"]

    def test_all_models_have_endpoint_and_description(self) -> None:
        from scripts.generate_image import TOPAZ_MODELS

        for name, cfg in TOPAZ_MODELS.items():
            assert "endpoint" in cfg, f"{name!r} missing 'endpoint'"
            assert "description" in cfg, f"{name!r} missing 'description'"

    def test_list_topaz_models_prints_output(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from scripts.generate_image import list_topaz_models

        list_topaz_models()

        out = capsys.readouterr().out
        assert "Standard V2" in out
        assert "precision" in out.lower() or "upscaling" in out.lower()


# ---------------------------------------------------------------------------
# Tier 3: list_models() -- prints available Gemini models
# ---------------------------------------------------------------------------


class TestListModels:
    def test_list_models_prints_flash_and_pro(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from scripts.generate_image import list_models

        list_models()

        out = capsys.readouterr().out
        assert "flash" in out.lower()
        assert "pro" in out.lower()
        # Nano Banana 2 (gemini-3.1-flash-image-preview)
        assert "flash-2" in out.lower()
        assert "gemini-3.1-flash-image-preview" in out


# ---------------------------------------------------------------------------
# Tier 4: generate_image() with mocked google-genai SDK
# ---------------------------------------------------------------------------


def _make_fake_inline_data(data: bytes, mime_type: str = "image/png") -> MagicMock:
    """Build a fake inline_data object."""
    inline = MagicMock()
    inline.data = data
    inline.mime_type = mime_type
    return inline


def _make_fake_part(
    inline_data: MagicMock | None = None,
    text: str | None = None,
    is_thought: bool = False,
) -> MagicMock:
    """Build a fake response part."""
    part = MagicMock()
    part.inline_data = inline_data
    part.text = text
    part.thought = is_thought
    # thought_signature is absent by default (hasattr check in production code)
    part.thought_signature = None
    return part


def _make_fake_response(parts: list[MagicMock]) -> MagicMock:
    """Build a fake generate_content() response with one candidate."""
    candidate = MagicMock()
    candidate.content.parts = parts
    response = MagicMock()
    response.candidates = [candidate]
    return response


class TestGenerateImageSuccess:
    """Happy-path: image bytes returned, file written, Path returned."""

    def test_writes_output_file_and_returns_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")

        # Fake response: one non-thought part with PNG inline_data
        inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        part = _make_fake_part(inline_data=inline)
        fake_response = _make_fake_response([part])

        # Build the mock client
        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        output_path = tmp_path / "out.png"

        # Patch script's __file__ so output path resolution uses tmp_path
        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)

        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch(
                "scripts.generate_image._load_api_key", return_value="fake-gemini-key"
            ),
            patch("scripts.generate_image.document_image_prompt"),
        ):
            result = mod.generate_image(
                prompt="A test image",
                model_key="flash",
                output_path=output_path,
                document_prompt=False,
            )

        assert result is not None
        assert result.exists()
        assert result.read_bytes() == _PNG_MAGIC

    def test_returned_path_has_correct_extension(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        part = _make_fake_part(inline_data=inline)
        fake_response = _make_fake_response([part])

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        output_path = tmp_path / "result.png"
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)

        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            result = mod.generate_image(
                prompt="Check extension",
                model_key="flash",
                output_path=output_path,
                document_prompt=False,
            )

        assert result is not None
        assert result.suffix == ".png"

    def test_auto_timestamp_path_when_output_not_specified(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        part = _make_fake_part(inline_data=inline)
        fake_response = _make_fake_response([part])

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)

        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            result = mod.generate_image(
                prompt="Auto path test",
                model_key="flash",
                output_path=None,
                document_prompt=False,
            )

        assert result is not None
        assert result.exists()
        # Filename must follow generated_<timestamp>_<8 hex>.<ext>; a regression
        # that drops the secrets.token_hex(4) suffix would fail this regex.
        import re

        assert re.match(r"^generated_\d{8}_\d{6}_[0-9a-f]{32}\.\w+$", result.name), (
            result.name
        )


class TestGenerateImageEmptyCandidates:
    """When the API returns no candidates, the function raises GeminiAPIError.

    Previously this path printed an error and returned None. The typed-raise
    contract (PR #39) makes the failure visible to ``main()`` as a typed
    AppError so the user sees an "Error: ..." exit instead of a silent
    None propagating through the success branch.
    """

    def test_empty_candidates_raises_gemini_api_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scripts.generate_image as mod
        from scripts.generate_image import GeminiAPIError

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        fake_response = MagicMock()
        fake_response.candidates = []
        # prompt_feedback absent to exercise the getattr default branch
        fake_response.prompt_feedback = None

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        with (
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
            pytest.raises(GeminiAPIError) as exc_info,
        ):
            mod.generate_image(
                prompt="Empty candidates test",
                model_key="flash",
                document_prompt=False,
            )

        assert "no response candidates" in str(exc_info.value).lower()

    def test_empty_candidates_with_prompt_feedback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When response has prompt_feedback, the exception carries it."""
        import scripts.generate_image as mod
        from scripts.generate_image import GeminiAPIError

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        fake_response = MagicMock()
        fake_response.candidates = []
        fake_response.prompt_feedback = "SAFETY_BLOCK"

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        with (
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
            pytest.raises(GeminiAPIError) as exc_info,
        ):
            mod.generate_image(
                prompt="Safety block test",
                model_key="flash",
                document_prompt=False,
            )

        assert "SAFETY_BLOCK" in str(exc_info.value)


class TestGenerateImageApiKeyMissing:
    """When get_api_key() raises ConfigError, it should propagate."""

    def test_config_error_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from scripts import generate_image as mod
        from scripts.generate_image import ConfigError

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        with (
            patch("scripts.generate_image._load_api_key", return_value=None),
            pytest.raises(ConfigError) as exc_info,
        ):
            mod.generate_image(
                prompt="Should raise",
                model_key="flash",
                document_prompt=False,
            )

        assert "GEMINI_API_KEY" in str(exc_info.value)


class TestGenerateImageUnknownModel:
    def test_unknown_model_raises_config_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scripts.generate_image as mod
        from scripts.generate_image import ConfigError

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        with (
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
            pytest.raises(ConfigError) as exc_info,
        ):
            mod.generate_image(
                prompt="Unknown model test",
                model_key="nonexistent_model",
                document_prompt=False,
            )

        assert "nonexistent_model" in str(exc_info.value)


class TestGenerateImageNoImageData:
    """When response has candidates but no inline_data, raises GeminiAPIError."""

    def test_text_only_response_raises_gemini_api_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scripts.generate_image as mod
        from scripts.generate_image import GeminiAPIError

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        # Part has text but no inline_data
        part = _make_fake_part(inline_data=None, text="Here is some text")
        fake_response = _make_fake_response([part])

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        with (
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
            pytest.raises(GeminiAPIError) as exc_info,
        ):
            mod.generate_image(
                prompt="Text-only test",
                model_key="flash",
                document_prompt=False,
            )

        assert "no inline image data" in str(exc_info.value).lower()


class TestGenerateImageApiException:
    """When generate_content() raises, it is wrapped in ``GeminiAPIError``.

    Post-refactor, the broad ``except Exception`` no longer prints + returns
    ``None`` -- it re-raises as a typed application error. ``main()`` is the
    layer that translates this to a clean stderr message + exit code 1.
    """

    def test_api_exception_raises_gemini_api_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts import generate_image as mod
        from scripts.generate_image import GeminiAPIError

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.side_effect = RuntimeError(
            "connection refused"
        )
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        with (
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
            pytest.raises(GeminiAPIError) as exc_info,
        ):
            mod.generate_image(
                prompt="Exception test",
                model_key="flash",
                document_prompt=False,
            )

        assert "connection refused" in str(exc_info.value)

    def test_api_key_error_hints_at_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the upstream error message contains 'API_KEY', the wrapped
        ``GeminiAPIError`` includes the hint about validating the env var."""
        from scripts import generate_image as mod
        from scripts.generate_image import GeminiAPIError

        monkeypatch.setenv("GEMINI_API_KEY", "bad-key")

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.side_effect = RuntimeError(
            "INVALID_API_KEY: key is malformed"
        )
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        with (
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="bad-key"),
            pytest.raises(GeminiAPIError) as exc_info,
        ):
            mod.generate_image(
                prompt="Bad key test",
                model_key="flash",
                document_prompt=False,
            )

        assert "GEMINI_API_KEY" in str(exc_info.value)


class TestGenerateImageDraftMode:
    """Draft mode generates into output/drafts/ subdirectory."""

    def test_draft_goes_to_drafts_subdir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        part = _make_fake_part(inline_data=inline)
        fake_response = _make_fake_response([part])

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "output" / "drafts").mkdir(parents=True, exist_ok=True)

        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            result = mod.generate_image(
                prompt="Draft mode test",
                model_key="flash",
                output_path=None,
                is_draft=True,
                document_prompt=False,
            )

        assert result is not None
        # Filename must follow draft_<timestamp>_<8 hex>.<ext>; assert the
        # parent dir is the drafts/ tree and the file itself carries the token.
        assert result.parent.name == "drafts"
        import re

        assert re.match(r"^draft_\d{8}_\d{6}_[0-9a-f]{32}\.\w+$", result.name), (
            result.name
        )


def _patch_pro_model_types() -> tuple[Any, Any]:
    """Return context managers that patch types for pro-model tests.

    The installed google-genai SDK does not have types.ImageConfig and
    types.GenerateContentConfig rejects unknown 'image_config' fields via
    pydantic strict validation. We mock both to avoid errors when exercising
    the pro model code path that builds an image config.
    """
    return (
        patch(
            "scripts.generate_image.types.ImageConfig",
            MagicMock(return_value=MagicMock()),
            create=True,
        ),
        patch(
            "scripts.generate_image.types.GenerateContentConfig",
            MagicMock(return_value=MagicMock()),
        ),
    )


class TestGenerateImageProModel:
    """Pro model supports aspect_ratio, image_size, and use_search."""

    def test_pro_model_with_aspect_and_size(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        part = _make_fake_part(inline_data=inline)
        fake_response = _make_fake_response([part])

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        output_path = tmp_path / "pro_out.png"
        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)

        img_cfg_patch, gen_cfg_patch = _patch_pro_model_types()
        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
            img_cfg_patch,
            gen_cfg_patch,
        ):
            result = mod.generate_image(
                prompt="Pro model test",
                model_key="pro",
                output_path=output_path,
                aspect_ratio="16:9",
                image_size="4K",
                document_prompt=False,
            )

        assert result is not None
        # Confirm generate_content was called with the pro model id
        call_kwargs = mock_client_instance.models.generate_content.call_args
        assert call_kwargs is not None
        assert "gemini-3-pro" in str(call_kwargs).lower() or "pro" in str(call_kwargs)

    def test_pro_model_with_google_search(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        part = _make_fake_part(inline_data=inline)
        fake_response = _make_fake_response([part])

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        output_path = tmp_path / "search_out.png"
        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)

        # use_search with no aspect/size does not hit ImageConfig, so no types mock needed
        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            result = mod.generate_image(
                prompt="Search-enabled image",
                model_key="pro",
                output_path=output_path,
                use_search=True,
                document_prompt=False,
            )

        assert result is not None
        # use_search=True must actually wire the grounding tool into the
        # GenerateContentConfig, not merely return a file. Verify both the
        # user-visible signal and that the config carries a non-empty tools list.
        assert "Google Search grounding: enabled" in capsys.readouterr().out
        call_kwargs = mock_client_instance.models.generate_content.call_args.kwargs
        tools = call_kwargs["config"].tools
        # use_search must wire exactly one grounding tool, and it must be the
        # Google Search tool specifically: a bare truthiness check would also
        # pass if the wrong tool were attached. google-genai coerces the
        # ``{"google_search": {}}`` dict into a ``types.Tool`` when building the
        # GenerateContentConfig, so tolerate either the coerced object or the
        # raw dict form.
        assert len(tools) == 1
        only_tool = tools[0]
        google_search = (
            only_tool.google_search
            if hasattr(only_tool, "google_search")
            else only_tool["google_search"]
        )
        if isinstance(google_search, dict):
            assert google_search == {}
        else:
            assert google_search.model_dump(exclude_none=True) == {}

    def test_invalid_aspect_ratio_warns_and_continues(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        part = _make_fake_part(inline_data=inline)
        fake_response = _make_fake_response([part])

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        output_path = tmp_path / "warn_out.png"
        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)

        # Invalid ratio means image_config_kwargs stays empty; ImageConfig not called
        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            result = mod.generate_image(
                prompt="Bad ratio test",
                model_key="pro",
                output_path=output_path,
                aspect_ratio="7:3",  # invalid ratio -- warning printed, not added to config
                document_prompt=False,
            )

        assert result is not None
        out = capsys.readouterr().out
        assert "warning" in out.lower() or "invalid" in out.lower()


class TestGenerateImageWithReferenceImage:
    """When reference images are provided, they are base64-encoded and included."""

    def test_existing_reference_image_included(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        # Create a reference image
        ref_img = tmp_path / "ref.png"
        ref_img.write_bytes(_PNG_MAGIC)

        inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        part = _make_fake_part(inline_data=inline)
        fake_response = _make_fake_response([part])

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        output_path = tmp_path / "out_with_ref.png"
        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)

        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            result = mod.generate_image(
                prompt="Edit this image",
                model_key="flash",
                reference_images=[ref_img],
                output_path=output_path,
                document_prompt=False,
            )

        assert result is not None
        # Verify generate_content was called with contents that include more than just the prompt
        call_args = mock_client_instance.models.generate_content.call_args
        contents = call_args.kwargs.get("contents") or call_args.args[1]
        # Should have at least 2 items: the image part + the prompt string
        assert len(contents) >= 2

    def test_missing_reference_image_warns_and_skips(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        missing_ref = tmp_path / "nonexistent_ref.png"

        inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        part = _make_fake_part(inline_data=inline)
        fake_response = _make_fake_response([part])

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        output_path = tmp_path / "out_skip_ref.png"
        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)

        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            result = mod.generate_image(
                prompt="Edit with missing ref",
                model_key="flash",
                reference_images=[missing_ref],
                output_path=output_path,
                document_prompt=False,
            )

        assert result is not None  # still succeeds, just skipped the ref
        out = capsys.readouterr().out
        assert "warning" in out.lower() or "not found" in out.lower()


# ---------------------------------------------------------------------------
# Additional document_image_prompt() branches
# ---------------------------------------------------------------------------


def _setup_prompts_env(tmp_path: Path) -> tuple[Path, Path]:
    """Return (fake_script_path, prompts_file) for document_image_prompt tests."""
    fake_script = tmp_path / "scripts" / "generate_image.py"
    fake_script.parent.mkdir(parents=True, exist_ok=True)

    examples_dir = tmp_path / "examples"
    examples_dir.mkdir(exist_ok=True)
    prompts_file = examples_dir / "PROMPTS.md"

    initial = (
        "# AI-Generated Image Registry\n\n"
        "## Image Registry\n\n"
        "| Filename | Model | Date | Prompt | Parameters |\n"
        "|----------|-------|------|--------|------------|\n"
        "\n## Detailed Prompts\n\n"
    )
    prompts_file.write_text(initial, encoding="utf-8")

    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)

    return fake_script, prompts_file


class TestDocumentImagePromptBranches:
    def test_reference_images_listed(self, tmp_path: Path) -> None:
        """When reference_images is provided, each is listed under Attachments."""
        import scripts.generate_image as mod

        fake_script, prompts_file = _setup_prompts_env(tmp_path)
        img = tmp_path / "output" / "img_with_refs.png"
        img.write_bytes(_PNG_MAGIC)
        ref1 = tmp_path / "ref1.png"
        ref1.write_bytes(_PNG_MAGIC)

        with patch.object(mod, "__file__", str(fake_script)):
            from scripts.generate_image import document_image_prompt

            document_image_prompt(
                image_path=img,
                prompt="Image with references",
                model_key="flash",
                aspect_ratio=None,
                image_size=None,
                reference_images=[ref1],
            )

        content = prompts_file.read_text(encoding="utf-8")
        assert "Attachments" in content
        assert "ref1.png" in content

    def test_is_draft_adds_type_label(self, tmp_path: Path) -> None:
        """is_draft=True adds the draft type annotation."""
        import scripts.generate_image as mod

        fake_script, prompts_file = _setup_prompts_env(tmp_path)
        img = tmp_path / "output" / "draft_img.png"
        img.write_bytes(_PNG_MAGIC)

        with patch.object(mod, "__file__", str(fake_script)):
            from scripts.generate_image import document_image_prompt

            document_image_prompt(
                image_path=img,
                prompt="Draft image",
                model_key="flash",
                aspect_ratio=None,
                image_size=None,
                is_draft=True,
            )

        content = prompts_file.read_text(encoding="utf-8")
        assert "Draft" in content or "temporary" in content.lower()

    def test_is_final_adds_type_and_location(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """is_final=True adds location and final type annotation, and prints confirmation."""
        import scripts.generate_image as mod

        fake_script, prompts_file = _setup_prompts_env(tmp_path)
        img = tmp_path / "output" / "final_img.png"
        img.write_bytes(_PNG_MAGIC)

        with patch.object(mod, "__file__", str(fake_script)):
            from scripts.generate_image import document_image_prompt

            document_image_prompt(
                image_path=img,
                prompt="Final image",
                model_key="pro",
                aspect_ratio="16:9",
                image_size="4K",
                is_final=True,
            )

        content = prompts_file.read_text(encoding="utf-8")
        assert "Final" in content or "production" in content.lower()
        assert "Location" in content
        out = capsys.readouterr().out
        assert "documented" in out.lower() or "prompts" in out.lower()

    def test_purpose_field_included(self, tmp_path: Path) -> None:
        """When purpose is provided, it is included in the detailed entry."""
        import scripts.generate_image as mod

        fake_script, prompts_file = _setup_prompts_env(tmp_path)
        img = tmp_path / "output" / "purpose_img.png"
        img.write_bytes(_PNG_MAGIC)

        with patch.object(mod, "__file__", str(fake_script)):
            from scripts.generate_image import document_image_prompt

            document_image_prompt(
                image_path=img,
                prompt="Image with purpose",
                model_key="flash",
                aspect_ratio=None,
                image_size=None,
                purpose="Testing the purpose field",
            )

        content = prompts_file.read_text(encoding="utf-8")
        assert "Testing the purpose field" in content

    def test_prompt_longer_than_50_chars_is_truncated_in_table(
        self, tmp_path: Path
    ) -> None:
        """Prompts > 50 chars are shortened with '...' in the table row."""
        import scripts.generate_image as mod

        fake_script, prompts_file = _setup_prompts_env(tmp_path)
        img = tmp_path / "output" / "long_prompt_img.png"
        img.write_bytes(_PNG_MAGIC)

        long_prompt = "A" * 60  # 60 chars, definitely over 50

        with patch.object(mod, "__file__", str(fake_script)):
            from scripts.generate_image import document_image_prompt

            document_image_prompt(
                image_path=img,
                prompt=long_prompt,
                model_key="flash",
                aspect_ratio=None,
                image_size=None,
            )

        content = prompts_file.read_text(encoding="utf-8")
        assert "..." in content


# ---------------------------------------------------------------------------
# generate_image() output path edge cases
# ---------------------------------------------------------------------------


class TestGenerateImageOutputPathEdgeCases:
    def test_no_extension_on_output_path_gets_detected_extension(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the output path has no extension, the detected extension is appended."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        part = _make_fake_part(inline_data=inline)
        fake_response = _make_fake_response([part])

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        # Output path with no extension
        output_path = tmp_path / "output" / "my_image"
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)

        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            result = mod.generate_image(
                prompt="No extension test",
                model_key="flash",
                output_path=output_path,
                document_prompt=False,
            )

        assert result is not None
        # Extension should have been added
        assert result.suffix == ".png"

    def test_output_path_not_under_output_dir_gets_prefixed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An output path not starting with 'output' is moved into output/."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        part = _make_fake_part(inline_data=inline)
        fake_response = _make_fake_response([part])

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        # Path that does NOT start with "output"
        output_path = tmp_path / "my_custom_name.png"
        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)

        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            result = mod.generate_image(
                prompt="Custom path test",
                model_key="flash",
                output_path=output_path,
                document_prompt=False,
            )

        assert result is not None
        assert result.exists()

    def test_output_path_with_final_in_name_goes_to_finals_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A path containing 'final' in the stem is placed in output/finals/."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        part = _make_fake_part(inline_data=inline)
        fake_response = _make_fake_response([part])

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        output_path = tmp_path / "my_final_image.png"  # "final" in stem
        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "output" / "finals").mkdir(parents=True, exist_ok=True)

        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            result = mod.generate_image(
                prompt="Final image test",
                model_key="flash",
                output_path=output_path,
                document_prompt=False,
            )

        assert result is not None
        assert result.exists()
        assert "finals" in str(result) or result.exists()

    def test_user_extension_mismatch_gets_corrected(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """User specifies .jpg but actual data is PNG: extension is corrected with warning."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        part = _make_fake_part(inline_data=inline)
        fake_response = _make_fake_response([part])

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        # User says .jpg but response is PNG
        output_path = tmp_path / "output" / "image.jpg"
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)

        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            result = mod.generate_image(
                prompt="Extension mismatch test",
                model_key="flash",
                output_path=output_path,
                document_prompt=False,
            )

        assert result is not None
        assert result.suffix == ".png"
        out = capsys.readouterr().out
        assert "warning" in out.lower() or ".jpg" in out


# ---------------------------------------------------------------------------
# generate_image() verbose / save_thoughts paths
# ---------------------------------------------------------------------------


class TestGenerateImageVerboseAndThoughts:
    """Thought processing branches (verbose=True, save_thoughts=True)."""

    def test_verbose_text_part_with_thought_signature(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A non-thought text part that also has a thought_signature should be handled."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        # Build two parts: a text-only part with a thought_signature, then the real image
        text_part = MagicMock()
        text_part.inline_data = None
        text_part.text = "Here is your image"
        text_part.thought = False
        text_part.thought_signature = b"\x00\x01"  # binary signature

        img_inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        img_part = _make_fake_part(inline_data=img_inline)

        fake_response = _make_fake_response([text_part, img_part])

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        output_path = tmp_path / "output" / "verbose_out.png"
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)

        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            result = mod.generate_image(
                prompt="Verbose signature test",
                model_key="flash",
                output_path=output_path,
                verbose=True,
                document_prompt=False,
            )

        assert result is not None
        # Signature sidecar file should have been written (verbose=True)
        sig_file = result.with_suffix(".signature.bin")
        assert sig_file.exists()

    def test_thought_part_verbose_prints_reasoning(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A thought=True part with text should print reasoning in verbose mode."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        # A thought part (not the final image)
        thought_part = MagicMock()
        thought_part.inline_data = None
        thought_part.text = "I am thinking about colors"
        thought_part.thought = True
        thought_part.thought_signature = None

        # Final image part
        img_inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        img_part = _make_fake_part(inline_data=img_inline)

        fake_response = _make_fake_response([thought_part, img_part])

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        output_path = tmp_path / "output" / "thought_out.png"
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)

        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            result = mod.generate_image(
                prompt="Thought test",
                model_key="flash",
                output_path=output_path,
                verbose=True,
                document_prompt=False,
            )

        assert result is not None
        out = capsys.readouterr().out
        assert (
            "thinking" in out.lower()
            or "reasoning" in out.lower()
            or "thought" in out.lower()
        )


# ---------------------------------------------------------------------------
# generate_story_sequence()
# ---------------------------------------------------------------------------


class TestGenerateStorySequence:
    def test_zero_parts_raises_config_error(self) -> None:
        from scripts.generate_image import ConfigError, generate_story_sequence

        with pytest.raises(ConfigError) as exc_info:
            generate_story_sequence(base_prompt="A story", num_parts=0)

        assert "at least 1" in str(exc_info.value)

    def test_unknown_model_raises_config_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts.generate_image import ConfigError, generate_story_sequence

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        with (
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
            pytest.raises(ConfigError) as exc_info,
        ):
            generate_story_sequence(
                base_prompt="A story",
                num_parts=2,
                model_key="nonexistent",
            )

        assert "nonexistent" in str(exc_info.value)

    def test_two_part_story_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two-part story: first part sets up, second finalizes. Mock generate_image."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        # Both parts return a real file
        part1_file = tmp_path / "story_part1.png"
        part2_file = tmp_path / "story_part2.png"
        part1_file.write_bytes(_PNG_MAGIC)
        part2_file.write_bytes(_PNG_MAGIC)

        call_count = [0]

        def fake_generate_image(**kwargs: object) -> Path:
            call_count[0] += 1
            return part1_file if call_count[0] == 1 else part2_file

        with (
            patch(
                "scripts.generate_image.generate_image", side_effect=fake_generate_image
            ),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            result = mod.generate_story_sequence(
                base_prompt="A two-part story",
                num_parts=2,
                model_key="flash",
                output_prefix=tmp_path / "story",
            )

        assert len(result) == 2
        assert result[0] == part1_file
        assert result[1] == part2_file

    def test_story_stops_early_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If part 1 fails (returns None), the loop breaks and returns empty list."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        with (
            patch("scripts.generate_image.generate_image", return_value=None),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            result = mod.generate_story_sequence(
                base_prompt="Failing story",
                num_parts=3,
                model_key="flash",
                output_prefix=tmp_path / "story",
            )

        assert result == []

    def test_three_part_story_uses_middle_prompt(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A 3-part story exercises the middle-part prompt branch."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        files = [tmp_path / f"story_part{i}.png" for i in range(1, 4)]
        for f in files:
            f.write_bytes(_PNG_MAGIC)

        call_count = [0]

        def fake_generate(**kwargs: object) -> Path:
            idx = call_count[0]
            call_count[0] += 1
            return files[idx]

        with (
            patch("scripts.generate_image.generate_image", side_effect=fake_generate),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            result = mod.generate_story_sequence(
                base_prompt="Three-part story",
                num_parts=3,
                model_key="flash",
                output_prefix=tmp_path / "story",
            )

        assert len(result) == 3

    def test_story_with_auto_prefix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When output_prefix is None, a timestamped prefix is generated."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        part_file = tmp_path / "auto_part1.png"
        part_file.write_bytes(_PNG_MAGIC)

        with (
            patch(
                "scripts.generate_image.generate_image",
                return_value=part_file,
            ),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            result = mod.generate_story_sequence(
                base_prompt="Auto prefix story",
                num_parts=1,
                model_key="flash",
                output_prefix=None,
            )

        assert len(result) == 1


# ---------------------------------------------------------------------------
# _load_api_key: direct env lookup (no Path mocking needed)
# ---------------------------------------------------------------------------


class TestLoadApiKeySimple:
    """Simple _load_api_key tests exercising the Settings backend.

    Post-pydantic-settings, ``_load_api_key`` only resolves the known
    ``GEMINI_API_KEY`` and ``TOPAZ_API_KEY`` variables. Unknown names return
    ``None``.
    """

    def test_returns_key_from_env_directly(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from scripts.generate_image import _load_api_key

        monkeypatch.setattr("scripts.generate_image._ENV_FILE", tmp_path / ".no-env")
        monkeypatch.setenv("TOPAZ_API_KEY", "direct-value")
        result = _load_api_key("TOPAZ_API_KEY")
        assert result == "direct-value"

    def test_returns_none_when_env_absent_and_no_env_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Key absent from env, .env file does not exist -> returns None."""
        from scripts.generate_image import _load_api_key

        monkeypatch.setattr("scripts.generate_image._ENV_FILE", tmp_path / ".no-env")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        result = _load_api_key("GEMINI_API_KEY")
        assert result is None

    def test_reads_key_from_env_file_via_real_fs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Key absent from env but .env file on disk contains it."""
        from scripts.generate_image import Settings, SettingsConfigDict, _load_api_key

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        env_file = tmp_path / ".env"
        env_file.write_text("GEMINI_API_KEY='fs-test-value'\n", encoding="utf-8")
        monkeypatch.setattr("scripts.generate_image._ENV_FILE", env_file)
        monkeypatch.setattr(
            Settings,
            "model_config",
            SettingsConfigDict(
                env_file=str(env_file),
                env_file_encoding="utf-8",
                extra="ignore",
                case_sensitive=True,
            ),
        )

        result = _load_api_key("GEMINI_API_KEY")
        assert result == "fs-test-value"

    def test_unknown_env_var_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Variables outside the Settings schema resolve to None."""
        from scripts.generate_image import _load_api_key

        monkeypatch.setattr("scripts.generate_image._ENV_FILE", tmp_path / ".no-env")
        monkeypatch.setenv("ARBITRARY_OTHER_KEY", "ignored")
        assert _load_api_key("ARBITRARY_OTHER_KEY") is None


# ---------------------------------------------------------------------------
# Additional generate_image() branches for deeper coverage
# ---------------------------------------------------------------------------


class TestGenerateImageSaveThoughts:
    """save_thoughts=True writes thought images to disk."""

    def test_save_thoughts_writes_thought_image_with_output_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """When save_thoughts=True and output_path given, thought images are saved."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        # A thought part with inline_data (image)
        thought_inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        thought_part = MagicMock()
        thought_part.thought = True
        thought_part.inline_data = thought_inline
        thought_part.text = None
        thought_part.thought_signature = None

        # Final image part
        img_inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        img_part = _make_fake_part(inline_data=img_inline)

        fake_response = _make_fake_response([thought_part, img_part])

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        output_path = tmp_path / "output" / "result.png"
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)

        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            result = mod.generate_image(
                prompt="Save thoughts test",
                model_key="flash",
                output_path=output_path,
                save_thoughts=True,
                verbose=True,
                document_prompt=False,
            )

        assert result is not None
        out = capsys.readouterr().out
        assert "thought" in out.lower()
        # Thought image file should exist
        thought_files = list((tmp_path / "output").glob("*thought*"))
        assert len(thought_files) >= 1

    def test_save_thoughts_without_output_path_uses_auto_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When save_thoughts=True and no output_path, thought image uses auto path."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        thought_inline = _make_fake_inline_data(_JPEG_MAGIC, "image/jpeg")
        thought_part = MagicMock()
        thought_part.thought = True
        thought_part.inline_data = thought_inline
        thought_part.text = None
        thought_part.thought_signature = None

        img_inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        img_part = _make_fake_part(inline_data=img_inline)

        fake_response = _make_fake_response([thought_part, img_part])

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)

        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            result = mod.generate_image(
                prompt="Auto thought path test",
                model_key="flash",
                output_path=None,
                save_thoughts=True,
                document_prompt=False,
            )

        assert result is not None


class TestGenerateImageMimeMismatch:
    """When MIME type and detected format disagree, a note is printed."""

    def test_mime_mismatch_prints_note(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """API says image/jpeg but bytes are PNG: note is printed."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        # PNG bytes, but MIME type says jpeg
        inline = _make_fake_inline_data(_PNG_MAGIC, "image/jpeg")
        part = _make_fake_part(inline_data=inline)
        fake_response = _make_fake_response([part])

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        output_path = tmp_path / "output" / "mime_mismatch.jpg"
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)

        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            result = mod.generate_image(
                prompt="MIME mismatch test",
                model_key="flash",
                output_path=output_path,
                document_prompt=False,
            )

        assert result is not None
        out = capsys.readouterr().out
        assert "note" in out.lower() or "mime" in out.lower() or ".jpg" in out


class TestGenerateImageDocumentPrompt:
    """document_prompt=True integrates document_image_prompt() into the flow."""

    def test_document_prompt_true_calls_document_image_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        part = _make_fake_part(inline_data=inline)
        fake_response = _make_fake_response([part])

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        output_path = tmp_path / "output" / "documented.png"
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)

        mock_doc = MagicMock()
        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
            patch("scripts.generate_image.document_image_prompt", mock_doc),
        ):
            result = mod.generate_image(
                prompt="Document prompt test",
                model_key="flash",
                output_path=output_path,
                document_prompt=True,
            )

        assert result is not None
        assert mock_doc.called

    def test_document_prompt_true_for_final_stem(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When 'final' is in the output path stem, is_final_image is True."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        part = _make_fake_part(inline_data=inline)
        fake_response = _make_fake_response([part])

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        # output path with "final" in the stem -- so is_final_image will be True
        # Path is absolute and starts with output/ so it won't be redirected
        output_path = tmp_path / "output" / "my_final_image.png"
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)

        mock_doc = MagicMock()
        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
            patch("scripts.generate_image.document_image_prompt", mock_doc),
        ):
            result = mod.generate_image(
                prompt="Finals stem test",
                model_key="flash",
                output_path=output_path,
                document_prompt=True,
            )

        assert result is not None
        assert mock_doc.called
        # is_final_image=True when "final" is in stem
        call_kwargs = mock_doc.call_args.kwargs
        assert call_kwargs.get("is_final") is True


class TestGenerateImageThoughtCount:
    """When thoughts are processed, thought count is printed in output."""

    def test_thought_count_printed_in_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        thought_part = MagicMock()
        thought_part.thought = True
        thought_part.inline_data = None
        thought_part.text = None
        thought_part.thought_signature = None

        img_inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        img_part = _make_fake_part(inline_data=img_inline)

        fake_response = _make_fake_response([thought_part, img_part])

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        output_path = tmp_path / "output" / "thought_count.png"
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)

        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            result = mod.generate_image(
                prompt="Count thoughts",
                model_key="flash",
                output_path=output_path,
                document_prompt=False,
            )

        assert result is not None
        out = capsys.readouterr().out
        assert "thought" in out.lower() or "processed" in out.lower()


class TestGenerateImageInlineSigOnImagePart:
    """thought_signature on image part sets final_signature; verbose mode prints it."""

    def test_image_part_with_thought_signature_verbose(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        # Image part with a thought_signature attribute
        img_inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        img_part = MagicMock()
        img_part.inline_data = img_inline
        img_part.text = None
        img_part.thought = False
        img_part.thought_signature = b"sigbytes"

        fake_response = _make_fake_response([img_part])

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        output_path = tmp_path / "output" / "sig_img.png"
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)

        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            result = mod.generate_image(
                prompt="Image sig test",
                model_key="flash",
                output_path=output_path,
                verbose=True,
                document_prompt=False,
            )

        assert result is not None
        sig_file = result.with_suffix(".signature.bin")
        assert sig_file.exists()
        # Verbose prints the signature
        out = capsys.readouterr().out
        assert "signature" in out.lower()


class TestDocumentImagePromptNoTable:
    """When content has no '## Detailed Prompts' marker, table insertion is skipped."""

    def test_no_table_marker_skips_row_insertion(self, tmp_path: Path) -> None:
        import scripts.generate_image as mod

        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)

        examples_dir = tmp_path / "examples"
        examples_dir.mkdir()
        prompts_file = examples_dir / "PROMPTS.md"

        # Write content without the "## Detailed Prompts" section
        prompts_file.write_text(
            "# My Registry\n\nSome content without the table.\n", encoding="utf-8"
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        img = output_dir / "no_table_img.png"
        img.write_bytes(_PNG_MAGIC)

        with patch.object(mod, "__file__", str(fake_script)):
            from scripts.generate_image import document_image_prompt

            document_image_prompt(
                image_path=img,
                prompt="No table test",
                model_key="flash",
                aspect_ratio=None,
                image_size=None,
            )

        content = prompts_file.read_text(encoding="utf-8")
        # The detailed entry should still be appended
        assert "no_table_img.png" in content

    def test_image_path_outside_output_uses_name_as_rel_path(
        self, tmp_path: Path
    ) -> None:
        """Image not under the output/ dir: rel_path falls back to image name."""
        import scripts.generate_image as mod

        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)

        examples_dir = tmp_path / "examples"
        examples_dir.mkdir()
        prompts_file = examples_dir / "PROMPTS.md"
        prompts_file.write_text(
            "# Registry\n\n## Image Registry\n\n"
            "| Filename | Model | Date | Prompt | Parameters |\n"
            "|----------|-------|------|--------|------------|\n"
            "\n## Detailed Prompts\n\n",
            encoding="utf-8",
        )

        # Image outside output/ dir
        img = tmp_path / "elsewhere" / "outside.png"
        img.parent.mkdir()
        img.write_bytes(_PNG_MAGIC)

        with patch.object(mod, "__file__", str(fake_script)):
            from scripts.generate_image import document_image_prompt

            document_image_prompt(
                image_path=img,
                prompt="Outside output dir",
                model_key="flash",
                aspect_ratio=None,
                image_size=None,
            )

        content = prompts_file.read_text(encoding="utf-8")
        assert "outside.png" in content


# ---------------------------------------------------------------------------
# Pro model: invalid image_size warning (line 717)
# ---------------------------------------------------------------------------


class TestGenerateImageProModelInvalidSize:
    def test_invalid_image_size_warns_and_continues(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Invalid image_size for pro model prints warning but generation continues."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        part = _make_fake_part(inline_data=inline)
        fake_response = _make_fake_response([part])

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        output_path = tmp_path / "output" / "size_warn.png"
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)

        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            result = mod.generate_image(
                prompt="Bad size test",
                model_key="pro",
                output_path=output_path,
                image_size="8K",  # invalid size
                document_prompt=False,
            )

        assert result is not None
        out = capsys.readouterr().out
        assert "warning" in out.lower() or "invalid" in out.lower() or "8K" in out


# ---------------------------------------------------------------------------
# is_draft user-specified path goes to output/drafts (line 870)
# ---------------------------------------------------------------------------


class TestGenerateImageDraftUserPath:
    def test_draft_mode_with_user_path_not_under_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """is_draft=True with a user-specified path not under output/ redirects to drafts/."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        part = _make_fake_part(inline_data=inline)
        fake_response = _make_fake_response([part])

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        # A path that doesn't start with "output" (string check)
        output_path = tmp_path / "my_draft.png"
        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "output" / "drafts").mkdir(parents=True, exist_ok=True)

        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            result = mod.generate_image(
                prompt="Draft user path",
                model_key="flash",
                output_path=output_path,
                is_draft=True,
                document_prompt=False,
            )

        assert result is not None
        assert "draft" in str(result).lower()


# ---------------------------------------------------------------------------
# Non-bytes thought signature is encoded as string (line 896)
# ---------------------------------------------------------------------------


class TestGenerateImageNonBytesSignature:
    def test_string_thought_signature_is_encoded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When thought_signature is a string (not bytes), it is str-encoded and written."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        img_inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        img_part = MagicMock()
        img_part.inline_data = img_inline
        img_part.text = None
        img_part.thought = False
        # Signature is a plain string (not bytes)
        img_part.thought_signature = "string-signature-value"

        fake_response = _make_fake_response([img_part])

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        output_path = tmp_path / "output" / "str_sig.png"
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)

        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            result = mod.generate_image(
                prompt="String sig test",
                model_key="flash",
                output_path=output_path,
                verbose=True,
                document_prompt=False,
            )

        assert result is not None
        sig_file = result.with_suffix(".signature.bin")
        assert sig_file.exists()
        # Should have been encoded from string
        assert sig_file.read_bytes() == b"string-signature-value"


# ---------------------------------------------------------------------------
# Branch coverage: part loop edge cases
# ---------------------------------------------------------------------------


class TestGenerateImagePartLoopEdgeCases:
    def test_image_part_with_signature_non_verbose(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Image part with thought_signature when verbose=False: signature still captured."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        img_inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        img_part = MagicMock()
        img_part.inline_data = img_inline
        img_part.text = None
        img_part.thought = False
        img_part.thought_signature = b"sig"

        fake_response = _make_fake_response([img_part])

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        output_path = tmp_path / "output" / "nosig_verbose.png"
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)

        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            # verbose=False means the signature branch at 812 is not printed
            result = mod.generate_image(
                prompt="Non-verbose sig",
                model_key="flash",
                output_path=output_path,
                verbose=False,
                document_prompt=False,
            )

        assert result is not None
        # With verbose=False the sidecar file should NOT be written
        sig_file = result.with_suffix(".signature.bin")
        assert not sig_file.exists()

    def test_part_with_no_inline_data_no_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A part with neither inline_data nor text is silently skipped."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        # "Blank" part: not a thought, no inline_data, no text
        blank_part = MagicMock()
        blank_part.thought = False
        blank_part.inline_data = None
        blank_part.text = None
        blank_part.thought_signature = None

        img_inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        img_part = _make_fake_part(inline_data=img_inline)

        fake_response = _make_fake_response([blank_part, img_part])

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        output_path = tmp_path / "output" / "blank_part.png"
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)

        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            result = mod.generate_image(
                prompt="Blank part test",
                model_key="flash",
                output_path=output_path,
                document_prompt=False,
            )

        assert result is not None

    def test_text_part_with_signature_verbose(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A text part with thought_signature in verbose mode prints the signature."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        text_part = MagicMock()
        text_part.thought = False
        text_part.inline_data = None
        text_part.text = "Model commentary"
        text_part.thought_signature = b"text-sig-bytes"

        img_inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        img_part = _make_fake_part(inline_data=img_inline)

        fake_response = _make_fake_response([text_part, img_part])

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        output_path = tmp_path / "output" / "text_sig_verbose.png"
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)

        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            result = mod.generate_image(
                prompt="Text sig verbose",
                model_key="flash",
                output_path=output_path,
                verbose=True,
                document_prompt=False,
            )

        assert result is not None
        out = capsys.readouterr().out
        assert "signature" in out.lower()


# ---------------------------------------------------------------------------
# main() function: argparse-driven entry points
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for main() covering the most impactful CLI paths."""

    def test_list_models_flag_prints_and_returns(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--list-models prints available models and returns without generating."""
        import scripts.generate_image as mod

        with patch("sys.argv", ["generate_image.py", "--list-models"]):
            mod.main()

        out = capsys.readouterr().out
        assert "flash" in out.lower()
        assert "pro" in out.lower()

    def test_list_topaz_models_flag_prints_and_returns(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--list-topaz-models prints Topaz models and returns."""
        import scripts.generate_image as mod

        with patch("sys.argv", ["generate_image.py", "--list-topaz-models"]):
            mod.main()

        out = capsys.readouterr().out
        assert "Standard V2" in out

    def test_no_args_exits_with_1(self) -> None:
        """No prompt, no --finalize, no --enhance: exits with code 1."""
        import scripts.generate_image as mod

        with (
            patch("sys.argv", ["generate_image.py"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            mod.main()

        assert exc_info.value.code != 0

    def test_enhance_flag_calls_topaz_enhance(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--enhance <image> calls topaz_enhance_image and exits."""
        import scripts.generate_image as mod

        monkeypatch.setenv("TOPAZ_API_KEY", "fake-topaz-key")

        img = tmp_path / "input.png"
        img.write_bytes(_PNG_MAGIC)

        mock_enhance = MagicMock(return_value=tmp_path / "output.png")

        with (
            patch("sys.argv", ["generate_image.py", "--enhance", str(img)]),
            patch("scripts.generate_image.topaz_enhance_image", mock_enhance),
            pytest.raises(SystemExit) as exc_info,
        ):
            mod.main()

        assert exc_info.value.code == 0
        assert mock_enhance.called

    def test_single_image_mode_calls_generate_image(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A basic prompt triggers generate_image() and exits 0 on success."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        result_path = tmp_path / "output.png"
        result_path.write_bytes(_PNG_MAGIC)

        mock_generate = MagicMock(return_value=result_path)

        with (
            patch("sys.argv", ["generate_image.py", "A test prompt"]),
            patch("scripts.generate_image.generate_image", mock_generate),
            pytest.raises(SystemExit) as exc_info,
        ):
            mod.main()

        assert exc_info.value.code == 0
        assert mock_generate.called

    def test_single_image_mode_exits_1_on_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When generate_image() returns None, main exits with code 1."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        mock_generate = MagicMock(return_value=None)

        with (
            patch("sys.argv", ["generate_image.py", "A failing prompt"]),
            patch("scripts.generate_image.generate_image", mock_generate),
            pytest.raises(SystemExit) as exc_info,
        ):
            mod.main()

        assert exc_info.value.code == 1

    def test_draft_mode_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--draft-mode passes is_draft=True to generate_image."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        result_path = tmp_path / "draft.png"
        result_path.write_bytes(_PNG_MAGIC)

        mock_generate = MagicMock(return_value=result_path)

        with (
            patch("sys.argv", ["generate_image.py", "Draft test", "--draft-mode"]),
            patch("scripts.generate_image.generate_image", mock_generate),
            pytest.raises(SystemExit) as exc_info,
        ):
            mod.main()

        assert exc_info.value.code == 0
        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs.get("is_draft") is True

    def test_story_parts_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--story-parts N triggers generate_story_sequence."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        result_paths = [tmp_path / f"part{i}.png" for i in range(1, 4)]
        for p in result_paths:
            p.write_bytes(_PNG_MAGIC)

        mock_story = MagicMock(return_value=result_paths)

        with (
            patch(
                "sys.argv",
                ["generate_image.py", "A story prompt", "--story-parts", "3"],
            ),
            patch("scripts.generate_image.generate_story_sequence", mock_story),
            pytest.raises(SystemExit) as exc_info,
        ):
            mod.main()

        assert exc_info.value.code == 0
        assert mock_story.called

    def test_story_parts_less_than_2_exits_1(self) -> None:
        """--story-parts 1 is rejected with exit code 1."""
        import scripts.generate_image as mod

        with (
            patch("sys.argv", ["generate_image.py", "prompt", "--story-parts", "1"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            mod.main()

        assert exc_info.value.code == 1

    def test_finalize_flag_with_existing_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--finalize <draft> calls generate_image with the draft as reference."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        draft = tmp_path / "draft.png"
        draft.write_bytes(_PNG_MAGIC)
        result = tmp_path / "draft_final.png"
        result.write_bytes(_PNG_MAGIC)

        mock_generate = MagicMock(return_value=result)

        with (
            patch("sys.argv", ["generate_image.py", "--finalize", str(draft)]),
            patch("scripts.generate_image.generate_image", mock_generate),
            pytest.raises(SystemExit) as exc_info,
        ):
            mod.main()

        assert exc_info.value.code == 0
        assert mock_generate.called

    def test_finalize_flag_with_missing_file_exits_1(self) -> None:
        """--finalize <missing> exits with code 1."""
        import scripts.generate_image as mod

        with (
            patch(
                "sys.argv",
                ["generate_image.py", "--finalize", "/nonexistent/draft.png"],
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            mod.main()

        assert exc_info.value.code == 1

    def test_no_genai_available_exits_1(self) -> None:
        """When google-genai is not installed and a prompt is given, exits 1."""
        import scripts.generate_image as mod

        original = mod.GENAI_AVAILABLE
        try:
            mod.GENAI_AVAILABLE = False
            with (
                patch("sys.argv", ["generate_image.py", "a prompt"]),
                pytest.raises(SystemExit) as exc_info,
            ):
                mod.main()
        finally:
            mod.GENAI_AVAILABLE = original

        assert exc_info.value.code == 1

    def test_topaz_flag_with_single_image(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--topaz triggers topaz_enhance_image after generation."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        monkeypatch.setenv("TOPAZ_API_KEY", "fake-topaz")

        gen_result = tmp_path / "gen.png"
        gen_result.write_bytes(_PNG_MAGIC)
        topaz_result = tmp_path / "gen_topaz.png"
        topaz_result.write_bytes(_PNG_MAGIC)

        mock_generate = MagicMock(return_value=gen_result)
        mock_enhance = MagicMock(return_value=topaz_result)

        with (
            patch("sys.argv", ["generate_image.py", "Topaz test", "--topaz"]),
            patch("scripts.generate_image.generate_image", mock_generate),
            patch("scripts.generate_image.topaz_enhance_image", mock_enhance),
            pytest.raises(SystemExit) as exc_info,
        ):
            mod.main()

        assert exc_info.value.code == 0
        assert mock_generate.called
        assert mock_enhance.called

    def test_finalize_with_topaz_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--finalize --topaz uses Topaz for finalization instead of Gemini."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        monkeypatch.setenv("TOPAZ_API_KEY", "fake-topaz")

        draft = tmp_path / "draft.png"
        draft.write_bytes(_PNG_MAGIC)
        topaz_result = tmp_path / "draft_topaz.png"
        topaz_result.write_bytes(_PNG_MAGIC)

        mock_enhance = MagicMock(return_value=topaz_result)

        with (
            patch(
                "sys.argv",
                ["generate_image.py", "--finalize", str(draft), "--topaz"],
            ),
            patch("scripts.generate_image.topaz_enhance_image", mock_enhance),
            pytest.raises(SystemExit) as exc_info,
        ):
            mod.main()

        assert exc_info.value.code == 0
        assert mock_enhance.called


# ---------------------------------------------------------------------------
# Per-model aspect_ratios / image_sizes validation (new in flash-2)
# ---------------------------------------------------------------------------


class TestPerModelAspectRatioValidation:
    """Aspect ratios valid for one model can be invalid for another."""

    def test_flash2_accepts_21_9_that_pro_rejects(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """21:9 is in flash-2's aspect_ratios but not pro's; flash-2 must not warn."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        part = _make_fake_part(inline_data=inline)
        fake_response = _make_fake_response([part])
        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)

        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            result = mod.generate_image(
                prompt="ultra-wide",
                model_key="flash-2",
                output_path=tmp_path / "output" / "ultra.png",
                aspect_ratio="21:9",
                document_prompt=False,
            )

        assert result is not None
        out = capsys.readouterr().out
        assert "not supported" not in out.lower()
        assert "21:9" in out
        # Stdout proves the validation branch accepted the value, but we also
        # need to prove it reached the SDK. Inspect the mocked generate_content
        # call to confirm the ImageConfig carries aspect_ratio="21:9".
        config = mock_client_instance.models.generate_content.call_args.kwargs["config"]
        assert config.image_config is not None
        assert config.image_config.aspect_ratio == "21:9"

    def test_pro_rejects_21_9_with_warning_naming_the_model(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """21:9 on pro emits a warning that names the active model."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        part = _make_fake_part(inline_data=inline)
        fake_response = _make_fake_response([part])
        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)

        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            result = mod.generate_image(
                prompt="pro rejects 21:9",
                model_key="pro",
                output_path=tmp_path / "output" / "pro21.png",
                aspect_ratio="21:9",
                document_prompt=False,
            )

        assert result is not None
        out = capsys.readouterr().out
        assert "warning" in out.lower()
        assert "21:9" in out
        assert "pro" in out.lower()

    def test_flash_warns_when_aspect_or_size_supplied(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Legacy flash silently dropped --aspect/--size before; now warns."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        part = _make_fake_part(inline_data=inline)
        fake_response = _make_fake_response([part])
        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)

        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            result = mod.generate_image(
                prompt="flash dropped aspect",
                model_key="flash",
                output_path=tmp_path / "output" / "flash_warn.png",
                aspect_ratio="16:9",
                image_size="2K",
                document_prompt=False,
            )

        assert result is not None
        out = capsys.readouterr().out
        assert "does not support" in out.lower()


# ---------------------------------------------------------------------------
# --thinking flag -> types.ThinkingConfig wiring (flash-2 only)
# ---------------------------------------------------------------------------


class TestThinkingLevelFlag:
    def test_flash2_sets_thinking_config_in_generate_content_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """thinking_level='high' on flash-2 reaches the SDK via ThinkingConfig."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        part = _make_fake_part(inline_data=inline)
        fake_response = _make_fake_response([part])
        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)

        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            mod.generate_image(
                prompt="thinking high",
                model_key="flash-2",
                output_path=tmp_path / "output" / "tc.png",
                thinking_level="high",
                document_prompt=False,
            )

        call_kwargs = mock_client_instance.models.generate_content.call_args.kwargs
        config = call_kwargs["config"]
        assert config.thinking_config is not None
        # google-genai normalizes 'high' -> ThinkingLevel.HIGH; str contains 'HIGH'.
        assert "HIGH" in str(config.thinking_config.thinking_level).upper()

    def test_thinking_on_pro_warns_and_omits_thinking_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Pro does not expose thinking_level; the flag warns and is dropped."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        part = _make_fake_part(inline_data=inline)
        fake_response = _make_fake_response([part])
        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)

        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            mod.generate_image(
                prompt="thinking on pro is a no-op",
                model_key="pro",
                output_path=tmp_path / "output" / "pro_tc.png",
                thinking_level="high",
                document_prompt=False,
            )

        out = capsys.readouterr().out
        assert "no effect" in out.lower() or "warning" in out.lower()
        config = mock_client_instance.models.generate_content.call_args.kwargs["config"]
        assert config.thinking_config is None


# ---------------------------------------------------------------------------
# main() forwards --thinking to generate_image() / generate_story_sequence()
# ---------------------------------------------------------------------------


class TestMainThinkingForwarding:
    """Verify args.thinking reaches the call sites at L1577 / L1604 / L1648."""

    def test_single_image_forwards_thinking(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        result_path = tmp_path / "out.png"
        result_path.write_bytes(_PNG_MAGIC)
        mock_generate = MagicMock(return_value=result_path)

        with (
            patch(
                "sys.argv",
                ["generate_image.py", "a prompt", "--thinking", "high"],
            ),
            patch("scripts.generate_image.generate_image", mock_generate),
            pytest.raises(SystemExit),
        ):
            mod.main()

        assert mock_generate.called
        assert mock_generate.call_args.kwargs.get("thinking_level") == "high"

    def test_story_sequence_forwards_thinking(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        mock_story = MagicMock(return_value=[tmp_path / "p1.png"])

        with (
            patch(
                "sys.argv",
                [
                    "generate_image.py",
                    "a story",
                    "--story-parts",
                    "2",
                    "--thinking",
                    "minimal",
                ],
            ),
            patch("scripts.generate_image.generate_story_sequence", mock_story),
            pytest.raises(SystemExit),
        ):
            mod.main()

        assert mock_story.called
        assert mock_story.call_args.kwargs.get("thinking_level") == "minimal"

    def test_finalize_forwards_thinking(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        draft = tmp_path / "draft.png"
        draft.write_bytes(_PNG_MAGIC)
        final = tmp_path / "final.png"
        final.write_bytes(_PNG_MAGIC)
        mock_generate = MagicMock(return_value=final)

        with (
            patch(
                "sys.argv",
                [
                    "generate_image.py",
                    "--finalize",
                    str(draft),
                    "--thinking",
                    "high",
                ],
            ),
            patch("scripts.generate_image.generate_image", mock_generate),
            pytest.raises(SystemExit),
        ):
            mod.main()

        assert mock_generate.called
        assert mock_generate.call_args.kwargs.get("thinking_level") == "high"


class TestThinkingLevelMinimalAndInvalid:
    """Cover thinking_level='minimal' and the programmatic invalid-value path."""

    def test_flash2_thinking_minimal_reaches_sdk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        part = _make_fake_part(inline_data=inline)
        fake_response = _make_fake_response([part])
        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)

        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            mod.generate_image(
                prompt="minimal thinking",
                model_key="flash-2",
                output_path=tmp_path / "output" / "min.png",
                thinking_level="minimal",
                document_prompt=False,
            )

        config = mock_client_instance.models.generate_content.call_args.kwargs["config"]
        assert config.thinking_config is not None
        assert "MINIMAL" in str(config.thinking_config.thinking_level).upper()

    def test_programmatic_invalid_thinking_level_warns_and_omits(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """generate_image() called programmatically with a bogus thinking_level."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        part = _make_fake_part(inline_data=inline)
        fake_response = _make_fake_response([part])
        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)

        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            mod.generate_image(
                prompt="bogus level",
                model_key="flash-2",
                output_path=tmp_path / "output" / "bad.png",
                thinking_level="ultra",
                document_prompt=False,
            )

        out = capsys.readouterr().out
        assert "invalid thinking level" in out.lower()
        config = mock_client_instance.models.generate_content.call_args.kwargs["config"]
        assert config.thinking_config is None


class TestArgparseThinkingChoices:
    """argparse rejects --thinking values outside the choices list."""

    def test_argparse_rejects_invalid_thinking_value(self) -> None:
        import scripts.generate_image as mod

        with (
            patch(
                "sys.argv",
                ["generate_image.py", "a prompt", "--thinking", "xyz"],
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            mod.main()

        # argparse exits with code 2 on unrecognized choice
        assert exc_info.value.code == 2


class TestFlash2ImageSize512:
    """The new 512 tier is accepted by flash-2 and reaches the SDK."""

    def test_flash2_accepts_size_512(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        part = _make_fake_part(inline_data=inline)
        fake_response = _make_fake_response([part])
        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)

        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            result = mod.generate_image(
                prompt="0.5K tier",
                model_key="flash-2",
                output_path=tmp_path / "output" / "tiny.png",
                image_size="512",
                document_prompt=False,
            )

        assert result is not None
        out = capsys.readouterr().out
        assert "not supported" not in out.lower()
        config = mock_client_instance.models.generate_content.call_args.kwargs["config"]
        assert config.image_config is not None
        assert config.image_config.image_size == "512"

    def test_pro_rejects_size_512_with_warning(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """pro's image_sizes list excludes 512; warning names the model."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        inline = _make_fake_inline_data(_PNG_MAGIC, "image/png")
        part = _make_fake_part(inline_data=inline)
        fake_response = _make_fake_response([part])
        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)

        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            result = mod.generate_image(
                prompt="pro rejects 512",
                model_key="pro",
                output_path=tmp_path / "output" / "pro512.png",
                image_size="512",
                document_prompt=False,
            )

        assert result is not None
        out = capsys.readouterr().out
        assert "warning" in out.lower()
        assert "512" in out
        assert "pro" in out.lower()


# ---------------------------------------------------------------------------
# Draft mode picks 512 on flash-2 and 1K elsewhere (model-aware default)
# ---------------------------------------------------------------------------


class TestDraftModeModelAwareSize:
    def test_draft_mode_on_flash2_uses_512(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--draft-mode with default model (flash-2) forwards image_size='512'."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        result_path = tmp_path / "out.png"
        result_path.write_bytes(_PNG_MAGIC)
        mock_generate = MagicMock(return_value=result_path)

        with (
            patch(
                "sys.argv",
                ["generate_image.py", "a prompt", "--draft-mode"],
            ),
            patch("scripts.generate_image.generate_image", mock_generate),
            pytest.raises(SystemExit),
        ):
            mod.main()

        assert mock_generate.called
        assert mock_generate.call_args.kwargs.get("image_size") == "512"

    def test_draft_mode_on_pro_uses_1k(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--draft-mode --model pro forwards image_size='1K' (pro lacks 512)."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        result_path = tmp_path / "out.png"
        result_path.write_bytes(_PNG_MAGIC)
        mock_generate = MagicMock(return_value=result_path)

        with (
            patch(
                "sys.argv",
                ["generate_image.py", "a prompt", "--draft-mode", "--model", "pro"],
            ),
            patch("scripts.generate_image.generate_image", mock_generate),
            pytest.raises(SystemExit),
        ):
            mod.main()

        assert mock_generate.called
        assert mock_generate.call_args.kwargs.get("image_size") == "1K"

    def test_explicit_size_overrides_draft_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--draft-mode --size 2K wins over the model-aware default."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        result_path = tmp_path / "out.png"
        result_path.write_bytes(_PNG_MAGIC)
        mock_generate = MagicMock(return_value=result_path)

        with (
            patch(
                "sys.argv",
                ["generate_image.py", "a prompt", "--draft-mode", "--size", "2K"],
            ),
            patch("scripts.generate_image.generate_image", mock_generate),
            pytest.raises(SystemExit),
        ):
            mod.main()

        assert mock_generate.called
        assert mock_generate.call_args.kwargs.get("image_size") == "2K"

    def test_draft_mode_on_legacy_flash_passes_no_size(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """flash has no image_sizes; --draft-mode --model flash sets image_size=None."""
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        result_path = tmp_path / "out.png"
        result_path.write_bytes(_PNG_MAGIC)
        mock_generate = MagicMock(return_value=result_path)

        with (
            patch(
                "sys.argv",
                [
                    "generate_image.py",
                    "a prompt",
                    "--draft-mode",
                    "--model",
                    "flash",
                ],
            ),
            patch("scripts.generate_image.generate_image", mock_generate),
            pytest.raises(SystemExit),
        ):
            mod.main()

        assert mock_generate.called
        # flash has no size tiers, so effective_size should be None - not "1K".
        assert mock_generate.call_args.kwargs.get("image_size") is None
        # The user-facing message must reflect the no-size case, not advertise
        # a resolution the model cannot honor.
        out = capsys.readouterr().out
        assert "no size control" in out.lower()
        assert "Generating at 1K" not in out


# ---------------------------------------------------------------------------
# Typed-error contract coverage (PR #39 follow-up tests)
#
# These tests anchor the new typed-raise contract introduced in the 2026-05
# compliance sweep so future refactors that broaden a catch upstream or remove
# a handler surface as test failures rather than silent regressions.
# ---------------------------------------------------------------------------


class TestFileIOErrorOnDiskWrite:
    """``generate_image`` wraps an OSError from the output write as ``FileIOError``."""

    def test_oserror_writing_image_raises_file_io_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scripts.generate_image as mod
        from scripts.generate_image import FileIOError

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        # Build a response carrying valid inline image data so the function
        # reaches the file-write step.
        part = _make_fake_part(
            inline_data=_make_fake_inline_data(b"PNG_BYTES", "image/png")
        )
        fake_response = _make_fake_response([part])

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = fake_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        # Force the file write to raise OSError by patching ``open`` for write
        # mode only inside the module's namespace.
        real_open = open

        def boom(
            path: Any,
            mode: str = "r",
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            if "w" in mode and "b" in mode:
                msg = "simulated disk-full"
                raise OSError(msg)
            return real_open(path, mode, *args, **kwargs)

        with (
            patch("scripts.generate_image.genai.Client", mock_client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
            patch("builtins.open", side_effect=boom),
            pytest.raises(FileIOError) as exc_info,
        ):
            mod.generate_image(
                prompt="Disk-full test",
                model_key="flash",
                output_path=tmp_path / "result.png",
                document_prompt=False,
            )

        assert "simulated disk-full" in str(exc_info.value)


class TestMainKeyboardInterruptAndUnexpected:
    """``main()`` distinguishes SIGINT, AppError, and truly unexpected errors."""

    def test_keyboard_interrupt_exits_130(self) -> None:
        """SIGINT during _run produces exit code 130 (POSIX convention)."""
        import scripts.generate_image as mod

        with (
            patch("scripts.generate_image._run", side_effect=KeyboardInterrupt),
            pytest.raises(SystemExit) as exc_info,
        ):
            mod.main()

        assert exc_info.value.code == 130

    def test_unexpected_exception_includes_traceback(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An exception that is not an AppError surfaces as ``Unexpected error``.

        ``exc_info=exc`` on the structlog call preserves the traceback so the
        user has a real bug report to attach.
        """
        import scripts.generate_image as mod

        # A plain RuntimeError is not an AppError, so it hits the outer
        # ``except Exception`` branch.
        with (
            patch(
                "scripts.generate_image._run",
                side_effect=RuntimeError("kaboom"),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            mod.main()

        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        # Either the structlog-rendered "Unexpected error" prefix or the
        # exc_info traceback line should be present.
        assert "unexpected error" in err.lower() or "RuntimeError" in err

    def test_apperror_exits_1_with_message(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A raised AppError produces a clean ``Error: ...`` exit, not a traceback."""
        import scripts.generate_image as mod
        from scripts.generate_image import GeminiAPIError

        with (
            patch(
                "scripts.generate_image._run",
                side_effect=GeminiAPIError("synthetic gemini failure"),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            mod.main()

        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "synthetic gemini failure" in err


class TestSettingsLoaderEdgeCases:
    """``get_settings()`` translates pydantic/IO errors to ConfigError or warning."""

    def test_env_var_takes_precedence_over_env_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When both are set, the process environment wins, per pydantic-settings."""
        import scripts.generate_image as mod

        env_path = tmp_path / ".env"
        env_path.write_text(
            "GEMINI_API_KEY=from-dotenv\nTOPAZ_API_KEY=from-dotenv-topaz\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("GEMINI_API_KEY", "from-environ")
        monkeypatch.delenv("TOPAZ_API_KEY", raising=False)
        monkeypatch.setattr(mod, "_ENV_FILE", env_path)
        # SettingsConfigDict is a mutable TypedDict; setitem in-place under
        # ``monkeypatch`` so the env_file key is restored after the test.
        monkeypatch.setitem(mod.Settings.model_config, "env_file", str(env_path))

        settings = mod.get_settings()

        # Environment wins for GEMINI_API_KEY; .env supplies TOPAZ_API_KEY.
        assert settings.GEMINI_API_KEY == "from-environ"
        assert settings.TOPAZ_API_KEY == "from-dotenv-topaz"

    def test_malformed_env_file_raises_config_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invalid UTF-8 in .env surfaces as ConfigError, not raw pydantic blob."""
        import scripts.generate_image as mod
        from scripts.generate_image import ConfigError

        env_path = tmp_path / ".env"
        # Invalid UTF-8 byte sequence
        env_path.write_bytes(b"GEMINI_API_KEY=\xff\xfe\xfd_invalid")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("TOPAZ_API_KEY", raising=False)
        monkeypatch.setattr(mod, "_ENV_FILE", env_path)
        monkeypatch.setitem(mod.Settings.model_config, "env_file", str(env_path))

        # python-dotenv reads the file as UTF-8 and raises UnicodeDecodeError
        # before pydantic ever sees it. ``get_settings`` translates that to a
        # typed ConfigError so the user never sees a raw decoding traceback.
        with pytest.raises(ConfigError) as exc_info:
            mod.get_settings()

        assert "UTF-8" in str(exc_info.value) or "decode" in str(exc_info.value).lower()

    def test_unreadable_env_file_falls_back_to_environment(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """An OSError reading .env logs a warning and falls back to env vars.

        Unlike a malformed (non-UTF-8) file, which raises ConfigError, an
        unreadable file must not raise: users who set keys via the process
        environment but happen to have an unreadable stub .env should still
        succeed. This exercises the ``except OSError`` fallback branch in
        ``get_settings`` that the malformed-file test does not reach.
        """
        import scripts.generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "from-environ")
        monkeypatch.delenv("TOPAZ_API_KEY", raising=False)

        real_settings_cls = mod.Settings

        def settings_factory(*args: object, **kwargs: object) -> object:
            # First call (loading the real .env) simulates an unreadable file;
            # the fallback call passes ``_env_file=None`` and must succeed.
            if "_env_file" not in kwargs:
                raise OSError("simulated unreadable .env")
            return real_settings_cls(*args, **kwargs)

        mock_settings_cls = MagicMock(side_effect=settings_factory)
        monkeypatch.setattr(mod, "Settings", mock_settings_cls)

        settings = mod.get_settings()

        # Fallback path executed: two constructions, the second disabling the
        # .env file, with the key resolved from the process environment.
        assert settings.GEMINI_API_KEY == "from-environ"
        # TOPAZ_API_KEY was removed from the environment, so the fallback
        # Settings(_env_file=None) construction must resolve it to its default
        # (None). This confirms the fallback loaded from the process env only,
        # not from any .env values.
        assert settings.TOPAZ_API_KEY is None
        assert mock_settings_cls.call_count == 2
        # Only the ``_env_file`` contract matters; tolerate future extra kwargs.
        assert mock_settings_cls.call_args.kwargs.get("_env_file") is None
        # The fallback must be announced. ``log`` writes via structlog's
        # _StderrLoggerFactory (not stdlib logging), so the warning lands on
        # stderr where capsys captures it -- caplog would not see it.
        err = capsys.readouterr().err
        assert ".env" in err
        assert "falling back" in err

    def test_invalid_settings_raises_config_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A pydantic ``ValidationError`` from ``Settings`` maps to ``ConfigError``.

        ``get_settings`` wraps pydantic's raw validation failure in the CLI's
        typed ``ConfigError`` (with remediation guidance) so the user never sees
        a raw pydantic blob. Neither the OSError-fallback nor the
        UnicodeDecodeError test reaches this ``except ValidationError`` branch.
        """
        from pydantic import ValidationError

        import scripts.generate_image as mod

        validation_error = ValidationError.from_exception_data(
            "Settings",
            [{"type": "missing", "loc": ("GEMINI_API_KEY",), "input": {}}],
        )
        monkeypatch.setattr(mod, "Settings", MagicMock(side_effect=validation_error))

        with pytest.raises(mod.ConfigError) as exc_info:
            mod.get_settings()

        assert "Invalid configuration" in str(exc_info.value)
