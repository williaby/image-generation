"""Tests for the helpers and module boundaries introduced by the GCA-001
follow-on refactor.

Covers:
- ``load_image_bytes`` (the raw-bytes loader that replaced the base64
  encode/decode round-trip on the generation path).
- The re-export contract: config/image names resolve identically from
  ``scripts.generate_image`` and from the new ``scripts._config`` /
  ``scripts._images`` modules.
- ``_configure_logging`` idempotency (satisfies the RAD ``#VERIFY`` note that
  re-calling the logging configuration is safe).
- The Topaz polling helper's terminal behaviors (completed / failed / timeout).
"""

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _mock_response(
    status_code: int = 200,
    json_data: dict | None = None,
    content: bytes = b"",
    text: str = "",
) -> MagicMock:
    """Build an httpx.Response-like mock (mirrors test_topaz_enhance helper)."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.content = content
    resp.text = text
    resp.raise_for_status = MagicMock()  # no-op by default
    return resp


class TestLoadImageBytes:
    """load_image_bytes returns raw bytes + detected mime and enforces the cap."""

    def test_returns_raw_bytes_and_mime(self, tmp_path: Path) -> None:
        from scripts import generate_image as mod

        raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
        png = tmp_path / "ok.png"
        png.write_bytes(raw)

        data, mime = mod.load_image_bytes(png)

        assert data == raw  # raw bytes, not base64
        assert isinstance(data, bytes)
        assert mime == "image/png"

    def test_oversize_raises_file_io_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts import generate_image as mod

        monkeypatch.setattr(mod, "MAX_INPUT_IMAGE_BYTES", 16)
        big = tmp_path / "big.png"
        big.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)

        with pytest.raises(mod.FileIOError, match="exceeds limit"):
            mod.load_image_bytes(big)

    def test_base64_wrapper_matches_bytes_loader(self, tmp_path: Path) -> None:
        from scripts import generate_image as mod

        raw = b"\xff\xd8\xff\xe0" + b"\x00" * 12  # JPEG magic
        jpg = tmp_path / "x.jpg"
        jpg.write_bytes(raw)

        b_data, b_mime = mod.load_image_bytes(jpg)
        s_data, s_mime = mod.load_image_as_base64(jpg)

        assert s_mime == b_mime == "image/jpeg"
        assert base64.standard_b64decode(s_data) == b_data


class TestReexportContract:
    """Public config/image names are the same object via both import paths."""

    def test_exceptions_are_identical_objects(self) -> None:
        from scripts import _config, generate_image

        assert generate_image.ConfigError is _config.ConfigError
        assert generate_image.AppError is _config.AppError
        assert issubclass(_config.ConfigError, _config.AppError)

    def test_image_helpers_are_identical_objects(self) -> None:
        from scripts import _images, generate_image

        assert generate_image.detect_image_format is _images.detect_image_format
        assert generate_image.get_extension_for_mime is _images.get_extension_for_mime

    def test_model_registry_shared(self) -> None:
        from scripts import _config, generate_image

        assert generate_image.MODELS is _config.MODELS
        assert generate_image.TOPAZ_MODELS is _config.TOPAZ_MODELS


class TestConfigureLoggingIdempotent:
    """_configure_logging can be called repeatedly without error (RAD #VERIFY)."""

    def test_repeated_calls_are_safe(self, capsys: pytest.CaptureFixture[str]) -> None:
        from scripts import generate_image as mod

        mod._configure_logging(verbose=False)
        mod._configure_logging(verbose=True)
        mod._configure_logging(verbose=False)

        # Logging still emits to the captured stderr after reconfiguration.
        mod.log.error("sentinel-after-reconfigure")
        assert "sentinel-after-reconfigure" in capsys.readouterr().err


class TestPollTopazStatus:
    """_poll_topaz_status terminal behaviors."""

    @patch("scripts.generate_image.time.sleep")
    def test_completed_returns_none(self, mock_sleep: MagicMock) -> None:
        from scripts import generate_image as mod

        completed = _mock_response(json_data={"status": "Completed"})
        with patch(
            "scripts.generate_image.httpx.get", MagicMock(return_value=completed)
        ):
            result = mod._poll_topaz_status("pid", {"X-API-KEY": "k"}, verbose=False)
        assert result is None

    @patch("scripts.generate_image.time.sleep")
    def test_failed_status_raises(self, mock_sleep: MagicMock) -> None:
        from scripts import generate_image as mod

        failed = _mock_response(json_data={"status": "Failed"})
        with (
            patch("scripts.generate_image.httpx.get", MagicMock(return_value=failed)),
            pytest.raises(mod.TopazAPIError, match="failed"),
        ):
            mod._poll_topaz_status("pid", {"X-API-KEY": "k"}, verbose=False)

    @patch("scripts.generate_image.time.sleep")
    def test_timeout_exhausts_iterations(
        self, mock_sleep: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts import generate_image as mod

        monkeypatch.setattr(mod, "_TOPAZ_POLL_ITERATIONS", 3)
        processing = _mock_response(json_data={"status": "Processing"})
        with (
            patch(
                "scripts.generate_image.httpx.get",
                MagicMock(return_value=processing),
            ),
            pytest.raises(mod.TopazAPIError, match="did not complete"),
        ):
            mod._poll_topaz_status("pid", {"X-API-KEY": "k"}, verbose=False)
