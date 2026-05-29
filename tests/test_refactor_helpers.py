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

import pytest


class TestLoadImageBytes:
    """load_image_bytes returns raw bytes + detected mime and enforces the cap."""

    def test_returns_raw_bytes_and_mime(self, tmp_path):
        from scripts import generate_image as mod

        raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
        png = tmp_path / "ok.png"
        png.write_bytes(raw)

        data, mime = mod.load_image_bytes(png)

        assert data == raw  # raw bytes, not base64
        assert isinstance(data, bytes)
        assert mime == "image/png"

    def test_oversize_raises_value_error(self, tmp_path, monkeypatch):
        from scripts import generate_image as mod

        monkeypatch.setattr(mod, "MAX_INPUT_IMAGE_BYTES", 16)
        big = tmp_path / "big.png"
        big.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)

        with pytest.raises(ValueError, match="exceeds limit"):
            mod.load_image_bytes(big)

    def test_base64_wrapper_matches_bytes_loader(self, tmp_path):
        import base64

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

    def test_exceptions_are_identical_objects(self):
        from scripts import _config, generate_image

        assert generate_image.ConfigError is _config.ConfigError
        assert generate_image.AppError is _config.AppError
        assert issubclass(_config.ConfigError, _config.AppError)

    def test_image_helpers_are_identical_objects(self):
        from scripts import _images, generate_image

        assert generate_image.detect_image_format is _images.detect_image_format
        assert generate_image.get_extension_for_mime is _images.get_extension_for_mime

    def test_model_registry_shared(self):
        from scripts import _config, generate_image

        assert generate_image.MODELS is _config.MODELS
        assert generate_image.TOPAZ_MODELS is _config.TOPAZ_MODELS


class TestConfigureLoggingIdempotent:
    """_configure_logging can be called repeatedly without error (RAD #VERIFY)."""

    def test_repeated_calls_are_safe(self, capsys):
        from scripts import generate_image as mod

        mod._configure_logging(verbose=False)
        mod._configure_logging(verbose=True)
        mod._configure_logging(verbose=False)

        # Logging still emits to the captured stderr after reconfiguration.
        mod.log.error("sentinel-after-reconfigure")
        assert "sentinel-after-reconfigure" in capsys.readouterr().err


class TestPollTopazStatus:
    """_poll_topaz_status terminal behaviors."""

    def _resp(self, *, status_code=200, payload=None):
        class _R:
            def __init__(self, code, data):
                self.status_code = code
                self._data = data

            def raise_for_status(self):
                if self.status_code >= 400:
                    import httpx

                    raise httpx.HTTPStatusError("err", request=None, response=None)

            def json(self):
                return self._data

        return _R(status_code, payload)

    def test_completed_returns_none(self, monkeypatch):
        from scripts import generate_image as mod

        completed = self._resp(payload={"status": "Completed"})
        monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
        monkeypatch.setattr(mod.httpx, "get", lambda *_a, **_k: completed)
        # returns without raising
        assert mod._poll_topaz_status("pid", {"X-API-KEY": "k"}, verbose=False) is None

    def test_failed_status_raises(self, monkeypatch):
        from scripts import generate_image as mod

        monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
        monkeypatch.setattr(
            mod.httpx,
            "get",
            lambda *_a, **_k: self._resp(payload={"status": "Failed"}),
        )
        with pytest.raises(mod.TopazAPIError, match="failed"):
            mod._poll_topaz_status("pid", {"X-API-KEY": "k"}, verbose=False)

    def test_timeout_exhausts_iterations(self, monkeypatch):
        from scripts import generate_image as mod

        monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
        monkeypatch.setattr(mod, "_TOPAZ_POLL_ITERATIONS", 3)
        monkeypatch.setattr(
            mod.httpx,
            "get",
            lambda *_a, **_k: self._resp(payload={"status": "Processing"}),
        )
        with pytest.raises(mod.TopazAPIError, match="did not complete"):
            mod._poll_topaz_status("pid", {"X-API-KEY": "k"}, verbose=False)
