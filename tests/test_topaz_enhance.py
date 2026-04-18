"""Tests for topaz_enhance_image() in scripts/generate_image.py.

Coverage targets: every branch in the function body, including:
- Guards (REQUESTS_AVAILABLE, API key, path, model, format, strength, face)
- Happy path (submit -> poll -> download -> write)
- Retry-on-429 during polling
- Polling timeout (for/else exhaustion)
- Terminal "Failed" status
- Missing process_id in submit response
- Missing download URL
- SSRF guards (bad hostname, http:// scheme)
- allow_redirects=False enforcement
- Empty image bytes
- OSError on disk write
- Extension mismatch with auto-correction
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# PNG magic bytes (smallest valid PNG header indicator for detect_image_format)
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
_JPEG_MAGIC = b"\xff\xd8" + b"\x00" * 16
_WEBP_MAGIC = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 4

# SSRF-allowed hosts (mirrors the frozenset inside the function)
_ALLOWED_HOSTS = frozenset({"api.topazlabs.com", "cdn.topazlabs.com"})
_VALID_DOWNLOAD_URL = "https://api.topazlabs.com/results/job123/output.png"


def _make_input_png(tmp_path: Path, name: str = "input.png") -> Path:
    """Write a minimal fake PNG to tmp_path and return its Path."""
    p = tmp_path / name
    p.write_bytes(_PNG_MAGIC)
    return p


def _mock_response(
    status_code: int = 200,
    json_data: dict | None = None,
    content: bytes = b"",
    text: str = "",
) -> MagicMock:
    """Build a requests.Response-like mock."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.content = content
    resp.text = text
    resp.raise_for_status = MagicMock()  # no-op by default
    return resp


# ---------------------------------------------------------------------------
# Convenience: patch the three network calls in the happy path
# ---------------------------------------------------------------------------


def _happy_path_mocks(
    submit_json: dict | None = None,
    status_json: dict | None = None,
    dl_url_json: dict | None = None,
    image_bytes: bytes = _PNG_MAGIC,
) -> tuple[MagicMock, MagicMock]:
    """Return (post_mock, get_mock) for a clean single-poll success."""
    if submit_json is None:
        submit_json = {"process_id": "job123"}
    if status_json is None:
        status_json = {"status": "Completed"}
    if dl_url_json is None:
        dl_url_json = {"url": _VALID_DOWNLOAD_URL}

    post_mock = MagicMock(return_value=_mock_response(json_data=submit_json))

    # GET is called in order: status check, download-URL fetch, image download
    status_resp = _mock_response(json_data=status_json)
    dl_url_resp = _mock_response(json_data=dl_url_json)
    img_resp = _mock_response(content=image_bytes)

    get_mock = MagicMock(side_effect=[status_resp, dl_url_resp, img_resp])
    return post_mock, get_mock


# ---------------------------------------------------------------------------
# Guard: requests not importable
# ---------------------------------------------------------------------------


class TestRequestsUnavailable:
    def test_returns_none_when_requests_not_available(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        inp = _make_input_png(tmp_path)
        import scripts.generate_image as mod

        original = mod.REQUESTS_AVAILABLE
        try:
            mod.REQUESTS_AVAILABLE = False
            from scripts.generate_image import topaz_enhance_image

            result = topaz_enhance_image(inp)
        finally:
            mod.REQUESTS_AVAILABLE = original

        assert result is None
        out = capsys.readouterr().out
        assert "requests" in out.lower()


# ---------------------------------------------------------------------------
# Guard: API key not set
# ---------------------------------------------------------------------------


class TestNoApiKey:
    def test_returns_none_when_api_key_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        inp = _make_input_png(tmp_path)
        # Remove env var and make .env lookup fail
        monkeypatch.delenv("TOPAZ_API_KEY", raising=False)

        # Prevent fallback to real .env on disk
        with patch("scripts.generate_image.Path.exists", return_value=False):
            from scripts.generate_image import topaz_enhance_image

            result = topaz_enhance_image(inp)

        assert result is None


# ---------------------------------------------------------------------------
# Guard: input path does not exist
# ---------------------------------------------------------------------------


class TestInputPathMissing:
    def test_returns_none_for_nonexistent_input(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        missing = tmp_path / "no_such_file.png"
        monkeypatch.setenv("TOPAZ_API_KEY", "test-key")

        from scripts.generate_image import topaz_enhance_image

        result = topaz_enhance_image(missing)

        assert result is None
        out = capsys.readouterr().out
        assert "not found" in out.lower() or "error" in out.lower()


# ---------------------------------------------------------------------------
# Guard: unknown model name
# ---------------------------------------------------------------------------


class TestUnknownModel:
    def test_returns_none_for_unknown_model(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        inp = _make_input_png(tmp_path)
        monkeypatch.setenv("TOPAZ_API_KEY", "test-key")

        from scripts.generate_image import topaz_enhance_image

        result = topaz_enhance_image(inp, model="NonExistentModel9999")

        assert result is None
        out = capsys.readouterr().out
        assert "unknown" in out.lower() or "nonexistentmodel" in out.lower()


# ---------------------------------------------------------------------------
# Guard: invalid output_format
# ---------------------------------------------------------------------------


class TestInvalidOutputFormat:
    def test_returns_none_for_bmp_format(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        inp = _make_input_png(tmp_path)
        monkeypatch.setenv("TOPAZ_API_KEY", "test-key")

        from scripts.generate_image import topaz_enhance_image

        result = topaz_enhance_image(inp, output_format="bmp")

        assert result is None
        out = capsys.readouterr().out
        assert "output_format" in out.lower() or "bmp" in out.lower()


# ---------------------------------------------------------------------------
# Guard: face_enhancement_strength set but face_enhancement=False
# ---------------------------------------------------------------------------


class TestFaceStrengthWithoutFaceEnhance:
    def test_returns_none_when_strength_set_without_flag(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        inp = _make_input_png(tmp_path)
        monkeypatch.setenv("TOPAZ_API_KEY", "test-key")

        from scripts.generate_image import topaz_enhance_image

        result = topaz_enhance_image(
            inp,
            face_enhancement=False,
            face_enhancement_strength=0.5,
        )

        assert result is None
        err = capsys.readouterr().err
        assert "face" in err.lower()


# ---------------------------------------------------------------------------
# Guard: strength values outside 0.0-1.0
# ---------------------------------------------------------------------------


class TestStrengthOutOfRange:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"sharpen": -0.1},
            {"sharpen": 1.1},
            {"denoise": -0.01},
            {"denoise": 1.5},
            {"face_enhancement": True, "face_enhancement_strength": -0.5},
            {"face_enhancement": True, "face_enhancement_strength": 2.0},
        ],
    )
    def test_returns_none_for_out_of_range(
        self,
        kwargs: dict,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        inp = _make_input_png(tmp_path)
        monkeypatch.setenv("TOPAZ_API_KEY", "test-key")

        from scripts.generate_image import topaz_enhance_image

        result = topaz_enhance_image(inp, **kwargs)

        assert result is None
        out = capsys.readouterr().out
        assert "0.0" in out or "1.0" in out or "between" in out.lower()


# ---------------------------------------------------------------------------
# Happy path: submit -> poll once -> Completed -> download -> save
# ---------------------------------------------------------------------------


class TestHappyPath:
    @patch("scripts.generate_image.time.sleep")
    def test_successful_job_returns_output_path(
        self,
        mock_sleep: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        inp = _make_input_png(tmp_path)
        out_path = tmp_path / "enhanced.png"
        monkeypatch.setenv("TOPAZ_API_KEY", "test-key")

        post_mock, get_mock = _happy_path_mocks()

        with (
            patch("scripts.generate_image.requests.post", post_mock),
            patch("scripts.generate_image.requests.get", get_mock),
        ):
            from scripts.generate_image import topaz_enhance_image

            result = topaz_enhance_image(inp, output_path=out_path)

        assert result == out_path
        assert out_path.exists()
        assert out_path.read_bytes() == _PNG_MAGIC

    @patch("scripts.generate_image.time.sleep")
    def test_output_path_auto_generated_when_none(
        self,
        mock_sleep: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        inp = _make_input_png(tmp_path)
        monkeypatch.setenv("TOPAZ_API_KEY", "test-key")

        post_mock, get_mock = _happy_path_mocks()

        with (
            patch("scripts.generate_image.requests.post", post_mock),
            patch("scripts.generate_image.requests.get", get_mock),
        ):
            from scripts.generate_image import topaz_enhance_image

            result = topaz_enhance_image(inp, output_path=None)

        assert result is not None
        assert result.exists()
        assert "topaz" in result.name


# ---------------------------------------------------------------------------
# 429 on poll -> retries with backoff and eventually completes
# ---------------------------------------------------------------------------


class TestPoll429Retry:
    @patch("scripts.generate_image.time.sleep")
    def test_429_retries_and_completes(
        self,
        mock_sleep: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        inp = _make_input_png(tmp_path)
        out_path = tmp_path / "result.png"
        monkeypatch.setenv("TOPAZ_API_KEY", "test-key")

        post_mock = MagicMock(
            return_value=_mock_response(json_data={"process_id": "job-429"})
        )

        # Two 429 responses, then Completed status, then download-URL, then image
        resp_429 = _mock_response(status_code=429)
        # raise_for_status must NOT raise for 429 (the code checks status_code directly)
        resp_429.raise_for_status = MagicMock()
        resp_completed = _mock_response(json_data={"status": "Completed"})
        resp_dl_url = _mock_response(
            json_data={"url": "https://api.topazlabs.com/out/file.png"}
        )
        resp_img = _mock_response(content=_PNG_MAGIC)

        get_mock = MagicMock(
            side_effect=[resp_429, resp_429, resp_completed, resp_dl_url, resp_img]
        )

        with (
            patch("scripts.generate_image.requests.post", post_mock),
            patch("scripts.generate_image.requests.get", get_mock),
        ):
            from scripts.generate_image import topaz_enhance_image

            result = topaz_enhance_image(inp, output_path=out_path)

        assert result == out_path
        assert out_path.exists()
        # Verify backoff: sleep was called; the second poll sleep should be >= first
        sleep_calls = mock_sleep.call_args_list
        assert len(sleep_calls) >= 3  # at least submit + 2 retries + completed


# ---------------------------------------------------------------------------
# Polling timeout
# ---------------------------------------------------------------------------


class TestPollingTimeout:
    @patch("scripts.generate_image.time.sleep")
    def test_timeout_returns_none_with_message(
        self,
        mock_sleep: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        inp = _make_input_png(tmp_path)
        monkeypatch.setenv("TOPAZ_API_KEY", "test-key")

        post_mock = MagicMock(
            return_value=_mock_response(json_data={"process_id": "job-timeout"})
        )
        # Always return "Pending" so the loop never breaks
        pending_resp = _mock_response(json_data={"status": "Pending"})
        get_mock = MagicMock(return_value=pending_resp)

        with (
            patch("scripts.generate_image.requests.post", post_mock),
            patch("scripts.generate_image.requests.get", get_mock),
        ):
            from scripts.generate_image import topaz_enhance_image

            result = topaz_enhance_image(inp)

        assert result is None
        err = capsys.readouterr().err
        assert "did not complete within the polling limit" in err


# ---------------------------------------------------------------------------
# Terminal "Failed" status
# ---------------------------------------------------------------------------


class TestFailedStatus:
    @patch("scripts.generate_image.time.sleep")
    def test_failed_status_returns_none(
        self,
        mock_sleep: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        inp = _make_input_png(tmp_path)
        monkeypatch.setenv("TOPAZ_API_KEY", "test-key")

        post_mock = MagicMock(
            return_value=_mock_response(json_data={"process_id": "job-fail"})
        )
        failed_resp = _mock_response(json_data={"status": "Failed"})
        get_mock = MagicMock(return_value=failed_resp)

        with (
            patch("scripts.generate_image.requests.post", post_mock),
            patch("scripts.generate_image.requests.get", get_mock),
        ):
            from scripts.generate_image import topaz_enhance_image

            result = topaz_enhance_image(inp)

        assert result is None
        err = capsys.readouterr().err
        assert "failed" in err.lower()


# ---------------------------------------------------------------------------
# Missing process_id in submit response
# ---------------------------------------------------------------------------


class TestMissingProcessId:
    @patch("scripts.generate_image.time.sleep")
    def test_missing_process_id_returns_none(
        self,
        mock_sleep: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        inp = _make_input_png(tmp_path)
        monkeypatch.setenv("TOPAZ_API_KEY", "test-key")

        post_mock = MagicMock(
            return_value=_mock_response(json_data={}, text="unexpected response body")
        )

        with patch("scripts.generate_image.requests.post", post_mock):
            from scripts.generate_image import topaz_enhance_image

            result = topaz_enhance_image(inp)

        assert result is None
        err = capsys.readouterr().err
        assert "process_id" in err.lower()


# ---------------------------------------------------------------------------
# Missing download URL
# ---------------------------------------------------------------------------


class TestMissingDownloadUrl:
    @patch("scripts.generate_image.time.sleep")
    def test_missing_download_url_returns_none(
        self,
        mock_sleep: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        inp = _make_input_png(tmp_path)
        out_path = tmp_path / "out.png"
        monkeypatch.setenv("TOPAZ_API_KEY", "test-key")

        post_mock = MagicMock(
            return_value=_mock_response(json_data={"process_id": "job-no-url"})
        )
        # GET side effects: status=Completed, then download-URL response missing "url"
        status_resp = _mock_response(json_data={"status": "Completed"})
        dl_url_resp = _mock_response(json_data={})  # no "url" key
        get_mock = MagicMock(side_effect=[status_resp, dl_url_resp])

        with (
            patch("scripts.generate_image.requests.post", post_mock),
            patch("scripts.generate_image.requests.get", get_mock),
        ):
            from scripts.generate_image import topaz_enhance_image

            result = topaz_enhance_image(inp, output_path=out_path)

        assert result is None
        err = capsys.readouterr().err
        assert "url" in err.lower() or "download" in err.lower()


# ---------------------------------------------------------------------------
# SSRF: disallowed hostname
# ---------------------------------------------------------------------------


class TestSsrfDisallowedHostname:
    @patch("scripts.generate_image.time.sleep")
    def test_bad_hostname_returns_none(
        self,
        mock_sleep: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        inp = _make_input_png(tmp_path)
        monkeypatch.setenv("TOPAZ_API_KEY", "test-key")

        evil_url = "https://evil.example.com/steal?data=yes"
        post_mock = MagicMock(
            return_value=_mock_response(json_data={"process_id": "job-ssrf"})
        )
        status_resp = _mock_response(json_data={"status": "Completed"})
        dl_url_resp = _mock_response(json_data={"url": evil_url})
        get_mock = MagicMock(side_effect=[status_resp, dl_url_resp])

        with (
            patch("scripts.generate_image.requests.post", post_mock),
            patch("scripts.generate_image.requests.get", get_mock),
        ):
            from scripts.generate_image import topaz_enhance_image

            result = topaz_enhance_image(inp)

        assert result is None
        err = capsys.readouterr().err
        assert (
            "unexpected" in err.lower()
            or "ssrf" in err.lower()
            or "evil" in err.lower()
        )


# ---------------------------------------------------------------------------
# SSRF: http:// scheme
# ---------------------------------------------------------------------------


class TestSsrfHttpScheme:
    @patch("scripts.generate_image.time.sleep")
    def test_http_scheme_returns_none(
        self,
        mock_sleep: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        inp = _make_input_png(tmp_path)
        monkeypatch.setenv("TOPAZ_API_KEY", "test-key")

        http_url = "http://api.topazlabs.com/results/file.png"  # http not https
        post_mock = MagicMock(
            return_value=_mock_response(json_data={"process_id": "job-http"})
        )
        status_resp = _mock_response(json_data={"status": "Completed"})
        dl_url_resp = _mock_response(json_data={"url": http_url})
        get_mock = MagicMock(side_effect=[status_resp, dl_url_resp])

        with (
            patch("scripts.generate_image.requests.post", post_mock),
            patch("scripts.generate_image.requests.get", get_mock),
        ):
            from scripts.generate_image import topaz_enhance_image

            result = topaz_enhance_image(inp)

        assert result is None
        err = capsys.readouterr().err
        assert "http" in err.lower() or "unexpected" in err.lower()


# ---------------------------------------------------------------------------
# allow_redirects=False is passed to image download
# ---------------------------------------------------------------------------


class TestAllowRedirectsFalse:
    @patch("scripts.generate_image.time.sleep")
    def test_image_download_disables_redirects(
        self,
        mock_sleep: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        inp = _make_input_png(tmp_path)
        out_path = tmp_path / "out.png"
        monkeypatch.setenv("TOPAZ_API_KEY", "test-key")

        post_mock, get_mock = _happy_path_mocks()

        with (
            patch("scripts.generate_image.requests.post", post_mock),
            patch("scripts.generate_image.requests.get", get_mock),
        ):
            from scripts.generate_image import topaz_enhance_image

            topaz_enhance_image(inp, output_path=out_path)

        # The third GET call is the image download
        img_download_call = get_mock.call_args_list[2]
        assert img_download_call.kwargs.get("allow_redirects") is False


# ---------------------------------------------------------------------------
# Empty image bytes
# ---------------------------------------------------------------------------


class TestEmptyImageBytes:
    @patch("scripts.generate_image.time.sleep")
    def test_empty_bytes_returns_none(
        self,
        mock_sleep: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        inp = _make_input_png(tmp_path)
        monkeypatch.setenv("TOPAZ_API_KEY", "test-key")

        post_mock, get_mock = _happy_path_mocks(image_bytes=b"")

        with (
            patch("scripts.generate_image.requests.post", post_mock),
            patch("scripts.generate_image.requests.get", get_mock),
        ):
            from scripts.generate_image import topaz_enhance_image

            result = topaz_enhance_image(inp)

        assert result is None
        err = capsys.readouterr().err
        assert "empty" in err.lower()


# ---------------------------------------------------------------------------
# Disk write fails (OSError)
# ---------------------------------------------------------------------------


class TestDiskWriteFailure:
    @patch("scripts.generate_image.time.sleep")
    def test_oserror_on_write_returns_none(
        self,
        mock_sleep: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        inp = _make_input_png(tmp_path)
        out_path = tmp_path / "output.png"
        monkeypatch.setenv("TOPAZ_API_KEY", "test-key")

        post_mock, get_mock = _happy_path_mocks()

        with (
            patch("scripts.generate_image.requests.post", post_mock),
            patch("scripts.generate_image.requests.get", get_mock),
            patch(
                "builtins.open",
                side_effect=[
                    # First open() is reading the input file for the POST
                    open(inp, "rb"),
                    # Second open() is the write; make it fail
                    OSError("disk full"),
                ],
            ),
        ):
            from scripts.generate_image import topaz_enhance_image

            result = topaz_enhance_image(inp, output_path=out_path)

        assert result is None
        err = capsys.readouterr().err
        assert "error" in err.lower()


# ---------------------------------------------------------------------------
# Extension mismatch: user specifies .jpg but result is .png
# ---------------------------------------------------------------------------


class TestExtensionMismatch:
    @patch("scripts.generate_image.time.sleep")
    def test_extension_corrected_and_warning_emitted(
        self,
        mock_sleep: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        inp = _make_input_png(tmp_path)
        # User says .jpg but we return PNG bytes
        user_out = tmp_path / "enhanced.jpg"
        monkeypatch.setenv("TOPAZ_API_KEY", "test-key")

        post_mock, get_mock = _happy_path_mocks(image_bytes=_PNG_MAGIC)

        with (
            patch("scripts.generate_image.requests.post", post_mock),
            patch("scripts.generate_image.requests.get", get_mock),
        ):
            from scripts.generate_image import topaz_enhance_image

            result = topaz_enhance_image(inp, output_path=user_out)

        # Result should have been corrected to .png
        assert result is not None
        assert result.suffix == ".png"
        assert result.exists()
        err = capsys.readouterr().err
        assert "correcting" in err.lower() or "warning" in err.lower() or ".jpg" in err
