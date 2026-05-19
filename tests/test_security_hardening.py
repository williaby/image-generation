"""Tests for the security hardening introduced in PR #17.

Covers:
- MAX_INPUT_IMAGE_BYTES enforcement in load_image_as_base64 and topaz_enhance_image
- stat() OSError surface in topaz_enhance_image
- Markdown sanitization in document_image_prompt (table escaping + indented block)
- Random-token filename property (regression prevention)

Style follows tests/test_generate_image.py: unittest.mock + tmp_path + monkeypatch.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# MAX_INPUT_IMAGE_BYTES enforcement
# ---------------------------------------------------------------------------


class TestLoadImageAsBase64SizeLimit:
    """load_image_as_base64 must reject inputs exceeding MAX_INPUT_IMAGE_BYTES."""

    def test_oversize_reference_image_raises_value_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts import generate_image as mod

        monkeypatch.setattr(mod, "MAX_INPUT_IMAGE_BYTES", 16)
        large = tmp_path / "huge.png"
        large.write_bytes(b"x" * 32)

        with pytest.raises(ValueError, match="exceeds limit"):
            mod.load_image_as_base64(large)

    def test_at_limit_boundary_does_not_raise(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Boundary is strict > so a file exactly equal to the limit must pass.
        from scripts import generate_image as mod

        monkeypatch.setattr(mod, "MAX_INPUT_IMAGE_BYTES", 16)
        png = tmp_path / "ok.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)  # 16 bytes, valid PNG

        data, mime = mod.load_image_as_base64(png)
        assert isinstance(data, str)
        assert mime == "image/png"


class TestTopazEnhanceSizeLimit:
    """topaz_enhance_image raises FileIOError (typed) on oversize / unreadable input."""

    def test_oversize_input_raises_file_io_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from scripts import generate_image as mod
        from scripts.generate_image import FileIOError

        monkeypatch.setenv("TOPAZ_API_KEY", "fake-key")
        monkeypatch.setattr(mod, "MAX_INPUT_IMAGE_BYTES", 16)
        large = tmp_path / "huge.png"
        large.write_bytes(b"x" * 32)

        with pytest.raises(FileIOError) as excinfo:
            mod.topaz_enhance_image(input_path=large, model="Standard V2")

        assert "exceeds limit" in str(excinfo.value)
        assert str(large) in str(excinfo.value)

    def test_stat_oserror_raises_file_io_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from scripts import generate_image as mod
        from scripts.generate_image import FileIOError

        monkeypatch.setenv("TOPAZ_API_KEY", "fake-key")
        # Disable .env discovery so pydantic-settings does not stat the project
        # root .env file during Settings construction (the global Path.stat
        # patch below would otherwise mis-fire there).
        monkeypatch.setattr(mod, "_ENV_FILE", tmp_path / ".no-env")

        target = tmp_path / "broken.png"
        target.write_bytes(b"\x89PNG")

        # stat() raises a non-FileNotFound OSError (e.g., EACCES on the parent
        # mount): the OSError handler now wraps it as FileIOError. Scope the
        # patch to the target path's resolved location so unrelated stat calls
        # (pydantic-settings .env discovery, output dirs) continue to work.
        target_resolved = target.resolve()
        real_stat = Path.stat

        def selective_stat(self: Path, *args: object, **kwargs: object) -> object:
            if self == target_resolved:
                raise PermissionError("simulated EACCES")
            return real_stat(self, *args, **kwargs)  # type: ignore[misc]

        with (
            patch.object(Path, "stat", selective_stat),
            pytest.raises(FileIOError) as excinfo,
        ):
            mod.topaz_enhance_image(input_path=target, model="Standard V2")

        assert "Cannot stat" in str(excinfo.value)

    def test_missing_input_raises_file_io_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The FileNotFoundError branch is now the canonical "missing file"
        # path; the contract is now a typed FileIOError, not a None return.
        from scripts import generate_image as mod
        from scripts.generate_image import FileIOError

        monkeypatch.setenv("TOPAZ_API_KEY", "fake-key")
        missing = tmp_path / "does-not-exist.png"

        with pytest.raises(FileIOError) as excinfo:
            mod.topaz_enhance_image(input_path=missing, model="Standard V2")

        assert "not found" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# document_image_prompt sanitization
# ---------------------------------------------------------------------------


def _prompts_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set up a temp examples/PROMPTS.md and return the path."""
    fake_script = tmp_path / "scripts" / "generate_image.py"
    fake_script.parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "examples").mkdir(parents=True, exist_ok=True)
    (tmp_path / "output").mkdir(parents=True, exist_ok=True)

    from scripts import generate_image as mod

    monkeypatch.setattr(mod, "__file__", str(fake_script))
    return tmp_path / "examples" / "PROMPTS.md"


def _run_document_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    prompt: str,
) -> str:
    from scripts import generate_image as mod

    prompts_file = _prompts_file(tmp_path, monkeypatch)
    image = tmp_path / "output" / "img.png"
    image.write_bytes(b"\x89PNG")
    mod.document_image_prompt(
        image_path=image,
        prompt=prompt,
        model_key="flash",
        aspect_ratio="1:1",
        image_size="2K",
    )
    return prompts_file.read_text(encoding="utf-8")


class TestDocumentImagePromptTableEscaping:
    """The table row must not break on user-supplied pipes, backslashes, newlines."""

    def test_pipe_in_prompt_is_escaped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        body = _run_document_prompt(tmp_path, monkeypatch, prompt="foo | bar | baz")
        # The table row contains the prompt cell; the pipes must be escaped.
        table_row = next(
            line for line in body.splitlines() if "foo" in line and line.startswith("|")
        )
        assert r"foo \| bar \| baz" in table_row

    def test_newline_in_prompt_collapsed_to_space(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        body = _run_document_prompt(tmp_path, monkeypatch, prompt="line1\nline2")
        table_row = next(
            line
            for line in body.splitlines()
            if "line1" in line and line.startswith("|")
        )
        # Exactly one row, single-cell content collapsed to one space.
        assert "line1 line2" in table_row

    def test_crlf_in_prompt_collapsed_to_single_space(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Guard against the chained .replace("\r", " ").replace("\n", " ")
        # pattern: that turns a single CRLF into TWO spaces, not one.
        body = _run_document_prompt(tmp_path, monkeypatch, prompt="line1\r\nline2")
        table_row = next(
            line
            for line in body.splitlines()
            if "line1" in line and line.startswith("|")
        )
        assert "line1 line2" in table_row
        assert "line1  line2" not in table_row

    @pytest.mark.parametrize(
        "prompt",
        [
            "line1\r\nline2",  # CRLF (single, most common Windows)
            "line1\n\rline2",  # LFCR (reversed order)
            "line1\n\nline2",  # consecutive LFs
            "line1\r\rline2",  # consecutive CRs
            "line1\n\n\nline2",  # three consecutive LFs
            "line1\r\nline2\r\nline3",  # multiple CRLFs (chained .replace would yield "line1 line2 line3" by coincidence; the second CRLF still tests the + quantifier)
        ],
        ids=[
            "single-crlf",
            "lfcr-reversed",
            "double-lf",
            "double-cr",
            "triple-lf",
            "multi-crlf",
        ],
    )
    def test_newline_runs_collapse_to_single_space(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        prompt: str,
    ) -> None:
        # The fix replaces `.replace("\r"," ").replace("\n"," ")` with
        # `re.sub(r"[\r\n]+", " ", ...)`. The `+` quantifier is what
        # distinguishes correct behavior across these inputs. A naive
        # `re.sub(r"[\r\n]{1,2}", " ", ...)` or any chained-replace variant
        # would produce more than one space on at least one of these cases.
        body = _run_document_prompt(tmp_path, monkeypatch, prompt=prompt)
        table_row = next(
            line
            for line in body.splitlines()
            if "line1" in line and line.startswith("|")
        )
        # No matter how many newlines were present, all runs collapse to
        # exactly one space. We assert "line1 line" (one space, then the
        # start of either "line2" or "line2 line3") and forbid any
        # consecutive-space pattern between "line1" and the next "line".
        assert "line1 line" in table_row
        assert "line1  " not in table_row
        assert "line1   " not in table_row

    def test_backslash_before_pipe_order_is_load_bearing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If the order of replace() calls flips ('|' before '\\'), the
        # backslash-escaped pipe would be double-escaped. Lock the order in.
        body = _run_document_prompt(tmp_path, monkeypatch, prompt=r"a\|b")
        table_row = next(
            line for line in body.splitlines() if "a\\" in line and line.startswith("|")
        )
        # Source: a\|b  (literal backslash, literal pipe)
        # After step 1 (\\ -> \\\\): a\\|b
        # After step 2 (| -> \|):    a\\\|b
        assert r"a\\\|b" in table_row


class TestDocumentImagePromptIndentedBlock:
    """The detailed entry must render the prompt as a real indented code block."""

    def test_six_space_indent_inside_list_item(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        body = _run_document_prompt(tmp_path, monkeypatch, prompt="hello world")
        # Every non-empty line of the prompt is indented by 6 spaces so that
        # CommonMark treats it as a code block within the surrounding list.
        assert "\n      hello world\n" in body

    def test_triple_backticks_in_prompt_survive_indented(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        body = _run_document_prompt(
            tmp_path, monkeypatch, prompt="```python\nprint('x')\n```"
        )
        # Each backtick line is indented; backticks survive as literal content,
        # not as a fence that would escape the surrounding list item.
        assert "      ```python" in body
        assert "      print('x')" in body
        assert "      ```" in body

    def test_empty_prompt_renders_placeholder_indent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        body = _run_document_prompt(tmp_path, monkeypatch, prompt="")
        # Exactly one indented line (the placeholder) so the section is not
        # an empty list-continuation paragraph.
        assert "- **Prompt**:\n\n      \n" in body


# ---------------------------------------------------------------------------
# Random-token filename property (regression prevention for S5)
# ---------------------------------------------------------------------------


_TIMESTAMP_TOKEN_PATTERN = re.compile(r"\d{8}_\d{6}_[0-9a-f]{32}\.")


class TestDefaultFilenameRandomToken:
    """Default filenames must contain a timestamp AND a random hex token."""

    @pytest.mark.parametrize(
        ("is_draft", "expected_prefix"),
        [(False, "generated_"), (True, "draft_")],
    )
    def test_default_output_path_includes_random_token(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        is_draft: bool,
        expected_prefix: str,
    ) -> None:
        from unittest.mock import MagicMock

        from scripts import generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        fake_inline = MagicMock()
        fake_inline.data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
        fake_inline.mime_type = "image/png"
        fake_part = MagicMock(inline_data=fake_inline, text=None, thought=None)
        fake_candidate = MagicMock()
        fake_candidate.content.parts = [fake_part]
        fake_response = MagicMock()
        fake_response.candidates = [fake_candidate]
        fake_response.prompt_feedback = None

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = fake_response
        client_cls = MagicMock(return_value=mock_client)

        fake_script = tmp_path / "scripts" / "generate_image.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "output" / "drafts").mkdir(parents=True, exist_ok=True)

        with (
            patch.object(mod, "__file__", str(fake_script)),
            patch("scripts.generate_image.genai.Client", client_cls),
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            result = mod.generate_image(
                prompt="random-token regression test",
                model_key="flash",
                output_path=None,
                is_draft=is_draft,
                document_prompt=False,
            )

        assert result is not None
        assert result.name.startswith(expected_prefix)
        assert _TIMESTAMP_TOKEN_PATTERN.search(result.name), (
            f"filename {result.name!r} does not match "
            f"<prefix>_<timestamp>_<token>.<ext>; a refactor likely dropped "
            f"the secrets.token_hex(16) suffix"
        )

    def test_story_auto_prefix_includes_random_token(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Drive generate_story_sequence with output_prefix=None and capture
        # the output_path that the production code builds for part 1. The
        # mock lets us inspect the prefix the production code derived,
        # rather than re-implementing the derivation in the test.
        from unittest.mock import patch

        from scripts import generate_image as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        part_file = tmp_path / "fake_part1.png"
        part_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)

        with (
            patch(
                "scripts.generate_image.generate_image",
                return_value=part_file,
            ) as gen_mock,
            patch("scripts.generate_image._load_api_key", return_value="fake-key"),
        ):
            mod.generate_story_sequence(
                base_prompt="random-token regression test",
                num_parts=1,
                model_key="flash",
                output_prefix=None,
            )

        # The output_path the production code passed to generate_image
        # encodes the auto-derived prefix as
        # story_<YYYYMMDD>_<HHMMSS>_<32hex>_part1.png
        assert gen_mock.call_args is not None
        captured_path = gen_mock.call_args.kwargs["output_path"]
        assert re.match(
            r"^story_\d{8}_\d{6}_[0-9a-f]{32}_part1$", captured_path.stem
        ), (
            f"output_path stem {captured_path.stem!r} does not match the "
            f"story_<timestamp>_<token>_partN format; a refactor likely "
            f"dropped the secrets.token_hex(16) suffix"
        )
