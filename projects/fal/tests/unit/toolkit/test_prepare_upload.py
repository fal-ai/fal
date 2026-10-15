from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image as PILImage

from fal.app import App
from fal.toolkit import File, Image
from fal.toolkit.file.types import FileData, FileRepository


class RecordingRepository(FileRepository):
    """Keeps what each upload received, so a test sees the bytes on the wire."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.saved: list[tuple[str, Any, str]] = []

    def save(self, data: FileData, **kwargs: Any) -> str:
        if self.fail:
            raise RuntimeError("primary down")
        self.saved.append(("save", data.data, data.content_type))
        return "https://example.com/bytes"

    def save_file(self, file_path, content_type, **kwargs: Any):
        if self.fail:
            raise RuntimeError("primary down")
        self.saved.append(("save_file", Path(file_path), content_type))
        return "https://example.com/path", None


class FakeApp:
    """Stands in for the running fal.App; the default hook is the real one."""

    prepare_upload = App.prepare_upload

    def __init__(self, hook=None, request_id: str | None = "req-1"):
        self.current_request = SimpleNamespace(
            request_id=request_id, lifecycle_preference=None, upload_policy=None
        )
        self.calls: list[tuple[Any, str, Any]] = []
        if hook is not None:

            def prepare_upload(source, content_type, request):
                self.calls.append((source, content_type, request))
                return hook(source, content_type)

            self.prepare_upload = prepare_upload  # type: ignore[method-assign]


@pytest.fixture
def run_app(monkeypatch):
    def install(app):
        for module in ("file", "_upload_policy"):
            monkeypatch.setattr(
                f"fal.toolkit.file.{module}.get_current_app", lambda: app
            )
        return app

    return install


def upload_bytes(data: bytes, repo: RecordingRepository, **kwargs: Any) -> File:
    return File.from_bytes(data, repository=repo, fallback_repository=[], **kwargs)


def test_from_bytes_uploads_what_the_hook_returns(run_app):
    app = run_app(FakeApp(lambda data, ct: (data + b"!", "text/markdown")))
    repo = RecordingRepository()

    file = upload_bytes(b"hello", repo, content_type="text/plain")

    assert app.calls == [(b"hello", "text/plain", app.current_request)]
    assert repo.saved == [("save", b"hello!", "text/markdown")]
    assert (file.file_size, file.file_data) == (6, b"hello!")
    assert file.content_type == "text/markdown"


def test_hook_sees_the_resolved_content_type(run_app):
    app = run_app(FakeApp(lambda data, ct: (data, ct)))

    upload_bytes(b"x", RecordingRepository(), file_name="notes.txt")
    upload_bytes(b"x", RecordingRepository())

    assert [ct for _, ct, _ in app.calls] == ["text/plain", "application/octet-stream"]


def test_explicit_request_wins_over_the_context(run_app):
    app = run_app(FakeApp(lambda data, ct: (data, ct)))
    request = SimpleNamespace(headers={})

    upload_bytes(b"x", RecordingRepository(), request=request)

    assert app.calls[0][2] is request


def test_from_path_hands_the_hook_the_path(run_app, tmp_path):
    source = tmp_path / "out.mp4"
    source.write_bytes(b"video")
    app = run_app(FakeApp(lambda path, ct: (path, ct)))
    repo = RecordingRepository()

    File.from_path(source, content_type="video/mp4", repository=repo)

    assert app.calls[0][:2] == (source, "video/mp4")
    assert repo.saved == [("save_file", source, "video/mp4")]


def test_path_in_bytes_out(run_app, tmp_path):
    source = tmp_path / "out.txt"
    source.write_bytes(b"on disk")
    run_app(FakeApp(lambda path, ct: (path.read_bytes().upper(), ct)))
    repo = RecordingRepository()

    file = File.from_path(source, content_type="text/plain", repository=repo)

    assert repo.saved == [("save", b"ON DISK", "text/plain")]
    assert (file.file_name, file.file_size) == ("out.txt", 7)


def test_bytes_in_path_out(run_app, tmp_path):
    staged = tmp_path / "staged.mp4"
    staged.write_bytes(b"large video")
    run_app(FakeApp(lambda data, ct: (staged, ct)))
    repo = RecordingRepository()

    file = upload_bytes(b"raw", repo, content_type="video/mp4")

    assert repo.saved == [("save_file", staged, "video/mp4")]
    assert (file.file_name, file.file_size) == ("staged.mp4", 11)


def test_image_from_pil_goes_through_the_hook(run_app):
    app = run_app(FakeApp(lambda data, ct: (b"watermarked", ct)))
    repo = RecordingRepository()

    Image.from_pil(
        PILImage.new("RGB", (2, 2)), "png", repository=repo, fallback_repository=[]
    )

    assert app.calls[0][1] == "image/png"
    assert repo.saved == [("save", b"watermarked", "image/png")]


@pytest.mark.asyncio
async def test_from_bytes_async_goes_through_the_hook(run_app):
    run_app(FakeApp(lambda data, ct: (data * 2, ct)))
    repo = RecordingRepository()

    await File.from_bytes_async(b"ab", repository=repo, fallback_repository=[])

    assert repo.saved[0][1] == b"abab"


def test_hook_runs_before_the_upload_policy(run_app, monkeypatch):
    run_app(FakeApp(lambda data, ct: (b"marked", ct)))
    sent = []
    monkeypatch.setattr(
        "fal.toolkit.file.file.get_upload_policy", lambda request: object()
    )
    monkeypatch.setattr(
        "fal.toolkit.file.file.save_with_policy",
        lambda policy, name, data, ct: sent.append(data) or "https://bucket/x",
    )

    File.from_bytes(b"raw", content_type="text/plain")

    assert sent == [b"marked"]


def test_no_app_means_no_hook(monkeypatch):
    monkeypatch.setattr("fal.toolkit.file.file.get_current_app", lambda: None)
    repo = RecordingRepository()

    upload_bytes(bytearray(b"as is"), repo, content_type="text/plain")

    assert repo.saved == [("save", b"as is", "text/plain")]


def test_default_hook_uploads_identical_bytes(run_app, tmp_path):
    run_app(FakeApp())
    source = tmp_path / "out.bin"
    source.write_bytes(b"\x00\x01")
    repo = RecordingRepository()

    upload_bytes(b"\x00\x01", repo, content_type="image/png")
    File.from_path(source, repository=repo)

    assert repo.saved == [
        ("save", b"\x00\x01", "image/png"),
        ("save_file", source, "application/octet-stream"),
    ]


def upload_lines(output: str) -> list[dict]:
    return [
        json.loads(line)["file_upload"]
        for line in output.splitlines()
        if line.startswith('{"file_upload"')
    ]


def test_one_log_line_per_successful_upload(run_app, capsys):
    run_app(FakeApp(lambda data, ct: (data + b"!", ct)))

    upload_bytes(b"hello", RecordingRepository(), content_type="text/plain")

    [line] = upload_lines(capsys.readouterr().out)
    assert line.pop("duration_ms") >= 0
    assert line == {
        "repository": "RecordingRepository",
        "outcome": "ok",
        "bytes": 6,
        "content_type": "text/plain",
        "request_id": "req-1",
    }


def test_fallback_logs_one_line_per_repository(run_app, capsys):
    run_app(FakeApp())

    File.from_bytes(
        b"hello",
        content_type="text/plain",
        repository=RecordingRepository(fail=True),
        fallback_repository="in_memory",
    )

    lines = upload_lines(capsys.readouterr().out)
    assert [(line["repository"], line["outcome"]) for line in lines] == [
        ("RecordingRepository", "fallback"),
        ("in_memory", "ok"),
    ]
    assert lines[0]["error"] == "RuntimeError: primary down"
    assert "error" not in lines[1]


def test_local_uploader_logs_as_local(run_app, capsys, monkeypatch):
    run_app(FakeApp())
    monkeypatch.setenv("FAL_USE_LOCAL_UPLOADER", "1")
    monkeypatch.setattr(
        "fal.toolkit.file.file.LocalFileRepository.save",
        lambda self, data, **kwargs: "https://example.com/local",
    )

    File.from_bytes(b"x", fallback_repository=[])

    [line] = upload_lines(capsys.readouterr().out)
    assert line["repository"] == "local"


def test_last_failure_logs_error(run_app, capsys):
    run_app(FakeApp())

    with pytest.raises(RuntimeError):
        upload_bytes(b"x", RecordingRepository(fail=True))

    [line] = upload_lines(capsys.readouterr().out)
    assert line["outcome"] == "error"
