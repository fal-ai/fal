from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from PIL import Image as PILImage

from fal.app import App
from fal.toolkit import File, Image
from fal.toolkit.exceptions import FileUploadFallbackException
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


def fake_app(hook=None) -> SimpleNamespace:
    """Stands in for the running fal.App; without a hook it runs the real one."""
    return SimpleNamespace(
        current_request=SimpleNamespace(
            request_id="req-1", lifecycle_preference=None, upload_policy=None
        ),
        prepare_upload=Mock(
            side_effect=hook or (lambda *args: App.prepare_upload(None, *args))
        ),
    )


@pytest.fixture
def run_app(monkeypatch):
    def install(app):
        for module in ("file", "_upload_policy"):
            monkeypatch.setattr(
                f"fal.toolkit.file.{module}.get_current_app", lambda: app
            )
        return app

    return install


@pytest.fixture
def on_disk(tmp_path) -> Path:
    path = tmp_path / "on_disk.txt"
    path.write_bytes(b"on disk")
    return path


def upload_bytes(data: bytes, repo: Any, **kwargs: Any) -> File:
    return File.from_bytes(data, repository=repo, fallback_repository=[], **kwargs)


@pytest.mark.parametrize(
    "upload, content_type",
    [
        pytest.param(
            lambda repo, path: upload_bytes(b"raw", repo, content_type="text/plain"),
            "text/plain",
            id="from_bytes",
        ),
        pytest.param(
            lambda repo, path: asyncio.run(
                File.from_bytes_async(b"raw", repository=repo, fallback_repository=[])
            ),
            "application/octet-stream",
            id="from_bytes_async",
        ),
        pytest.param(
            lambda repo, path: Image.from_pil(
                PILImage.new("RGB", (2, 2)),
                "png",
                repository=repo,
                fallback_repository=[],
            ),
            "image/png",
            id="Image.from_pil",
        ),
        pytest.param(
            lambda repo, path: File.from_path(
                path, content_type="text/plain", repository=repo
            ),
            "text/plain",
            id="from_path-path_in_bytes_out",
        ),
    ],
)
def test_repository_receives_what_the_hook_returns(
    run_app, on_disk, upload, content_type
):
    app = run_app(fake_app(lambda source, ct, request: (b"hooked", ct)))
    repo = RecordingRepository()

    file = upload(repo, on_disk)

    app.prepare_upload.assert_called_once()
    assert repo.saved == [("save", b"hooked", content_type)]
    assert file.file_size == 6


def test_bytes_in_path_out(run_app, on_disk):
    run_app(fake_app(lambda data, ct, request: (on_disk, "text/markdown")))
    repo = RecordingRepository()

    file = upload_bytes(b"raw", repo, content_type="text/plain")

    assert repo.saved == [("save_file", on_disk, "text/markdown")]
    assert (file.file_name, file.file_size) == ("on_disk.txt", 7)


def test_hook_gets_the_resolved_content_type_and_request(run_app):
    app = run_app(fake_app(lambda data, ct, request: (data, ct)))
    request = SimpleNamespace(headers={})

    upload_bytes(b"x", RecordingRepository(), file_name="notes.txt")
    upload_bytes(b"x", RecordingRepository(), request=request)

    assert [call.args for call in app.prepare_upload.call_args_list] == [
        (b"x", "text/plain", app.current_request),
        (b"x", "application/octet-stream", request),
    ]


def test_hook_runs_before_the_upload_policy(run_app, monkeypatch):
    run_app(fake_app(lambda data, ct, request: (b"marked", ct)))
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


@pytest.mark.parametrize("app", [None, fake_app()], ids=["no_app", "default_hook"])
def test_without_a_custom_hook_the_input_is_uploaded_unchanged(run_app, on_disk, app):
    run_app(app)
    repo = RecordingRepository()

    upload_bytes(bytearray(b"as is"), repo, content_type="text/plain")
    File.from_path(on_disk, repository=repo)

    assert repo.saved == [
        ("save", b"as is", "text/plain"),
        ("save_file", on_disk, "application/octet-stream"),
    ]


def upload_lines(output: str) -> list[dict]:
    return [
        json.loads(line)["file_upload"]
        for line in output.splitlines()
        if line.startswith('{"file_upload"')
    ]


@pytest.mark.parametrize("local", [False, True], ids=["repository", "local_uploader"])
def test_one_log_line_per_successful_upload(run_app, capsys, monkeypatch, local):
    run_app(fake_app(lambda data, ct, request: (data + b"!", ct)))
    repo: Any = RecordingRepository()
    if local:
        monkeypatch.setenv("FAL_USE_LOCAL_UPLOADER", "1")
        monkeypatch.setattr(
            "fal.toolkit.file.file.LocalFileRepository.save",
            lambda self, data, **kwargs: "https://example.com/local",
        )
        repo = "fal_v3"

    upload_bytes(b"hello", repo, content_type="text/plain")

    [line] = upload_lines(capsys.readouterr().out)
    assert line.pop("duration_ms") >= 0
    assert line == {
        "repository": "local" if local else "RecordingRepository",
        "outcome": "ok",
        "bytes": 6,
        "content_type": "text/plain",
        "request_id": "req-1",
    }


def test_each_failed_repository_logs_a_line(run_app, capsys):
    run_app(fake_app())

    with pytest.raises(FileUploadFallbackException):
        File.from_bytes(
            b"hello",
            repository=RecordingRepository(fail=True),
            fallback_repository=[RecordingRepository(fail=True)],
        )

    lines = upload_lines(capsys.readouterr().out)
    assert [(line["outcome"], line["error"]) for line in lines] == [
        ("fallback", "RuntimeError: primary down"),
        ("error", "RuntimeError: primary down"),
    ]
