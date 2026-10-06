from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from rich.progress import Progress

from fal.files import FalFileSystem


class RecordingProgress(Progress):
    """A real rich Progress that also keeps the trace of reported positions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, disable=True, **kwargs)
        self.completions = []

    def update(self, task_id, **kwargs):
        super().update(task_id, **kwargs)
        if "completed" in kwargs:
            self.completions.append(kwargs["completed"])


def fake_multipart(monkeypatch, etag, md5, emit=()):
    uploader = Mock()
    uploader.content_md5 = md5

    def upload_file(lpath, on_bytes_uploaded=None, **kwargs):
        for uploaded in emit:
            on_bytes_uploaded(uploaded)
        return etag

    uploader.upload_file.side_effect = upload_file
    monkeypatch.setattr(
        "fal.files.DataFileMultipartUpload", Mock(return_value=uploader)
    )
    return uploader


def test_multipart_task_is_sized_in_bytes_and_tracks_the_transfer(
    monkeypatch, tmp_path
):
    size = 100 * 1024 * 1024
    path = tmp_path / "model.bin"
    fake_multipart(monkeypatch, "etag", "etag", emit=(25_000, 60_000_000, size))

    progress = RecordingProgress()
    FalFileSystem._put_file_multipart(
        SimpleNamespace(_client=object()), str(path), "/data/model.bin", size, progress
    )

    task = progress.tasks[0]
    assert task.description == "Uploading model.bin"
    assert task.total == size
    assert task.completed == size
    # The symptom being fixed: a single jump from 0 to done. The trailing
    # repeat is the end-of-upload settle.
    assert progress.completions == [25_000, 60_000_000, size, size]


def test_multipart_raises_when_the_server_etag_differs_from_the_local_digest(
    monkeypatch, tmp_path
):
    path = tmp_path / "model.bin"
    fake_multipart(monkeypatch, "server-etag", "local-md5")

    with pytest.raises(RuntimeError, match="MD5 mismatch"):
        FalFileSystem._put_file_multipart(
            SimpleNamespace(_client=object()),
            str(path),
            "/data/model.bin",
            32,
            RecordingProgress(),
        )


def test_multipart_raises_when_no_digest_was_produced(monkeypatch, tmp_path):
    path = tmp_path / "model.bin"
    fake_multipart(monkeypatch, "server-etag", None)

    with pytest.raises(RuntimeError, match="MD5 mismatch"):
        FalFileSystem._put_file_multipart(
            SimpleNamespace(_client=object()),
            str(path),
            "/data/model.bin",
            32,
            RecordingProgress(),
        )


def test_small_file_upload_advances_while_the_body_is_written(monkeypatch, tmp_path):
    path = tmp_path / "small.bin"
    path.write_bytes(b"x" * 200_000)

    recorded = {}

    def fake_request(self, method, url, **kwargs):
        _, reader = kwargs["files"]["file_upload"]
        while reader.read(65_536):
            pass
        recorded["url"] = url
        return SimpleNamespace(status_code=200)

    progresses = []

    def make_progress(*args, **kwargs):
        progress = RecordingProgress(*args, **kwargs)
        progresses.append(progress)
        return progress

    monkeypatch.setattr(FalFileSystem, "_request", fake_request)
    monkeypatch.setattr("rich.progress.Progress", make_progress)

    fs = FalFileSystem(skip_instance_cache=True)
    fs.put_file(str(path), "small.bin")

    assert recorded["url"] == "/files/file/local//data/small.bin"
    progress = progresses[0]
    assert progress.tasks[0].total == 200_000
    assert progress.tasks[0].completed == 200_000
    assert len([c for c in progress.completions if 0 < c < 200_000]) >= 2


def test_empty_file_upload_finishes_the_bar(monkeypatch, tmp_path):
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")

    def fake_request(self, method, url, **kwargs):
        _, reader = kwargs["files"]["file_upload"]
        while reader.read(65_536):
            pass
        return SimpleNamespace(status_code=200)

    progresses = []

    def make_progress(*args, **kwargs):
        progress = RecordingProgress(*args, **kwargs)
        progresses.append(progress)
        return progress

    monkeypatch.setattr(FalFileSystem, "_request", fake_request)
    monkeypatch.setattr("rich.progress.Progress", make_progress)

    fs = FalFileSystem(skip_instance_cache=True)
    fs.put_file(str(path), "empty.bin")

    task = progresses[0].tasks[0]
    assert task.total == 1
    assert task.finished


def test_multipart_bar_settles_on_the_full_size_after_a_stale_callback(
    monkeypatch, tmp_path
):
    """Concurrent parts can deliver a lower running total last.

    The tracker computes totals under a lock but invokes the callback outside
    it, so delivery order is not guaranteed. The bar must still end at the
    uploaded size rather than wherever the last callback happened to land.
    """
    size = 300_000
    path = tmp_path / "payload.bin"
    # Highest total delivered mid-stream, a lower one arriving last.
    fake_multipart(monkeypatch, "etag", "etag", emit=(100_000, size, 200_000))

    progress = RecordingProgress()
    FalFileSystem._put_file_multipart(
        SimpleNamespace(_client=object()),
        str(path),
        "/data/payload.bin",
        size,
        progress,
    )

    task = progress.tasks[0]
    assert progress.completions[-2] == 200_000, "expected the stale callback last"
    assert progress.completions[-1] == size
    assert task.completed == size
    assert task.finished


@pytest.mark.parametrize(
    "filename",
    ["x[bold]y.bin", "x[not a tag]y.bin", "[bold]lead.bin", "movie[1080p].mkv"],
    ids=["style-tag", "tag-like-text", "leading-tag", "plain-brackets"],
)
def test_bracketed_filename_is_shown_verbatim(monkeypatch, tmp_path, filename):
    """A task description is parsed as Rich markup.

    Any bracketed span that looks like a tag is dropped from the rendered
    label, so `x[bold]y.bin` would otherwise be shown as `xy.bin`. A closing
    tag cannot occur here because `/` is the path separator and
    `os.path.basename` removes anything before it.
    """
    from rich.text import Text

    size = 300_000
    fake_multipart(monkeypatch, "etag", "etag", emit=(size,))

    progress = RecordingProgress()
    FalFileSystem._put_file_multipart(
        SimpleNamespace(_client=object()),
        str(tmp_path / filename),
        f"/data/{filename}",
        size,
        progress,
    )

    rendered = Text.from_markup(progress.tasks[0].description).plain
    assert rendered == f"Uploading {filename}"
