from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from fal.files import _MD5_CHUNK_SIZE, FalFileSystem, _compute_md5


class RecordingProgress:
    """Stands in for ``rich.Progress`` and records the order of calls.

    The bug being guarded against is an ordering bug, not a rendering bug: rich
    draws nothing until the first task exists, so what matters is that a task is
    added before the checksum starts.
    """

    def __init__(self):
        self.events = []
        self._next_id = 0

    def add_task(self, description, total=None):
        task_id = self._next_id
        self._next_id += 1
        self.events.append(("add_task", task_id, description, total))
        return task_id

    def advance(self, task_id, amount=1):
        self.events.append(("advance", task_id, amount))


@pytest.fixture
def big_file(tmp_path):
    """A file over ``MULTIPART_THRESHOLD`` (10MB), so the multipart path is taken."""
    path = tmp_path / "model.bin"
    path.write_bytes(b"\xab" * (12 * 1024 * 1024))
    return path


def test_compute_md5_reports_every_byte_through_on_chunk(big_file):
    seen = []

    digest = _compute_md5(str(big_file), on_chunk=seen.append)

    assert digest == hashlib.md5(big_file.read_bytes()).hexdigest()
    assert sum(seen) == big_file.stat().st_size
    assert all(count > 0 for count in seen)


@pytest.mark.parametrize("chunk_size", [8192, 1024 * 1024, _MD5_CHUNK_SIZE])
def test_compute_md5_digest_does_not_depend_on_chunk_size(big_file, chunk_size):
    """MD5 is a streaming hash, so the etag comparison is unaffected by the
    chunk size. This is what makes the chunk size free to tune."""
    seen = []

    digest = _compute_md5(str(big_file), chunk_size=chunk_size, on_chunk=seen.append)

    assert digest == hashlib.md5(big_file.read_bytes()).hexdigest()
    assert sum(seen) == big_file.stat().st_size
    assert max(seen) <= chunk_size


def test_compute_md5_without_callback_still_works(big_file):
    assert _compute_md5(str(big_file)) == hashlib.md5(big_file.read_bytes()).hexdigest()


def _put_multipart(big_file, progress, monkeypatch, etag=None):
    """Drive ``_put_file_multipart`` with a stubbed uploader, no network."""
    digest = hashlib.md5(big_file.read_bytes()).hexdigest()
    uploaded = {}

    class StubMultipartUpload:
        def __init__(self, **kwargs):
            uploaded["kwargs"] = kwargs

        def upload_file(self, lpath, on_part_complete=None):
            uploaded["lpath"] = lpath
            if on_part_complete is not None:
                on_part_complete(1)
            return digest if etag is None else etag

    monkeypatch.setattr("fal.files.DataFileMultipartUpload", StubMultipartUpload)

    fs = SimpleNamespace(_client=object())
    FalFileSystem._put_file_multipart(
        fs, str(big_file), "/data/model.bin", big_file.stat().st_size, progress
    )
    return digest, uploaded


def test_checksum_progress_is_visible_before_hashing_finishes(big_file, monkeypatch):
    """Regression for SERV-1567.

    Before the fix the whole file was hashed before the first ``add_task``, so a
    20GB upload showed an empty terminal for minutes. The checksum task must be
    created first and advance while hashing.
    """
    progress = RecordingProgress()

    _put_multipart(big_file, progress, monkeypatch)

    kinds = [event[0] for event in progress.events]
    add_task_positions = [i for i, kind in enumerate(kinds) if kind == "add_task"]
    assert len(add_task_positions) == 2, (
        "expected a checksum task and then an upload task; only one task means "
        "the checksum phase rendered nothing, which is the SERV-1567 blank terminal"
    )
    assert add_task_positions[0] == 0, "nothing was rendered before hashing started"

    checksum_task_id = progress.events[0][1]
    size = big_file.stat().st_size
    advances_while_hashing = [
        event
        for event in progress.events[: add_task_positions[1]]
        if event[0] == "advance" and event[1] == checksum_task_id
    ]
    assert advances_while_hashing, "the checksum phase reported no progress"
    assert progress.events[0][3] == size, "checksum task total is not the file size"
    assert sum(event[2] for event in advances_while_hashing) == size


def test_upload_task_is_still_added_and_advanced(big_file, monkeypatch):
    progress = RecordingProgress()

    _put_multipart(big_file, progress, monkeypatch)

    add_tasks = [event for event in progress.events if event[0] == "add_task"]
    assert len(add_tasks) == 2, "expected a checksum task and an upload task"
    upload_task_id = add_tasks[1][1]
    assert any(
        event[0] == "advance" and event[1] == upload_task_id
        for event in progress.events
    )


def test_etag_mismatch_still_raises(big_file, monkeypatch):
    """The chunk size changed, so pin the comparison it feeds."""
    progress = RecordingProgress()

    with pytest.raises(RuntimeError, match="MD5 mismatch"):
        _put_multipart(big_file, progress, monkeypatch, etag="0" * 32)
