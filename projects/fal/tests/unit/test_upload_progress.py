import hashlib
import io
import os
import threading

import httpx
import pytest

from fal.upload import BaseMultipartUpload, ProgressFileReader

FINAL_ETAG = "final-etag"


class FakeMultipartUpload(BaseMultipartUpload):
    @property
    def initiate_url(self):
        return "/initiate"

    @property
    def part_url(self):
        return f"/{self.upload_id}/part"

    @property
    def complete_url(self):
        return f"/{self.upload_id}/complete"

    @property
    def cancel_url(self):
        return f"/{self.upload_id}/cancel"


def make_client(fail_parts=None):
    """A transport that answers the multipart lifecycle and drains each body.

    `fail_parts` maps a part number to how many times it should return 500
    before succeeding, which exercises the retry path.
    """
    remaining_failures = dict(fail_parts or {})
    lock = threading.Lock()

    def handler(request):
        path = request.url.path
        if path.endswith("/initiate"):
            return httpx.Response(200, json={"upload_id": "upload-1"})
        if path.endswith("/complete"):
            return httpx.Response(200, json={"etag": FINAL_ETAG})
        if path.endswith("/cancel"):
            return httpx.Response(200, json={})

        part_number = int(path.rsplit("/", 1)[-1])
        with lock:
            left = remaining_failures.get(part_number, 0)
            if left:
                remaining_failures[part_number] = left - 1
                return httpx.Response(500, json={"detail": "boom"})
        return httpx.Response(
            200, json={"part_number": part_number, "etag": f"etag-{part_number}"}
        )

    return httpx.Client(
        base_url="http://testserver", transport=httpx.MockTransport(handler)
    )


def write_payload(tmp_path, size):
    payload = os.urandom(size)
    path = tmp_path / "payload.bin"
    path.write_bytes(payload)
    return str(path), payload


def test_progress_reader_reports_absolute_offset():
    seen = []
    reader = ProgressFileReader(io.BytesIO(b"0123456789"), seen.append)

    assert reader.read(4) == b"0123"
    assert reader.read(4) == b"4567"
    assert seen == [4, 8]

    # A retry rewinds the body; the reader must re-report from the start
    # rather than continue climbing.
    reader.seek(0)
    assert reader.read(4) == b"0123"
    assert seen == [4, 8, 4]


def test_byte_progress_is_monotonic_and_ends_at_file_size(tmp_path):
    # One part per 100_000 bytes, read by httpx in 64KiB slices, so every
    # part reports several times on its way up.
    path, payload = write_payload(tmp_path, 300_000)
    reports = []

    uploader = FakeMultipartUpload(
        client=make_client(), chunk_size=100_000, max_concurrency=1
    )
    etag = uploader.upload_file(path, on_bytes_uploaded=reports.append)

    assert etag == FINAL_ETAG
    assert reports == sorted(reports)
    assert reports[-1] == len(payload)
    # Parts-based accounting would have produced exactly 3 updates.
    assert len(reports) > 3


def test_byte_progress_never_exceeds_file_size_under_concurrency(tmp_path):
    path, payload = write_payload(tmp_path, 400_000)
    reports = []

    uploader = FakeMultipartUpload(
        client=make_client(), chunk_size=100_000, max_concurrency=4
    )
    uploader.upload_file(path, on_bytes_uploaded=reports.append)

    assert max(reports) == len(payload)
    assert min(reports) > 0


def test_retried_part_does_not_inflate_byte_progress(tmp_path, monkeypatch):
    monkeypatch.setattr("fal.upload.time.sleep", lambda _: None)
    path, payload = write_payload(tmp_path, 300_000)
    reports = []

    uploader = FakeMultipartUpload(
        client=make_client(fail_parts={2: 1}), chunk_size=100_000, max_concurrency=1
    )
    uploader.upload_file(path, on_bytes_uploaded=reports.append)

    assert max(reports) == len(payload)
    assert reports[-1] == len(payload)


def test_content_md5_matches_file_without_a_separate_read(tmp_path):
    path, payload = write_payload(tmp_path, 250_000)

    uploader = FakeMultipartUpload(
        client=make_client(), chunk_size=100_000, max_concurrency=2
    )
    uploader.upload_file(path, compute_md5=True)

    assert uploader.content_md5 == hashlib.md5(payload).hexdigest()


def test_empty_file_still_reports_a_digest(tmp_path):
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    reports = []

    uploader = FakeMultipartUpload(client=make_client(), chunk_size=100_000)
    parts = []
    etag = uploader.upload_file(
        str(path),
        on_part_complete=parts.append,
        on_bytes_uploaded=reports.append,
        compute_md5=True,
    )

    assert etag == FINAL_ETAG
    assert parts == [1]
    assert reports == [0]
    assert uploader.content_md5 == hashlib.md5(b"").hexdigest()


def test_part_callback_still_fires_for_every_part(tmp_path):
    path, _ = write_payload(tmp_path, 300_000)
    parts = []

    uploader = FakeMultipartUpload(
        client=make_client(), chunk_size=100_000, max_concurrency=2
    )
    uploader.upload_file(path, on_part_complete=parts.append)

    assert sorted(parts) == [1, 2, 3]


def test_reader_failure_propagates_and_leaves_no_digest(tmp_path, monkeypatch):
    path, _ = write_payload(tmp_path, 300_000)

    real_open = open

    def exploding_open(file, *args, **kwargs):
        if str(file) == path:
            raise OSError("disk went away")
        return real_open(file, *args, **kwargs)

    uploader = FakeMultipartUpload(
        client=make_client(), chunk_size=100_000, max_concurrency=2
    )
    monkeypatch.setattr("builtins.open", exploding_open)

    with pytest.raises(OSError, match="disk went away"):
        uploader.upload_file(path)

    assert uploader.content_md5 is None


def make_recording_client(fail_parts=None):
    """Like `make_client`, but keeps the body of every part attempt."""
    remaining_failures = dict(fail_parts or {})
    attempts = []
    lock = threading.Lock()

    def handler(request):
        path = request.url.path
        if path.endswith("/initiate"):
            return httpx.Response(200, json={"upload_id": "upload-1"})
        if path.endswith("/complete"):
            return httpx.Response(200, json={"etag": FINAL_ETAG})
        if path.endswith("/cancel"):
            return httpx.Response(200, json={})

        part_number = int(path.rsplit("/", 1)[-1])
        with lock:
            attempts.append((part_number, request.content))
            left = remaining_failures.get(part_number, 0)
            if left:
                remaining_failures[part_number] = left - 1
                return httpx.Response(500, json={"detail": "boom"})
        return httpx.Response(
            200, json={"part_number": part_number, "etag": f"etag-{part_number}"}
        )

    client = httpx.Client(
        base_url="http://testserver", transport=httpx.MockTransport(handler)
    )
    return client, attempts


def test_request_rebuilds_a_single_use_body_per_attempt(monkeypatch):
    """Every attempt must get a fresh body, not one left at EOF by the last.

    Asserted against the retry loop rather than the wire: whether a consumed
    stream would actually resend as an empty part depends on whether the HTTP
    client rewinds it, which is not a guarantee this code should rest on.
    """
    monkeypatch.setattr("fal.upload.time.sleep", lambda _: None)
    built = []

    def files_factory():
        stream = io.BytesIO(b"chunk-payload")
        built.append(stream)
        return {"file_upload": ("chunk", stream, "application/octet-stream")}

    client, attempts = make_recording_client(fail_parts={1: 1})
    uploader = FakeMultipartUpload(client=client, chunk_size=100_000)
    uploader._upload_id = "upload-1"
    uploader._request("PUT", "/upload-1/part/1", files_factory=files_factory)

    assert len(built) == 2, "expected one body per attempt"
    assert built[0] is not built[1], "the retry reused the consumed stream"
    for _, body in attempts:
        assert b"chunk-payload" in body


def test_retried_part_still_sends_the_whole_chunk(tmp_path, monkeypatch):
    monkeypatch.setattr("fal.upload.time.sleep", lambda _: None)
    path, payload = write_payload(tmp_path, 300_000)
    client, attempts = make_recording_client(fail_parts={2: 1})

    uploader = FakeMultipartUpload(client=client, chunk_size=100_000, max_concurrency=1)
    uploader.upload_file(path, on_bytes_uploaded=lambda _: None)

    part_two = [body for number, body in attempts if number == 2]
    assert len(part_two) == 2, "expected one failed attempt and one retry"
    expected_chunk = payload[100_000:200_000]
    for body in part_two:
        assert expected_chunk in body
    # Bodies differ only by the multipart boundary, which is regenerated per
    # request, so compare sizes rather than bytes.
    assert len(part_two[0]) == len(part_two[1])


def test_digest_is_skipped_unless_requested(tmp_path):
    path, _ = write_payload(tmp_path, 300_000)

    uploader = FakeMultipartUpload(
        client=make_client(), chunk_size=100_000, max_concurrency=2
    )
    etag = uploader.upload_file(path)

    assert etag == FINAL_ETAG
    assert uploader.content_md5 is None


def test_file_shrinking_mid_upload_is_rejected(tmp_path, monkeypatch):
    path, payload = write_payload(tmp_path, 300_000)
    real_open = open

    class ShortReader(io.BytesIO):
        """Serves the first chunk, then behaves as if the file ended."""

        def __init__(self, data):
            super().__init__(data)
            self._reads = 0

        def read(self, size=-1):
            self._reads += 1
            if self._reads > 1:
                return b""
            return super().read(size)

    def truncating_open(file, *args, **kwargs):
        if str(file) == path:
            return ShortReader(payload)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", truncating_open)
    uploader = FakeMultipartUpload(
        client=make_client(), chunk_size=100_000, max_concurrency=1
    )

    with pytest.raises(RuntimeError, match="changed while uploading"):
        uploader.upload_file(path, compute_md5=True)
