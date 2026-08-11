import hashlib
import json
import math
import os
import tempfile

import httpx
import pytest

from fal.exceptions import FalServerlessException
from fal.upload import (
    MULTIPART_CHUNK_SIZE,
    MULTIPART_MAX_PART_SIZE,
    MULTIPART_MAX_PARTS,
    DataFileMultipartUpload,
    compute_multipart_chunk_size,
)

MB = 1024 * 1024
GB = 1024 * MB


def _num_parts(size, chunk):
    return math.ceil(size / chunk)


@pytest.mark.parametrize("size", [1, MB, 50 * MB, 100 * GB, 200 * GB, 900 * GB])
def test_adaptive_chunk_size_stays_within_limits(size):
    chunk = compute_multipart_chunk_size(size)
    assert chunk >= MULTIPART_CHUNK_SIZE
    assert chunk <= MULTIPART_MAX_PART_SIZE
    assert chunk % MB == 0
    assert _num_parts(size, chunk) <= MULTIPART_MAX_PARTS


def test_small_file_keeps_default_chunk():
    assert compute_multipart_chunk_size(5 * MB) == MULTIPART_CHUNK_SIZE


def test_200gb_would_exceed_cap_with_old_fixed_chunk():
    # Regression guard for SERV-1404: the old fixed 10MB chunk overflowed the
    # 10k-part cap for a 200GB file; the adaptive size must not.
    assert _num_parts(200 * GB, MULTIPART_CHUNK_SIZE) > MULTIPART_MAX_PARTS
    adaptive = compute_multipart_chunk_size(200 * GB)
    assert _num_parts(200 * GB, adaptive) <= MULTIPART_MAX_PARTS


def test_file_over_ceiling_raises():
    over = MULTIPART_MAX_PART_SIZE * MULTIPART_MAX_PARTS + 1
    with pytest.raises(FalServerlessException):
        compute_multipart_chunk_size(over)


class _FakeServer:
    """Minimal stateful multipart server mirroring the controller behavior."""

    def __init__(self):
        self.store = {}  # upload_id -> {part_number: etag}
        self.put_log = []  # part numbers PUT across all runs
        self.fail_after = None  # fail once more than N successful PUTs happened

    def transport(self):
        return httpx.MockTransport(self._handle)

    def _handle(self, req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path.endswith("/initiate"):
            body = json.loads(req.content)
            uid = hashlib.sha256(
                f"{body.get('content_md5')}:{body.get('chunk_size')}".encode()
            ).hexdigest()
            self.store.setdefault(uid, {})
            parts = [
                {"part_number": n, "etag": e}
                for n, e in sorted(self.store[uid].items())
            ]
            return httpx.Response(200, json={"upload_id": uid, "parts": parts})
        if path.endswith("/complete"):
            return httpx.Response(200, json={"etag": "final"})
        segs = path.rstrip("/").split("/")
        part_number, uid = int(segs[-1]), segs[-2]
        self.put_log.append(part_number)
        if self.fail_after is not None and len(self.put_log) > self.fail_after:
            return httpx.Response(500, json={"detail": "boom"})
        etag = hashlib.md5(f"p{part_number}".encode()).hexdigest()
        self.store[uid][part_number] = etag
        return httpx.Response(200, json={"part_number": part_number, "etag": etag})


def _make_upload(client, md5):
    m = DataFileMultipartUpload(
        client=client, target_path="ckpt.bin", chunk_size=MB
    )
    m._content_md5 = md5
    return m


def test_resume_skips_already_uploaded_parts():
    server = _FakeServer()
    client = httpx.Client(transport=server.transport(), base_url="http://x")

    data = os.urandom(10 * MB + 123)  # 11 parts at 1MB
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(data)
        path = f.name
    md5 = hashlib.md5(data).hexdigest()

    try:
        # Run 1: fail after 4 successful parts.
        server.fail_after = 4
        with pytest.raises(Exception):
            _make_upload(client, md5).upload_file(path)

        stored = sorted(next(iter(server.store.values())).keys())
        assert len(stored) == 4

        # Run 2: succeeds, resuming and uploading only the missing parts.
        server.put_log.clear()
        server.fail_after = None
        etag = _make_upload(client, md5).upload_file(path)

        assert etag == "final"
        run2 = set(server.put_log)
        assert not (run2 & set(stored)), "resume re-uploaded already-stored parts"
        assert run2 | set(stored) == set(range(1, 12))
    finally:
        os.unlink(path)
