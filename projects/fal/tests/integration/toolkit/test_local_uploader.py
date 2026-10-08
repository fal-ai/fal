"""Opt-in local process tests: real Rust uploader and a controlled HTTP peer.

No cloud credentials are used. Set CDN_UPLOADER_TEST_BINARY to a locally built
isolate-cloud/monorust/target/debug/cdn-uploader binary to run these tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import urlparse

import httpx
import pytest

from fal.toolkit.file import File, _local_uploader
from fal.toolkit.file._local_uploader import LocalUploadError
from fal.toolkit.file.providers import fal as remote
from fal.toolkit.file.providers.local import LocalFileRepository

pytestmark = pytest.mark.skipif(
    not os.environ.get("CDN_UPLOADER_TEST_BINARY"),
    reason="Set CDN_UPLOADER_TEST_BINARY for local uploader process tests",
)


def _wait_healthy(socket_path, process, log):
    deadline = time.monotonic() + 10
    with httpx.Client(
        transport=httpx.HTTPTransport(uds=socket_path), timeout=0.2
    ) as client:
        while True:
            try:
                if client.get("http://localhost/health").status_code == 200:
                    return
            except httpx.TransportError:
                pass
            if process.poll() is not None or time.monotonic() > deadline:
                log.seek(0)
                pytest.fail("Uploader failed to start: " + log.read())
            time.sleep(0.02)


@pytest.fixture
def uploader(monkeypatch):
    release = threading.Event()
    receiving = threading.Event()
    done = threading.Event()
    state = {
        "parts": {},
        "multipart": False,
        "headers": None,
        "reservations": 0,
        "reject": None,
    }

    class Handler(BaseHTTPRequestHandler):
        def body(self):
            return self.rfile.read(int(self.headers.get("Content-Length", 0)))

        def respond(self, data, *, status=200, etag=None):
            encoded = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            if etag:
                self.send_header("ETag", etag)
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self):
            body = self.body()
            if self.path.startswith("/storage/upload/initiate"):
                state["reservations"] += 1
                number = state["reservations"]
                multipart = "multipart" in self.path
                if number == 1:
                    state["headers"] = self.headers
                    state["multipart"] = multipart
                suffix = "/multipart/session" if multipart else ""
                self.respond(
                    {
                        "file_url": origin + f"/file/{number}",
                        "upload_url": origin
                        + f"/file/{number}"
                        + suffix
                        + "?signature=test",
                    }
                )
            else:
                assert urlparse(self.path).path == "/file/1/multipart/session/complete"
                assert json.loads(body)["parts"]
                done.set()
                self.respond({})

        def do_PUT(self):
            body = self.body()
            path = urlparse(self.path).path
            number = int(path.rsplit("/", 1)[-1]) if state["multipart"] else 1
            state["parts"][number] = body
            receiving.set()
            if not release.wait(20):
                return
            if state["reject"]:
                return self.respond({}, status=state["reject"])
            if not state["multipart"]:
                done.set()
            self.respond({}, etag=f'"part-{number}"')

        def do_GET(self):
            assert self.path == "/file/1/inspect"
            self.respond(
                {
                    "metadata": {
                        "kind": "async-multipart" if state["multipart"] else "async",
                        "state": "done" if done.is_set() else "pending",
                        "signature": "test",
                    }
                }
            )

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as peer:
        origin = f"http://127.0.0.1:{peer.server_port}"
        thread = threading.Thread(target=peer.serve_forever, daemon=True)
        thread.start()
        with tempfile.TemporaryDirectory(prefix="fal-upl-", dir="/tmp") as directory:
            root = Path(directory)
            spool = root / "spool"
            spool.mkdir()
            socket_path = str(root / "upload.sock")
            env = {
                **os.environ,
                "CDN_UPLOADER_SOCKET_PATH": socket_path,
                "CDN_UPLOADER_SPOOL_DIR": str(spool),
                "CDN_UPLOADER_REST_BASE_URL": origin,
                "CDN_UPLOADER_METRICS_ADDR": "127.0.0.1:0",
                "CDN_UPLOADER_MAX_SPOOL_FILES": "1",
                "CDN_UPLOADER_MAX_SPOOL_BYTES": str(256 * 1024 * 1024),
                "CDN_UPLOADER_MIN_FREE_BYTES": "0",
                "CDN_UPLOADER_MAX_QUEUE_AGE_SECONDS": "60",
                "CDN_UPLOADER_FAILED_RETENTION_SECONDS": "60",
                "CDN_UPLOADER_SWEEP_INTERVAL_SECONDS": "1",
            }
            env.pop("CDN_UPLOADER_CDN_BASE_URL", None)
            with (root / "uploader.log").open("w+") as log:
                process = subprocess.Popen(
                    [env["CDN_UPLOADER_TEST_BINARY"]], env=env, stdout=log, stderr=log
                )
                try:
                    _wait_healthy(socket_path, process, log)
                    monkeypatch.setenv("CDN_UPLOADER_SOCKET_PATH", socket_path)
                    monkeypatch.setenv("FAL_USE_LOCAL_UPLOADER", "1")
                    monkeypatch.setenv("FAL_KEY", "local:test")
                    yield SimpleNamespace(
                        release=release, receiving=receiving, done=done, state=state
                    )
                finally:
                    release.set()
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                    peer.shutdown()
                    thread.join(10)


@pytest.mark.parametrize("size_mib", [80, 101])
def test_runner_exit_after_acceptance(uploader, tmp_path, monkeypatch, size_mib):
    source = tmp_path / "video.bin"
    chunk = bytes(range(256)) * 4096
    digest = hashlib.sha256()
    with source.open("wb") as output:
        for _ in range(size_mib):
            output.write(chunk)
            digest.update(chunk)
    runner = subprocess.run(
        [
            sys.executable,
            "-c",
            "from fal.toolkit import File; import sys; "
            "print(File.from_path(sys.argv[1], content_type='video/mp4').url)",
            str(source),
        ],
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert runner.returncode == 0, runner.stderr
    assert b"/file/1" in runner.stdout
    # The retained file claims the sole spool slot. Another large submission is
    # definitely rejected, retried, then sent directly to CDN.
    direct = Mock(return_value=("https://direct.example/video.bin", None))
    sleep = Mock()
    monkeypatch.setattr(remote.FalFileRepositoryV3, "save_file", direct)
    monkeypatch.setattr("fal.toolkit.utils.retry.time.sleep", sleep)
    assert File.from_path(source, multipart=True).url == direct.return_value[0]
    direct.assert_called_once()
    assert direct.call_args.args[0] == source
    assert [call.args[0] for call in sleep.call_args_list] == [0.1, 0.2]
    source.unlink()
    assert uploader.receiving.wait(10)
    assert not uploader.done.is_set()  # caller exited while transfer is blocked
    assert uploader.state["multipart"] == (size_mib > 100)
    assert uploader.state["headers"]["Authorization"] == "Key local:test"
    uploader.release.set()
    assert uploader.done.wait(10)
    assert uploader.state["reservations"] == 1
    actual = hashlib.sha256()
    for _, part in sorted(uploader.state["parts"].items()):
        actual.update(part)
    assert actual.digest() == digest.digest()


def test_wait_holds_until_cdn_completion(uploader, monkeypatch):
    # Short status waits make the client see pending and wait again.
    monkeypatch.setattr(_local_uploader, "_WAIT_SECONDS", 1)
    result = {}

    def save():
        result["file"] = File.from_bytes(
            b"hello", save_kwargs={"wait_for_completion": True}
        )

    saving = threading.Thread(target=save)
    saving.start()
    assert uploader.receiving.wait(10)
    saving.join(2.5)
    assert saving.is_alive()
    uploader.release.set()
    saving.join(10)
    assert uploader.done.is_set()
    assert result["file"].url.endswith("/file/1")


def test_wait_raises_after_failed_transfer(uploader, monkeypatch):
    uploader.state["reject"] = 403
    uploader.release.set()
    direct = Mock()
    monkeypatch.setattr(remote.FalFileRepositoryV3, "save", direct)
    with pytest.raises(LocalUploadError, match="failed"):
        File.from_bytes(b"hello", save_kwargs={"wait_for_completion": True})
    direct.assert_not_called()
    assert uploader.state["parts"] == {1: b"hello"}


def test_generated_stream_waits_for_completion(uploader):
    uploader.release.set()
    url = LocalFileRepository().save_stream(
        iter([b"hello", b" world"]),
        "generated.txt",
        "text/plain",
        wait_for_completion=True,
    )
    assert url.endswith("/file/1")
    assert uploader.done.is_set()
    assert uploader.state["parts"] == {1: b"hello world"}
