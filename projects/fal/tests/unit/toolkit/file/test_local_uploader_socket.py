"""Real Unix HTTP transport and runner serialization, without remote services."""

from __future__ import annotations

import asyncio
import json
import os
import socketserver
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fal.compat import run_in_thread
from fal.toolkit.file import File
from fal.toolkit.file.providers.fal import FalFileRepositoryV3

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="Node-local uploader requires a POSIX host"
)


@pytest.fixture
def uploader(monkeypatch, request):
    requests = []
    received = threading.Event()
    accept = threading.Event()
    accept.set()
    responded = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append((dict(self.headers), body))
            if len(requests) <= getattr(request, "param", 0):
                self.send_response(429)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            received.set()
            if not accept.wait(10):
                return
            payload = json.dumps(
                {
                    "upload_id": "accepted-id",
                    "file_url": "https://fal.media/file.bin",
                    "state": "accepted_local",
                }
            ).encode()
            self.send_response(202)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            responded.set()

        def log_message(self, *args):
            pass

    # pytest's nested temporary path can exceed sockaddr_un's path limit.
    with tempfile.TemporaryDirectory(prefix="fal-upl-", dir="/tmp") as directory:
        path = str(Path(directory) / "upload.sock")
        with socketserver.UnixStreamServer(path, Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            monkeypatch.setenv("FAL_USE_LOCAL_UPLOADER", "1")
            monkeypatch.setenv("CDN_UPLOADER_SOCKET_PATH", path)
            monkeypatch.setenv("FAL_KEY", "local:test")
            try:
                yield SimpleNamespace(
                    requests=requests,
                    received=received,
                    accept=accept,
                    responded=responded,
                )
            finally:
                accept.set()
                server.shutdown()
                thread.join(10)


def test_real_socket_ignores_http_proxy(uploader, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("SSL_CERT_FILE", "/nonexistent/certificate.pem")
    result = File.from_bytes(b"hello", file_name="hello.txt")
    assert result.url == "https://fal.media/file.bin"
    assert uploader.requests[0][1] == b"hello"
    assert uploader.requests[0][0]["Authorization"] == "Key local:test"


def test_missing_socket_falls_back_to_direct_cdn(monkeypatch, capsys):
    monkeypatch.setenv("FAL_USE_LOCAL_UPLOADER", "1")
    monkeypatch.setenv("CDN_UPLOADER_SOCKET_PATH", "/tmp/fal-missing-uploader/socket")
    monkeypatch.setenv("FAL_KEY", "local:test")
    direct = Mock(return_value="https://direct/file")
    monkeypatch.setattr(FalFileRepositoryV3, "save", direct)
    assert File.from_bytes(b"hello").url == "https://direct/file"
    direct.assert_called_once()
    assert "Cannot connect" in capsys.readouterr().out


@pytest.mark.parametrize("uploader", [2], indirect=True)
def test_serialized_sdk_reads_runner_environment(uploader, tmp_path):
    target = tmp_path / "file.pkl"
    # Serialize in a clean deploy process with the flag OFF. The new runner
    # enables it only after deserializing the app's copy of File.
    deploy = subprocess.run(
        [
            sys.executable,
            "-c",
            "from fal._serialization import patch_pickle; patch_pickle(); "
            "import cloudpickle; from fal.toolkit import File; import sys; "
            "open(sys.argv[1], 'wb').write(cloudpickle.dumps(File))",
            str(target),
        ],
        env={**os.environ, "FAL_USE_LOCAL_UPLOADER": "0"},
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert deploy.returncode == 0, deploy.stderr
    runner = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pickle, sys; "
            "File = pickle.load(open(sys.argv[1], 'rb')); "
            "print(File.from_bytes(b'from runner').url)",
            str(target),
        ],
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert runner.returncode == 0, runner.stderr
    assert b"https://fal.media/file.bin" in runner.stdout
    assert [body for _, body in uploader.requests] == [b"from runner"] * 3


@pytest.mark.asyncio
async def test_cancelled_await_does_not_cancel_or_retry_submission(uploader, tmp_path):
    uploader.accept.clear()
    path = tmp_path / "source.bin"
    path.write_bytes(b"hello")
    task = asyncio.create_task(File.from_path_async(path, multipart=True))
    try:
        assert await run_in_thread(uploader.received.wait, 10)
        assert not task.done()  # request body alone is not acceptance
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        path.unlink()  # worker still owns the open source handle
    finally:
        uploader.accept.set()
        assert await run_in_thread(uploader.responded.wait, 10)
    assert len(uploader.requests) == 1
    assert uploader.requests[0][1] == b"hello"
