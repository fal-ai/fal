import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from fal.dev.remote import proxy

pytestmark = pytest.mark.usefixtures("dev_account")


def test_forward_refuses_child_control_paths():
    for path in ("/__fal_dev_child/reload", "/%5F_fal_dev_child/reload", "x"):
        with pytest.raises(ValueError):
            proxy.forward({"path": path}, 1)


def _fake_child(monkeypatch, statuses, times):
    calls: list[dict] = []

    def forward(request, timeout):
        index = len(calls)
        calls.append({**request, "timeout": timeout})
        return {
            "status_code": statuses[index],
            "content_type": "application/json",
            "elapsed_seconds": times[index],
            "body": json.dumps({"run": index}),
        }

    monkeypatch.setattr(proxy, "forward", forward)
    return calls


def test_forward_runs_times_warmups_and_runs_separately(monkeypatch):
    calls = _fake_child(monkeypatch, [200] * 5, [9.0, 1.0, 2.0, 3.0, 4.0])
    result = proxy.forward_runs({"path": "/", "warmups": 1, "repeat": 4})

    assert len(calls) == 5
    assert [run["elapsed_seconds"] for run in result["warmup_runs"]] == [9.0]
    assert [run["elapsed_seconds"] for run in result["runs"]] == [1, 2, 3, 4]
    # The last response comes back, not all of them.
    assert json.loads(result["body"]) == {"run": 4}


def test_forward_runs_stops_at_the_first_failure(monkeypatch):
    calls = _fake_child(monkeypatch, [200, 422, 200], [1.0, 1.0, 1.0])
    result = proxy.forward_runs({"path": "/", "repeat": 3})

    assert len(calls) == 2
    assert result["status_code"] == 422
    assert [run["status_code"] for run in result["runs"]] == [200, 422]


@pytest.mark.parametrize(
    "payload",
    [{"repeat": 0}, {"repeat": 101}, {"warmups": -1}, {"warmups": "2"}],
)
def test_forward_runs_checks_counts(payload):
    with pytest.raises(ValueError):
        proxy.forward_runs({"path": "/", **payload})


def test_forward_runs_stops_at_the_time_budget(monkeypatch):
    calls = _fake_child(monkeypatch, [200] * 3, [1.0] * 3)
    clock = iter([0.0, 10.0, 4000.0])
    monkeypatch.setattr(proxy.time, "monotonic", lambda: next(clock))
    result = proxy.forward_runs({"path": "/", "repeat": 3})
    assert "time budget" in result["stopped"]
    assert len(result["runs"]) == 2
    # The second run only gets what is left of the budget.
    assert calls[1]["timeout"] == proxy.RUNS_BUDGET_SECONDS - 10


def test_forward_runs_keeps_runs_when_a_later_run_fails_to_connect(monkeypatch):
    calls = _fake_child(monkeypatch, [200], [1.0])
    forward = proxy.forward

    def flaky(request, timeout):
        if calls:
            raise TimeoutError("timed out")
        return forward(request, timeout)

    monkeypatch.setattr(proxy, "forward", flaky)
    result = proxy.forward_runs({"path": "/", "repeat": 3})
    assert result["stopped"] == "run 2 did not complete: timed out"
    assert [run["elapsed_seconds"] for run in result["runs"]] == [1.0]


class _ChunkedChild(BaseHTTPRequestHandler):
    """Answers with three separate chunks and records what it was sent."""

    protocol_version = "HTTP/1.1"
    seen: list = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers["content-length"]))
        _ChunkedChild.seen.append((self.path, dict(self.headers), body))
        status = 503 if self.path == "/fail" else 200
        self.send_response(status)
        self.send_header("content-type", "text/event-stream")
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()
        for chunk in (b"data: 1\n\n", b"data: 2\n\n", b"data: [DONE]\n\n"):
            self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")


@pytest.fixture
def chunked_child(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ChunkedChild)
    monkeypatch.setattr(proxy, "CHILD_PORT", server.server_address[1])
    _ChunkedChild.seen = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield _ChunkedChild
    server.shutdown()
    server.server_close()


def test_stream_passes_the_body_and_streams_the_response(chunked_child):
    body = json.dumps({"stream": True}).encode()
    status, content_type, chunks, close = proxy.stream(
        {
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": {"Content-Type": "application/json", "Authorization": "Key k"},
            "body_base64": base64.b64encode(body).decode(),
        }
    )
    assert (status, content_type) == (200, "text/event-stream")
    assert b"".join(chunks) == b"data: 1\n\ndata: 2\n\ndata: [DONE]\n\n"
    path, headers, sent = chunked_child.seen[0]
    assert (path, sent) == ("/v1/chat/completions", body)
    # fal auth never reaches the app.
    assert "Authorization" not in headers
    assert headers["Content-Type"] == "application/json"


def test_stream_refuses_child_control_paths_and_bad_bodies(chunked_child):
    with pytest.raises(ValueError):
        proxy.stream({"path": "/__fal_dev_child/reload"})
    with pytest.raises(ValueError):
        proxy.stream({"path": "/", "body_base64": "not base64!"})
    assert chunked_child.seen == []


def test_closing_a_stream_stops_a_read_in_progress(monkeypatch):
    import socket
    import threading
    import time

    listener = socket.create_server(("127.0.0.1", 0))
    monkeypatch.setattr(proxy, "CHILD_PORT", listener.getsockname()[1])

    def stalled_app():
        conn, _ = listener.accept()
        conn.recv(65536)
        conn.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n"
        )
        time.sleep(5)
        conn.close()

    threading.Thread(target=stalled_app, daemon=True).start()
    status, _, chunks, close = proxy.stream({"method": "GET", "path": "/"})
    assert next(chunks) == b"hello"
    finished = []

    def read_rest():
        try:
            finished.append(list(chunks))
        except Exception as exc:  # an aborted read may raise instead
            finished.append(exc)

    reader = threading.Thread(target=read_rest)
    reader.start()
    time.sleep(0.2)
    started = time.monotonic()
    close()
    reader.join(timeout=2)
    assert time.monotonic() - started < 1.5 and finished, finished
    listener.close()
