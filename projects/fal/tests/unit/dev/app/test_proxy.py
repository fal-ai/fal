import base64
import json
import socket
import threading

import httpx
import pytest

from dev.app.fakes import make_session
from fal.dev.app import proxy

pytestmark = pytest.mark.usefixtures("dev_account")


def test_envelope_carries_the_token_and_only_content_headers():
    session = make_session()
    payload = proxy.envelope(
        session,
        "POST",
        "/v1/chat/completions",
        {"Content-Type": "application/json", "Authorization": "Bearer local"},
        b'{"a": 1}',
    )
    assert payload["token"] == "t"
    assert payload["headers"] == {"Content-Type": "application/json"}
    assert base64.b64decode(payload["body_base64"]) == b'{"a": 1}'


def test_proxy_streams_the_box_response_to_a_local_client(monkeypatch):
    monkeypatch.setattr(proxy._Auth, "headers", lambda self: {"Authorization": "Key k"})
    sent = []

    def box(request: httpx.Request) -> httpx.Response:
        sent.append(
            (str(request.url), dict(request.headers), json.loads(request.content))
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream", "x-other": "dropped"},
            content=b"data: hi\n\ndata: [DONE]\n\n",
        )

    client = httpx.Client(transport=httpx.MockTransport(box))
    lines = []
    server = proxy.serve(make_session(), "127.0.0.1", 0, report=lines.append)
    server.RequestHandlerClass = proxy.make_handler(
        make_session(), client, lines.append
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        response = httpx.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={"stream": True},
            headers={"Authorization": "Bearer local"},
        )
    finally:
        server.shutdown()
        server.server_close()
    assert response.status_code == 200
    assert response.text == "data: hi\n\ndata: [DONE]\n\n"
    assert "x-other" not in response.headers
    url, headers, payload = sent[0]
    assert url == "https://fal.run/me/dev-demo-me/__fal_dev/proxy"
    assert headers["authorization"] == "Key k"
    assert headers["x-fal-store-io"] == "0"
    assert payload["path"] == "/v1/chat/completions"
    assert json.loads(base64.b64decode(payload["body_base64"])) == {"stream": True}
    assert lines and "POST /v1/chat/completions  200" in lines[0]


def _serve(monkeypatch, box):
    monkeypatch.setattr(proxy._Auth, "headers", lambda self: {"Authorization": "Key k"})
    client = httpx.Client(transport=httpx.MockTransport(box))
    lines: list[str] = []
    server = proxy.serve(make_session(), "127.0.0.1", 0, report=lines.append)
    server.RequestHandlerClass = proxy.make_handler(
        make_session(), client, lines.append
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, lines


@pytest.mark.parametrize(
    "headers, status",
    [
        ({"Origin": "https://evil.example"}, 403),
        ({"Sec-Fetch-Site": "cross-site"}, 403),
        ({"Sec-Fetch-Site": "cross-site"}, 403),
        ({"Host": "evil.example:8899"}, 403),
        ({"Transfer-Encoding": "chunked"}, 411),
        ({"Content-Length": "nope"}, 400),
    ],
)
def test_proxy_refuses_browsers_foreign_hosts_and_bad_bodies(
    monkeypatch, headers, status
):
    sent = []
    server, lines = _serve(monkeypatch, lambda request: sent.append(1))
    try:
        with socket.create_connection(server.server_address) as conn:
            fields = {"Host": "127.0.0.1", "Content-Length": "2", **headers}
            head = "".join(f"{k}: {v}\r\n" for k, v in fields.items())
            conn.sendall(f"POST /v1/models HTTP/1.1\r\n{head}\r\n{{}}".encode())
            reply = conn.recv(4096).decode()
    finally:
        server.shutdown()
        server.server_close()
    assert reply.startswith(f"HTTP/1.1 {status} ")
    assert sent == [] and "refused" in lines[0]


def test_proxy_ends_a_stream_that_fails_midway_without_a_second_status(monkeypatch):
    class Broken(httpx.SyncByteStream):
        def __iter__(self):
            yield b"data: 1\n\n"
            raise httpx.ReadError("box went away")

    server, lines = _serve(
        monkeypatch,
        lambda request: httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Broken()
        ),
    )
    try:
        port = server.server_address[1]
        with pytest.raises(httpx.RemoteProtocolError):
            httpx.get(f"http://127.0.0.1:{port}/v1/models")
    finally:
        server.shutdown()
        server.server_close()
    assert "error: box went away" in lines[0]


def test_proxy_serves_a_page_it_served_itself(monkeypatch):
    server, _ = _serve(monkeypatch, lambda request: httpx.Response(200, content=b"{}"))
    try:
        port = server.server_address[1]
        response = httpx.get(
            f"http://127.0.0.1:{port}/openapi.json",
            headers={
                "Sec-Fetch-Site": "same-origin",
                "Origin": f"http://localhost:{port}",
            },
        )
    finally:
        server.shutdown()
        server.server_close()
    assert response.status_code == 200


def _serve_with_reboot(monkeypatch, box, reboot):
    monkeypatch.setattr(proxy._Auth, "headers", lambda self: {"Authorization": "Key k"})
    client = httpx.Client(transport=httpx.MockTransport(box))
    lines: list[str] = []
    server = proxy.serve(make_session(), "127.0.0.1", 0, report=lines.append)
    server.RequestHandlerClass = proxy.make_handler(
        make_session(), client, lines.append, reboot=reboot
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_proxy_boots_an_app_the_box_lost_and_retries(monkeypatch):
    phases = ["idle", None]
    reboots = []

    def box(request):
        phase = phases.pop(0)
        if phase:
            return httpx.Response(400, headers={"X-Fal-Dev-App-Phase": phase})
        return httpx.Response(200, content=b"ok")

    server = _serve_with_reboot(monkeypatch, box, lambda: reboots.append(1))
    try:
        port = server.server_address[1]
        response = httpx.get(f"http://127.0.0.1:{port}/v1/models")
    finally:
        server.shutdown()
        server.server_close()
    assert response.status_code == 200 and response.text == "ok"
    assert reboots == [1] and phases == []


def test_proxy_passes_other_refusals_through_and_retries_only_once(monkeypatch):
    calls = []

    def box(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(400, json={"detail": "invalid path"})
        return httpx.Response(400, headers={"X-Fal-Dev-App-Phase": "exited"})

    reboots = []
    server = _serve_with_reboot(monkeypatch, box, lambda: reboots.append(1))
    try:
        port = server.server_address[1]
        first = httpx.get(f"http://127.0.0.1:{port}/x")
        second = httpx.get(f"http://127.0.0.1:{port}/x")
    finally:
        server.shutdown()
        server.server_close()
    assert first.status_code == 400 and reboots == [1]
    assert second.status_code == 400 and len(calls) == 3


@pytest.mark.parametrize(
    "host, loopback",
    [
        ("127.0.0.1", True),
        ("127.0.0.2", True),
        ("::1", True),
        ("localhost", True),
        ("0.0.0.0", False),
        ("192.168.1.5", False),
    ],
)
def test_is_loopback(host, loopback):
    assert proxy.is_loopback(host) is loopback


def test_serve_binds_ipv6_loopback():
    if not socket.has_ipv6:
        pytest.skip("no IPv6")
    try:
        server = proxy.serve(make_session(), "::1", 0)
    except OSError:
        pytest.skip("IPv6 loopback unavailable")
    try:
        assert server.server_address[0] == "::1"
    finally:
        server.server_close()
    assert proxy.server_url("::1", 8899) == "http://[::1]:8899"
    assert proxy.server_url("127.0.0.1", 8899) == "http://127.0.0.1:8899"
