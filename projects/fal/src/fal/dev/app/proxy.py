"""`fal dev app proxy`: a local HTTP endpoint for the app on a dev box.

Any local client (an OpenAI SDK, a coding agent, curl) talks plain HTTP to
127.0.0.1. Each request is wrapped in the box's JSON envelope with the session
token, signed with fresh fal credentials, and sent to the box's
``/__fal_dev/proxy`` route, which passes it to the app process and streams the
response back. The client never holds a fal key or the box token.

A box scaled to zero comes back on a new runner without the app process; the
first request that finds it so boots the app again and is then retried.
"""

from __future__ import annotations

import base64
import ipaddress
import json
import socket
import sys
import threading
import time
import urllib.parse
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx

from fal.dev.app.session import Session
from fal.dev.remote.protocol import (
    APP_PHASE_HEADER,
    REQUEST_TIMEOUT_SECONDS,
    ROUTE_PREFIX,
    STREAM_HEADERS,
    TOKEN_FIELD,
)

# Re-signing every request costs a token check; reuse headers for this long.
AUTH_TTL_SECONDS = 300.0
# The box's holder passes these back; everything else is fal.run's own.
RESPONSE_HEADERS = ("content-type", "x-fal-dev-app-status")
# A request body is held in memory and sent base64-encoded in JSON.
MAX_REQUEST_BYTES = 32 * 1024 * 1024
LOOPBACK = ("127.0.0.1", "localhost", "::1")


class _Auth:
    """fal auth headers for the box's account, refreshed now and then."""

    def __init__(self, session: Session) -> None:
        self._team = session.account.fal_team
        self._headers: dict[str, str] = {}
        self._at = 0.0
        self._lock = threading.Lock()

    def headers(self) -> dict[str, str]:
        from fal.sdk import get_credentials

        with self._lock:
            if time.monotonic() - self._at > AUTH_TTL_SECONDS:
                self._headers = get_credentials(team=self._team).to_headers()
                self._at = time.monotonic()
            return self._headers

    def expire(self) -> None:
        """Sign the next request afresh (after a 401, say)."""
        with self._lock:
            self._at = 0.0


def envelope(session: Session, method: str, path: str, headers: Any, body: bytes):
    """The box's JSON form of one client request."""
    return {
        TOKEN_FIELD: session.token,
        "method": method,
        "path": path,
        "headers": {
            key: value
            for key, value in headers.items()
            if key.lower() in STREAM_HEADERS
        },
        "body_base64": base64.b64encode(body).decode(),
        "timeout": REQUEST_TIMEOUT_SECONDS,
    }


def _host_name(host: str) -> str:
    """The name part of a Host header: `[::1]:8899` -> `::1`."""
    if host.startswith("["):
        return host[1:].split("]", 1)[0]
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def make_handler(
    session: Session,
    client: httpx.Client,
    report: Callable[[str], None],
    allowed_hosts: tuple[str, ...] | None = LOOPBACK,
    reboot: Callable[[], None] | None = None,
) -> type[BaseHTTPRequestHandler]:
    """The request handler; ``allowed_hosts`` are the names a Host may carry
    (None, for a server bound beyond loopback: any). ``reboot`` boots the app
    on the box again; requests the box refuses for want of it wait for it.

    A browser page can reach 127.0.0.1 too (directly, or by DNS rebinding a
    name of its own), so requests from a browser (an Origin header) or for a
    host that is not ours are refused: each request is signed as the user.
    """
    auth = _Auth(session)
    url = f"{session.run_url}{ROUTE_PREFIX}proxy"
    # One boot at a time; requests that find the app down meanwhile wait.
    booting = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            return None

        def do_GET(self) -> None:
            self._forward()

        def do_POST(self) -> None:
            self._forward()

        def do_PUT(self) -> None:
            self._forward()

        def do_PATCH(self) -> None:
            self._forward()

        def do_DELETE(self) -> None:
            self._forward()

        def _refusal(self) -> tuple[int, str] | None:
            host = _host_name(self.headers.get("host", ""))
            if allowed_hosts is not None and host not in allowed_hosts:
                return 403, "unexpected Host header"
            # Browsers send these; curl, SDKs and agents do not. A cross-site
            # GET carries no Origin, but does carry Sec-Fetch-Site. A page the
            # proxy itself served (the app's /docs) may still fetch.
            fetch_site = self.headers.get("sec-fetch-site", "none")
            origin = self.headers.get("origin")
            origin_host = _host_name(urllib.parse.urlsplit(origin or "").netloc)
            if fetch_site not in ("none", "same-origin") or (
                origin is not None
                and allowed_hosts is not None
                and origin_host not in allowed_hosts
            ):
                return 403, "requests from another site are refused"
            if "transfer-encoding" in self.headers:
                return 411, "send the request body with a Content-Length"
            try:
                length = int(self.headers.get("content-length") or 0)
            except ValueError:
                return 400, "invalid Content-Length"
            if not 0 <= length <= MAX_REQUEST_BYTES:
                return 413, "the request body is over the 32 MiB proxy limit"
            return None

        def _forward(self) -> None:
            started = time.perf_counter()
            refused = self._refusal()
            if refused is not None:
                self._fail(*refused)
                report(f"{self.command} {self.path}  refused: {refused[1]}")
                return
            length = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(length) if length else b""
            payload = envelope(session, self.command, self.path, self.headers, body)
            first_byte = None
            sent = 0
            headers_sent = False
            response = None
            try:
                response = self._open(payload)
                self.send_response(response.status_code)
                for key in RESPONSE_HEADERS:
                    if key in response.headers:
                        self.send_header(key, response.headers[key])
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                headers_sent = True
                for chunk in response.iter_bytes():
                    if first_byte is None:
                        first_byte = time.perf_counter() - started
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                    self.wfile.flush()
                    sent += len(chunk)
                self.wfile.write(b"0\r\n\r\n")
                status = response.status_code
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
                report(f"{self.command} {self.path}  client went away")
                return
            except Exception as exc:
                # Once the response has started, a 502 cannot follow: end the
                # connection unterminated, so the client sees the stream fail.
                if headers_sent:
                    self.close_connection = True
                else:
                    self._fail(502, f"could not reach the dev box: {exc}")
                report(f"{self.command} {self.path}  error: {exc}")
                return
            finally:
                if response is not None:
                    response.close()
            ttfb = f"{first_byte:.2f}s" if first_byte is not None else "-"
            total = time.perf_counter() - started
            report(
                f"{self.command} {self.path}  {status}  "
                f"first byte {ttfb}  total {total:.1f}s  {sent / 1024:.0f} KiB"
            )

        def _send(self, payload: dict) -> httpx.Response:
            headers = {**auth.headers(), "X-Fal-Store-IO": "0"}
            request = client.build_request("POST", url, json=payload, headers=headers)
            response = client.send(request, stream=True)
            if response.status_code == 401:
                auth.expire()
            return response

        def _open(self, payload: dict) -> httpx.Response:
            """The box's response; when the app is down, boot it and retry once."""
            response = self._send(payload)
            if reboot is None or APP_PHASE_HEADER not in response.headers:
                return response
            response.close()
            with booting:
                # A no-op when another request booted the app meanwhile.
                reboot()
            return self._send(payload)

        def _fail(self, status: int, message: str) -> None:
            data = json.dumps({"detail": message}).encode()
            self.close_connection = True
            try:
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except OSError:
                pass

    return Handler


class _Server6(ThreadingHTTPServer):
    address_family = socket.AF_INET6


def serve(
    session: Session,
    host: str,
    port: int,
    report: Callable[[str], None] = lambda line: print(line, file=sys.stderr),
    reboot: Callable[[], None] | None = None,
) -> ThreadingHTTPServer:
    """A server for the box's app on host:port; call ``serve_forever()`` on it.

    Raises OSError when it cannot bind there (the port is taken, say).
    """
    client = httpx.Client(
        timeout=httpx.Timeout(
            connect=30, read=REQUEST_TIMEOUT_SECONDS, write=60, pool=60
        ),
        limits=httpx.Limits(max_connections=64),
    )
    allowed = (*LOOPBACK, host) if is_loopback(host) else None
    server_class = _Server6 if ":" in host else ThreadingHTTPServer
    try:
        server = server_class(
            (host, port), make_handler(session, client, report, allowed, reboot)
        )
    except BaseException:
        client.close()
        raise
    server.daemon_threads = True
    return server


def server_url(host: str, port: int) -> str:
    return f"http://[{host}]:{port}" if ":" in host else f"http://{host}:{port}"
