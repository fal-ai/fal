"""Client for the node-local uploader's HTTP API over a Unix socket.

Acceptance is durable on this node, not CDN completion; callers that need the
CDN copy wait for completion. Do not serialize live connections.
"""

from __future__ import annotations

import os
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Iterable, Iterator
from urllib.parse import quote

from fal.toolkit.exceptions import FileUploadException

if TYPE_CHECKING:
    import httpx

# Keep in sync with infra/modules/nomad_jobs/jobs.tf and isolate-cloud's
# projects/isolate_controller/src/isolate_controller/scheduler/nomad/forger.py.
DEFAULT_SOCKET_PATH = "/run/fal-upload/upload.sock"
_TIMEOUT = 300
_CONNECT_TIMEOUT = 5
# How long each status request holds while the upload is pending.
_WAIT_SECONDS = 60
_PRINTABLE_ASCII = "".join(map(chr, range(0x20, 0x7F)))


class LocalUploadError(FileUploadException):
    """The node-local uploader failed or could not be reached."""


class LocalUploadRejected(LocalUploadError):
    """The uploader is full or draining and did not accept the work."""


def upload(
    file_name: str,
    body: bytes | Iterable[bytes],
    size_bytes: int,
    headers: dict[str, str],
    *,
    wait_for_completion: bool = False,
) -> str:
    """Return the URL once the complete known-length body is durably accepted,
    or once the CDN has it when waiting for completion."""
    with _new_client() as http:
        response = _request(
            http,
            "POST",
            "/uploads",
            202,
            headers={
                **headers,
                "X-Fal-File-Name": _header_file_name(file_name),
                "Content-Length": str(size_bytes),
            },
            content=body,
        )
        upload_id, file_url = _upload_info(response, "accepted_local")
        if wait_for_completion:
            _wait_for_completion(http, upload_id)
    return file_url


def upload_stream(
    file_name: str,
    chunks: Iterable[bytes],
    headers: dict[str, str],
    *,
    wait_for_completion: bool = False,
) -> str:
    """Return the URL once a body of unknown size is durably accepted, or once
    the CDN has it when waiting for completion.

    The uploader accepts only on an explicit finish carrying the final byte
    count. Any failure, including a raising producer, aborts the session.
    """
    size = 0

    def counted() -> Iterator[bytes]:
        nonlocal size
        for chunk in chunks:
            size += len(chunk)
            yield chunk

    with _new_client() as http:
        response = _request(
            http,
            "POST",
            "/upload-sessions",
            201,
            headers={**headers, "X-Fal-File-Name": _header_file_name(file_name)},
        )
        upload_id, _ = _upload_info(response, "receiving")
        path = "/upload-sessions/" + quote(upload_id, safe="")
        try:
            _request(http, "PUT", path + "/body", 204, content=counted())
            response = _request(
                http,
                "POST",
                path + "/finish",
                202,
                json={"size_bytes": size},
            )
            upload_id, file_url = _upload_info(response, "accepted_local")
        except BaseException:
            # Aborting must not mask the original failure.
            with suppress(LocalUploadError):
                _request(http, "DELETE", path, 202)
            raise
        if wait_for_completion:
            _wait_for_completion(http, upload_id)
    return file_url


def _wait_for_completion(http: httpx.Client, upload_id: str) -> None:
    """Return once the CDN has the file.

    The uploader fails pending uploads at their queue deadline, so this ends. A
    404 means the uploader restarted or dropped the record: the outcome is unknown.
    """
    path = "/uploads/" + quote(upload_id, safe="")
    while True:
        response = _request(http, "GET", path, 200, params={"wait": _WAIT_SECONDS})
        try:
            status = response.json()
            state = status["state"]
        except (ValueError, KeyError, TypeError):
            raise LocalUploadError(
                "Local uploader returned an invalid response."
            ) from None
        if state == "completed":
            return
        if state != "pending":
            raise LocalUploadError(f"Local upload {state} ({status.get('failure')}).")


def _request(
    http: httpx.Client,
    method: str,
    path: str,
    expected_status: int,
    **kwargs: Any,
) -> httpx.Response:
    import httpx  # noqa: PLC0415 -- see _new_client

    # Name the failure class but drop its text, which can carry credentials
    # or signed URLs.
    try:
        response = http.request(method, path, **kwargs)
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
        raise LocalUploadError(
            f"Cannot connect to the local uploader ({type(exc).__name__})."
        ) from None
    except httpx.RequestError as exc:
        raise LocalUploadError(
            f"Local uploader connection was interrupted ({type(exc).__name__})."
        ) from None
    if response.status_code != expected_status:
        message = f"Local uploader returned HTTP {response.status_code}."
        if response.status_code == 429:
            raise LocalUploadRejected(message)
        raise LocalUploadError(message)
    return response


def _header_file_name(file_name: str) -> str:
    """Percent-encode only what cannot travel in an HTTP header.

    Header values are ASCII: httpx refuses to encode anything else, and both the
    uploader and the CDN reject it. Printable ASCII names pass unchanged, as the
    Rust client sends them.
    """
    return quote(file_name, safe=_PRINTABLE_ASCII)


def _new_client() -> httpx.Client:
    # Local, not module scope, for the reason given in _upload_policy._new_client.
    import httpx  # noqa: PLC0415

    # Resolve the socket on the runner, not when serializing the app.
    socket_path = os.environ.get("CDN_UPLOADER_SOCKET_PATH", DEFAULT_SOCKET_PATH)
    return httpx.Client(
        transport=httpx.HTTPTransport(
            uds=socket_path, verify=False, retries=0, trust_env=False
        ),
        base_url="http://localhost",
        trust_env=False,
        follow_redirects=False,
        timeout=httpx.Timeout(_TIMEOUT, connect=_CONNECT_TIMEOUT),
    )


def _upload_info(response: httpx.Response, state: str) -> tuple[str, str]:
    try:
        result = response.json()
        if result["state"] == state:
            return result["upload_id"], result["file_url"]
    except (ValueError, KeyError, TypeError):
        pass
    raise LocalUploadError("Local uploader returned an invalid response.")
