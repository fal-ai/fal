"""Client for the node-local uploader on the fal API socket.

Acceptance is durable on this node, not CDN completion; callers that need the
CDN copy wait for completion. Do not serialize live connections.

Uploads carry the in-flight request's x-fal-* headers, which only the gateway
sets, and every request names the runner's app in x-fal-application.
"""

from __future__ import annotations

import os
import time
from contextlib import suppress
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator
from urllib.parse import quote

from fal.ref import get_current_app
from fal.toolkit.exceptions import FileUploadException
from fal.toolkit.file.types import Reservation

if TYPE_CHECKING:
    import httpx

# Keep in sync with infra/modules/nomad_jobs/nomad_cdn_uploader_job.hcl.tftpl
# and projects/isolate_controller/src/isolate_controller/scheduler/nomad/forger.py
# in isolate-cloud.
DEFAULT_SOCKET_PATH = "/run/fal/api.sock"
_TIMEOUT = 300
_CONNECT_TIMEOUT = 5
_CHUNK_SIZE = 64 * 1024
_REJECTION_DELAYS = (0.1, 0.2)
# How long each status request holds while the upload is pending.
_WAIT_SECONDS = 60
# Uploader inputs come only from the upload itself, never from the request: a
# policy upload must stay credential-free, and an explicit lifecycle choice wins.
_UPLOADER_INPUTS = {
    "x-fal-application",
    "x-fal-cdn-token",
    "x-fal-file-name",
    "x-fal-object-lifecycle",
    "x-fal-object-lifecycle-preference",
    "x-fal-provenance",
    "x-fal-request-id",
}


class LocalUploadError(FileUploadException):
    """The uploader failed after it may have accepted the bytes."""

    falls_back = False


class LocalUploadRefused(LocalUploadError):
    """The uploader did not take the work, so a direct upload cannot duplicate it."""

    falls_back = True


class LocalUploadRejected(LocalUploadRefused):
    """The uploader is full or draining and may accept a replay shortly."""


class LocalUploadInvalid(LocalUploadRefused):
    """The uploader refused this upload's content or metadata (HTTP 413 or 422),
    not for lack of capacity."""


def upload(
    file_name: str,
    content: bytes | Path,
    headers: dict[str, str],
    *,
    wait_for_completion: bool = False,
    request: Any = None,
) -> str:
    """Return the URL once the complete body is durably accepted, or once the
    CDN has it when waiting for completion.

    A rejected submission is replayed briefly; a file is reopened for each
    attempt.
    """
    size = len(content) if isinstance(content, bytes) else content.stat().st_size
    headers = {
        **_with_request_headers(headers, request),
        "X-Fal-File-Name": _header_file_name(file_name),
        "Content-Length": str(size),
    }
    with _new_client() as http:
        upload_id, file_url = _accept(http, content, headers)
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
    """Return the URL once a body of unknown size is durably accepted, or once the
    CDN has it when waiting for completion.
    """
    with _new_client() as http:
        session_id, _, _ = _open_session(http, file_name, headers)
        upload_id, file_url = _send(http, session_id, chunks)
        if wait_for_completion:
            _wait_for_completion(http, upload_id)
    return file_url


def reserve(
    file_name: str, headers: dict[str, str], request: Any = None
) -> Reservation:
    with _new_client() as http:
        session_id, url, cdn_signing_requested = _open_session(
            http, file_name, headers, request
        )
    return Reservation(
        url,
        partial(_complete_session, session_id),
        partial(_delete_session, session_id),
        cdn_signing_requested,
    )


def _complete_session(session_id: str, content: bytes | Path) -> None:
    body = [content] if isinstance(content, bytes) else _chunks(content)
    with _new_client() as http:
        _send(http, session_id, body)


def _delete_session(session_id: str) -> None:
    with _new_client() as http, suppress(LocalUploadError):
        _request(http, "DELETE", _session_path(session_id), 202)


def _with_request_headers(
    headers: dict[str, str], request: Any = None
) -> dict[str, str]:
    """Add the request's x-fal-* context headers, by default the in-flight one's.

    Values that cannot travel as ASCII are skipped rather than fail the upload.
    """
    if request is None:
        request = getattr(get_current_app(), "current_request", None)
    context = request.headers if request else {}
    explicit_names = {name.lower() for name in headers}
    forwarded = {
        name: value
        for name, value in context.items()
        if name.lower().startswith("x-fal-")
        and name.lower() not in _UPLOADER_INPUTS
        and name.lower() not in explicit_names
        and value.isascii()
    }
    return {**forwarded, **headers}


def _accept(
    http: httpx.Client, content: bytes | Path, headers: dict[str, str]
) -> tuple[str, str]:
    for delay in _REJECTION_DELAYS:
        try:
            return _submit(http, content, headers)
        except LocalUploadRejected:
            time.sleep(delay)
    return _submit(http, content, headers)


def _submit(
    http: httpx.Client, content: bytes | Path, headers: dict[str, str]
) -> tuple[str, str]:
    body: bytes | Iterator[bytes] = (
        content if isinstance(content, bytes) else _chunks(content)
    )
    response = _request(http, "POST", "/v1/uploads", 202, headers=headers, content=body)
    return _receipt(response, "accepted_local")


def _open_session(
    http: httpx.Client, file_name: str, headers: dict[str, str], request: Any = None
) -> tuple[str, str, bool]:
    headers = {
        **_with_request_headers(headers, request),
        "X-Fal-File-Name": _header_file_name(file_name),
    }
    response = _request(http, "POST", "/v1/upload-sessions", 201, headers=headers)
    session_id, url = _receipt(response, "receiving")
    return session_id, url, response.json().get("cdn_signing_requested") is True


def _send(
    http: httpx.Client, session_id: str, chunks: Iterable[bytes]
) -> tuple[str, str]:
    """Accept only on an explicit finish carrying the final byte count; any
    failure, including a raising producer, aborts the session.
    """
    size = 0

    def counted() -> Iterator[bytes]:
        nonlocal size
        for chunk in chunks:
            size += len(chunk)
            yield chunk

    path = _session_path(session_id)
    try:
        _request(http, "PUT", path + "/body", 204, content=counted())
        response = _request(
            http, "POST", path + "/finish", 202, json={"size_bytes": size}
        )
        return _receipt(response, "accepted_local")
    except BaseException:
        # Aborting must not mask the original failure.
        with suppress(LocalUploadError):
            _request(http, "DELETE", path, 202)
        raise


def _session_path(session_id: str) -> str:
    return "/v1/upload-sessions/" + quote(session_id, safe="")


def _request(
    http: httpx.Client,
    method: str,
    path: str,
    expected_status: int,
    *,
    headers: dict[str, str] | None = None,
    content: bytes | Iterator[bytes] | None = None,
    json: dict[str, int] | None = None,
) -> httpx.Response:
    import httpx  # noqa: PLC0415 -- see _new_client

    # Name the failure class but drop its text, which can carry credentials
    # or signed URLs.
    try:
        response = http.request(
            method, path, headers=headers, content=content, json=json
        )
    except (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.PoolTimeout,
        httpx.WriteError,
        UnicodeEncodeError,  # a caller header that cannot travel over HTTP
    ) as exc:
        raise LocalUploadRefused(
            f"Cannot send to the local uploader ({type(exc).__name__})."
        ) from None
    except httpx.RequestError as exc:
        raise LocalUploadError(
            f"Local uploader connection was interrupted ({type(exc).__name__})."
        ) from None

    status = response.status_code
    if status == 429:
        raise LocalUploadRejected("Local uploader returned HTTP 429.")
    if status in (413, 422):
        raise LocalUploadInvalid(f"Local uploader returned HTTP {status}.")
    if status != expected_status:
        message = f"Local uploader returned HTTP {status}."
        raise (LocalUploadRefused if status < 500 else LocalUploadError)(message)
    return response


def _receipt(response: httpx.Response, state: str) -> tuple[str, str]:
    try:
        result = response.json()
        if result["state"] == state:
            return result["upload_id"], result["file_url"]
    except (ValueError, KeyError, TypeError):
        pass
    raise LocalUploadError("Local uploader returned an invalid response.")


def _wait_for_completion(http: httpx.Client, upload_id: str) -> None:
    """Return once the CDN has the file; raise if it failed or is unknown.

    The uploader fails pending uploads at their queue deadline, so this ends. A
    404 means the uploader restarted or dropped the record.
    """
    import httpx  # noqa: PLC0415 -- see _new_client

    path = "/v1/uploads/" + quote(upload_id, safe="")
    while True:
        try:
            response = http.get(path, params={"wait": _WAIT_SECONDS})
        except httpx.RequestError as exc:
            raise LocalUploadError(
                f"Lost the local uploader while waiting ({type(exc).__name__})."
            ) from None
        if response.status_code != 200:
            raise LocalUploadError(
                f"Local upload state is unknown (HTTP {response.status_code})."
            )
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


def _chunks(path: Path) -> Iterator[bytes]:
    # Older httpx versions iterate file objects by line, without a size limit.
    with path.open("rb") as source:
        yield from iter(lambda: source.read(_CHUNK_SIZE), b"")


def _header_file_name(file_name: str) -> str:
    """Sanitize as the REST API does, so both upload paths publish the same name.

    Non-ASCII becomes "?" exactly as REST rewrites it before naming the CDN
    object. Control characters and surrounding whitespace cannot travel in a
    header at all.
    """
    return "".join(c if " " <= c <= "~" else "?" for c in file_name).strip()


def _new_client() -> httpx.Client:
    # Local, not module scope, for the reason given in _upload_policy._new_client.
    try:
        import httpx  # noqa: PLC0415
    except ModuleNotFoundError:
        # Plain function runners may not have httpx installed.
        raise LocalUploadRefused(
            "Local uploader dependencies are unavailable."
        ) from None

    # Resolve the socket on the runner, not when serializing the app.
    socket_path = os.environ.get("FAL_API_SOCKET", DEFAULT_SOCKET_PATH)
    app_name = os.environ.get("FAL_APP_NAME")
    return httpx.Client(
        headers={"x-fal-application": app_name} if app_name else None,
        transport=httpx.HTTPTransport(
            uds=socket_path, verify=False, retries=0, trust_env=False
        ),
        base_url="http://localhost",
        trust_env=False,
        follow_redirects=False,
        timeout=httpx.Timeout(_TIMEOUT, connect=_CONNECT_TIMEOUT),
    )
