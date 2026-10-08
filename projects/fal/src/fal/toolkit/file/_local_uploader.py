"""Client for the node-local uploader's HTTP API over a Unix socket.

Acceptance is durable on this node, not CDN completion; callers that need the
CDN copy wait for completion. Do not serialize live connections.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Iterator
from urllib.parse import quote

from fal.toolkit.exceptions import FileUploadException

if TYPE_CHECKING:
    import httpx

# Keep in sync with infra/modules/nomad_jobs/jobs.tf and isolate-cloud's
# projects/isolate_controller/src/isolate_controller/scheduler/nomad/forger.py.
DEFAULT_SOCKET_PATH = "/run/fal-upload/upload.sock"
_TIMEOUT = 300
_CONNECT_TIMEOUT = 5
_CHUNK_SIZE = 64 * 1024
_REJECTION_DELAYS = (0.1, 0.2)
# How long each status request holds while the upload is pending.
_WAIT_SECONDS = 60


class LocalUploadError(FileUploadException):
    """The uploader failed after it may have accepted the bytes."""

    falls_back = False


class LocalUploadRefused(LocalUploadError):
    """The uploader did not take the work, so a direct upload cannot duplicate it."""

    falls_back = True


class LocalUploadRejected(LocalUploadRefused):
    """The uploader is full or draining and may accept a replay shortly."""


def upload(
    file_name: str,
    content: bytes | Path,
    headers: dict[str, str],
    *,
    wait_for_completion: bool = False,
) -> str:
    """Return the URL once the complete body is durably accepted, or once the
    CDN has it when waiting for completion.

    A rejected submission is replayed briefly; a file is reopened for each
    attempt.
    """
    size = len(content) if isinstance(content, bytes) else content.stat().st_size
    headers = {
        **headers,
        "X-Fal-File-Name": _header_file_name(file_name),
        "Content-Length": str(size),
    }
    with _new_client() as http:
        upload_id, file_url = _accept(http, content, headers)
        if wait_for_completion:
            _wait_for_completion(http, upload_id)
    return file_url


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
    import httpx  # noqa: PLC0415 -- see _new_client

    body = content if isinstance(content, bytes) else _chunks(content)
    # Name the failure class but drop its text, which can carry credentials
    # or signed URLs.
    try:
        response = http.post("/uploads", headers=headers, content=body)
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
    if status != 202:
        message = f"Local uploader returned HTTP {status}."
        raise (LocalUploadRefused if status < 500 else LocalUploadError)(message)
    try:
        result = response.json()
        if result["state"] == "accepted_local":
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

    path = "/uploads/" + quote(upload_id, safe="")
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
