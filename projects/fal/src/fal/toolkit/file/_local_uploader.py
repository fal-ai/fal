"""Client for the node-local uploader's HTTP API over a Unix socket.

Acceptance is durable on this node, not CDN completion. Submissions are never
retried: losing a response can leave accepted work, and resending creates a new
URL. Keep this client scoped to an upload; do not serialize live connections.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable
from urllib.parse import quote

from fal.toolkit.exceptions import FileUploadException

if TYPE_CHECKING:
    import httpx

DEFAULT_SOCKET_PATH = "/run/fal-upload/upload.sock"
_TIMEOUT = 300
_CONNECT_TIMEOUT = 5


class LocalUploadError(FileUploadException):
    """A local failure, optionally with an unknown acceptance outcome."""

    def __init__(self, message: str, *, acceptance_uncertain: bool = False):
        self.acceptance_uncertain = acceptance_uncertain
        if acceptance_uncertain:
            message += " Acceptance is uncertain; resubmitting may create another URL."
        super().__init__(message)


@dataclass(frozen=True)
class AcceptedUpload:
    """Receipt for durable local acceptance, before CDN completion."""

    upload_id: str
    file_url: str


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
    raise LocalUploadError(
        "Local uploader returned an invalid response.",
        acceptance_uncertain=state == "accepted_local",
    )


class LocalUploader:
    """Upload over a Unix socket without retries; scope connections with ``with``."""

    def __init__(self):
        self._http = _new_client()

    def __enter__(self) -> LocalUploader:
        return self

    def __exit__(self, *exc: Any) -> None:
        self._http.close()

    def _request(
        self,
        method: str,
        path: str,
        expected_status: int,
        *,
        accepting: bool = False,
        **kwargs: Any,
    ) -> httpx.Response:
        import httpx  # noqa: PLC0415 -- see _new_client

        try:
            response = self._http.request(method, path, **kwargs)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout):
            raise LocalUploadError("Cannot connect to the local uploader.") from None
        except httpx.RequestError:
            # Transport errors can contain credentials or signed URLs.
            raise LocalUploadError(
                "Local uploader connection was interrupted.",
                acceptance_uncertain=accepting,
            ) from None
        if response.status_code != expected_status:
            raise LocalUploadError(
                f"Local uploader returned HTTP {response.status_code}.",
                # A shutdown or disk failure can leave committed work behind.
                acceptance_uncertain=accepting
                and (response.status_code >= 500 or response.is_success),
            )
        return response

    def upload(
        self,
        file_name: str,
        body: bytes | Iterable[bytes],
        size_bytes: int,
        headers: dict[str, str],
    ) -> AcceptedUpload:
        """Wait for the complete known-length body to be durably accepted."""
        response = self._request(
            "POST",
            "/uploads",
            202,
            accepting=True,
            headers={
                **headers,
                "X-Fal-File-Name": file_name,
                "Content-Length": str(size_bytes),
            },
            content=body,
        )
        return AcceptedUpload(*_upload_info(response, "accepted_local"))

    def begin_stream(self, file_name: str, headers: dict[str, str]) -> UploadSession:
        """Reserve a URL for a body whose final size is not yet known."""
        response = self._request(
            "POST",
            "/upload-sessions",
            201,
            headers={**headers, "X-Fal-File-Name": file_name},
        )
        upload_id, file_url = _upload_info(response, "receiving")
        return UploadSession(self, upload_id, file_url)


class UploadSession:
    """Use as a context manager to abort unfinished work on exit.

    The reserved URL is not accepted yet. Send one body, await its completion, then
    explicitly finish with the producer's final byte count. Closing a session never
    finishes it. Aborting cannot undo an acceptance already in progress.
    """

    def __init__(self, client: LocalUploader, upload_id: str, file_url: str):
        self._client = client
        self._path = "/upload-sessions/" + quote(upload_id, safe="")
        self.file_url = file_url
        self._body_sent = False
        self._closed = False

    def __enter__(self) -> UploadSession:
        return self

    def __exit__(self, *exc: Any) -> None:
        self._abort_quietly()

    def _abort_quietly(self) -> None:
        try:
            self.abort()
        except LocalUploadError:
            # Cleanup must not mask a producer failure or lost finish response.
            pass

    def send_body(self, body: bytes | Iterable[bytes]) -> None:
        """Send the sole body without accepting it; abort if the producer fails."""
        if self._closed or self._body_sent:
            raise ValueError("A session accepts exactly one body")
        try:
            self._client._request("PUT", self._path + "/body", 204, content=body)
        except BaseException:
            self._abort_quietly()
            raise
        self._body_sent = True

    def finish(self, size_bytes: int) -> AcceptedUpload:
        """Validate the final byte count and wait for durable local acceptance."""
        if self._closed or not self._body_sent:
            raise ValueError("Finish requires a successfully received body")
        response = self._client._request(
            "POST",
            self._path + "/finish",
            202,
            accepting=True,
            json={"size_bytes": size_bytes},
        )
        accepted = AcceptedUpload(*_upload_info(response, "accepted_local"))
        self._closed = True
        return accepted

    def abort(self) -> None:
        """Discard unfinished work; an upload already accepted cannot be undone."""
        if not self._closed:
            self._closed = True
            self._client._request("DELETE", self._path, 202)
