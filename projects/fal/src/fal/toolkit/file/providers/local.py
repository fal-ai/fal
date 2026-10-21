"""Repository adapter for opt-in, node-local uploads.

A submission the uploader refused is sent directly to the CDN instead, or
in-process to the caller's bucket under an upload policy. A failure after the
uploader may have accepted the bytes is raised, because a second upload could
publish a duplicate. Small files retain bytes for File.as_bytes().
"""

from __future__ import annotations

import json
from functools import partial, wraps
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from fal.flags import bool_envvar
from fal.toolkit.file._local_uploader import (
    LocalUploadInvalid,
    LocalUploadRefused,
    reserve,
    upload,
    upload_stream,
)
from fal.toolkit.file._upload_policy import (
    UPLOAD_POLICY_KEY,
    UploadPolicy,
    UploadPolicyError,
    _prepare_upload,
    _request_id,
    _upload_file_name,
    _validate_size,
    upload_bytes_with_policy,
    upload_path_with_policy,
)
from fal.toolkit.file.providers.fal import (
    FalFileRepositoryV3,
    MultipartUploadV3,
    _object_lifecycle_headers,
    _v3_auth_headers,
)
from fal.toolkit.file.types import FileData, FileRepository, Reservation


def _or_direct(method):
    """Upload directly to the CDN when the local uploader did not take the work."""

    @wraps(method)
    def save(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except LocalUploadRefused as exc:
            print(f"{exc} Uploading directly to CDN.")
            return getattr(FalFileRepositoryV3(), method.__name__)(*args, **kwargs)

    return save


def _headers(
    content_type: str,
    object_lifecycle_preference: dict[str, str] | None,
    request: Any = None,
) -> dict[str, str]:
    headers = {**_v3_auth_headers(request), "Content-Type": content_type}
    _object_lifecycle_headers(headers, object_lifecycle_preference)
    return headers


class LocalFileRepository(FileRepository):
    """Hand uploads to the node-local service and wait for durable acceptance,
    or for CDN completion with `wait_for_completion`.

    Cancelling an async wrapper's await does not stop the uploading thread.
    """

    @_or_direct
    def save(
        self,
        data: FileData,
        multipart: bool | None = None,
        multipart_threshold: int | None = None,
        multipart_chunk_size: int | None = None,
        multipart_max_concurrency: int | None = None,
        object_lifecycle_preference: dict[str, str] | None = None,
        wait_for_completion: bool = False,
    ) -> str:
        """Return the URL once accepted, or once on the CDN when waiting."""
        headers = _headers(data.content_type, object_lifecycle_preference)
        return upload(
            data.file_name,
            data.data,
            headers,
            wait_for_completion=wait_for_completion,
        )

    @_or_direct
    def save_file(
        self,
        file_path: str | Path,
        content_type: str,
        multipart: bool | None = None,
        multipart_threshold: int | None = None,
        multipart_chunk_size: int | None = None,
        multipart_max_concurrency: int | None = None,
        object_lifecycle_preference: dict[str, str] | None = None,
        wait_for_completion: bool = False,
    ) -> tuple[str, FileData | None]:
        """Stream large files; small ones are read once and returned as FileData."""
        path = Path(file_path)
        if multipart is None:
            threshold = multipart_threshold or MultipartUploadV3.MULTIPART_THRESHOLD
            multipart = path.stat().st_size > threshold
        headers = _headers(content_type, object_lifecycle_preference)
        if multipart:
            url = upload(
                path.name, path, headers, wait_for_completion=wait_for_completion
            )
            return url, None
        data = FileData(path.read_bytes(), content_type, path.name)
        url = upload(
            path.name, data.data, headers, wait_for_completion=wait_for_completion
        )
        return url, data

    @_or_direct
    def reserve(
        self,
        file_name: str,
        content_type: str,
        object_lifecycle_preference: dict[str, str] | None = None,
        request: Any = None,
        extra_headers: dict[str, str] | None = None,
    ) -> Reservation:
        upload_headers = _headers(content_type, object_lifecycle_preference, request)
        upload_headers.update(extra_headers or {})
        return reserve(file_name, upload_headers, request)

    def save_stream(
        self,
        chunks: Iterable[bytes],
        file_name: str,
        content_type: str,
        object_lifecycle_preference: dict[str, str] | None = None,
        wait_for_completion: bool = False,
    ) -> str:
        """Upload output whose size is unknown until the producer is exhausted.

        Never sent directly to the CDN: the producer cannot be replayed.
        """
        headers = _headers(content_type, object_lifecycle_preference)
        return upload_stream(
            file_name, chunks, headers, wait_for_completion=wait_for_completion
        )


def reserve_with_policy(
    policy: UploadPolicy, file_name: str, content_type: str, request: Any
) -> Reservation:
    """The URL from the policy's key now; complete runs save_with_policy, which
    does not wait for the bucket."""
    name = _upload_file_name(policy, file_name, content_type)
    url, _ = _prepare_upload(policy, file_name, content_type, name=name)
    complete = partial(
        save_with_policy,
        policy,
        file_name,
        content_type=content_type,
        name=name,
        request=request,
    )
    return Reservation(url, complete)


def save_with_policy(
    policy: UploadPolicy,
    file_name: str,
    content: bytes | Path,
    content_type: str,
    *,
    name: str | None = None,
    request: Any = None,
) -> str:
    """Upload to the caller's bucket and return the URL the object will have.

    Opted-in runners hand the upload to the node-local uploader. Any other
    runner, or a refusal, uploads in-process. Neither waits for the bucket, so
    no runner time is spent on the caller's storage.
    """
    name = name or _upload_file_name(policy, file_name, content_type)
    if url := _uploader_with_policy(policy, name, content, content_type, request):
        return url
    if isinstance(content, bytes):
        return upload_bytes_with_policy(
            policy, file_name, content, content_type, name=name, request=request
        )
    return upload_path_with_policy(
        policy, content, file_name, content_type, name=name, request=request
    )


def _uploader_with_policy(
    policy: UploadPolicy,
    name: str,
    content: bytes | Path,
    content_type: str,
    request: Any = None,
) -> str | None:
    """The URL from the node-local uploader, or None when it did not take the work."""
    if not bool_envvar("FAL_USE_LOCAL_UPLOADER"):
        return None
    try:
        return _upload_with_policy(policy, name, content, content_type, request)
    except LocalUploadInvalid as exc:
        raise UploadPolicyError(
            f"Upload via {UPLOAD_POLICY_KEY} was refused: {exc}"
        ) from None
    except LocalUploadRefused as exc:
        print(f"{exc} Uploading in-process.")
        return None


def _upload_with_policy(
    policy: UploadPolicy,
    name: str,
    content: bytes | Path,
    content_type: str,
    request: Any = None,
) -> str:
    _validate_size(
        len(content) if isinstance(content, bytes) else content.stat().st_size
    )
    # Percent-encoded so a non-ASCII name can travel in the header; the
    # uploader decodes it, so the key matches the in-process path.
    name = quote(name, safe="")
    # No credentials: an uploader without policy support then refuses the
    # upload rather than send it to the fal CDN.
    headers = {
        "Content-Type": content_type,
        UPLOAD_POLICY_KEY: json.dumps({"url": policy.url, "fields": policy.fields}),
    }
    if request_id := _request_id(request):
        headers["X-Fal-Request-ID"] = request_id
    return upload(name, content, headers, request=request)
