"""Repository adapter for opt-in, node-local uploads.

The uploader owns remote multipart transfers, concurrency, and retries. The
multipart arguments only decide whether save_file returns the bytes it sent,
preserving File.as_bytes() behavior.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from fal._user_agent import USER_AGENT
from fal.auth import fetch_auth_credentials
from fal.exceptions.auth import UnauthenticatedException
from fal.toolkit.exceptions import FileUploadException
from fal.toolkit.file._local_uploader import LocalUploader
from fal.toolkit.file.providers.fal import (
    MultipartUploadV3,
    _caller_cdn_header,
    _object_lifecycle_headers,
)
from fal.toolkit.file.types import FileData, FileRepository

_READ_SIZE = 1024 * 1024


def _headers(
    content_type: str, object_lifecycle_preference: dict[str, str] | None
) -> dict[str, str]:
    headers = {"Content-Type": content_type, "User-Agent": USER_AGENT}
    _caller_cdn_header(headers)
    _object_lifecycle_headers(headers, object_lifecycle_preference)
    try:
        headers["Authorization"] = fetch_auth_credentials().header_value
    except UnauthenticatedException:
        if "X-Fal-CDN-Token" not in headers:
            raise FileUploadException(
                "Local upload requires fal credentials."
            ) from None
        # The uploader decides whether the token suffices or REST is required.
    return headers


def _upload(
    file_name: str,
    body: bytes | Iterable[bytes],
    size: int,
    content_type: str,
    object_lifecycle_preference: dict[str, str] | None,
) -> str:
    headers = _headers(content_type, object_lifecycle_preference)
    with LocalUploader() as client:
        return client.upload(file_name, body, size, headers).file_url


class LocalFileRepository(FileRepository):
    """Hand uploads to the node-local service and wait for durable acceptance.

    Cancelling an async wrapper's await does not stop the uploading thread.
    """

    # A lost response may already have accepted the bytes; trying another
    # destination could publish them twice.
    falls_back = False

    def save(
        self,
        data: FileData,
        multipart: bool | None = None,
        multipart_threshold: int | None = None,
        multipart_chunk_size: int | None = None,
        multipart_max_concurrency: int | None = None,
        object_lifecycle_preference: dict[str, str] | None = None,
    ) -> str:
        """Return the accepted URL; the service completes the CDN transfer."""
        return _upload(
            data.file_name,
            data.data,
            len(data.data),
            data.content_type,
            object_lifecycle_preference,
        )

    def save_file(
        self,
        file_path: str | Path,
        content_type: str,
        multipart: bool | None = None,
        multipart_threshold: int | None = None,
        multipart_chunk_size: int | None = None,
        multipart_max_concurrency: int | None = None,
        object_lifecycle_preference: dict[str, str] | None = None,
    ) -> tuple[str, FileData | None]:
        """Stream large files; small ones are read once and returned as FileData."""
        size = os.path.getsize(file_path)
        if multipart is None:
            threshold = multipart_threshold or MultipartUploadV3.MULTIPART_THRESHOLD
            multipart = size > threshold
        if not multipart:
            return super().save_file(
                file_path,
                content_type,
                object_lifecycle_preference=object_lifecycle_preference,
            )

        with open(file_path, "rb") as source:
            body = iter(lambda: source.read(_READ_SIZE), b"")
            url = _upload(
                Path(file_path).name,
                body,
                size,
                content_type,
                object_lifecycle_preference,
            )
        return url, None
