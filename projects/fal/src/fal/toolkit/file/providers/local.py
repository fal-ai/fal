"""Repository adapter for opt-in, node-local uploads.

The uploader owns remote multipart transfers, concurrency, and retries. The
existing multipart arguments only determine whether save_file retains bytes in
the returned FileData, preserving File.as_bytes() behavior.
"""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Iterator

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


class LocalFileRepository(FileRepository):
    """Hand uploads to the node-local service and wait for durable acceptance."""

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
        headers = _headers(data.content_type, object_lifecycle_preference)
        with LocalUploader() as client:
            return client.upload(
                data.file_name, data.data, len(data.data), headers
            ).file_url

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
        """Stream the file, retaining bytes only for the legacy non-multipart case."""
        headers = _headers(content_type, object_lifecycle_preference)
        file_name = Path(file_path).name
        with open(file_path, "rb") as source:
            size = os.fstat(source.fileno()).st_size
            if multipart is None:
                threshold = multipart_threshold or MultipartUploadV3.MULTIPART_THRESHOLD
                retain_data = size <= threshold
            else:
                retain_data = not multipart
            retained = BytesIO() if retain_data else None

            def body() -> Iterator[bytes]:
                while chunk := source.read(_READ_SIZE):
                    if retained is not None:
                        retained.write(chunk)
                    yield chunk

            # Cancelling the async wrapper's await does not stop this thread.
            with LocalUploader() as client:
                accepted = client.upload(file_name, body(), size, headers)
        data = (
            FileData(retained.getvalue(), content_type, file_name)
            if retained is not None
            else None
        )
        return accepted.file_url, data
