"""Repository adapter for opt-in, node-local uploads.

Rejections are retried briefly. Any uploader failure, including a failed or
unknown transfer while waiting for completion, then falls back to a direct CDN
upload, which can publish a second URL if the uploader had accepted the bytes.
Small files retain bytes for File.as_bytes().
"""

from __future__ import annotations

import os
from functools import wraps
from pathlib import Path
from typing import Iterable

from fal._user_agent import USER_AGENT
from fal.auth import fetch_auth_credentials
from fal.exceptions.auth import UnauthenticatedException
from fal.toolkit.exceptions import FileUploadException
from fal.toolkit.file._local_uploader import (
    LocalUploadError,
    LocalUploadRejected,
    upload,
    upload_stream,
)
from fal.toolkit.file.providers.fal import (
    FalFileRepositoryV3,
    MultipartUploadV3,
    _caller_cdn_header,
    _object_lifecycle_headers,
)
from fal.toolkit.file.types import FileData, FileRepository
from fal.toolkit.utils.retry import retry


def _retry_or_fallback(method):
    """Replay the save after a rejection, then fall back to a direct CDN upload."""
    replayed = retry(
        max_retries=3,
        base_delay=0.1,
        should_retry=lambda exc: isinstance(exc, LocalUploadRejected),
    )(method)

    @wraps(method)
    def save(self, *args, **kwargs):
        try:
            return replayed(self, *args, **kwargs)
        except LocalUploadError as exc:
            print(f"{exc} Uploading directly to CDN.")
            return getattr(FalFileRepositoryV3(), method.__name__)(*args, **kwargs)

    return save


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
    """Hand uploads to the node-local service and wait for durable acceptance,
    or for CDN completion with `wait_for_completion`.

    Cancelling an async wrapper's await does not stop the uploading thread.
    """

    @_retry_or_fallback
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
        """Return the accepted URL; the service completes the CDN transfer."""
        return upload(
            data.file_name,
            data.data,
            len(data.data),
            _headers(data.content_type, object_lifecycle_preference),
            wait_for_completion,
        )

    @_retry_or_fallback
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
        size = os.path.getsize(file_path)
        if multipart is None:
            threshold = multipart_threshold or MultipartUploadV3.MULTIPART_THRESHOLD
            multipart = size > threshold
        name = Path(file_path).name
        headers = _headers(content_type, object_lifecycle_preference)
        if not multipart:
            data = FileData(Path(file_path).read_bytes(), content_type, name)
            url = upload(name, data.data, len(data.data), headers, wait_for_completion)
            return url, data

        with open(file_path, "rb") as source:
            # httpx streams file objects in fixed-size reads.
            return upload(name, source, size, headers, wait_for_completion), None

    def save_stream(
        self,
        chunks: Iterable[bytes],
        file_name: str,
        content_type: str,
        object_lifecycle_preference: dict[str, str] | None = None,
        wait_for_completion: bool = False,
    ) -> str:
        """Upload output whose size is unknown until the producer is exhausted.

        Not retried and no fallback: the producer cannot be replayed.
        """
        return upload_stream(
            file_name,
            chunks,
            _headers(content_type, object_lifecycle_preference),
            wait_for_completion,
        )
