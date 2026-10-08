"""Repository adapter for opt-in, node-local uploads.

Definite admission rejections are retried briefly, then sent directly to CDN.
Accepted uploads remain the uploader's responsibility. Small files retain bytes
for File.as_bytes().
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
    """Replay the whole save operation only after a definite rejection."""
    replayed = retry(
        max_retries=3,
        base_delay=0.1,
        should_retry=lambda exc: isinstance(exc, LocalUploadRejected),
    )(method)

    @wraps(method)
    def save(self, *args, **kwargs):
        try:
            return replayed(self, *args, **kwargs)
        except LocalUploadRejected:
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
    """Hand uploads to the node-local service and wait for durable acceptance.

    Cancelling an async wrapper's await does not stop the uploading thread.
    """

    # A lost response may already have accepted the bytes; trying another
    # destination could publish them twice.
    falls_back = False

    @_retry_or_fallback
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
        return upload(
            data.file_name,
            data.data,
            len(data.data),
            _headers(data.content_type, object_lifecycle_preference),
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
    ) -> tuple[str, FileData | None]:
        """Stream large files; small ones are read once and returned as FileData."""
        size = os.path.getsize(file_path)
        if multipart is None:
            threshold = multipart_threshold or MultipartUploadV3.MULTIPART_THRESHOLD
            multipart = size > threshold
        if not multipart:
            # self.save handles rejections; forward options to its fallback.
            return super().save_file(
                file_path,
                content_type,
                multipart=False,
                multipart_threshold=multipart_threshold,
                multipart_chunk_size=multipart_chunk_size,
                multipart_max_concurrency=multipart_max_concurrency,
                object_lifecycle_preference=object_lifecycle_preference,
            )

        with open(file_path, "rb") as source:
            # httpx streams file objects in fixed-size reads.
            url = upload(
                Path(file_path).name,
                source,
                size,
                _headers(content_type, object_lifecycle_preference),
            )
        return url, None

    def save_stream(
        self,
        chunks: Iterable[bytes],
        file_name: str,
        content_type: str,
        object_lifecycle_preference: dict[str, str] | None = None,
    ) -> str:
        """Upload output whose size is unknown until the producer is exhausted."""
        return upload_stream(
            file_name, chunks, _headers(content_type, object_lifecycle_preference)
        )
