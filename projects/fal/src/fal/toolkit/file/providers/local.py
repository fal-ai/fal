"""Repository adapter for opt-in, node-local uploads.

The uploader owns remote multipart transfers, concurrency, and retries. The
multipart arguments only decide whether save_file returns the bytes it sent,
preserving File.as_bytes() behavior.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Iterable, Iterator, TypeVar

from fal._user_agent import USER_AGENT
from fal.auth import fetch_auth_credentials
from fal.exceptions.auth import UnauthenticatedException
from fal.toolkit.exceptions import FileUploadException
from fal.toolkit.file._local_uploader import LocalUploader, LocalUploadRejected
from fal.toolkit.file.providers.fal import (
    FalFileRepositoryV3,
    MultipartUploadV3,
    _caller_cdn_header,
    _object_lifecycle_headers,
)
from fal.toolkit.file.types import FileData, FileRepository

_READ_SIZE = 1024 * 1024
_RETRY_DELAYS = (0.1, 0.2)
_T = TypeVar("_T")


def _retry_rejections(upload: Callable[[], _T]) -> _T:
    for delay in _RETRY_DELAYS:
        try:
            return upload()
        except LocalUploadRejected:
            time.sleep(delay)
    return upload()


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

    # Definite rejections use direct CDN here. Other failures must not chain
    # through File's fallback list, since a lost response can leave accepted work.
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
        try:
            return _retry_rejections(
                lambda: _upload(
                    data.file_name,
                    data.data,
                    len(data.data),
                    data.content_type,
                    object_lifecycle_preference,
                )
            )
        except LocalUploadRejected:
            return FalFileRepositoryV3().save(
                data,
                multipart=multipart,
                multipart_threshold=multipart_threshold,
                multipart_chunk_size=multipart_chunk_size,
                multipart_max_concurrency=multipart_max_concurrency,
                object_lifecycle_preference=object_lifecycle_preference,
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
                multipart=False,
                multipart_threshold=multipart_threshold,
                multipart_chunk_size=multipart_chunk_size,
                multipart_max_concurrency=multipart_max_concurrency,
                object_lifecycle_preference=object_lifecycle_preference,
            )

        with open(file_path, "rb") as source:

            def upload() -> str:
                source.seek(0)
                body = iter(lambda: source.read(_READ_SIZE), b"")
                return _upload(
                    Path(file_path).name,
                    body,
                    size,
                    content_type,
                    object_lifecycle_preference,
                )

            try:
                return _retry_rejections(upload), None
            except LocalUploadRejected:
                pass

        return FalFileRepositoryV3().save_file(
            file_path,
            content_type,
            multipart=multipart,
            multipart_threshold=multipart_threshold,
            multipart_chunk_size=multipart_chunk_size,
            multipart_max_concurrency=multipart_max_concurrency,
            object_lifecycle_preference=object_lifecycle_preference,
        )

    def save_stream(
        self,
        chunks: Iterable[bytes],
        file_name: str,
        content_type: str,
        object_lifecycle_preference: dict[str, str] | None = None,
    ) -> str:
        """Upload output whose size is unknown until the producer is exhausted.

        The URL is accepted only after the uploader has confirmed the full byte
        count; a producer that raises aborts the upload instead.
        """
        headers = _headers(content_type, object_lifecycle_preference)
        size = 0

        def counted() -> Iterator[bytes]:
            nonlocal size
            for chunk in chunks:
                size += len(chunk)
                yield chunk

        with LocalUploader() as client:
            try:
                session = _retry_rejections(
                    lambda: client.begin_stream(file_name, headers)
                )
            except LocalUploadRejected:
                # Direct CDN needs a known size. Spool only after rejection,
                # while the producer is still untouched.
                with TemporaryDirectory() as directory:
                    path = Path(directory) / Path(file_name).name
                    with path.open("wb") as target:
                        for chunk in chunks:
                            target.write(chunk)
                    return FalFileRepositoryV3().save_file(
                        path,
                        content_type,
                        object_lifecycle_preference=object_lifecycle_preference,
                    )[0]
            with session:
                session.send_body(counted())
                return session.finish(size).file_url
