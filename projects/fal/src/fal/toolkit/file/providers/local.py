"""Repository adapter for opt-in, node-local uploads.

A submission the uploader refused is sent directly to the CDN instead. A
failure after the uploader may have accepted the bytes is raised, because a
second upload could publish a duplicate. Small files retain bytes for
File.as_bytes().
"""

from __future__ import annotations

from functools import wraps
from pathlib import Path

from fal.toolkit.file._local_uploader import LocalUploadRefused, upload
from fal.toolkit.file.providers.fal import (
    FalFileRepositoryV3,
    MultipartUploadV3,
    _object_lifecycle_headers,
)
from fal.toolkit.file.types import FileData, FileRepository


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
    content_type: str, object_lifecycle_preference: dict[str, str] | None
) -> dict[str, str]:
    headers = {**FalFileRepositoryV3().auth_headers, "Content-Type": content_type}
    _object_lifecycle_headers(headers, object_lifecycle_preference)
    return headers


class LocalFileRepository(FileRepository):
    """Hand uploads to the node-local service and wait for durable acceptance.

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
    ) -> str:
        """Return the accepted URL; the service completes the CDN transfer."""
        headers = _headers(data.content_type, object_lifecycle_preference)
        return upload(data.file_name, data.data, headers)

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
    ) -> tuple[str, FileData | None]:
        """Stream large files; small ones are read once and returned as FileData."""
        path = Path(file_path)
        if multipart is None:
            threshold = multipart_threshold or MultipartUploadV3.MULTIPART_THRESHOLD
            multipart = path.stat().st_size > threshold
        headers = _headers(content_type, object_lifecycle_preference)
        if multipart:
            return upload(path.name, path, headers), None
        data = FileData(path.read_bytes(), content_type, path.name)
        return upload(path.name, data.data, headers), data
