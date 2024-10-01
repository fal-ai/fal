from __future__ import annotations

from dataclasses import dataclass
from mimetypes import guess_extension, guess_type
from pathlib import Path
from typing import Literal, Optional
from uuid import uuid4


class FileData:
    data: bytes
    content_type: str
    file_name: str

    def __init__(
        self, data: bytes, content_type: str | None = None, file_name: str | None = None
    ):
        self.data = data
        if content_type is None and file_name is not None:
            content_type, _ = guess_type(file_name or "")

        # Ultimately fallback to a generic binary file mime type
        self.content_type = content_type or "application/octet-stream"

        if file_name is None:
            extension = guess_extension(self.content_type, strict=False) or ".bin"
            self.file_name = f"{uuid4().hex}{extension}"
        else:
            self.file_name = file_name


RepositoryId = Literal[
    "fal", "fal_v2", "fal_v3", "in_memory", "gcp_storage", "r2", "cdn"
]


@dataclass
class FileRepository:
    def save(
        self,
        data: FileData,
        object_lifecycle_preference: Optional[dict[str, str]] = None,
    ) -> str:
        raise NotImplementedError()

    def save_file(
        self,
        file_path: str | Path,
        content_type: str,
        multipart: bool | None = None,
        multipart_threshold: int | None = None,
        multipart_chunk_size: int | None = None,
        multipart_max_concurrency: int | None = None,
        object_lifecycle_preference: Optional[dict[str, str]] = None,
    ) -> tuple[str, FileData | None]:
        if multipart:
            raise NotImplementedError()

        with open(file_path, "rb") as fobj:
            data = FileData(fobj.read(), content_type, Path(file_path).name)

        return self.save(data, object_lifecycle_preference), data
