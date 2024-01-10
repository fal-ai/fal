from __future__ import annotations

from dataclasses import dataclass
from io import FileIO, IOBase
from mimetypes import guess_extension, guess_type
from os import remove as remove_file
from tempfile import mkdtemp
from typing import Literal
from uuid import uuid4

from fal.toolkit.mainify import mainify
from fal.toolkit.utils.download_utils import download_file


RepositoryId = Literal["fal", "in_memory", "gcp_storage", "r2"]


@mainify
class RemoteFileIO(IOBase):
    url: str
    _file: FileIO | None

    def __init__(self, url: str):
        self.url = url
        self._file = None

    def _ensure_file_is_downloaded(self):
        temp_dir = mkdtemp(prefix="fal_file_", suffix="", dir=None)
        file_path = download_file(self.url, temp_dir)

        self._file = FileIO(file_path, mode="rb")

    def read(self, size: int = -1) -> bytes:
        if not self.is_downloaded():
            self._ensure_file_is_downloaded()

        return self._file.read(size)  # type: ignore

    def is_downloaded(self) -> bool:
        return self._file is not None

    def close(self) -> None:
        if self._file:
            self._file.close()
            remove_file(str(self._file.name))
            self._file = None


@mainify
class FileData:
    data: IOBase
    content_type: str | None = None
    file_name: str | None = None
    file_size: int | None = None

    _cached_content: bytes | None = None

    def __init__(
        self,
        data: IOBase,
        content_type: str | None = None,
        file_name: str | None = None,
    ):
        self.data = data

        if content_type is None and file_name is not None:
            content_type, _ = guess_type(file_name or "")

        # If the data is a remote file, try to guess the content type from the url
        url = getattr(data, "url", None)
        if url and content_type is None:
            content_type, _ = guess_type(url)

        # Ultimately fallback to a generic binary file mime type
        self.content_type = content_type or "application/octet-stream"

        if file_name is None:
            extension = guess_extension(self.content_type, strict=False) or ".bin"
            self.file_name = f"{uuid4().hex}{extension}"
        else:
            self.file_name = file_name

    def as_bytes(self) -> bytes:
        if self._cached_content:
            return self._cached_content

        content: bytes | str = self.data.read()
        # For files that open in text mode, convert to bytes
        if isinstance(content, str):
            content = content.encode()

        self._cached_content = content

        if content:
            self.file_size = len(content)

        return content


@mainify
@dataclass
class FileRepository:
    def save(self, data: FileData) -> str:
        raise NotImplementedError()
