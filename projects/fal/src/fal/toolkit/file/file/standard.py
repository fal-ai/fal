from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, PrivateAttr

from fal.toolkit.file.types import FileData, FileRepository, RepositoryId
from fal.toolkit.mainify import mainify
from fal.toolkit.utils.download_utils import download_file

from .repositories import DEFAULT_REPOSITORY, get_builtin_repository

__all__ = ["File"]


@mainify
class File(BaseModel):
    # public properties
    _file_data: FileData = PrivateAttr()
    url: str = Field(
        description="The URL where the file can be downloaded from.",
    )
    content_type: Optional[str] = Field(
        description="The mime type of the file.",
        examples=["image/png"],
    )
    file_name: Optional[str] = Field(
        description="The name of the file. It will be auto-generated if not provided.",
        examples=["z9RV14K95DvU.png"],
    )
    file_size: Optional[int] = Field(
        description="The size of the file in bytes.", examples=[4404019]
    )

    def __init__(self, **kwargs):
        has_fd = "file_data" in kwargs
        if has_fd:
            data: FileData = kwargs.pop("file_data")
            repository = kwargs.pop("repository", None)
            repo = (
                repository
                if isinstance(repository, FileRepository)
                else get_builtin_repository(repository)
            )
            kwargs.update(
                {
                    "url": repo.save(data),
                    "content_type": data.content_type,
                    "file_name": data.file_name,
                    "file_size": len(data.data),
                }
            )
        super().__init__(**kwargs)
        if has_fd:
            self._file_data = data

    # Pydantic custom validator for input type conversion
    @classmethod
    def __get_validators__(cls):
        yield cls.__convert_from_str

    @classmethod
    def __convert_from_str(cls, value: Any):
        if isinstance(value, str):
            parsed_url = urlparse(value)
            if parsed_url.scheme not in ["http", "https", "data"]:
                raise ValueError(f"value must be a valid URL")
            return cls._from_url(parsed_url.geturl())

        return value

    @classmethod
    def _from_url(
        cls,
        url: str,
    ) -> File:
        return cls(
            url=url,
            content_type=None,
            file_name=None,
            repository=DEFAULT_REPOSITORY,
        )

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        content_type: Optional[str] = None,
        file_name: Optional[str] = None,
        repository: FileRepository | RepositoryId = DEFAULT_REPOSITORY,
    ) -> File:
        return cls(
            file_data=FileData(data, content_type, file_name),
            repository=repository,
        )

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        content_type: Optional[str] = None,
        repository: FileRepository | RepositoryId = DEFAULT_REPOSITORY,
    ) -> File:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")
        with open(file_path, "rb") as f:
            data = f.read()
        return File.from_bytes(
            data, content_type, file_name=file_path.name, repository=repository
        )

    def as_bytes(self) -> bytes:
        if getattr(self, "_file_data", None) is None:
            raise ValueError("File has not been downloaded")

        return self._file_data.data

    def save(self, path: str | Path, overwrite: bool = False) -> Path:
        file_path = Path(path).resolve()

        if file_path.exists() and not overwrite:
            raise FileExistsError(f"File {file_path} already exists")

        downloaded_path = download_file(self.url, target_dir=file_path.parent)
        downloaded_path.rename(file_path)

        return file_path
