from __future__ import annotations

from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field, PrivateAttr

from fal.toolkit.file.providers.fal import FalFileRepository, InMemoryRepository
from fal.toolkit.file.providers.gcp import GoogleStorageRepository
from fal.toolkit.file.providers.r2 import R2Repository
from fal.toolkit.file.types import FileData, FileRepository, RepositoryId
from fal.toolkit.mainify import mainify

FileRepositoryFactory = Callable[[], FileRepository]

BUILT_IN_REPOSITORIES: dict[RepositoryId, FileRepositoryFactory] = {
    "fal": lambda: FalFileRepository(),
    "in_memory": lambda: InMemoryRepository(),
    "gcp_storage": lambda: GoogleStorageRepository(),
    "r2": lambda: R2Repository(),
}


def get_builtin_repository(id: RepositoryId) -> FileRepository:
    if id not in BUILT_IN_REPOSITORIES.keys():
        raise ValueError(f'"{id}" is not a valid built-in file repository')
    return BUILT_IN_REPOSITORIES[id]()


get_builtin_repository.__module__ = "__main__"

DEFAULT_REPOSITORY: FileRepository | RepositoryId = "fal"


@mainify
class File(BaseModel):
    # public properties
    _file_data: FileData = PrivateAttr()
    url: str = Field(
        description="The URL where the file can be downloaded from.",
        examples=["https://url.to/generated/file/z9RV14K95DvU.png"],
    )
    content_type: str = Field(
        description="The mime type of the file.",
        examples=["image/png"],
    )
    file_name: str = Field(
        description="The name of the file. It will be auto-generated if not provided.",
        examples=["z9RV14K95DvU.png"],
    )
    file_size: int = Field(
        description="The size of the file in bytes.", examples=[4404019]
    )

    def __init__(self, **kwargs):
        if "file_data" in kwargs:
            data = kwargs.pop("file_data")
            repository = kwargs.pop("repository", None)

            repo = (
                repository
                if isinstance(repository, FileRepository)
                else get_builtin_repository(repository)
            )
            self._file_data = data

            kwargs.update(
                {
                    "url": repo.save(data),
                    "content_type": data.content_type,
                    "file_name": data.file_name,
                    "file_size": len(data.data),
                }
            )

        super().__init__(**kwargs)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        content_type: str | None = None,
        file_name: str | None = None,
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
        content_type: str | None = None,
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
        return self._file_data.data
