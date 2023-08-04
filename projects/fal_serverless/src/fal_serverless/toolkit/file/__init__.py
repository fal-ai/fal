from __future__ import annotations

from base64 import b64encode
from pathlib import Path
from typing import Callable, Literal

from fal_serverless.toolkit import mainify
from fal_serverless.toolkit.file.providers.fal import (
    FalFileRepository,
    InMemoryRepository,
)
from fal_serverless.toolkit.file.providers.gcp import GoogleStorageRepository
from fal_serverless.toolkit.file.types import FileData, FileRepository, RepositoryId
from pydantic import Field
from pydantic.dataclasses import dataclass

BuiltInRepositoryId = Literal["fal", "in_memory", "gcp_storage"]
FileRepositoryFactory = Callable[[], FileRepository]

BUILT_IN_REPOSITORIES: dict[BuiltInRepositoryId, FileRepositoryFactory] = {
    "fal": lambda: FalFileRepository(),
    "in_memory": lambda: InMemoryRepository(),
    "gcp_storage": lambda: GoogleStorageRepository(),
}


def get_builtin_repository(id: BuiltInRepositoryId) -> FileRepository:
    if id not in BUILT_IN_REPOSITORIES.keys():
        raise ValueError(f'"{id}" is not a valid built-in file repository')
    return BUILT_IN_REPOSITORIES[id]()


get_builtin_repository.__module__ = "__main__"

DEFAULT_REPOSITORY: FileRepository | BuiltInRepositoryId = "fal"


@mainify
@dataclass
class File:

    # internal properties
    # _file_data: FileData = Field(exclude=True)

    # public properties
    url: str = Field(
        description="The URL where the file can be downloaded from.",
    )
    content_type: str = Field(description="The mime type of the file.")
    file_name: str = Field(
        description="The name of the file. It will be auto-generated if not provided."
    )
    file_size: int = Field(
        description="The size of the file in bytes.",
    )

    def __init__(self, data: FileData, repository: FileRepository | RepositoryId):
        self._file_data = data

        repo = (
            repository
            if isinstance(repository, FileRepository)
            else get_builtin_repository(repository)
        )
        self.url = repo.save(data)

        self.content_type = data.content_type
        self.file_name = data.file_name
        self.file_size = len(data.data)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        content_type: str | None = None,
        file_name: str | None = None,
        repository: FileRepository | RepositoryId = DEFAULT_REPOSITORY,
    ) -> File:
        return cls(FileData(data, content_type, file_name), repository)

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

    def as_base64(self) -> str:
        return b64encode(self.as_bytes()).decode("utf-8")
