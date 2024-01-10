from __future__ import annotations

from io import BytesIO, FileIO, IOBase

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable
from urllib.parse import urlparse
from zipfile import ZipFile

from fal.toolkit.file.providers.fal import FalFileRepository, InMemoryRepository
from fal.toolkit.file.providers.gcp import GoogleStorageRepository
from fal.toolkit.file.providers.r2 import R2Repository
from fal.toolkit.file.types import FileData, FileRepository, RepositoryId, RemoteFileIO
from fal.toolkit.mainify import mainify
from pydantic import BaseModel, Field, PrivateAttr

FileRepositoryFactory = Callable[[], FileRepository]

DEFAULT_REPOSITORY: FileRepository | RepositoryId = "fal"

BUILT_IN_REPOSITORIES: dict[RepositoryId, FileRepositoryFactory] = {
    "fal": lambda: FalFileRepository(),
    "in_memory": lambda: InMemoryRepository(),
    "gcp_storage": lambda: GoogleStorageRepository(),
    "r2": lambda: R2Repository(),
}


@mainify
def get_builtin_repository(id: RepositoryId) -> FileRepository:
    if id not in BUILT_IN_REPOSITORIES.keys():
        raise ValueError(f'"{id}" is not a valid built-in file repository')
    return BUILT_IN_REPOSITORIES[id]()


@mainify
def get_repository(id: FileRepository | RepositoryId) -> FileRepository:
    if isinstance(id, FileRepository):
        return id
    return get_builtin_repository(id)


@mainify
class File(BaseModel):
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
    file_size: int | None = Field(
        description="The size of the file in bytes, when available.", examples=[4404019]
    )

    def __init__(self, **kwargs):
        if "file_data" in kwargs:
            self._file_data = kwargs.pop("file_data")
        super().__init__(**kwargs)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        content_type: str | None = None,
        file_name: str | None = None,
        repository: FileRepository | RepositoryId = DEFAULT_REPOSITORY,
    ) -> File:
        return cls.from_fileobj(
            BytesIO(data),
            content_type=content_type,
            file_name=file_name,
            repository=repository,
        )

    @classmethod
    def from_fileobj(
        cls,
        fileobj: IOBase,
        content_type: str | None = None,
        file_name: str | None = None,
        repository: FileRepository | RepositoryId = DEFAULT_REPOSITORY,
    ) -> File:
        file_data = FileData(fileobj, content_type, file_name)

        file_repository = get_repository(repository)
        url = file_repository.save(file_data)

        return cls(
            file_data=file_data,
            url=url,
            content_type=file_data.content_type,
            file_name=file_data.file_name,
            file_size=file_data.file_size,
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

        file_data = FileData(FileIO(file_path), content_type, file_path.name)

        file_repository = get_repository(repository)
        url = file_repository.save(file_data)

        return cls(
            file_data=file_data,
            url=url,
            content_type=file_data.content_type,
            file_name=file_data.file_name,
            file_size=file_data.file_size,
        )

    # Pydantic custom validator for input type conversion
    @classmethod
    def __get_validators__(cls):
        yield cls.__convert_from_str

    @classmethod
    def __convert_from_str(cls, value: Any):
        if isinstance(value, str):
            url = urlparse(value)
            if url.scheme not in ["http", "https", "data"]:
                raise ValueError(f"value must be a valid URL")
            return cls._from_url(url.geturl())
        return value

    @classmethod
    def _from_url(
        cls,
        url: str,
    ) -> File:
        remote_url = RemoteFileIO(url)
        file_data = FileData(remote_url)

        return cls(
            file_data=file_data,
            url=url,
            content_type=file_data.content_type,
            file_name=file_data.file_name,
            file_size=file_data.file_size,
        )

    def as_bytes(self) -> bytes:
        content = self._file_data.as_bytes()
        self.file_size = len(content)

        if not content:
            raise Exception("File is empty")
        return content

    def save(self, path: str | Path):
        file_path = Path(path)
        file_path.write_bytes(self.as_bytes())


@mainify
class CompressedFile(File):
    _extract_dir: TemporaryDirectory | None = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._extract_dir = None

    def __iter__(self):
        if not self._extract_dir:
            self._extract_files(self.as_bytes())

        files = Path(self._extract_dir.name).iterdir()  # type: ignore
        return iter(files)

    def _extract_files(self, file_bytes: bytes):
        self._extract_dir = TemporaryDirectory()

        with ZipFile(BytesIO(file_bytes)) as zip_file:
            zip_file.extractall(self._extract_dir.name)

    def glob(self, pattern: str):
        if not self._extract_dir:
            self._extract_files(self.as_bytes())

        return Path(self._extract_dir.name).glob(pattern)  # type: ignore

    def __del__(self):
        if self._extract_dir:
            self._extract_dir.cleanup()
