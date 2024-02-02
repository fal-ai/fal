from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any, Callable
from urllib.parse import urlparse
from zipfile import ZipFile

from pydantic import BaseModel, Field, PrivateAttr
from pydantic.typing import Optional

from fal.toolkit.file.providers.fal import (
    FalCDNFileRepository,
    FalFileRepository,
    InMemoryRepository,
)
from fal.toolkit.file.providers.gcp import GoogleStorageRepository
from fal.toolkit.file.providers.r2 import R2Repository
from fal.toolkit.file.types import FileData, FileRepository, RepositoryId
from fal.toolkit.mainify import mainify
from fal.toolkit.utils.download_utils import download_file

FileRepositoryFactory = Callable[[], FileRepository]

BUILT_IN_REPOSITORIES: dict[RepositoryId, FileRepositoryFactory] = {
    "fal": lambda: FalFileRepository(),
    "in_memory": lambda: InMemoryRepository(),
    "gcp_storage": lambda: GoogleStorageRepository(),
    "r2": lambda: R2Repository(),
    "cdn": lambda: FalCDNFileRepository(),
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
        if "file_data" in kwargs:
            data: FileData = kwargs.pop("file_data")
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


@mainify
class CompressedFile(File):
    _extract_dir: Optional[TemporaryDirectory] = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._extract_dir = None

    def __iter__(self):
        if not self._extract_dir:
            self._extract_files()

        files = Path(self._extract_dir.name).iterdir()  # type: ignore
        return iter(files)

    def _extract_files(self):
        self._extract_dir = TemporaryDirectory()

        with NamedTemporaryFile() as temp_file:
            file_path = temp_file.name
            self.save(file_path, overwrite=True)

            with ZipFile(file_path) as zip_file:
                zip_file.extractall(self._extract_dir.name)

    def glob(self, pattern: str):
        if not self._extract_dir:
            self._extract_files()

        return Path(self._extract_dir.name).glob(pattern)  # type: ignore

    def __del__(self):
        if self._extract_dir:
            self._extract_dir.cleanup()
