from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp
from typing import Any, Callable, Optional
from urllib.parse import urlparse
from zipfile import ZipFile

import pydantic
from fastapi import Request

# https://github.com/pydantic/pydantic/pull/2573
if not hasattr(pydantic, "__version__") or pydantic.__version__.startswith("1."):
    IS_PYDANTIC_V2 = False
else:
    from pydantic import GetCoreSchemaHandler
    from pydantic_core import CoreSchema, core_schema

    IS_PYDANTIC_V2 = True

from pydantic import BaseModel, Field

from fal.toolkit.file.providers.fal import (
    FalCDNFileRepository,
    FalFileRepository,
    FalFileRepositoryV2,
    FalFileRepositoryV3,
    InMemoryRepository,
)
from fal.toolkit.file.providers.gcp import GoogleStorageRepository
from fal.toolkit.file.providers.r2 import R2Repository
from fal.toolkit.file.types import FileData, FileRepository, RepositoryId
from fal.toolkit.utils.download_utils import download_file

FileRepositoryFactory = Callable[[], FileRepository]

BUILT_IN_REPOSITORIES: dict[RepositoryId, FileRepositoryFactory] = {
    "fal": lambda: FalFileRepository(),
    "fal_v2": lambda: FalFileRepositoryV2(),
    "fal_v3": lambda: FalFileRepositoryV3(),
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

DEFAULT_REPOSITORY: FileRepository | RepositoryId = "fal_v3"
FALLBACK_REPOSITORY: FileRepository | RepositoryId = "cdn"
OBJECT_LIFECYCLE_PREFERENCE_KEY = "x-fal-object-lifecycle-preference"


class File(BaseModel):
    # public properties
    url: str = Field(
        description="The URL where the file can be downloaded from.",
    )
    content_type: Optional[str] = Field(
        None,
        description="The mime type of the file.",
        examples=["image/png"],
    )
    file_name: Optional[str] = Field(
        None,
        description="The name of the file. It will be auto-generated if not provided.",
        examples=["z9RV14K95DvU.png"],
    )
    file_size: Optional[int] = Field(
        None, description="The size of the file in bytes.", examples=[4404019]
    )
    file_data: Optional[bytes] = Field(
        None,
        description="File data",
        exclude=True,
        repr=False,
    )

    # Pydantic custom validator for input type conversion
    if IS_PYDANTIC_V2:

        @classmethod
        def __get_pydantic_core_schema__(
            cls, source_type: Any, handler: GetCoreSchemaHandler
        ) -> CoreSchema:
            return core_schema.no_info_before_validator_function(
                cls.__convert_from_str,
                handler(source_type),
            )

    else:

        @classmethod
        def __get_validators__(cls):
            yield cls.__convert_from_str

    @classmethod
    def __convert_from_str(cls, value: Any):
        if isinstance(value, str):
            parsed_url = urlparse(value)
            if parsed_url.scheme not in ["http", "https", "data"]:
                raise ValueError("value must be a valid URL")
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
        fallback_repository: Optional[
            FileRepository | RepositoryId
        ] = FALLBACK_REPOSITORY,
        request: Optional[Request] = None,
        save_kwargs: Optional[dict] = None,
        fallback_save_kwargs: Optional[dict] = None,
    ) -> File:
        repo = (
            repository
            if isinstance(repository, FileRepository)
            else get_builtin_repository(repository)
        )

        save_kwargs = save_kwargs or {}
        fallback_save_kwargs = fallback_save_kwargs or {}

        fdata = FileData(data, content_type, file_name)

        object_lifecycle_preference = get_lifecycle_preference(request)

        try:
            url = repo.save(fdata, object_lifecycle_preference, **save_kwargs)
        except Exception:
            if not fallback_repository:
                raise

            fallback_repo = (
                fallback_repository
                if isinstance(fallback_repository, FileRepository)
                else get_builtin_repository(fallback_repository)
            )

            url = fallback_repo.save(
                fdata, object_lifecycle_preference, **fallback_save_kwargs
            )

        return cls(
            url=url,
            content_type=fdata.content_type,
            file_name=fdata.file_name,
            file_size=len(data),
            file_data=data,
        )

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        content_type: Optional[str] = None,
        repository: FileRepository | RepositoryId = DEFAULT_REPOSITORY,
        multipart: bool | None = None,
        fallback_repository: Optional[
            FileRepository | RepositoryId
        ] = FALLBACK_REPOSITORY,
        request: Optional[Request] = None,
        save_kwargs: Optional[dict] = None,
        fallback_save_kwargs: Optional[dict] = None,
    ) -> File:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")

        repo = (
            repository
            if isinstance(repository, FileRepository)
            else get_builtin_repository(repository)
        )

        save_kwargs = save_kwargs or {}
        fallback_save_kwargs = fallback_save_kwargs or {}

        content_type = content_type or "application/octet-stream"
        object_lifecycle_preference = get_lifecycle_preference(request)

        try:
            url, data = repo.save_file(
                file_path,
                content_type=content_type,
                multipart=multipart,
                object_lifecycle_preference=object_lifecycle_preference,
                **save_kwargs,
            )
        except Exception:
            if not fallback_repository:
                raise

            fallback_repo = (
                fallback_repository
                if isinstance(fallback_repository, FileRepository)
                else get_builtin_repository(fallback_repository)
            )

            url, data = fallback_repo.save_file(
                file_path,
                content_type=content_type,
                multipart=multipart,
                object_lifecycle_preference=object_lifecycle_preference,
                **fallback_save_kwargs,
            )

        return cls(
            url=url,
            file_data=data.data if data else None,
            content_type=content_type,
            file_name=file_path.name,
            file_size=file_path.stat().st_size,
        )

    def as_bytes(self) -> bytes:
        if self.file_data is None:
            raise ValueError("File has not been downloaded")

        return self.file_data

    def save(self, path: str | Path, overwrite: bool = False) -> Path:
        file_path = Path(path).resolve()

        if file_path.exists() and not overwrite:
            raise FileExistsError(f"File {file_path} already exists")

        downloaded_path = download_file(self.url, target_dir=file_path.parent)
        downloaded_path.rename(file_path)

        return file_path


class CompressedFile(File):
    extract_dir: Optional[str] = Field(default=None, exclude=True, repr=False)

    def __iter__(self):
        if not self.extract_dir:
            self._extract_files()

        files = Path(self.extract_dir).iterdir()  # type: ignore
        return iter(files)

    def _extract_files(self):
        self.extract_dir = mkdtemp()

        with NamedTemporaryFile() as temp_file:
            file_path = temp_file.name
            self.save(file_path, overwrite=True)

            with ZipFile(file_path) as zip_file:
                zip_file.extractall(self.extract_dir)

    def glob(self, pattern: str):
        if not self.extract_dir:
            self._extract_files()

        return Path(self.extract_dir).glob(pattern)  # type: ignore

    def __del__(self):
        if self.extract_dir:
            shutil.rmtree(self.extract_dir)


def get_lifecycle_preference(request: Request) -> dict[str, str] | None:
    import json

    preference_str = (
        request.headers.get(OBJECT_LIFECYCLE_PREFERENCE_KEY)
        if request is not None
        else None
    )
    if preference_str is None:
        return None

    object_lifecycle_preference = {}
    try:
        object_lifecycle_preference = json.loads(preference_str)
        return object_lifecycle_preference
    except Exception as e:
        print(f"Failed to parse object lifecycle preference: {e}")
        return None
