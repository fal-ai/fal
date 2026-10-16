from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from functools import wraps
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
    from pydantic import model_validator

    IS_PYDANTIC_V2 = True

from pydantic import BaseModel, Field

from fal.compat import run_in_thread
from fal.flags import bool_envvar
from fal.ref import get_current_app
from fal.toolkit.exceptions import FileUploadFallbackException
from fal.toolkit.file._upload_policy import (
    UPLOAD_POLICY_KEY,
    UploadPolicyInputError,
    _current_request,
    get_upload_policy,
)
from fal.toolkit.file.providers.fal import (
    # Re-exported for backwards compatibility; both resolve to FalFileRepositoryV3.
    FalCDNFileRepository,  # noqa: F401
    FalFileRepository,
    FalFileRepositoryV2,  # noqa: F401
    FalFileRepositoryV3,
    InMemoryRepository,
    MultipartUploadV3,
)
from fal.toolkit.file.providers.gcp import GoogleStorageRepository
from fal.toolkit.file.providers.local import (
    LocalFileRepository,
    reserve_with_policy,
    save_with_policy,
)
from fal.toolkit.file.providers.r2 import R2Repository
from fal.toolkit.file.types import FileData, FileRepository, RepositoryId, Reservation
from fal.toolkit.utils.download_utils import download_file

FileRepositoryFactory = Callable[[], FileRepository]

BUILT_IN_REPOSITORIES: dict[RepositoryId, FileRepositoryFactory] = {
    "fal": lambda: FalFileRepository(),
    # Read on the runner at upload time, never when serializing the app.
    "fal_v3": lambda: (
        LocalFileRepository()
        if bool_envvar("FAL_USE_LOCAL_UPLOADER")
        else FalFileRepositoryV3()
    ),
    "in_memory": lambda: InMemoryRepository(),
    "gcp_storage": lambda: GoogleStorageRepository(),
    "r2": lambda: R2Repository(),
}


def get_builtin_repository(id: RepositoryId | FileRepository) -> FileRepository:
    if isinstance(id, FileRepository):
        return id

    if id in ["fal_v2", "cdn"]:
        import warnings  # noqa: PLC0415

        warnings.warn(
            f"`{id}` file repository is deprecated, falling back to `fal_v3`.",
            DeprecationWarning,
            stacklevel=2,
        )
        id = "fal_v3"

    if id not in BUILT_IN_REPOSITORIES.keys():
        raise ValueError(f'"{id}" is not a valid built-in file repository')
    return BUILT_IN_REPOSITORIES[id]()


get_builtin_repository.__module__ = "__main__"

DEFAULT_REPOSITORY: FileRepository | RepositoryId = "fal_v3"
FALLBACK_REPOSITORY: list[FileRepository | RepositoryId] = ["fal"]
OBJECT_LIFECYCLE_PREFERENCE_KEY = "x-fal-object-lifecycle-preference"


# get_builtin_repository folds both deprecated ids onto fal_v3, so all three name
# the default destination.
_DEFAULT_REPOSITORY_IDS = frozenset({DEFAULT_REPOSITORY, "fal_v2", "cdn"})
# By name, not by isinstance: `fal` is registered pickle-by-value
# (_serialization.include_module("fal")), so on a runner the class object here and
# the one an app's own import produced are different objects and isinstance would
# reject a genuinely default repository. FalFileRepositoryV2 and
# FalCDNFileRepository are aliases of the same class, so one name covers all three.
# LocalFileRepository is what FAL_USE_LOCAL_UPLOADER substitutes for "fal_v3".
_DEFAULT_REPOSITORY_TYPES = frozenset({"FalFileRepositoryV3", "LocalFileRepository"})


def _is_default_repository(repository: FileRepository | RepositoryId) -> bool:
    if isinstance(repository, str):
        return repository in _DEFAULT_REPOSITORY_IDS
    return type(repository).__name__ in _DEFAULT_REPOSITORY_TYPES


def _require_no_repository_conflict(
    repository: FileRepository | RepositoryId,
) -> None:
    """An app that named a destination other than the default has said where its
    output belongs, often a fal-owned bucket its own catalogue reads back from.
    Refuse rather than move someone else's artifacts."""
    if _is_default_repository(repository):
        return
    # Never repr() the repository here. GoogleStorageRepository and R2Repository
    # are dataclasses whose fields carry credentials, and this message reaches
    # the caller verbatim through AppException's handler.
    named = repository if isinstance(repository, str) else type(repository).__name__
    raise UploadPolicyInputError(
        f"Invalid {UPLOAD_POLICY_KEY}: this output is written to {named}, which a "
        "caller-supplied upload policy cannot override. Remove the header to use "
        "the destination the app chose."
    )


@wraps(Field)
def FileField(*args, **kwargs):
    if IS_PYDANTIC_V2:
        # Pydantic v2: use json_schema_extra
        json_schema_extra = kwargs.pop("json_schema_extra", None) or {}
        if callable(json_schema_extra):
            # If it's a callable, wrap it to also add ui.field
            original_func = json_schema_extra

            def merged_schema_extra(schema):
                original_func(schema)
                schema.setdefault("ui", {}).setdefault("field", "file")

            kwargs["json_schema_extra"] = merged_schema_extra
        else:
            json_schema_extra.setdefault("ui", {}).setdefault("field", "file")
            kwargs["json_schema_extra"] = json_schema_extra
    else:
        # Pydantic v1: use ui kwarg (stored in extra)
        ui = kwargs.get("ui", {})
        ui.setdefault("field", "file")
        kwargs["ui"] = ui
    return Field(*args, **kwargs)


def _repo_label(repo: FileRepository | RepositoryId) -> str:
    # Never format the object: S3Repository, R2Repository and
    # GoogleStorageRepository are dataclasses holding credentials.
    return repo if isinstance(repo, str) else type(repo).__name__


def _try_with_fallback(
    func: str,
    args: list[Any],
    repository: FileRepository | RepositoryId,
    fallback_repository: Optional[
        FileRepository | RepositoryId | list[FileRepository | RepositoryId]
    ],
    save_kwargs: dict,
    fallback_save_kwargs: dict,
) -> Any:
    if fallback_repository is None:
        fallback_repository = []
    elif isinstance(fallback_repository, list):
        pass
    else:
        fallback_repository = [fallback_repository]

    attempts: list[tuple[FileRepository | RepositoryId, dict]] = [
        (repository, save_kwargs),
        *((fallback, fallback_save_kwargs) for fallback in fallback_repository),
    ]
    failures: list[str] = []
    for idx, (repo, kwargs) in enumerate(attempts):
        repo_obj = get_builtin_repository(repo)
        try:
            return getattr(repo_obj, func)(*args, **kwargs)
        except Exception as exc:
            # An uncertain upload must not be replayed through another repository.
            if not getattr(exc, "falls_back", True):
                raise

            failure = f"{_repo_label(repo)}: {type(exc).__name__}: {exc}"
            failures.append(failure)
            if idx < len(attempts) - 1:
                print(
                    f"Failed to {func} to repository {failure}, "
                    f"falling back to {_repo_label(attempts[idx + 1][0])}"
                )
                continue

            if len(attempts) == 1:
                raise

            # The last error alone misleads: a denied fallback would hide the
            # failure that triggered it.
            raise FileUploadFallbackException(
                f"Failed to {func} to all {len(failures)} repositories. "
                + "; ".join(failures)
            ) from exc


def _get_object_lifecycle_preference_from_context() -> dict[str, str] | None:
    current_app = get_current_app()
    if current_app is None or current_app.current_request is None:
        return None
    return current_app.current_request.lifecycle_preference


def _lifecycle_preference(request: Optional[Request]) -> dict[str, str] | None:
    if request:
        return request_lifecycle_preference(request)
    return _get_object_lifecycle_preference_from_context()


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

        @model_validator(mode="before")
        @classmethod
        def __convert_from_str_v2(cls, value: Any):
            if isinstance(value, str):
                parsed_url = urlparse(value)
                if parsed_url.scheme not in ["http", "https", "data"]:
                    raise ValueError("value must be a valid URL")
                # Return a mapping so the model can be constructed normally
                return {"url": parsed_url.geturl()}
            return value

    else:

        @classmethod
        def __convert_from_dict(cls, value: Any):
            if isinstance(value, dict):
                return cls(**value)
            return value

        @classmethod
        def __get_validators__(cls):
            yield cls.__convert_from_dict
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
            file_size=None,
            file_data=None,
        )

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        content_type: Optional[str] = None,
        file_name: Optional[str] = None,
        repository: FileRepository | RepositoryId = DEFAULT_REPOSITORY,
        fallback_repository: Optional[
            FileRepository | RepositoryId | list[FileRepository | RepositoryId]
        ] = FALLBACK_REPOSITORY,
        request: Optional[Request] = None,
        save_kwargs: Optional[dict] = None,
        fallback_save_kwargs: Optional[dict] = None,
    ) -> File:
        save_kwargs = save_kwargs or {}
        fallback_save_kwargs = fallback_save_kwargs or {}

        fdata = FileData(data, content_type, file_name)

        # A caller's policy takes precedence over the default destination, but
        # never over one the app named explicitly. See
        # fal.toolkit.file._upload_policy.
        upload_policy = get_upload_policy(request)
        if upload_policy is not None:
            _require_no_repository_conflict(repository)
            url = save_with_policy(
                upload_policy, fdata.file_name, data, fdata.content_type
            )
            return cls(
                url=url,
                content_type=fdata.content_type,
                file_name=fdata.file_name,
                file_size=len(data),
                file_data=data,
            )

        object_lifecycle_preference = _lifecycle_preference(request)

        save_kwargs.setdefault(
            "object_lifecycle_preference", object_lifecycle_preference
        )
        fallback_save_kwargs.setdefault(
            "object_lifecycle_preference", object_lifecycle_preference
        )

        url = _try_with_fallback(
            "save",
            [fdata],
            repository=repository,
            fallback_repository=fallback_repository,
            save_kwargs=save_kwargs,
            fallback_save_kwargs=fallback_save_kwargs,
        )

        return cls(
            url=url,
            content_type=fdata.content_type,
            file_name=fdata.file_name,
            file_size=len(data),
            file_data=data,
        )

    @classmethod
    async def from_bytes_async(
        cls,
        data: bytes,
        content_type: Optional[str] = None,
        file_name: Optional[str] = None,
        repository: FileRepository | RepositoryId = DEFAULT_REPOSITORY,
        fallback_repository: Optional[
            FileRepository | RepositoryId | list[FileRepository | RepositoryId]
        ] = FALLBACK_REPOSITORY,
        request: Optional[Request] = None,
        save_kwargs: Optional[dict] = None,
        fallback_save_kwargs: Optional[dict] = None,
    ) -> File:
        return await run_in_thread(
            cls.from_bytes,
            data,
            content_type=content_type,
            file_name=file_name,
            repository=repository,
            fallback_repository=fallback_repository,
            request=request,
            save_kwargs=save_kwargs,
            fallback_save_kwargs=fallback_save_kwargs,
        )

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        content_type: Optional[str] = None,
        repository: FileRepository | RepositoryId = DEFAULT_REPOSITORY,
        multipart: bool | None = None,
        fallback_repository: Optional[
            FileRepository | RepositoryId | list[FileRepository | RepositoryId]
        ] = FALLBACK_REPOSITORY,
        request: Optional[Request] = None,
        save_kwargs: Optional[dict] = None,
        fallback_save_kwargs: Optional[dict] = None,
    ) -> File:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")

        save_kwargs = save_kwargs or {}
        fallback_save_kwargs = fallback_save_kwargs or {}

        content_type = content_type or "application/octet-stream"

        upload_policy = get_upload_policy(request)
        if upload_policy is not None:
            _require_no_repository_conflict(repository)
            file_size = file_path.stat().st_size
            # The upload-policy path is always a single S3 POST (<=5 GB), never a
            # multipart upload; this flag only picks how the payload is held for
            # it. Same threshold as the repository path, so .as_bytes() stays
            # consistent.
            stage_to_disk = (
                multipart
                if multipart is not None
                else file_size > MultipartUploadV3.MULTIPART_THRESHOLD
            )
            # Reusing small files' bytes for file_data avoids a second full read
            # and the staging copy for the common small-output case.
            file_data = None if stage_to_disk else file_path.read_bytes()
            url = save_with_policy(
                upload_policy,
                file_path.name,
                file_path if file_data is None else file_data,
                content_type,
            )
            return cls(
                url=url,
                file_data=file_data,
                content_type=content_type,
                file_name=file_path.name,
                file_size=file_size,
            )

        object_lifecycle_preference = _lifecycle_preference(request)

        save_kwargs.setdefault(
            "object_lifecycle_preference", object_lifecycle_preference
        )
        fallback_save_kwargs.setdefault(
            "object_lifecycle_preference", object_lifecycle_preference
        )

        save_kwargs.setdefault("multipart", multipart)
        fallback_save_kwargs.setdefault("multipart", multipart)

        save_kwargs.setdefault("content_type", content_type)
        fallback_save_kwargs.setdefault("content_type", content_type)

        url, data = _try_with_fallback(
            "save_file",
            [file_path],
            repository=repository,
            fallback_repository=fallback_repository,
            save_kwargs=save_kwargs,
            fallback_save_kwargs=fallback_save_kwargs,
        )

        return cls(
            url=url,
            file_data=data.data if data else None,
            content_type=content_type,
            file_name=file_path.name,
            file_size=file_path.stat().st_size,
        )

    @classmethod
    async def from_path_async(
        cls,
        path: str | Path,
        content_type: Optional[str] = None,
        repository: FileRepository | RepositoryId = DEFAULT_REPOSITORY,
        multipart: bool | None = None,
        fallback_repository: Optional[
            FileRepository | RepositoryId | list[FileRepository | RepositoryId]
        ] = FALLBACK_REPOSITORY,
        request: Optional[Request] = None,
        save_kwargs: Optional[dict] = None,
        fallback_save_kwargs: Optional[dict] = None,
    ) -> File:
        return await run_in_thread(
            cls.from_path,
            path,
            content_type=content_type,
            repository=repository,
            multipart=multipart,
            fallback_repository=fallback_repository,
            request=request,
            save_kwargs=save_kwargs,
            fallback_save_kwargs=fallback_save_kwargs,
        )

    @classmethod
    def reserve(
        cls,
        content_type: Optional[str] = None,
        file_name: Optional[str] = None,
        request: Optional[Request] = None,
        repository: FileRepository | RepositoryId = DEFAULT_REPOSITORY,
        extra_headers: dict[str, str] | None = None,
    ) -> PendingFile:
        """Get the file's URL now; send the bytes later with
        ``pending.complete(data)``, or give the URL up with ``pending.abort()``.

        The request context is read here, not at ``complete``. If ``complete``
        fails, the URL stays dead: nothing falls back once a URL is out.
        """
        fdata = FileData(b"", content_type, file_name)
        upload_policy = get_upload_policy(request)
        # The caller, fixed now: complete may run after the request, or under another.
        caller = request if request is not None else _current_request()
        if upload_policy is not None:
            _require_no_repository_conflict(repository)
            reservation = reserve_with_policy(
                upload_policy, fdata.file_name, fdata.content_type, caller
            )
        else:
            reservation = get_builtin_repository(repository).reserve(
                fdata.file_name,
                fdata.content_type,
                _lifecycle_preference(request),
                caller,
                extra_headers=extra_headers,
            )
        return PendingFile(fdata.content_type, fdata.file_name, reservation)

    def as_bytes(self) -> bytes:
        if self.file_data is None:
            raise ValueError("File has not been downloaded")

        return self.file_data

    def save(
        self,
        path: str | Path,
        overwrite: bool = False,
    ) -> Path:
        file_path = Path(path).resolve()

        if file_path.exists() and not overwrite:
            raise FileExistsError(f"File {file_path} already exists")

        downloaded_path = download_file(
            self.url,
            target_dir=file_path.parent,
        )
        downloaded_path.replace(file_path)

        return file_path

    async def save_async(
        self,
        path: str | Path,
        overwrite: bool = False,
    ) -> Path:
        return await run_in_thread(self.save, path, overwrite)


@dataclass
class PendingFile:
    """See File.reserve."""

    content_type: str
    file_name: str
    _reservation: Reservation = field(repr=False)

    @property
    def url(self) -> str:
        return self._reservation.url

    @property
    def _cdn_signing_requested(self) -> bool:
        return self._reservation.cdn_signing_requested

    def complete(self, content: bytes | str | Path) -> File:
        if not isinstance(content, bytes):
            content = Path(content)
        self._reservation.complete(content)
        return File(
            url=self.url,
            content_type=self.content_type,
            file_name=self.file_name,
            file_size=(
                len(content) if isinstance(content, bytes) else content.stat().st_size
            ),
            file_data=content if isinstance(content, bytes) else None,
        )

    def abort(self) -> None:
        self._reservation.abort()


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


def request_lifecycle_preference(request: Request) -> dict[str, str] | None:
    import json  # noqa: PLC0415

    preference_str = request.headers.get(OBJECT_LIFECYCLE_PREFERENCE_KEY)
    if preference_str is None:
        return None

    try:
        return json.loads(preference_str)
    except Exception as e:
        print(f"Failed to parse object lifecycle preference: {e}")
        return None
