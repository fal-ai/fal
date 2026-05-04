from __future__ import annotations

import json
import secrets
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp
from threading import BoundedSemaphore
from typing import Any, Callable, Optional
from urllib.error import HTTPError
from urllib.parse import quote, urlparse
from urllib.request import Request as URLRequest
from urllib.request import urlopen
from uuid import uuid4
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
from fal.ref import get_current_app
from fal.toolkit.exceptions import FileUploadException
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


def get_builtin_repository(id: RepositoryId | FileRepository) -> FileRepository:
    if isinstance(id, FileRepository):
        return id

    if id not in BUILT_IN_REPOSITORIES.keys():
        raise ValueError(f'"{id}" is not a valid built-in file repository')
    return BUILT_IN_REPOSITORIES[id]()


get_builtin_repository.__module__ = "__main__"

DEFAULT_REPOSITORY: FileRepository | RepositoryId = "fal_v3"
FALLBACK_REPOSITORY: list[FileRepository | RepositoryId] = ["cdn", "fal"]
OBJECT_LIFECYCLE_PREFERENCE_KEY = "x-fal-object-lifecycle-preference"
UPLOAD_POLICY_KEY = "x-app-fal-upload-policy"
UPLOAD_POLICY_POST_TIMEOUT = 5 * 60
UPLOAD_POLICY_FILENAME_PLACEHOLDER = "${filename}"
UPLOAD_POLICY_MAX_PENDING = 32
UPLOAD_POLICY_EXECUTOR = ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix="fal-upload-policy",
)
UPLOAD_POLICY_PENDING = BoundedSemaphore(UPLOAD_POLICY_MAX_PENDING)


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
    for idx, (repo, kwargs) in enumerate(attempts):
        repo_obj = get_builtin_repository(repo)
        try:
            return getattr(repo_obj, func)(*args, **kwargs)
        except Exception as exc:
            if idx >= len(attempts) - 1:
                raise

            traceback.print_exc()
            print(
                f"Failed to {func} to repository {repo}: {exc}, "
                f"falling back to {attempts[idx + 1][0]}"
            )


def _get_object_lifecycle_preference_from_context() -> dict[str, str] | None:
    current_app = get_current_app()
    if current_app is None or current_app.current_request is None:
        return None
    return current_app.current_request.lifecycle_preference


def parse_upload_policy(headers: Any) -> dict | None:
    """Parse the X-App-Fal-Upload-Policy header.

    Expected value: the JSON object returned by S3's generate_presigned_post(),
    which has the shape::

        {"url": "https://bucket.s3.<region>.amazonaws.com/",
         "fields": {"key": "...", "policy": "...", "x-amz-signature": "...", ...}}

    Returns the parsed policy dict, or None if the header is absent.
    Raises FileUploadException on malformed input.
    """
    raw = headers.get(UPLOAD_POLICY_KEY)
    if raw is None:
        return None

    try:
        policy = json.loads(raw)
    except Exception as exc:
        raise FileUploadException(
            f"Invalid {UPLOAD_POLICY_KEY} header: not valid JSON ({exc})"
        )

    if not isinstance(policy, dict):
        raise FileUploadException(
            f"Invalid {UPLOAD_POLICY_KEY} header: expected a JSON object"
        )

    url = policy.get("url")
    fields = policy.get("fields")
    if not isinstance(url, str) or not isinstance(fields, dict):
        raise FileUploadException(
            f"Invalid {UPLOAD_POLICY_KEY} header: must contain string 'url' and "
            "object 'fields'"
        )

    if not url.lower().startswith(("http://", "https://")):
        raise FileUploadException(
            f"Invalid {UPLOAD_POLICY_KEY} 'url': must start with http:// or https://"
        )

    return policy


def _get_upload_policy() -> dict | None:
    """Return the parsed upload policy for the current request, or None."""
    current_app = get_current_app()
    if current_app is None or current_app.current_request is None:
        return None
    return getattr(current_app.current_request, "upload_policy", None)


def _escape_multipart_header(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", "%0D")
        .replace("\n", "%0A")
    )


def _validate_multipart_header_value(name: str, value: str) -> None:
    if "\r" in value or "\n" in value:
        raise FileUploadException(f"Invalid multipart {name}: contains CR/LF")


def _build_multipart_form(
    fields: dict[str, str],
    file_bytes: bytes,
    file_name: str,
    file_content_type: str,
) -> tuple[bytes, str]:
    """Build a multipart/form-data body matching S3's pre-signed POST contract.

    Per S3 spec, the 'file' field MUST be last. All policy fields come first.
    Returns (body_bytes, content_type_header).
    """
    boundary = f"----FalUploadBoundary{secrets.token_hex(16)}"
    lines: list[bytes] = []

    for name, value in fields.items():
        lines.append(f"--{boundary}".encode())
        escaped_name = _escape_multipart_header(str(name))
        lines.append(f'Content-Disposition: form-data; name="{escaped_name}"'.encode())
        lines.append(b"")
        lines.append(str(value).encode())

    lines.append(f"--{boundary}".encode())
    escaped_file_name = _escape_multipart_header(file_name)
    lines.append(
        f'Content-Disposition: form-data; name="file"; '
        f'filename="{escaped_file_name}"'.encode()
    )
    _validate_multipart_header_value("content type", file_content_type)
    lines.append(f"Content-Type: {file_content_type}".encode())
    lines.append(b"")
    lines.append(file_bytes)

    lines.append(f"--{boundary}--".encode())
    lines.append(b"")

    body = b"\r\n".join(lines)
    return body, f"multipart/form-data; boundary={boundary}"


def _build_upload_policy_request(
    policy: dict,
    file_name: str,
    data: bytes,
    content_type: str,
) -> tuple[str, URLRequest]:
    fields = dict(policy["fields"])
    key_template = fields.get("key")
    if (
        not isinstance(key_template, str)
        or UPLOAD_POLICY_FILENAME_PLACEHOLDER not in key_template
    ):
        raise FileUploadException(
            f"Invalid {UPLOAD_POLICY_KEY} header: fields.key must contain "
            f"{UPLOAD_POLICY_FILENAME_PLACEHOLDER!r}"
        )

    upload_file_name = f"{uuid4().hex}-{file_name}"
    final_key = key_template.replace(
        UPLOAD_POLICY_FILENAME_PLACEHOLDER, upload_file_name
    )
    fields["key"] = final_key
    fields["Content-Type"] = content_type

    body, content_type_header = _build_multipart_form(
        fields, data, file_name, content_type
    )

    request = URLRequest(
        policy["url"],
        data=body,
        headers={"Content-Type": content_type_header},
        method="POST",
    )
    encoded_key = quote(final_key.lstrip("/"), safe="/~")
    return f"{policy['url'].rstrip('/')}/{encoded_key}", request


def _post_upload_policy_request(request: URLRequest) -> None:
    try:
        with urlopen(request, timeout=UPLOAD_POLICY_POST_TIMEOUT):
            pass
    except HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        raise FileUploadException(
            f"Upload via {UPLOAD_POLICY_KEY} failed. "
            f"Status {exc.code}: {exc.reason}. {detail}"
        )
    except Exception as exc:
        raise FileUploadException(f"Upload via {UPLOAD_POLICY_KEY} failed: {exc}")


def _complete_upload_policy_request(future) -> None:
    UPLOAD_POLICY_PENDING.release()
    try:
        future.result()
    except Exception:
        traceback.print_exc()


def _reserve_upload_policy_slot() -> None:
    if not UPLOAD_POLICY_PENDING.acquire(blocking=False):
        raise FileUploadException(f"Too many pending {UPLOAD_POLICY_KEY} uploads")


def _enqueue_upload_via_policy(
    policy: dict, file_name: str, data: bytes, content_type: str
) -> str:
    try:
        url, request = _build_upload_policy_request(
            policy, file_name, data, content_type
        )
        future = UPLOAD_POLICY_EXECUTOR.submit(_post_upload_policy_request, request)
    except Exception:
        UPLOAD_POLICY_PENDING.release()
        raise

    future.add_done_callback(_complete_upload_policy_request)
    return url


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

        upload_policy = _get_upload_policy()
        if upload_policy is not None:
            _reserve_upload_policy_slot()
            url = _enqueue_upload_via_policy(
                upload_policy, fdata.file_name, data, fdata.content_type
            )
            return cls(
                url=url,
                content_type=fdata.content_type,
                file_name=fdata.file_name,
                file_size=len(data),
                file_data=data,
            )

        if request:
            object_lifecycle_preference = request_lifecycle_preference(request)
        else:
            object_lifecycle_preference = (
                _get_object_lifecycle_preference_from_context()
            )

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

        upload_policy = _get_upload_policy()
        if upload_policy is not None:
            _reserve_upload_policy_slot()
            try:
                with open(file_path, "rb") as f:
                    file_bytes = f.read()
            except Exception:
                UPLOAD_POLICY_PENDING.release()
                raise
            url = _enqueue_upload_via_policy(
                upload_policy, file_path.name, file_bytes, content_type
            )
            return cls(
                url=url,
                file_data=file_bytes,
                content_type=content_type,
                file_name=file_path.name,
                file_size=file_path.stat().st_size,
            )

        if request:
            object_lifecycle_preference = request_lifecycle_preference(request)
        else:
            object_lifecycle_preference = (
                _get_object_lifecycle_preference_from_context()
            )

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

    async def save_async(self, path: str | Path, overwrite: bool = False) -> Path:
        return await run_in_thread(self.save, path, overwrite)


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
