from __future__ import annotations

import json
import math
import os
import threading
from base64 import b64encode
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Generic, TypeVar
from urllib.error import HTTPError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

from fal.auth import key_credentials
from fal.toolkit.exceptions import FileUploadException
from fal.toolkit.file.types import FileData, FileRepository
from fal.toolkit.utils.retry import retry

_FAL_CDN = "https://fal.media"
_FAL_CDN_V3 = "https://v3.fal.media"


@dataclass
class FalV2Token:
    token: str
    token_type: str
    base_upload_url: str
    expires_at: datetime

    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) >= self.expires_at


class FalV3Token(FalV2Token):
    pass


class FalV2TokenManager:
    token_cls: type[FalV2Token] = FalV2Token
    storage_type: str = "fal-cdn"
    upload_prefix = "upload."

    def __init__(self):
        self._token: FalV2Token = self.token_cls(
            token="",
            token_type="",
            base_upload_url="",
            expires_at=datetime.min.replace(tzinfo=timezone.utc),
        )
        self._lock: threading.Lock = threading.Lock()

    def get_token(self) -> FalV2Token:
        with self._lock:
            if self._token.is_expired():
                self._refresh_token()
            return self._token

    def _refresh_token(self) -> None:
        key_creds = key_credentials()
        if not key_creds:
            raise FileUploadException("FAL_KEY must be set")

        key_id, key_secret = key_creds
        headers = {
            "Authorization": f"Key {key_id}:{key_secret}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        grpc_host = os.environ.get("FAL_HOST", "api.alpha.fal.ai")
        rest_host = grpc_host.replace("api", "rest", 1)
        url = f"https://{rest_host}/storage/auth/token?storage_type={self.storage_type}"

        req = Request(
            url,
            headers=headers,
            data=b"{}",
            method="POST",
        )
        with urlopen(req) as response:
            result = json.load(response)

        parsed_base_url = urlparse(result["base_url"])
        base_upload_url = urlunparse(
            parsed_base_url._replace(netloc=self.upload_prefix + parsed_base_url.netloc)
        )

        self._token = self.token_cls(
            token=result["token"],
            token_type=result["token_type"],
            base_upload_url=base_upload_url,
            expires_at=datetime.fromisoformat(result["expires_at"]),
        )


class FalV3TokenManager(FalV2TokenManager):
    token_cls: type[FalV2Token] = FalV3Token
    storage_type: str = "fal-cdn-v3"
    upload_prefix = ""


fal_v2_token_manager = FalV2TokenManager()
fal_v3_token_manager = FalV3TokenManager()


VariableType = TypeVar("VariableType")


class VariableReference(Generic[VariableType]):
    def __init__(self, value: VariableType) -> None:
        self.set(value)

    def get(self) -> VariableType:
        return self.value

    def set(self, value: VariableType) -> None:
        self.value = value


LIFECYCLE_PREFERENCE: VariableReference[dict[str, str] | None] = VariableReference(None)


@dataclass
class FalFileRepositoryBase(FileRepository):
    @retry(max_retries=3, base_delay=1, backoff_type="exponential", jitter=True)
    def _save(self, file: FileData, storage_type: str) -> str:
        key_creds = key_credentials()
        if not key_creds:
            raise FileUploadException("FAL_KEY must be set")

        key_id, key_secret = key_creds
        headers = {
            "Authorization": f"Key {key_id}:{key_secret}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        grpc_host = os.environ.get("FAL_HOST", "api.alpha.fal.ai")
        rest_host = grpc_host.replace("api", "rest", 1)
        storage_url = (
            f"https://{rest_host}/storage/upload/initiate?storage_type={storage_type}"
        )

        try:
            req = Request(
                storage_url,
                data=json.dumps(
                    {
                        "file_name": file.file_name,
                        "content_type": file.content_type,
                    }
                ).encode(),
                headers=headers,
                method="POST",
            )
            with urlopen(req) as response:
                result = json.load(response)

            upload_url = result["upload_url"]
        except HTTPError as e:
            raise FileUploadException(
                f"Error initiating upload. Status {e.status}: {e.reason}"
            )

        try:
            req = Request(
                upload_url,
                method="PUT",
                data=file.data,
                headers={"Content-Type": file.content_type},
            )

            with urlopen(req):
                pass

            return result["file_url"]
        except HTTPError as e:
            raise FileUploadException(
                f"Error uploading file. Status {e.status}: {e.reason}"
            )


@dataclass
class FalFileRepository(FalFileRepositoryBase):
    def save(
        self, file: FileData, object_lifecycle_preference: dict[str, str] | None = None
    ) -> str:
        return self._save(file, "gcs")


@dataclass
class FalFileRepositoryV3(FalFileRepositoryBase):
    def save(
        self, file: FileData, object_lifecycle_preference: dict[str, str] | None = None
    ) -> str:
        return self._save(file, "fal-cdn-v3")


class MultipartUpload:
    MULTIPART_THRESHOLD = 100 * 1024 * 1024
    MULTIPART_CHUNK_SIZE = 100 * 1024 * 1024
    MULTIPART_MAX_CONCURRENCY = 10

    def __init__(
        self,
        file_name: str,
        chunk_size: int | None = None,
        content_type: str | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        self.file_name = file_name
        self.chunk_size = chunk_size or self.MULTIPART_CHUNK_SIZE
        self.content_type = content_type or "application/octet-stream"
        self.max_concurrency = max_concurrency or self.MULTIPART_MAX_CONCURRENCY

        self._parts: list[dict] = []

    def create(self):
        token = fal_v2_token_manager.get_token()
        try:
            req = Request(
                f"{token.base_upload_url}/upload/initiate-multipart",
                method="POST",
                headers={
                    "Authorization": f"{token.token_type} {token.token}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                data=json.dumps(
                    {
                        "file_name": self.file_name,
                        "content_type": self.content_type,
                    }
                ).encode(),
            )
            with urlopen(req) as response:
                result = json.load(response)
                self._upload_url = result["upload_url"]
                self._file_url = result["file_url"]
        except HTTPError as exc:
            raise FileUploadException(
                f"Error initiating upload. Status {exc.status}: {exc.reason}"
            )

    def upload_part(self, part_number: int, data: bytes) -> None:
        url = f"{self._upload_url}&part_number={part_number}"

        req = Request(
            url,
            method="PUT",
            headers={"Content-Type": self.content_type},
            data=data,
        )

        try:
            with urlopen(req) as resp:
                self._parts.append(
                    {
                        "part_number": part_number,
                        "etag": resp.headers["ETag"],
                    }
                )
        except HTTPError as exc:
            raise FileUploadException(
                f"Error uploading part {part_number} to {url}. "
                f"Status {exc.status}: {exc.reason}"
            )

    def complete(self):
        url = self._upload_url
        try:
            req = Request(
                url,
                method="POST",
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                data=json.dumps({"parts": self._parts}).encode(),
            )
            with urlopen(req):
                pass
        except HTTPError as e:
            raise FileUploadException(
                f"Error completing upload {url}. Status {e.status}: {e.reason}"
            )

        return self._file_url

    @classmethod
    def save(
        cls,
        file: FileData,
        chunk_size: int | None = None,
        max_concurrency: int | None = None,
    ):
        import concurrent.futures

        multipart = cls(
            file.file_name,
            chunk_size=chunk_size,
            content_type=file.content_type,
            max_concurrency=max_concurrency,
        )
        multipart.create()

        parts = math.ceil(len(file.data) / multipart.chunk_size)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=multipart.max_concurrency
        ) as executor:
            futures = []
            for part_number in range(1, parts + 1):
                start = (part_number - 1) * multipart.chunk_size
                data = file.data[start : start + multipart.chunk_size]
                futures.append(
                    executor.submit(multipart.upload_part, part_number, data)
                )

            for future in concurrent.futures.as_completed(futures):
                future.result()

        return multipart.complete()

    @classmethod
    def save_file(
        cls,
        file_path: str | Path,
        chunk_size: int | None = None,
        content_type: str | None = None,
        max_concurrency: int | None = None,
    ) -> str:
        import concurrent.futures

        file_name = os.path.basename(file_path)
        size = os.path.getsize(file_path)

        multipart = cls(
            file_name,
            chunk_size=chunk_size,
            content_type=content_type,
            max_concurrency=max_concurrency,
        )
        multipart.create()

        parts = math.ceil(size / multipart.chunk_size)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=multipart.max_concurrency
        ) as executor:
            futures = []
            for part_number in range(1, parts + 1):

                def _upload_part(pn: int) -> None:
                    with open(file_path, "rb") as f:
                        start = (pn - 1) * multipart.chunk_size
                        f.seek(start)
                        data = f.read(multipart.chunk_size)
                        multipart.upload_part(pn, data)

                futures.append(executor.submit(_upload_part, part_number))

            for future in concurrent.futures.as_completed(futures):
                future.result()

        return multipart.complete()


class InternalMultipartUploadV3:
    MULTIPART_THRESHOLD = 100 * 1024 * 1024
    MULTIPART_CHUNK_SIZE = 10 * 1024 * 1024
    MULTIPART_MAX_CONCURRENCY = 10

    def __init__(
        self,
        file_name: str,
        chunk_size: int | None = None,
        content_type: str | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        self.file_name = file_name
        self.chunk_size = chunk_size or self.MULTIPART_CHUNK_SIZE
        self.content_type = content_type or "application/octet-stream"
        self.max_concurrency = max_concurrency or self.MULTIPART_MAX_CONCURRENCY
        self._access_url: str | None = None
        self._upload_id: str | None = None

        self._parts: list[dict] = []

    @property
    def access_url(self) -> str:
        if not self._access_url:
            raise FileUploadException("Upload not initiated")
        return self._access_url

    @property
    def upload_id(self) -> str:
        if not self._upload_id:
            raise FileUploadException("Upload not initiated")
        return self._upload_id

    @property
    def auth_headers(self) -> dict[str, str]:
        token = fal_v3_token_manager.get_token()
        return {
            "Authorization": f"{token.token_type} {token.token}",
            "User-Agent": "fal/0.1.0",
        }

    def create(self):
        token = fal_v3_token_manager.get_token()
        try:
            req = Request(
                f"{token.base_upload_url}/files/upload/multipart",
                method="POST",
                headers={
                    **self.auth_headers,
                    "Accept": "application/json",
                    "Content-Type": self.content_type,
                    "X-Fal-File-Name": self.file_name,
                },
            )
            with urlopen(req) as response:
                result = json.load(response)
                self._access_url = result["access_url"]
                self._upload_id = result["uploadId"]

        except HTTPError as exc:
            raise FileUploadException(
                f"Error initiating upload. Status {exc.status}: {exc.reason}"
            )

    @retry(max_retries=5, base_delay=1, backoff_type="exponential", jitter=True)
    def upload_part(self, part_number: int, data: bytes) -> None:
        url = f"{self.access_url}/multipart/{self.upload_id}/{part_number}"

        req = Request(
            url,
            method="PUT",
            headers={
                **self.auth_headers,
                "Content-Type": self.content_type,
            },
            data=data,
        )

        try:
            with urlopen(req) as resp:
                self._parts.append(
                    {
                        "partNumber": part_number,
                        "etag": resp.headers["ETag"],
                    }
                )
        except HTTPError as exc:
            raise FileUploadException(
                f"Error uploading part {part_number} to {url}. "
                f"Status {exc.status}: {exc.reason}"
            )

    def complete(self) -> str:
        url = f"{self.access_url}/multipart/{self.upload_id}/complete"
        try:
            req = Request(
                url,
                method="POST",
                headers={
                    **self.auth_headers,
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                data=json.dumps({"parts": self._parts}).encode(),
            )
            with urlopen(req):
                pass
        except HTTPError as e:
            raise FileUploadException(
                f"Error completing upload {url}. Status {e.status}: {e.reason}"
            )

        return self.access_url

    @classmethod
    def save(
        cls,
        file: FileData,
        chunk_size: int | None = None,
        max_concurrency: int | None = None,
    ):
        import concurrent.futures

        multipart = cls(
            file.file_name,
            chunk_size=chunk_size,
            content_type=file.content_type,
            max_concurrency=max_concurrency,
        )
        multipart.create()

        parts = math.ceil(len(file.data) / multipart.chunk_size)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=multipart.max_concurrency
        ) as executor:
            futures = []
            for part_number in range(1, parts + 1):
                start = (part_number - 1) * multipart.chunk_size
                data = file.data[start : start + multipart.chunk_size]
                futures.append(
                    executor.submit(multipart.upload_part, part_number, data)
                )

            for future in concurrent.futures.as_completed(futures):
                future.result()

        return multipart.complete()

    @classmethod
    def save_file(
        cls,
        file_path: str | Path,
        chunk_size: int | None = None,
        content_type: str | None = None,
        max_concurrency: int | None = None,
    ) -> str:
        import concurrent.futures

        file_name = os.path.basename(file_path)
        size = os.path.getsize(file_path)

        multipart = cls(
            file_name,
            chunk_size=chunk_size,
            content_type=content_type,
            max_concurrency=max_concurrency,
        )
        multipart.create()

        parts = math.ceil(size / multipart.chunk_size)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=multipart.max_concurrency
        ) as executor:
            futures = []
            for part_number in range(1, parts + 1):

                def _upload_part(pn: int) -> None:
                    with open(file_path, "rb") as f:
                        start = (pn - 1) * multipart.chunk_size
                        f.seek(start)
                        data = f.read(multipart.chunk_size)
                        multipart.upload_part(pn, data)

                futures.append(executor.submit(_upload_part, part_number))

            for future in concurrent.futures.as_completed(futures):
                future.result()

        return multipart.complete()


@dataclass
class FalFileRepositoryV2(FalFileRepositoryBase):
    @retry(max_retries=3, base_delay=1, backoff_type="exponential", jitter=True)
    def save(
        self,
        file: FileData,
        multipart: bool | None = None,
        multipart_threshold: int | None = None,
        multipart_chunk_size: int | None = None,
        multipart_max_concurrency: int | None = None,
        object_lifecycle_preference: dict[str, str] | None = None,
    ) -> str:
        if multipart is None:
            threshold = multipart_threshold or MultipartUpload.MULTIPART_THRESHOLD
            multipart = len(file.data) > threshold

        if multipart:
            return MultipartUpload.save(
                file,
                chunk_size=multipart_chunk_size,
                max_concurrency=multipart_max_concurrency,
            )

        token = fal_v2_token_manager.get_token()
        headers = {
            "Authorization": f"{token.token_type} {token.token}",
            "Accept": "application/json",
            "X-Fal-File-Name": file.file_name,
            "Content-Type": file.content_type,
        }

        storage_url = f"{token.base_upload_url}/upload"

        try:
            req = Request(
                storage_url,
                data=file.data,
                headers=headers,
                method="PUT",
            )
            with urlopen(req) as response:
                result = json.load(response)

            return result["file_url"]
        except HTTPError as e:
            raise FileUploadException(
                f"Error initiating upload. Status {e.status}: {e.reason}"
            )

    def save_file(
        self,
        file_path: str | Path,
        content_type: str,
        multipart: bool | None = None,
        multipart_threshold: int | None = None,
        multipart_chunk_size: int | None = None,
        multipart_max_concurrency: int | None = None,
        object_lifecycle_preference: dict[str, str] | None = None,
    ) -> tuple[str, FileData | None]:
        if multipart is None:
            threshold = multipart_threshold or MultipartUpload.MULTIPART_THRESHOLD
            multipart = os.path.getsize(file_path) > threshold

        if multipart:
            url = MultipartUpload.save_file(
                file_path,
                chunk_size=multipart_chunk_size,
                content_type=content_type,
                max_concurrency=multipart_max_concurrency,
            )
            data = None
        else:
            with open(file_path, "rb") as f:
                data = FileData(
                    f.read(),
                    content_type=content_type,
                    file_name=os.path.basename(file_path),
                )
            url = self.save(data, object_lifecycle_preference)

        return url, data


@dataclass
class InMemoryRepository(FileRepository):
    def save(
        self,
        file: FileData,
        object_lifecycle_preference: dict[str, str] | None = None,
    ) -> str:
        return f'data:{file.content_type};base64,{b64encode(file.data).decode("utf-8")}'


@dataclass
class FalCDNFileRepository(FileRepository):
    def _object_lifecycle_headers(
        self,
        headers: dict[str, str],
        object_lifecycle_preference: dict[str, str] | None,
    ):
        if object_lifecycle_preference:
            headers["X-Fal-Object-Lifecycle-Preference"] = json.dumps(
                object_lifecycle_preference
            )

    @retry(max_retries=3, base_delay=1, backoff_type="exponential", jitter=True)
    def save(
        self,
        file: FileData,
        object_lifecycle_preference: dict[str, str] | None = None,
    ) -> str:
        headers = {
            **self.auth_headers,
            "Accept": "application/json",
            "Content-Type": file.content_type,
            "X-Fal-File-Name": file.file_name,
        }

        self._object_lifecycle_headers(headers, object_lifecycle_preference)

        url = os.getenv("FAL_CDN_HOST", _FAL_CDN) + "/files/upload"
        request = Request(url, headers=headers, method="POST", data=file.data)
        try:
            with urlopen(request) as response:
                result = json.load(response)
        except HTTPError as e:
            raise FileUploadException(
                f"Error initiating upload. Status {e.status}: {e.reason}"
            )

        access_url = result["access_url"]
        return access_url

    @property
    def auth_headers(self) -> dict[str, str]:
        key_creds = key_credentials()
        if not key_creds:
            raise FileUploadException("FAL_KEY must be set")

        key_id, key_secret = key_creds
        return {
            "Authorization": f"Bearer {key_id}:{key_secret}",
            "User-Agent": "fal/0.1.0",
        }


# This is only available for internal users to have long-lived access tokens
@dataclass
class InternalFalFileRepositoryV3(FileRepository):
    """
    InternalFalFileRepositoryV3 is a file repository that uses the FAL CDN V3.
    But generates and uses long-lived access tokens.
    That way it can avoid the need to refresh the token for every upload.
    """

    def _object_lifecycle_headers(
        self,
        headers: dict[str, str],
        object_lifecycle_preference: dict[str, str] | None,
    ):
        if object_lifecycle_preference:
            headers["X-Fal-Object-Lifecycle"] = json.dumps(object_lifecycle_preference)

    @retry(max_retries=3, base_delay=1, backoff_type="exponential", jitter=True)
    def save(
        self,
        file: FileData,
        multipart: bool | None = None,
        multipart_threshold: int | None = None,
        multipart_chunk_size: int | None = None,
        multipart_max_concurrency: int | None = None,
        object_lifecycle_preference: dict[str, str] | None = None,
    ) -> str:
        if multipart is None:
            threshold = (
                multipart_threshold or InternalMultipartUploadV3.MULTIPART_THRESHOLD
            )
            multipart = len(file.data) > threshold

        if multipart:
            return InternalMultipartUploadV3.save(
                file,
                chunk_size=multipart_chunk_size,
                max_concurrency=multipart_max_concurrency,
            )

        headers = {
            **self.auth_headers,
            "Accept": "application/json",
            "Content-Type": file.content_type,
            "X-Fal-File-Name": file.file_name,
        }

        self._object_lifecycle_headers(headers, object_lifecycle_preference)

        url = os.getenv("FAL_CDN_V3_HOST", _FAL_CDN_V3) + "/files/upload"
        request = Request(url, headers=headers, method="POST", data=file.data)
        try:
            with urlopen(request) as response:
                result = json.load(response)
        except HTTPError as e:
            raise FileUploadException(
                f"Error initiating upload. Status {e.status}: {e.reason}"
            )

        access_url = result["access_url"]
        return access_url

    @property
    def auth_headers(self) -> dict[str, str]:
        token = fal_v3_token_manager.get_token()
        return {
            "Authorization": f"{token.token_type} {token.token}",
            "User-Agent": "fal/0.1.0",
        }

    def save_file(
        self,
        file_path: str | Path,
        content_type: str,
        multipart: bool | None = None,
        multipart_threshold: int | None = None,
        multipart_chunk_size: int | None = None,
        multipart_max_concurrency: int | None = None,
        object_lifecycle_preference: dict[str, str] | None = None,
    ) -> tuple[str, FileData | None]:
        if multipart is None:
            threshold = multipart_threshold or MultipartUpload.MULTIPART_THRESHOLD
            multipart = os.path.getsize(file_path) > threshold

        if multipart:
            url = MultipartUpload.save_file(
                file_path,
                chunk_size=multipart_chunk_size,
                content_type=content_type,
                max_concurrency=multipart_max_concurrency,
            )
            data = None
        else:
            with open(file_path, "rb") as f:
                data = FileData(
                    f.read(),
                    content_type=content_type,
                    file_name=os.path.basename(file_path),
                )
            url = self.save(data, object_lifecycle_preference)

        return url, data
