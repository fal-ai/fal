from __future__ import annotations

import dataclasses
import json
import math
import os
import threading
from base64 import b64encode
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
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


@dataclass
class ObjectLifecyclePreference:
    expiration_duration_seconds: int


GLOBAL_LIFECYCLE_PREFERENCE = ObjectLifecyclePreference(
    expiration_duration_seconds=86400
)


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
            self._upload_file(upload_url, file)

            return result["file_url"]
        except HTTPError as e:
            raise FileUploadException(
                f"Error initiating upload. Status {e.status}: {e.reason}"
            )

    def _upload_file(self, upload_url: str, file: FileData):
        req = Request(
            upload_url,
            method="PUT",
            data=file.data,
            headers={"Content-Type": file.content_type},
        )

        with urlopen(req):
            return


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
        file_path: str | Path,
        chunk_size: int | None = None,
        content_type: str | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        self.file_path = file_path
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
                        "file_name": os.path.basename(self.file_path),
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

    def _upload_part(self, url: str, part_number: int) -> dict:
        with open(self.file_path, "rb") as f:
            start = (part_number - 1) * self.chunk_size
            f.seek(start)
            data = f.read(self.chunk_size)
            req = Request(
                url,
                method="PUT",
                headers={"Content-Type": self.content_type},
                data=data,
            )

            try:
                with urlopen(req) as resp:
                    return {
                        "part_number": part_number,
                        "etag": resp.headers["ETag"],
                    }
            except HTTPError as exc:
                raise FileUploadException(
                    f"Error uploading part {part_number} to {url}. "
                    f"Status {exc.status}: {exc.reason}"
                )

    def upload(self) -> None:
        import concurrent.futures

        parts = math.ceil(os.path.getsize(self.file_path) / self.chunk_size)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_concurrency
        ) as executor:
            futures = []
            for part_number in range(1, parts + 1):
                upload_url = f"{self._upload_url}&part_number={part_number}"
                futures.append(
                    executor.submit(self._upload_part, upload_url, part_number)
                )

            for future in concurrent.futures.as_completed(futures):
                entry = future.result()
                self._parts.append(entry)

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


class MultipartUploadV3:
    MULTIPART_THRESHOLD = 100 * 1024 * 1024
    MULTIPART_CHUNK_SIZE = 10 * 1024 * 1024
    MULTIPART_MAX_CONCURRENCY = 10

    def __init__(
        self,
        file_path: str | Path,
        chunk_size: int | None = None,
        content_type: str | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        self.file_path = file_path
        self.chunk_size = chunk_size or self.MULTIPART_CHUNK_SIZE
        self.content_type = content_type or "application/octet-stream"
        self.max_concurrency = max_concurrency or self.MULTIPART_MAX_CONCURRENCY
        self.access_url = None
        self.upload_id = None

        self._parts: list[dict] = []

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
                    "X-Fal-File-Name": os.path.basename(self.file_path),
                },
            )
            with urlopen(req) as response:
                result = json.load(response)
                self.access_url = result["access_url"]
                self.upload_id = result["uploadId"]
        except HTTPError as exc:
            raise FileUploadException(
                f"Error initiating upload. Status {exc.status}: {exc.reason}"
            )

    @retry(max_retries=5, base_delay=1, backoff_type="exponential", jitter=True)
    def _upload_part(self, url: str, part_number: int) -> dict:
        with open(self.file_path, "rb") as f:
            start = (part_number - 1) * self.chunk_size
            f.seek(start)
            data = f.read(self.chunk_size)
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
                    return {
                        "partNumber": part_number,
                        "etag": resp.headers["ETag"],
                    }
            except HTTPError as exc:
                raise FileUploadException(
                    f"Error uploading part {part_number} to {url}. "
                    f"Status {exc.status}: {exc.reason}"
                )

    def upload(self) -> None:
        import concurrent.futures

        parts = math.ceil(os.path.getsize(self.file_path) / self.chunk_size)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_concurrency
        ) as executor:
            futures = []
            for part_number in range(1, parts + 1):
                upload_url = (
                    f"{self.access_url}/multipart/{self.upload_id}/{part_number}"
                )
                futures.append(
                    executor.submit(self._upload_part, upload_url, part_number)
                )

            for future in concurrent.futures.as_completed(futures):
                entry = future.result()
                self._parts.append(entry)

    def complete(self):
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


@dataclass
class FalFileRepositoryV2(FalFileRepositoryBase):
    @retry(max_retries=3, base_delay=1, backoff_type="exponential", jitter=True)
    def save(
        self, file: FileData, object_lifecycle_preference: dict[str, str] | None = None
    ) -> str:
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

    def _save_multipart(
        self,
        file_path: str | Path,
        chunk_size: int | None = None,
        content_type: str | None = None,
        max_concurrency: int | None = None,
    ) -> str:
        multipart = MultipartUpload(
            file_path,
            chunk_size=chunk_size,
            content_type=content_type,
            max_concurrency=max_concurrency,
        )
        multipart.create()
        multipart.upload()
        return multipart.complete()

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
            url = self._save_multipart(
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
            "X-Fal-Object-Lifecycle-Preference": json.dumps(
                dataclasses.asdict(GLOBAL_LIFECYCLE_PREFERENCE)
            ),
        }
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

    @retry(max_retries=3, base_delay=1, backoff_type="exponential", jitter=True)
    def save(
        self, file: FileData, object_lifecycle_preference: dict[str, str] | None
    ) -> str:
        lifecycle = dataclasses.asdict(GLOBAL_LIFECYCLE_PREFERENCE)
        if object_lifecycle_preference is not None:
            lifecycle = {
                key: object_lifecycle_preference[key]
                if key in object_lifecycle_preference
                else value
                for key, value in lifecycle.items()
            }

        headers = {
            **self.auth_headers,
            "Accept": "application/json",
            "Content-Type": file.content_type,
            "X-Fal-File-Name": file.file_name,
            "X-Fal-Object-Lifecycle-Preference": json.dumps(lifecycle),
        }
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

    def _save_multipart(
        self,
        file_path: str | Path,
        chunk_size: int | None = None,
        content_type: str | None = None,
        max_concurrency: int | None = None,
    ) -> str:
        multipart = MultipartUploadV3(
            file_path,
            chunk_size=chunk_size,
            content_type=content_type,
            max_concurrency=max_concurrency,
        )
        multipart.create()
        multipart.upload()
        return multipart.complete()

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
            url = self._save_multipart(
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
