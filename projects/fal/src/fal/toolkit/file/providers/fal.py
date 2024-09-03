from __future__ import annotations

import dataclasses
import json
import math
import os
from base64 import b64encode
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from fal.auth import key_credentials
from fal.toolkit.exceptions import FileUploadException
from fal.toolkit.file.types import FileData, FileRepository
from fal.toolkit.utils.retry import retry

_FAL_CDN = "https://fal.media"


@dataclass
class ObjectLifecyclePreference:
    expriation_duration_seconds: int


GLOBAL_LIFECYCLE_PREFERENCE = ObjectLifecyclePreference(
    expriation_duration_seconds=86400
)


@dataclass
class FalFileRepositoryBase(FileRepository):
    def _save(self, file: FileData, storage_type: str) -> str:
        @retry(max_retries=3, base_delay=1, backoff_type="exponential", jitter=True)
        def __save():
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
            storage_url = f"https://{rest_host}/storage/upload/initiate?storage_type={storage_type}"

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

        return __save()

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
    def save(self, file: FileData) -> str:
        return self._save(file, "gcs")


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

        key_creds = key_credentials()
        if not key_creds:
            raise FileUploadException("FAL_KEY must be set")

        key_id, key_secret = key_creds

        self._auth_headers = {
            "Authorization": f"Key {key_id}:{key_secret}",
        }
        grpc_host = os.environ.get("FAL_HOST", "api.alpha.fal.ai")
        rest_host = grpc_host.replace("api", "rest", 1)
        self._storage_upload_url = f"https://{rest_host}/storage/upload"

    def create(self):
        try:
            req = Request(
                f"{self._storage_upload_url}/initiate-multipart",
                method="POST",
                headers={
                    **self._auth_headers,
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
                self._upload_id = result["upload_id"]
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
                upload_url = (
                    f"{self._file_url}?upload_id={self._upload_id}"
                    f"&part_number={part_number}"
                )
                futures.append(
                    executor.submit(self._upload_part, upload_url, part_number)
                )

            for future in concurrent.futures.as_completed(futures):
                entry = future.result()
                self._parts.append(entry)

    def complete(self):
        url = f"{self._file_url}?upload_id={self._upload_id}"
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


@dataclass
class FalFileRepositoryV2(FalFileRepositoryBase):
    def save(self, file: FileData) -> str:
        return self._save(file, "fal-cdn")

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
            url = self.save(data)

        return url, data


@dataclass
class InMemoryRepository(FileRepository):
    def save(
        self,
        file: FileData,
    ) -> str:
        return f'data:{file.content_type};base64,{b64encode(file.data).decode("utf-8")}'


@dataclass
class FalCDNFileRepository(FileRepository):
    def save(
        self,
        file: FileData,
    ) -> str:
        @retry(max_retries=3, base_delay=1, backoff_type="exponential", jitter=True)
        def _save():
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

        return _save()

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
