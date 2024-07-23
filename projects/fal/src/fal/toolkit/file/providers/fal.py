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
    def save(self, file: FileData) -> str:
        return self._save(file, "gcs")


class MultipartUpload:
    MULTIPART_THRESHOLD = 2**30

    def __init__(self, file_path: str | Path, content_type: str):
        self.file_path = file_path
        self.content_type = content_type

        self._etags: dict = {}

        key_creds = key_credentials()
        if not key_creds:
            raise FileUploadException("FAL_KEY must be set")

        key_id, key_secret = key_creds

        self._headers = {
            "Authorization": f"Key {key_id}:{key_secret}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        grpc_host = os.environ.get("FAL_HOST", "api.alpha.fal.ai")
        rest_host = grpc_host.replace("api", "rest", 1)
        self._storage_upload_url = f"https://{rest_host}/storage/upload"

    def create(self):
        parts_number = math.ceil(
            os.path.getsize(self.file_path) / self.MULTIPART_THRESHOLD
        )
        try:
            req = Request(
                f"{self._storage_upload_url}/create-multipart",
                data=json.dumps(
                    {
                        "file_name": os.path.basename(self.file_path),
                        "content_type": self.content_type,
                        "parts_number": parts_number,
                    }
                ).encode(),
                headers=self._headers,
                method="POST",
            )
            with urlopen(req) as response:
                result = json.load(response)
                self._file_id = result["file_id"]
                self._upload_id = result["upload_id"]
                self._file_url = result["file_url"]
                self._parts = result["parts"]
        except HTTPError as e:
            raise FileUploadException(
                f"Error initiating upload. Status {e.status}: {e.reason}"
            )

    def _upload_part(self, url: str, part_number: int) -> tuple:
        with open(self.file_path, "rb") as f:
            start = (part_number - 1) * self.MULTIPART_THRESHOLD
            f.seek(start)
            data = f.read(self.MULTIPART_THRESHOLD)
            req = Request(
                url,
                method="PUT",
                data=data,
                headers={"Content-Type": self.content_type},
            )

            with urlopen(req) as resp:
                return part_number, resp.headers["ETag"]

    def upload(self) -> None:
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for part in self._parts:
                part_number = part["part_number"]
                upload_url = part["upload_url"]
                futures.append(
                    executor.submit(self._upload_part, upload_url, part_number)
                )

            for future in concurrent.futures.as_completed(futures):
                part_number, etag = future.result()
                self._etags[part_number] = etag

    def complete(self):
        try:
            req = Request(
                f"{self._storage_upload_url}/complete-multipart",
                method="POST",
                headers=self._headers,
                data=json.dumps(
                    {
                        "file_id": self._file_id,
                        "upload_id": self._upload_id,
                        "parts": [
                            {"part_number": part_number, "etag": etag}
                            for part_number, etag in self._etags.items()
                        ],
                    }
                ).encode(),
            )
            with urlopen(req):
                pass
        except HTTPError as e:
            raise FileUploadException(
                f"Error completing upload. Status {e.status}: {e.reason}"
            )

        return self._file_url


@dataclass
class FalFileRepositoryV2(FalFileRepositoryBase):
    def save(self, file: FileData) -> str:
        return self._save(file, "fal-cdn")

    def _save_multipart(self, file_path: str | Path, content_type: str) -> str:
        multipart = MultipartUpload(file_path, content_type)
        multipart.create()
        multipart.upload()
        return multipart.complete()

    def save_file(
        self,
        file_path: str | Path,
        content_type: str,
        multipart: bool | None = None,
    ) -> str:
        if multipart is None:
            multipart = os.path.getsize(file_path) > MultipartUpload.MULTIPART_THRESHOLD

        if multipart:
            return self._save_multipart(file_path, content_type=content_type)

        with open(file_path, "rb") as f:
            data = FileData(
                f.read(),
                content_type=content_type,
                file_name=os.path.basename(file_path),
            )

        return self.save(data)


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
