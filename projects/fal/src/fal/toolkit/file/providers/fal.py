from __future__ import annotations

import json
import os
from base64 import b64encode
from dataclasses import dataclass
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from fal.auth import key_credentials
from fal.toolkit.exceptions import FileUploadException
from fal.toolkit.file.types import FileData, FileRepository
from fal.toolkit.mainify import mainify

_FAL_CDN = "https://fal-cdn.batuhan-941.workers.dev"


@mainify
@dataclass
class FalFileRepository(FileRepository):
    def save(self, file: FileData) -> str:
        key_creds = key_credentials()
        if not key_creds:
            raise FileUploadException("FAL_KEY must be set")

        key_id, key_secret = key_creds
        headers = {
            "Authorization": f"Key {key_id}:{key_secret}",
            "Accept": "application/json",
            "Content-Type": f"application/json",
        }

        grpc_host = os.environ.get("FAL_HOST", "api.alpha.fal.ai")
        rest_host = grpc_host.replace("api", "rest", 1)
        storage_url = f"https://{rest_host}/storage/upload/initiate"

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


@mainify
@dataclass
class InMemoryRepository(FileRepository):
    def save(self, file: FileData) -> str:
        return f'data:{file.content_type};base64,{b64encode(file.data).decode("utf-8")}'


@mainify
@dataclass
class FalCDNFileRepository(FileRepository):
    def save(self, file: FileData) -> str:
        headers = {
            **self.auth_headers,
            "Accept": "application/json",
            "Content-Type": file.content_type,
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
