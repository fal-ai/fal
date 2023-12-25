from __future__ import annotations

import json
import os
from base64 import b64encode
from dataclasses import dataclass
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from fal.toolkit.exceptions import FileUploadException
from fal.toolkit.file.types import FileData, FileRepository
from fal.toolkit.mainify import mainify


@mainify
@dataclass
class FalFileRepository(FileRepository):
    def save(self, file: FileData) -> str:
        key_id = os.environ.get("FAL_KEY_ID")
        key_secret = os.environ.get("FAL_KEY_SECRET")

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
            headers={"Content-Type": file.content_type},  # type: ignore
        )

        with urlopen(req):
            return


@mainify
@dataclass
class InMemoryRepository(FileRepository):
    def save(self, file: FileData) -> str:
        return (
            f"data:{file.content_type};base64,"
            f'{b64encode(file.as_bytes()).decode("utf-8")}'  # type: ignore
        )
