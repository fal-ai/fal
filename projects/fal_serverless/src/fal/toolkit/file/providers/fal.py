from __future__ import annotations

import os
from base64 import b64encode
from dataclasses import dataclass

from fal.toolkit.file.types import FileData, FileRepository
from fal.toolkit.mainify import mainify


@mainify
@dataclass
class FalFileRepository(FileRepository):
    def save(self, file: FileData) -> str:
        from json import load as load_json
        from urllib.error import HTTPError
        from urllib.request import Request, urlopen
        from uuid import uuid4

        key_id = os.environ.get("FAL_KEY_ID")
        key_secret = os.environ.get("FAL_KEY_SECRET")

        boundary = f"----FormBoundary-{uuid4().hex}"
        headers = {
            "Authorization": f"Key {key_id}:{key_secret}",
            "Accept": "application/json",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        }

        grpc_host = os.environ.get("FAL_HOST", "api.alpha.fal.ai")
        rest_host = grpc_host.replace("api", "rest", 1)
        storage_url = f"https://{rest_host}/storage/upload"

        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: file; name="file"; filename="{file.file_name}"\r\n'
            f"Content-Type: {file.content_type}\r\n\r\n".encode()
            + file.data
            + f"\r\n--{boundary}--\r\n".encode()
        )

        try:
            req = Request(storage_url, data=body, headers=headers, method="POST")
            with urlopen(req) as response:
                result = load_json(response)
        except HTTPError as e:
            # TODO sometimes the error body is JSON
            # What's the appropriate way to handle it and forward the payload?
            raise Exception(f"Error uploading file. Status {e.status}: {e.reason}")

        return result["url"]


@mainify
@dataclass
class InMemoryRepository(FileRepository):
    def save(self, file: FileData) -> str:
        return f'data:{file.content_type};base64,{b64encode(file.data).decode("utf-8")}'
