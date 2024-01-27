from __future__ import annotations

import json
import os
from base64 import b64encode
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from fal.toolkit.exceptions import FileUploadException
from fal.toolkit.file.types import FileData, FileRepository
from fal.toolkit.mainify import mainify

# Don't allow more than 24 uploads to be in progress at once, if we are stuck
# then execute the next upload synchronously.
MAX_BACKGROUND_UPLOADS = 24


@mainify
@dataclass
class FalFileRepository(FileRepository):
    thread_pool: ThreadPoolExecutor = field(default_factory=ThreadPoolExecutor)
    uploads: set[Future] = field(default_factory=set)

    def __post_init__(self):
        self.allow_background_uploads = os.environ.get(
            "FAL_ALLOW_BACKGROUND_UPLOADS", False
        )

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

        self.gc_futures()
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
            if (
                not self.allow_background_uploads
                or len(self.uploads) >= MAX_BACKGROUND_UPLOADS
            ):
                self._upload_file(upload_url, file)
            else:
                future = self.thread_pool.submit(self._upload_file, upload_url, file)
                self.uploads.add(future)

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

    def gc_futures(self):
        import traceback

        for future in self.uploads.copy():
            if not future.done():
                continue

            if future in self.uploads:
                self.uploads.remove(future)

            exception = future.exception()
            if exception is not None:
                print("[Warning] Failed to upload file")
                traceback.print_exception(
                    type(exception), exception, exception.__traceback__
                )


@mainify
@dataclass
class InMemoryRepository(FileRepository):
    def save(self, file: FileData) -> str:
        return f'data:{file.content_type};base64,{b64encode(file.data).decode("utf-8")}'
