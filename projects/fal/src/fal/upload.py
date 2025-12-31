import concurrent.futures
import math
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, cast

import httpx

from fal.exceptions import FalServerlessException

MULTIPART_CHUNK_SIZE = 10 * 1024 * 1024  # 10MB per part
MULTIPART_MAX_CONCURRENCY = 10
MULTIPART_THRESHOLD = 10 * 1024 * 1024  # 10MB


class BaseMultipartUpload(ABC):
    def __init__(
        self,
        client: httpx.Client,
        chunk_size: int = MULTIPART_CHUNK_SIZE,
        max_concurrency: int = MULTIPART_MAX_CONCURRENCY,
    ):
        self.client = client
        self.chunk_size = chunk_size
        self.max_concurrency = max_concurrency
        self._upload_id: Optional[str] = None
        self._parts: List[Dict[str, object]] = []

    @property
    def upload_id(self) -> str:
        if not self._upload_id:
            raise FalServerlessException("Upload not initiated")
        return self._upload_id

    @property
    @abstractmethod
    def initiate_url(self) -> str:
        pass

    @property
    @abstractmethod
    def part_url(self) -> str:
        pass

    @property
    @abstractmethod
    def complete_url(self) -> str:
        pass

    @property
    def cancel_url(self) -> Optional[str]:
        return None

    def get_initiate_payload(self) -> Optional[dict]:
        return None

    def get_complete_payload(self, parts: List[Dict[str, object]]) -> dict:
        return {"parts": parts}

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        response = self.client.request(method, path, **kwargs)
        if response.status_code == 409:
            raise FileExistsError("File already exists on server")
        elif response.status_code == 404:
            raise FalServerlessException("Not Found")
        elif response.status_code != 200:
            try:
                detail = response.json()["detail"]
            except Exception:
                detail = response.text
            raise FalServerlessException(detail)
        return response

    def initiate(self) -> str:
        payload = self.get_initiate_payload()
        kwargs: Dict[str, Any] = {"json": payload} if payload else {}
        response = self._request("POST", self.initiate_url, **kwargs)
        data = response.json()
        self._upload_id = data["upload_id"]
        return self.upload_id

    def upload_part(
        self, part_number: int, data: bytes, filename: str = ""
    ) -> Dict[str, object]:
        file_name = filename or "chunk"
        response = self._request(
            "PUT",
            f"{self.part_url}/{part_number}",
            files={"file_upload": (file_name, data, "application/octet-stream")},
        )
        result = response.json()
        part_info = {
            "part_number": result["part_number"],
            "etag": result["etag"],
        }
        self._parts.append(part_info)
        return part_info

    def complete(self) -> str:
        sorted_parts = sorted(self._parts, key=lambda p: cast(int, p["part_number"]))
        payload = self.get_complete_payload(sorted_parts)
        response = self._request("POST", self.complete_url, json=payload)
        data = response.json()
        return data.get("etag", "")

    def cancel(self) -> None:
        if self._upload_id and self.cancel_url:
            try:
                self._request("POST", self.cancel_url)
            except Exception:
                pass

    def upload_file(
        self,
        file_path: str,
        on_part_complete: Optional[Callable[[int], None]] = None,
    ) -> str:
        size = os.path.getsize(file_path)
        num_parts = max(1, math.ceil(size / self.chunk_size))

        try:
            self.initiate()
        except FileExistsError:
            return ""

        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_concurrency
            ) as executor:
                futures = []

                for part_number in range(1, num_parts + 1):

                    def _upload_part(pn: int) -> Dict[str, object]:
                        with open(file_path, "rb") as f:
                            start = (pn - 1) * self.chunk_size
                            f.seek(start)
                            chunk = f.read(self.chunk_size)
                            result = self.upload_part(pn, chunk)
                            if on_part_complete:
                                on_part_complete(pn)
                            return result

                    futures.append(executor.submit(_upload_part, part_number))

                for future in concurrent.futures.as_completed(futures):
                    future.result()

            return self.complete()
        except FileExistsError:
            return ""
        except Exception:
            self.cancel()
            raise


class AppFileMultipartUpload(BaseMultipartUpload):
    def __init__(
        self,
        client: httpx.Client,
        file_hash: str,
        metadata: dict,
        chunk_size: int = MULTIPART_CHUNK_SIZE,
        max_concurrency: int = MULTIPART_MAX_CONCURRENCY,
    ):
        super().__init__(client, chunk_size, max_concurrency)
        self.file_hash = file_hash
        self.metadata = metadata

    @property
    def initiate_url(self) -> str:
        return f"/files/app/multipart/{self.file_hash}/initiate"

    @property
    def part_url(self) -> str:
        return f"/files/app/multipart/{self.file_hash}/{self.upload_id}"

    @property
    def complete_url(self) -> str:
        return f"/files/app/multipart/{self.file_hash}/{self.upload_id}/complete"

    @property
    def cancel_url(self) -> Optional[str]:
        return f"/files/app/multipart/{self.file_hash}/{self.upload_id}/cancel"

    def get_initiate_payload(self) -> Optional[dict]:
        return self.metadata

    def get_complete_payload(self, parts: List[Dict[str, object]]) -> dict:
        return {
            "parts": parts,
            "metadata": self.metadata,
        }


class DataFileMultipartUpload(BaseMultipartUpload):
    def __init__(
        self,
        client: httpx.Client,
        target_path: str,
        chunk_size: int = MULTIPART_CHUNK_SIZE,
        max_concurrency: int = MULTIPART_MAX_CONCURRENCY,
    ):
        super().__init__(client, chunk_size, max_concurrency)
        self.target_path = target_path

    @property
    def initiate_url(self) -> str:
        return f"/files/file/multipart/{self.target_path}/initiate"

    @property
    def part_url(self) -> str:
        return f"/files/file/multipart/{self.target_path}/{self.upload_id}"

    @property
    def complete_url(self) -> str:
        return f"/files/file/multipart/{self.target_path}/{self.upload_id}/complete"

    @property
    def cancel_url(self) -> Optional[str]:
        return f"/files/file/multipart/{self.target_path}/{self.upload_id}/cancel"
