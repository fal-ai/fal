import concurrent.futures
import logging
import math
import os
import queue
import time
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import httpx

from fal.exceptions import FalServerlessException

logger = logging.getLogger(__name__)

MULTIPART_CHUNK_SIZE = 10 * 1024 * 1024  # 10MB per part
MULTIPART_MAX_CONCURRENCY = 10
MULTIPART_THRESHOLD = 10 * 1024 * 1024  # 10MB


class BaseMultipartUpload:
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
        self._parts_lock = Lock()

    @property
    def upload_id(self) -> str:
        if not self._upload_id:
            raise FalServerlessException("Upload not initiated")
        return self._upload_id

    @property
    def initiate_url(self) -> str:
        raise NotImplementedError("Subclasses must implement initiate_url")

    @property
    def part_url(self) -> str:
        raise NotImplementedError("Subclasses must implement part_url")

    @property
    def complete_url(self) -> str:
        raise NotImplementedError("Subclasses must implement complete_url")

    @property
    def cancel_url(self) -> Optional[str]:
        return None

    def get_initiate_payload(self) -> Optional[dict]:
        return None

    def get_complete_payload(self, parts: List[Dict[str, object]]) -> dict:
        return {"parts": parts}

    def _request(
        self,
        method: str,
        path: str,
        max_retries: int = 3,
        **kwargs,
    ) -> httpx.Response:
        last_exception = None

        for attempt in range(max_retries):
            try:
                response = self.client.request(method, path, **kwargs)

                if response.status_code in (200, 201, 204):
                    return response
                elif response.status_code == 409:
                    raise FileExistsError("File already exists on server")
                elif response.status_code == 404:
                    raise FalServerlessException("Not Found")
                elif response.status_code == 429:
                    # Rate limited, retry after if available
                    retry_after = int(response.headers.get("Retry-After", 2))
                    if attempt < max_retries - 1:
                        logger.warning(f"Rate limited, retrying after {retry_after}s")
                        time.sleep(retry_after)
                        continue
                    raise FalServerlessException("Rate limit exceeded")
                elif response.status_code >= 500:
                    # Server error, retry with exponential backoff
                    if attempt < max_retries - 1:
                        backoff = 2**attempt
                        logger.warning(
                            f"Server error {response.status_code}, "
                            f"retrying in {backoff}s ({attempt + 1} of {max_retries})"
                        )
                        time.sleep(backoff)
                        continue
                    # Last attempt failed
                    try:
                        detail = response.json()["detail"]
                    except Exception:
                        detail = response.text
                    raise FalServerlessException(detail)
                else:
                    # Client error (4xx) - don't retry
                    try:
                        detail = response.json()["detail"]
                    except Exception:
                        detail = response.text
                    raise FalServerlessException(detail)

            except (httpx.TimeoutException, httpx.NetworkError) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    backoff = 2**attempt
                    logger.warning(
                        f"Network error: {e}, "
                        f"retrying in {backoff}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(backoff)
                    continue
                raise FalServerlessException(
                    f"Network error after {max_retries} retries: {e}"
                )

        # Should not reach here, but handle it anyway
        raise FalServerlessException(
            f"Request failed after {max_retries} retries: {last_exception}"
        )

    def initiate(self) -> str:
        payload = self.get_initiate_payload()
        kwargs: Dict[str, Any] = {"json": payload} if payload else {}
        response = self._request("POST", self.initiate_url, **kwargs)
        data = response.json()
        self._upload_id = data["upload_id"]
        return self.upload_id

    def _upload_part(
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
        with self._parts_lock:
            self._parts.append(part_info)
        return part_info

    def complete(self) -> str:
        with self._parts_lock:
            sorted_parts = sorted(
                self._parts, key=lambda p: cast(int, p["part_number"])
            )
        payload = self.get_complete_payload(sorted_parts)
        response = self._request("POST", self.complete_url, json=payload)
        data = response.json()
        return data.get("etag", "")

    def cancel(self) -> None:
        if self._upload_id and self.cancel_url:
            try:
                self._request("POST", self.cancel_url)
            except Exception as e:
                logger.warning(f"Failed to cancel upload {self._upload_id}: {e}")

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

        chunk_queue: queue.Queue[Optional[Tuple[int, bytes]]] = queue.Queue(
            maxsize=self.max_concurrency * 2
        )
        read_error: List[Exception] = []

        def reader_thread():
            """Reads file chunks and puts them in bounded queue"""
            try:
                with open(file_path, "rb") as f:
                    for part_number in range(1, num_parts + 1):
                        chunk = f.read(self.chunk_size)
                        if chunk:
                            chunk_queue.put((part_number, chunk))
                # Sentinel to signal completion
                chunk_queue.put(None)
            except Exception as e:
                read_error.append(e)
                chunk_queue.put(None)

        reader = Thread(target=reader_thread, daemon=True)
        reader.start()

        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_concurrency
            ) as executor:
                futures = []

                while True:
                    item = chunk_queue.get()
                    if item is None:
                        break

                    part_number, chunk = item
                    future = executor.submit(
                        self._upload_part,
                        part_number,
                        chunk,
                    )
                    futures.append((part_number, future))

                # Wait for all uploads to complete
                for part_number, future in futures:
                    future.result()
                    if on_part_complete:
                        on_part_complete(part_number)

            reader.join()

            if read_error:
                raise read_error[0]

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
