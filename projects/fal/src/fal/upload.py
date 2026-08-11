import concurrent.futures
import logging
import math
import os
import time
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Set, cast

import httpx

from fal.exceptions import FalServerlessException

logger = logging.getLogger(__name__)

MULTIPART_CHUNK_SIZE = 10 * 1024 * 1024  # 10MB per part
MULTIPART_MAX_CONCURRENCY = 10
MULTIPART_THRESHOLD = 10 * 1024 * 1024  # 10MB

# These must stay in sync with the server (isolate_controller ... files/utils.py):
# MAX_PART_SIZE and MAX_PART_NUMBER.
MULTIPART_MAX_PART_SIZE = 100 * 1024 * 1024  # 100MB per part
MULTIPART_MAX_PARTS = 10000  # maximum number of parts the server accepts
_MB = 1024 * 1024


def compute_multipart_chunk_size(size: int) -> int:
    """Pick a chunk size so a file of ``size`` bytes fits within
    ``MULTIPART_MAX_PARTS`` parts, each no larger than ``MULTIPART_MAX_PART_SIZE``.

    Small files keep the default 10MB chunk; large files (e.g. multi-hundred-GB
    checkpoints) scale the chunk up so the part count stays under the server cap.
    """
    if size <= 0:
        return MULTIPART_CHUNK_SIZE

    # Smallest chunk that keeps the part count within the server limit.
    min_chunk = math.ceil(size / MULTIPART_MAX_PARTS)
    chunk = max(MULTIPART_CHUNK_SIZE, min_chunk)
    # Round up to a whole MB for stable, predictable part boundaries.
    chunk = math.ceil(chunk / _MB) * _MB

    if chunk > MULTIPART_MAX_PART_SIZE:
        max_size = MULTIPART_MAX_PART_SIZE * MULTIPART_MAX_PARTS
        raise FalServerlessException(
            f"File is too large to upload ({size} bytes): exceeds the maximum "
            f"supported size of {max_size} bytes."
        )
    return chunk


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
        # Populated from the initiate response so already-uploaded parts can be
        # skipped when resuming an interrupted upload.
        self._completed_parts: Set[int] = set()
        # File identity sent to initiate so the server can resume a prior upload.
        self._content_md5: Optional[str] = None
        self._file_size: Optional[int] = None

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

        # The server may return parts that were already uploaded in a previous
        # (interrupted) run so we can resume instead of starting over.
        existing = data.get("parts") or []
        with self._parts_lock:
            self._parts = [
                {"part_number": p["part_number"], "etag": p["etag"]}
                for p in existing
            ]
            self._completed_parts = {int(p["part_number"]) for p in self._parts}
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

    def _upload_part_from_file(
        self, file_path: str, part_number: int
    ) -> Dict[str, object]:
        """Read a single part from ``file_path`` by offset and upload it.

        Each worker opens its own file handle and seeks to the part offset so
        only the parts that still need uploading are read from disk (important
        when resuming a large upload that is already mostly complete).
        """
        offset = (part_number - 1) * self.chunk_size
        with open(file_path, "rb") as f:
            f.seek(offset)
            chunk = f.read(self.chunk_size)
        return self._upload_part(part_number, chunk)

    def upload_file(
        self,
        file_path: str,
        on_part_complete: Optional[Callable[[int], None]] = None,
    ) -> str:
        self._file_size = os.path.getsize(file_path)
        size = self._file_size

        # Handle empty files specially - upload single empty part
        if size == 0:
            try:
                self.initiate()
            except FileExistsError:
                return ""

            try:
                if 1 not in self._completed_parts:
                    self._upload_part(1, b"")
                if on_part_complete:
                    on_part_complete(1)
                return self.complete()
            except FileExistsError:
                return ""

        num_parts = max(1, math.ceil(size / self.chunk_size))

        try:
            self.initiate()
        except FileExistsError:
            return ""

        # Reflect parts the server already has so progress reporting is accurate.
        if on_part_complete:
            for part_number in sorted(self._completed_parts):
                on_part_complete(part_number)

        parts_to_upload = [
            part_number
            for part_number in range(1, num_parts + 1)
            if part_number not in self._completed_parts
        ]

        # NOTE: we deliberately do NOT cancel the upload on failure. The parts
        # uploaded so far are kept server-side, so re-running the upload resumes
        # from where it left off instead of starting over.
        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_concurrency
            ) as executor:
                future_to_part = {
                    executor.submit(
                        self._upload_part_from_file, file_path, part_number
                    ): part_number
                    for part_number in parts_to_upload
                }

                for future in concurrent.futures.as_completed(future_to_part):
                    part_number = future_to_part[future]
                    future.result()
                    if on_part_complete:
                        on_part_complete(part_number)

            return self.complete()
        except FileExistsError:
            return ""


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

    def get_initiate_payload(self) -> Optional[dict]:
        # Sent so the server can derive a deterministic upload id and resume a
        # previously interrupted upload of the same file to the same path.
        payload: Dict[str, Any] = {"chunk_size": self.chunk_size}
        if self._content_md5 is not None:
            payload["content_md5"] = self._content_md5
        if self._file_size is not None:
            payload["size"] = self._file_size
        return payload
