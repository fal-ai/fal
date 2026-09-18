import concurrent.futures
import hashlib
import io
import logging
import math
import os
import queue
import time
from threading import Lock, Thread
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Tuple, cast

import httpx

from fal.exceptions import FalServerlessException

logger = logging.getLogger(__name__)

MULTIPART_CHUNK_SIZE = 10 * 1024 * 1024  # 10MB per part
MULTIPART_MAX_CONCURRENCY = 10
MULTIPART_THRESHOLD = 10 * 1024 * 1024  # 10MB


class ProgressFileReader:
    """Read-through view of a binary stream that reports the absolute offset.

    Reporting the offset rather than a delta keeps the count honest across a
    retry, which rewinds the body and resends it from the start.
    """

    def __init__(
        self,
        fobj: BinaryIO,
        on_progress: Callable[[int], None],
    ) -> None:
        self._fobj = fobj
        self._on_progress = on_progress

    def read(self, size: int = -1) -> bytes:
        chunk = self._fobj.read(size)
        self._on_progress(self._fobj.tell())
        return chunk

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        return self._fobj.seek(offset, whence)

    def tell(self) -> int:
        return self._fobj.tell()


class _BytesUploadedTracker:
    """Folds concurrent per-part offsets into one cumulative byte count.

    The total is recomputed from the latest offset of every part rather than
    accumulated, so a rewound part cannot inflate it.
    """

    def __init__(self, on_bytes_uploaded: Callable[[int], None]) -> None:
        self._on_bytes_uploaded = on_bytes_uploaded
        self._offsets: Dict[int, int] = {}
        self._lock = Lock()

    def for_part(self, part_number: int) -> Callable[[int], None]:
        def report(offset: int) -> None:
            with self._lock:
                self._offsets[part_number] = offset
                total = sum(self._offsets.values())
            self._on_bytes_uploaded(total)

        return report


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
        self._content_md5: Optional[str] = None

    @property
    def content_md5(self) -> Optional[str]:
        """MD5 of the bytes sent by the last successful `upload_file` call.

        `None` until a call has read the file through, so callers must treat a
        missing digest as "not verified" rather than "verified".
        """
        return self._content_md5

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
        files_factory: Optional[Callable[[], Dict[str, Any]]] = None,
        **kwargs,
    ) -> httpx.Response:
        last_exception = None

        for attempt in range(max_retries):
            # A single-use body must be rebuilt per attempt: a stream left at
            # EOF by a failed attempt would otherwise resend nothing.
            if files_factory is not None:
                kwargs["files"] = files_factory()
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
        self,
        part_number: int,
        data: bytes,
        filename: str = "",
        on_progress: Optional[Callable[[int], None]] = None,
    ) -> Dict[str, object]:
        file_name = filename or "chunk"
        if on_progress is None:
            response = self._request(
                "PUT",
                f"{self.part_url}/{part_number}",
                files={"file_upload": (file_name, data, "application/octet-stream")},
            )
        else:
            # Bound outside the closure: narrowing does not carry into a nested
            # function, which could be called after the name was rebound.
            report: Callable[[int], None] = on_progress

            def build_files() -> Dict[str, Any]:
                reader = ProgressFileReader(io.BytesIO(data), report)
                return {"file_upload": (file_name, reader, "application/octet-stream")}

            response = self._request(
                "PUT",
                f"{self.part_url}/{part_number}",
                files_factory=build_files,
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
        on_bytes_uploaded: Optional[Callable[[int], None]] = None,
        compute_md5: bool = False,
    ) -> str:
        """Upload `file_path` and return the server etag.

        `on_part_complete` fires once per finished part. `on_bytes_uploaded`
        fires as the body of each part is handed to the transport, with the
        running total of payload bytes sent across all parts. `compute_md5`
        populates `content_md5`; it costs a hash over the whole file, so it is
        opt-in for callers that verify the etag.
        """
        size = os.path.getsize(file_path)

        tracker = (
            _BytesUploadedTracker(on_bytes_uploaded)
            if on_bytes_uploaded is not None
            else None
        )

        # Handle empty files specially - upload single empty part
        if size == 0:
            try:
                self.initiate()
            except FileExistsError:
                return ""

            try:
                self._upload_part(1, b"")
                if on_part_complete:
                    on_part_complete(1)
                if on_bytes_uploaded:
                    on_bytes_uploaded(0)
                if compute_md5:
                    self._content_md5 = hashlib.md5(b"").hexdigest()
                return self.complete()
            except FileExistsError:
                return ""
            except Exception:
                self.cancel()
                raise

        num_parts = max(1, math.ceil(size / self.chunk_size))

        try:
            self.initiate()
        except FileExistsError:
            return ""

        chunk_queue: queue.Queue[Optional[Tuple[int, bytes]]] = queue.Queue(
            maxsize=self.max_concurrency * 2
        )
        read_error: List[Exception] = []
        hasher = hashlib.md5() if compute_md5 else None
        bytes_read: List[int] = [0]

        def reader_thread():
            """Reads file chunks and puts them in bounded queue"""
            try:
                with open(file_path, "rb") as f:
                    for part_number in range(1, num_parts + 1):
                        chunk = f.read(self.chunk_size)
                        if chunk:
                            # Hashing here rides along with the read the upload
                            # already needs, instead of a second full-file pass.
                            if hasher is not None:
                                hasher.update(chunk)
                            bytes_read[0] += len(chunk)
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
                        on_progress=(
                            tracker.for_part(part_number) if tracker else None
                        ),
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

            # The part count is fixed from the size sampled before the read, so
            # a file that changes underneath us would otherwise upload a prefix
            # and report success.
            if bytes_read[0] != size:
                raise RuntimeError(
                    f"{file_path} changed while uploading: read "
                    f"{bytes_read[0]} bytes, expected {size}"
                )

            if hasher is not None:
                self._content_md5 = hasher.hexdigest()
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
