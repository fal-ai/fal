import asyncio
import hashlib
import os
import threading
from io import BufferedReader
from pathlib import Path
from typing import Callable, Dict, List, Optional

import aiohttp
import aiotus
from tqdm import tqdm

import fal.tusd.cache as cache
from fal.exceptions import FileTooLargeError


def _noop_log_before(s: str):
    return lambda retry_state: None


def _noop_log_before_sleep(s: str):
    return lambda retry_state: None


aiotus.retry._make_log_before_function = _noop_log_before
aiotus.retry._make_log_before_sleep_function = _noop_log_before_sleep

from fal.tusd.retry import upload_single  # noqa: E402

setattr(aiotus.retry, "upload", upload_single)

USER_AGENT = "fal-sdk/1.14.0 (python)"
TUS_SERVER_URL = os.environ.get("TUS_SERVER_URL")


class LimitedReader:
    """Simple file reader that reads a limited range."""

    def __init__(
        self,
        file_path: Path,
        start: int,
        size: int,
        parent_hash: str,
        progress_callback: Optional[Callable[[int], None]] = None,
    ):
        self.file_path = file_path
        self.start = start
        self.size = size
        self.position = 0
        self._file: Optional[BufferedReader] = None
        self.parent_hash = parent_hash
        self.progress_callback = progress_callback

    def __enter__(self):
        self._file = self.file_path.open("rb")
        self._file.seek(self.start)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size:
            return b""

        remaining = self.size - self.position
        if size < 0 or size > remaining:
            size = remaining

        if self._file is None:
            raise ValueError("File not opened")
        data = self._file.read(size)
        self.position += len(data)

        if self.progress_callback and len(data) > 0:
            self.progress_callback(len(data))

        return data

    def seek(self, offset: int, whence: int = 0) -> int:
        if self._file is None:
            raise ValueError("File not opened")
        if whence == 0:
            self.position = offset
            self._file.seek(self.start + offset)
        elif whence == 1:
            self.position += offset
            self._file.seek(offset, 1)
        elif whence == 2:
            self.position = self.size + offset
            self._file.seek(self.start + self.size + offset)

        return self.position

    def tell(self) -> int:
        if self._file is None:
            raise ValueError("File not opened")
        return self._file.tell()

    def close(self):
        if self._file:
            self._file.close()


def create_file_readers(
    file_path: Path,
    file_size: int,
    num_parts: int,
    parent_hash: str,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> List[LimitedReader]:
    """Creates LimitedReader instances for file chunks."""
    part_size = file_size // num_parts
    part_readers: List[LimitedReader] = []
    for i in range(num_parts):
        start = i * part_size
        size = file_size - start if i == num_parts - 1 else part_size

        reader = LimitedReader(
            file_path=file_path,
            start=start,
            size=size,
            parent_hash=parent_hash,
            progress_callback=progress_callback,
        )
        part_readers.append(reader)
    return part_readers


def compute_hash(file_path: Path) -> str:
    """Compute sha256 hash using the first 16KB of file content."""
    file_hash = hashlib.sha256()

    SAMPLE_SIZE = 16 * 1024
    with file_path.open("rb") as f:
        content = f.read(SAMPLE_SIZE)
        file_hash.update(content)

    return file_hash.hexdigest()


def prepare_metadata(file_path: Path, file_dir: str) -> Dict:
    """Prepares metadata for file upload."""
    stat = file_path.stat()

    metadata = {
        "mode": str(stat.st_mode).encode(),
        "mtime": str(stat.st_mtime).encode(),
        "size": str(stat.st_size).encode(),
        "filename": (file_dir).encode(),
    }

    return metadata


class TusdUploader:
    """TUS uploader for parallel file uploads."""

    MAX_CHUNK_SIZE = 256 * 1024 * 1024
    MAX_FILE_SIZE = 100 * 1024 * 1024 * 1024

    def __init__(
        self,
        server_url: str,
        headers: Dict[str, str],
        n_parallel: int = 20,
        chunk_size: int = 100 * 1024 * 1024,
        show_progress: bool = True,
    ):
        self.server_url = server_url
        self.headers = headers
        self.n_parallel = n_parallel
        self.chunk_size = chunk_size
        self.show_progress = show_progress

    def _calculate_chunk_size(self, file_size: int) -> int:
        """Calculate optimal chunk size based on file size."""
        if file_size < 100 * 1024 * 1024:
            return self.chunk_size

        calculated_size = file_size // self.n_parallel * 10
        min_chunk = 100 * 1024 * 1024
        calculated_size = max(calculated_size, min_chunk)
        calculated_size = min(calculated_size, self.MAX_CHUNK_SIZE)

        return calculated_size

    def upload(
        self,
        lpath: str,
        file_dir: str,
    ) -> str:
        """Synchronously upload a file using TUS parallel upload."""
        file_path = Path(lpath)
        return asyncio.run(self._upload_async(file_path, file_dir))

    async def _upload_async(
        self,
        file_path: Path,
        file_dir: str,
    ) -> str:
        uploaded_path = await self._do_upload(file_path, file_dir)
        return uploaded_path

    async def _do_upload(
        self,
        file_path: Path,
        file_dir: str,
    ) -> str:
        metadata = prepare_metadata(file_path, file_dir=file_dir)
        file_hash = compute_hash(file_path)
        metadata["hash"] = file_hash.encode()

        file_size = int(metadata.get("size", "0").decode())

        if file_size > self.MAX_FILE_SIZE:
            raise FileTooLargeError(
                f"File size ({file_size / (1024**3):.2f} GB) exceeds "
                f"maximum allowed size ({self.MAX_FILE_SIZE / (1024**3):.0f} GB)"
            )

        chunk_size = self._calculate_chunk_size(file_size)

        pbar = None
        pbar_lock = threading.Lock()

        if self.show_progress:
            pbar = tqdm(
                total=file_size,
                unit="B",
                unit_scale=True,
                desc=f"Uploading {file_path.name}",
            )

        def update_progress(n: int):
            if pbar:
                with pbar_lock:
                    pbar.update(n)

        file = LimitedReader(
            file_path=file_path,
            start=0,
            size=file_size,
            parent_hash=file_hash,
            progress_callback=update_progress if self.show_progress else None,
        )

        try:
            file.__enter__()

            timeout = aiohttp.ClientTimeout(
                total=7200,
                connect=30,
                sock_read=300,
            )
            async with aiohttp.ClientSession(timeout=timeout) as session:
                location = await upload_single(
                    endpoint=self.server_url,
                    file=file,
                    metadata=metadata,
                    client_session=session,
                    headers=self.headers,
                    chunksize=chunk_size,
                )

                if location is None:
                    raise RuntimeError("Upload failed - no location returned")

                if pbar and pbar.n < file_size:
                    with pbar_lock:
                        pbar.update(file_size - pbar.n)

                await cache.remove_from_cache(file_hash=file_hash)

                uploaded_path = f"/data/.uploads/{file_path.name}"

                return uploaded_path

        finally:
            try:
                file.__exit__(None, None, None)
            except Exception:
                pass

            if pbar:
                pbar.close()
