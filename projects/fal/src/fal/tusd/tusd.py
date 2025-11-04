import asyncio
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
import aiotus

import fal.tusd.cache as cache

# Suppress retry logging by replacing the logging functions with no-ops
# This must be done BEFORE importing upload_single, since upload_single uses
# _make_retrying which calls these functions to configure tenacity logging
def _noop_log_before(s: str):
    return lambda retry_state: None

def _noop_log_before_sleep(s: str):
    return lambda retry_state: None

aiotus.retry._make_log_before_function = _noop_log_before
aiotus.retry._make_log_before_sleep_function = _noop_log_before_sleep

# Now import and monkey-patch aiotus.retry.upload with our cached version
from fal.tusd.retry import upload_single

# Monkey-patch aiotus.retry.upload so other functions use our cached version
setattr(aiotus.retry, "upload", upload_single)

USER_AGENT = "fal-sdk/1.14.0 (python)"
TUS_SERVER_URL = os.environ.get("TUS_SERVER_URL")


class LimitedReader:
    """
    Simple file reader that reads a limited range.

    aiotus already uses asyncio.to_thread() internally, so we just need
    regular synchronous file operations here.
    """

    def __init__(self, file_path: Path, start: int, size: int, parent_hash: str):
        self.file_path = file_path
        self.start = start
        self.size = size
        self.position = 0
        self._file = None
        self.parent_hash = parent_hash

    def __enter__(self):
        """Open file handle."""
        self._file = self.file_path.open("rb")
        self._file.seek(self.start)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close file handle."""
        if self._file:
            self._file.close()

    def read(self, size: int = -1) -> bytes:
        """Read up to size bytes from the file."""
        if self.position >= self.size:
            return b""

        # Calculate how much to read
        remaining = self.size - self.position
        if size < 0 or size > remaining:
            size = remaining

        data = self._file.read(size)
        self.position += len(data)
        return data

    def seek(self, offset: int, whence: int = 0) -> int:
        """Seek to position in the file."""
        if whence == 0:  # SEEK_SET - absolute position
            self.position = offset
            self._file.seek(self.start + offset)
        elif whence == 1:  # SEEK_CUR - relative to current position
            self.position += offset
            self._file.seek(offset, 1)
        elif whence == 2:  # SEEK_END - relative to end
            self.position = self.size + offset
            self._file.seek(self.start + self.size + offset)

        return self.position

    def tell(self) -> int:
        return self._file.tell()

    def close(self):
        """Close the file."""
        if self._file:
            self._file.close()


def create_file_readers(
    file_path: Path, file_size: int, num_parts: int, parent_hash: str
) -> List[LimitedReader]:
    """
    Creates a list of LimitedReader instances, each responsible for reading a distinct chunk of a file.

    Divides the file located at 'file_path' into 'num_parts' sequential chunks and returns a LimitedReader
    for each chunk. The parent file's hash ('parent_hash') is attached to each reader for resumption or verification.

    :param file_path: Path to the input file.
    :param file_size: Total size of the file in bytes.
    :param num_parts: Number of parts (chunks) to split the file into.
    :param parent_hash: Precomputed hash of the parent file for resumption/cache purposes.
    :return: List of LimitedReader objects, covering the entire file in non-overlapping segments.
    """

    part_size = file_size // num_parts
    part_readers: List[LimitedReader] = []
    # Create readers for each part
    for i in range(num_parts):
        start = i * part_size
        size = file_size - start if i == num_parts - 1 else part_size

        reader = LimitedReader(
            file_path=file_path, start=start, size=size, parent_hash=parent_hash
        )
        part_readers.append(reader)
    return part_readers


def compute_hash(file_path: Path) -> str:
    """
    Compute sha256 hash using the filename and the first 16KB of file content.

    :param Path file_path: Path of the file.
    :return str: The hash value as a string of hexadecimal digits.
    """
    file_hash = hashlib.sha256()

    # Add filename
    file_hash.update(file_path.name.encode("utf-8"))

    # Add first 16KB of file content
    SAMPLE_SIZE = 16 * 1024  # 16KB
    with file_path.open("rb") as f:
        content = f.read(SAMPLE_SIZE)
        file_hash.update(content)

    return file_hash.hexdigest()


def prepare_metadata(file_path: Path) -> Dict:
    """
    Takes a file path and its hash. Then prepares metadata for file upload.

    :param Path file_path: Path of the file.
    :param str file_hash: The hash value of the file.
    :return Dict: A mapping that contains the metadata (values as bytes).
    """
    stat = file_path.stat()

    metadata = {
        "mode": str(stat.st_mode).encode(),
        "mtime": str(stat.st_mtime).encode(),
        "size": str(stat.st_size).encode(),
        "filename": str(file_path.name).encode(),
    }

    return metadata


class TusdUploader:
    """
    TUS uploader class for parallel file uploads.

    Wraps the async upload_multiple functionality in a synchronous interface
    and handles error cases with proper retry logic.
    """

    def __init__(
        self,
        server_url: str,
        headers: Dict[str, str],
        n_parallel: int = 20,
        chunk_size: int = 20 * 1024 * 1024,
    ):
        """
        Initialize TusdUploader with server URL and headers.

        :param server_url: TUS server URL endpoint
        :param headers: HTTP headers to use for requests
        :param n_parallel: Number of parallel upload parts
        :param chunk_size: Size of each chunk in bytes
        """
        self.server_url = server_url
        self.headers = headers
        self.n_parallel = n_parallel
        self.chunk_size = chunk_size

    def upload(
        self,
        lpath: str,
    ) -> str:
        """
        Synchronously upload a file using TUS parallel upload.

        Note: Only the filename (from lpath) is sent to TUS, not the full rpath.
        The file will be moved to rpath after upload completes.

        :param lpath: Local file path
        :return: The uploaded file path (before move to rpath)
        """
        file_path = Path(lpath)
        return asyncio.run(self._upload_async(file_path))

    async def _upload_async(
        self,
        file_path: Path,
    ) -> str:
        """
        Internal async upload method that performs the actual upload.

        Only the filename (file_path.name) is sent to TUS in metadata.
        Exceptions are not handled here - they propagate to the caller.
        """
        uploaded_path = await self._do_upload(file_path)
        return uploaded_path

    async def _do_upload(
        self,
        file_path: Path,
    ) -> str:
        """
        Perform the actual upload operation.

        :param file_path: Local file path to upload
        :return: The uploaded file path (with original filename)
        """
        # Prepare the file metadata - use original filename (not rpath)
        metadata = prepare_metadata(file_path)
        file_hash = compute_hash(file_path)
        # Add hash to metadata (required by server)
        metadata["hash"] = file_hash.encode()
        # Filename is already set by prepare_metadata to file_path.name

        file_size = int(metadata.get("size").decode())
        part_readers = create_file_readers(
            file_path=file_path,
            file_size=file_size,
            num_parts=self.n_parallel,
            parent_hash=file_hash,
        )

        try:
            # Open all file handles
            part_files = [reader.__enter__() for reader in part_readers]

            timeout = aiohttp.ClientTimeout(
                total=7200,  # 1 hour total timeout
                connect=30,  # 30s to establish connection
                sock_read=300,  # 5 min to read response after request sent
            )
            async with aiohttp.ClientSession(timeout=timeout) as session:
                location = await aiotus.upload_multiple(
                    endpoint=self.server_url,
                    files=part_files,
                    metadata=metadata,
                    client_session=session,
                    headers=self.headers,
                    chunksize=self.chunk_size,
                    parallel_uploads=self.n_parallel,
                )

                if location is None:
                    raise RuntimeError("Upload failed - no location returned")

                await cache.remove_from_cache(file_hash=file_hash)

                # The uploaded file is at /data/.uploads/{filename}
                uploaded_path = f"/data/.uploads/{file_path.name}"

                return uploaded_path

        finally:
            # Clean up file handles
            for reader in part_readers:
                try:
                    reader.__exit__(None, None, None)
                except Exception:
                    pass
