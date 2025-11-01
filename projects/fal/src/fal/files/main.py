import argparse
import asyncio
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import aiohttp
import aiotus
from cache import remove_from_cache
from retry import upload_single

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


def compute_file_hash(file_path: Path) -> str:
    """
    Computes sha256 hash using metadata and minimal content sampling. Uses ctime, mtime, filename, file size.

    :param Path file_path: Path of the file.
    :return str: The hash value as a string of hexadecimal digits.
    """

    hasher = hashlib.sha256()
    stat = file_path.stat()
    file_size = stat.st_size

    # Hash metadata: creation time, modification time, filename, size
    metadata_string = f"{stat.st_ctime}|{stat.st_mtime}|{file_path.name}|{file_size}"
    hasher.update(metadata_string.encode("utf-8"))

    hash_result = hasher.hexdigest()

    return hash_result


def prepare_metadata(file_path: Path, file_hash: str) -> Dict:
    """
    Takes a file path and its hash. Then prepares metadata for file upload.

    :param Path file_path: Path of the file.
    :param str file_hash: The hash value of the file.
    :return Dict: A mapping that contains the metadata (values as bytes).
    """
    stat = file_path.stat()

    metadata = {
        "hash": str(file_hash).encode(),
        "mode": str(stat.st_mode).encode(),
        "mtime": str(stat.st_mtime).encode(),
        "size": str(stat.st_size).encode(),
        "filename": str(file_path.name).encode(),
    }

    return metadata


async def upload(
    file_path: Path,
    n_parallel: int = 20,
    server_url: str = TUS_SERVER_URL,
    chunk_size: int = 20 * 1024 * 1024,
) -> str:
    """
    Parallel upload using aiotus. aiotus handles threading internally via asyncio.to_thread(),
    so we just provide simple synchronous file readers.
    """

    # Prepare the file metadata
    file_hash = compute_file_hash(file_path)
    metadata = prepare_metadata(file_path, file_hash)
    print(metadata)

    headers = {
        "User-Agent": USER_AGENT,
    }

    part_readers = create_file_readers(
        file_path=file_path,
        file_size=int(metadata.get("size").decode()),
        num_parts=n_parallel,
        parent_hash=file_hash,
    )

    print("Starting upload...")
    start_time = time.time()

    try:
        # Open all file handles
        part_files = [reader.__enter__() for reader in part_readers]

        async with aiohttp.ClientSession() as session:
            location = await aiotus.upload_multiple(
                endpoint=server_url,
                files=part_files,
                metadata=metadata,
                client_session=session,
                headers=headers,
                chunksize=chunk_size,
                parallel_uploads=n_parallel,
            )

            if location is None:
                raise RuntimeError("Upload failed - no location returned")

            elapsed = time.time() - start_time
            speed = int(metadata.get("size").decode()) / elapsed / (1024**2)

            await remove_from_cache(file_hash=file_hash)

            print("Upload completed")
            print(f"Location: {location}")
            print(f"Time: {elapsed:.2f}s")
            print(f"Speed: {speed:.2f} MB/s")

            return str(location)

    finally:
        # Clean up file handles
        for reader in part_readers:
            try:
                reader.__exit__(None, None, None)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command")

    # Upload command
    upload_parser = subparsers.add_parser("upload")
    upload_parser.add_argument("file", type=Path)
    upload_parser.add_argument(
        "--server", default=os.getenv("TUS_SERVER_URL", TUS_SERVER_URL)
    )
    upload_parser.add_argument("--chunk-size", type=int, default=50)

    # Parallel upload command
    parallel_parser = subparsers.add_parser("upload-parallel")
    parallel_parser.add_argument("file", type=Path)
    parallel_parser.add_argument("--parallel", type=int, default=10)
    parallel_parser.add_argument(
        "--server", default=os.getenv("TUS_SERVER_URL", TUS_SERVER_URL)
    )
    parallel_parser.add_argument("--chunk-size", type=int, default=50)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if not args.file.exists():
            print(f"File not found: {args.file}")
            return 1

        chunk_size = args.chunk_size * 1024 * 1024
        location = asyncio.run(
            upload(
                file_path=args.file,
                n_parallel=args.parallel,
                server_url=args.server,
                chunk_size=chunk_size,
            )
        )
        print(f"Upload URL: {location}")
        return 0

    except Exception as e:
        print(f"\n Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
