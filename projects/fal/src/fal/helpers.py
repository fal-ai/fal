from __future__ import annotations

import concurrent.futures
import os
from os import PathLike
from pathlib import Path
from typing import Iterator

DEFAULT_WARM_DIR_PARALLELISM = 32
DEFAULT_WARM_FILE_CHUNK_SIZE = 1024 * 1024


def _iter_regular_files(directory: Path) -> Iterator[Path]:
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file(follow_symlinks=False):
                yield Path(entry.path)
            elif entry.is_dir(follow_symlinks=False):
                yield from _iter_regular_files(Path(entry.path))


def warm_file(
    file_path: str | PathLike[str],
    chunk_size: int = DEFAULT_WARM_FILE_CHUNK_SIZE,
) -> None:
    """Pre-read a file into the OS page cache."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be greater than or equal to 1")

    path = Path(file_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Expected a file: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a regular file: {path}")

    with path.open("rb") as file:
        while file.read(chunk_size):
            pass


def warm_dir(
    directory: str | PathLike[str],
    parallelism: int = DEFAULT_WARM_DIR_PARALLELISM,
) -> None:
    """Pre-read all files in a directory into the OS page cache."""

    if parallelism < 1:
        raise ValueError("parallelism must be greater than or equal to 1")

    path = Path(directory).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Expected a directory: {path}")

    if parallelism == 1:
        for file_path in _iter_regular_files(path):
            warm_file(file_path)
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
        pending = set()

        for file_path in _iter_regular_files(path):
            pending.add(executor.submit(warm_file, file_path))

            # Keep a small bounded queue of work instead of materializing the
            # full directory tree before starting any reads.
            if len(pending) >= parallelism * 2:
                done, pending = concurrent.futures.wait(
                    pending,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    future.result()

        for future in concurrent.futures.as_completed(pending):
            future.result()
