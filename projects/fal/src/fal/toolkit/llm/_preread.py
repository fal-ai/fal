from __future__ import annotations

import fnmatch
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

DEFAULT_WORKERS = 48
DEFAULT_CHUNK_BYTES = 64 * 1024 * 1024
_READ_BLOCK = 8 * 1024 * 1024


@dataclass
class PrereadStats:
    """Result of ``preread_weights``.

    Attributes:
        files (`int`): Number of files read.
        bytes (`int`): Total bytes read.
        seconds (`float`): Wall-clock duration of the read.
    """

    files: int
    bytes: int
    seconds: float

    @property
    def gb_per_s(self) -> float:
        return self.bytes / 1e9 / self.seconds if self.seconds > 0 else 0.0


def _read_range(path: str, offset: int, length: int) -> int:
    done = 0
    buf = bytearray(min(_READ_BLOCK, length))
    view = memoryview(buf)
    if not hasattr(os, "preadv"):
        # Windows: no positional reads, so seek a private handle instead.
        with open(path, "rb", buffering=0) as f:
            f.seek(offset)
            while done < length:
                n = f.readinto(view[: min(len(buf), length - done)])
                if not n:
                    break
                done += n
        return done
    fd = os.open(path, os.O_RDONLY)
    try:
        while done < length:
            n = os.preadv(fd, [view[: min(len(buf), length - done)]], offset + done)
            if n == 0:
                break
            done += n
    finally:
        os.close(fd)
    return done


def preread_weights(
    path: str | os.PathLike[str],
    *,
    workers: int = DEFAULT_WORKERS,
    patterns: Sequence[str] = ("*.safetensors",),
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    stop: Optional[threading.Event] = None,
) -> PrereadStats:
    """Read weight shards with many parallel streams to warm the page cache.

    A single sequential stream on the JuiceFS ``/data`` volume is slow, and
    engines read shards serially. Splitting every shard into ``chunk_bytes``
    ranges read by ``workers`` threads saturates the volume, so the engine's
    own load then hits cache. ``path`` is a directory (searched recursively)
    or a single file. Blocking: call it from a thread in async code. Setting
    ``stop`` skips the chunks not yet started, so the pool (whose threads the
    interpreter joins at exit) drains quickly.

    Args:
        path: Directory to search recursively, or a single file.
        workers: Number of parallel reader threads.
        patterns: Filename patterns of the files to read.
        chunk_bytes: Size in bytes of each range read by one worker.
        stop: Event that, once set, skips the chunks not yet started.

    Returns:
        Counts and timing of the read.
    """
    root = Path(path)
    if root.is_file():
        files: Iterable[Path] = [root]
    else:
        files = sorted(
            p
            for p in root.rglob("*")
            if p.is_file() and any(fnmatch.fnmatch(p.name, pat) for pat in patterns)
        )
    tasks: list[tuple[str, int, int]] = []
    n_files = 0
    for f in files:
        size = f.stat().st_size
        n_files += 1
        for off in range(0, size, chunk_bytes):
            tasks.append((str(f), off, min(chunk_bytes, size - off)))

    def read(task: tuple[str, int, int]) -> int:
        if stop is not None and stop.is_set():
            return 0
        return _read_range(*task)

    start = time.monotonic()
    total = 0
    if tasks:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for n in pool.map(read, tasks):
                total += n
    return PrereadStats(files=n_files, bytes=total, seconds=time.monotonic() - start)
