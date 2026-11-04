from __future__ import annotations

import fnmatch
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_WORKERS = 48
DEFAULT_CHUNK_BYTES = 64 * 1024 * 1024
_READ_BLOCK = 8 * 1024 * 1024


@dataclass
class PrereadStats:
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
) -> PrereadStats:
    """Read weight shards with many parallel streams to warm the page cache.

    A single sequential stream on the JuiceFS ``/data`` volume is slow, and
    engines read shards serially. Splitting every shard into ``chunk_bytes``
    ranges read by ``workers`` threads saturates the volume, so the engine's
    own load then hits cache. ``path`` is a directory (searched recursively)
    or a single file. Blocking: call it from a thread in async code.
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

    start = time.monotonic()
    total = 0
    if tasks:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for n in pool.map(lambda t: _read_range(*t), tasks):
                total += n
    return PrereadStats(files=n_files, bytes=total, seconds=time.monotonic() - start)
