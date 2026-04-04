from __future__ import annotations

import os
import subprocess
from os import PathLike
from pathlib import Path

DEFAULT_WARM_DIR_PARALLELISM = 32


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

    # Skip spawning subprocesses when there are no files to warm.
    if not any(child.is_file() for child in path.rglob("*")):
        return

    subprocess.check_call(
        [
            "bash",
            "-lc",
            'find "$1" -type f -print0 | '
            'xargs -0 -P "$2" -I {} sh -c \'cat "$1" > /dev/null\' sh "{}"',
            "bash",
            os.fspath(path),
            str(parallelism),
        ]
    )
