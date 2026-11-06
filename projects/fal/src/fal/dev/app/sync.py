"""Local side of `fal dev app` file sync: pick, hash, diff and pack files.

Inside a git checkout the file list is `git ls-files` (tracked plus untracked,
minus ignored); elsewhere it is a directory walk that skips the usual virtualenv
and cache folders. Key, credential and env files (see
``fal.dev.files.secret_like``) and files over 10 MiB are never sent. Only
files that changed since the last sync are sent.

The box runs the app from this synced tree. The app's ``app_files_ignore``
patterns apply here too, matched like `fal deploy` matches them: against the
path relative to the app files context folder.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import re
import subprocess
import tarfile
from pathlib import Path

from fal.dev.files import SKIP_DIRS, secret_like
from fal.dev.remote.protocol import MAX_FILE_BYTES

SKIP_SUFFIXES = (".pyc", ".pyo")
BATCH_BYTES = 25 * 1024 * 1024


def _git_files(root: Path) -> list[str] | None:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "-co",
                "--exclude-standard",
                "-z",
                ".",
            ],
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return [p for p in result.stdout.decode().split("\0") if p]


def _walk_files(root: Path) -> list[str]:
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for filename in filenames:
            found.append(str((Path(dirpath) / filename).relative_to(root)))
    return found


def _ignored(name: str, patterns: list[re.Pattern[str]], base: str) -> bool:
    if not patterns:
        return False
    if base not in ("", "."):
        prefix = base.rstrip("/") + "/"
        if not name.startswith(prefix):
            return False
        name = name[len(prefix) :]
    return any(pattern.search(name) for pattern in patterns)


def collect(
    root: Path,
    skipped: list[str] | None = None,
    *,
    ignore: list[str] | None = None,
    ignore_base: str = ".",
    withheld: list[str] | None = None,
) -> dict[str, Path]:
    """Files to sync, by POSIX path relative to root.

    Files left out for their size are appended to ``skipped``, key and
    credential files to ``withheld``. ``ignore`` holds regex patterns matched
    against the path relative to ``ignore_base``.
    """
    root = root.resolve()
    patterns = [re.compile(pattern) for pattern in ignore or []]
    names = _git_files(root)
    if names is None:
        names = _walk_files(root)
    files = {}
    for name in names:
        path = root / name
        parts = Path(name).parts
        if name.endswith(SKIP_SUFFIXES) or _ignored(
            Path(name).as_posix(), patterns, ignore_base
        ):
            continue
        if secret_like(parts):
            if withheld is not None:
                withheld.append(name)
            continue
        try:
            if SKIP_DIRS.intersection(parts) or not path.is_file():
                continue
            # A symlink is sent as its target, so the target must pass too.
            resolved = path.resolve()
            target = (
                resolved.relative_to(root).parts
                if resolved.is_relative_to(root)
                else ()
            )
            if not target or SKIP_DIRS.intersection(target) or secret_like(target):
                continue
            size = path.stat().st_size
        except OSError:
            # Editors create and delete temp files while we look.
            continue
        if size > MAX_FILE_BYTES:
            if skipped is not None:
                skipped.append(name)
            continue
        files[Path(name).as_posix()] = path
    return files


def stat_signature(files: dict[str, Path]) -> dict[str, tuple[int, int]]:
    """Cheap change detection for --watch: (mtime, size) per file."""
    signature = {}
    for rel, path in files.items():
        try:
            st = path.stat()
        except OSError:
            continue
        signature[rel] = (st.st_mtime_ns, st.st_size)
    return signature


def manifest(files: dict[str, Path]) -> dict[str, str]:
    """sha256 per file; files that vanished since ``collect`` are dropped."""
    digests = {}
    for rel, path in sorted(files.items()):
        try:
            digests[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            files.pop(rel)
    return digests


def diff(old: dict[str, str], new: dict[str, str]) -> tuple[list[str], list[str]]:
    """(changed or added, deleted) between two manifests."""
    changed = sorted(rel for rel, digest in new.items() if old.get(rel) != digest)
    deleted = sorted(rel for rel in old if rel not in new)
    return changed, deleted


def batches(
    files: dict[str, Path],
    names: list[str],
    max_bytes: int | None = None,
    max_count: int | None = None,
) -> list[list[str]]:
    """Split names into groups of about ``max_bytes`` (default BATCH_BYTES, and
    at most ``max_count`` names), for one request each."""
    max_bytes = max_bytes or BATCH_BYTES
    groups: list[list[str]] = [[]]
    size = 0
    for rel in names:
        file_size = files[rel].stat().st_size
        full = max_count is not None and len(groups[-1]) >= max_count
        if groups[-1] and (full or size + file_size > max_bytes):
            groups.append([])
            size = 0
        groups[-1].append(rel)
        size += file_size
    return groups


def pack(files: dict[str, Path], names: list[str]) -> str:
    """A base64 tar.gz of the named files."""
    buffer = io.BytesIO()
    # Symlinks are sent as the files they point to.
    with tarfile.open(
        fileobj=buffer, mode="w:gz", compresslevel=1, dereference=True
    ) as archive:
        for rel in names:
            archive.add(files[rel], arcname=rel, recursive=False)
    return base64.b64encode(buffer.getvalue()).decode()
