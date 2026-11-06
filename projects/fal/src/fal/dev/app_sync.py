"""Local side of `fal dev app` file sync: pick files, diff them, pack them.

Inside a git checkout the file list is `git ls-files` (tracked plus untracked,
minus ignored); elsewhere it is a directory walk that skips the usual virtualenv
and cache folders. `.env` files or folders and files over 10 MiB are never
sent. Only files that changed since the last sync are sent.

The box runs the app from this synced tree. The app's own ``app_files`` and
``app_files_ignore`` settings apply to deploys, not to this sync.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import subprocess
import tarfile
from importlib import metadata
from pathlib import Path

SKIP_DIRS = {
    ".git",
    ".hg",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
}
SKIP_SUFFIXES = (".pyc", ".pyo")
# Local secrets stay local.
SKIP_PREFIXES = (".env",)
MAX_FILE_BYTES = 10 * 1024 * 1024
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


def _skipped(parts: tuple[str, ...]) -> bool:
    return bool(SKIP_DIRS.intersection(parts)) or any(
        part.lower().startswith(SKIP_PREFIXES) for part in parts
    )


def collect(root: Path, skipped: list[str] | None = None) -> dict[str, Path]:
    """Files to sync, by POSIX path relative to root.

    Files left out for their size are appended to ``skipped``.
    """
    root = root.resolve()
    names = _git_files(root)
    if names is None:
        names = _walk_files(root)
    files = {}
    for name in names:
        path = root / name
        parts = Path(name).parts
        if name.endswith(SKIP_SUFFIXES):
            continue
        try:
            if _skipped(parts) or not path.is_file():
                continue
            # A symlink is sent as its target, so the target must pass too.
            resolved = path.resolve()
            if not resolved.is_relative_to(root) or _skipped(
                resolved.relative_to(root).parts
            ):
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


def batches(files: dict[str, Path], names: list[str]) -> list[list[str]]:
    """Split names into groups of about BATCH_BYTES, for one request each."""
    groups: list[list[str]] = [[]]
    size = 0
    for rel in names:
        file_size = files[rel].stat().st_size
        if groups[-1] and size + file_size > BATCH_BYTES:
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


def sdk_files() -> dict[str, Path]:
    """The local fal package, which the runner does not have installed."""
    package = Path(__file__).resolve().parent.parent
    files = {}
    for path in package.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            files[(Path("fal") / path.relative_to(package)).as_posix()] = path
    return files


def sdk_requirements() -> list[str]:
    """fal's own runtime dependencies, installed on the runner only if missing."""
    try:
        requires = metadata.requires("fal") or []
    except metadata.PackageNotFoundError:
        return []
    return [r for r in requires if "extra ==" not in r]
