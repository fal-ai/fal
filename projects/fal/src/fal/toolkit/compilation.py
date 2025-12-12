"""PyTorch compilation cache management utilities.

This module provides utilities for managing PyTorch Inductor compilation caches
across workers. When using torch.compile(), PyTorch generates optimized CUDA kernels
on first run, which can take 20-30 seconds. By sharing these compiled kernels across
workers, subsequent workers can load pre-compiled kernels in ~2 seconds instead of
recompiling.

Typical usage in a model setup:

    Manual cache management:
        dir_hash = load_inductor_cache("mymodel/v1")
        self.model = torch.compile(self.model)
        self.warmup()  # Triggers compilation
        sync_inductor_cache("mymodel/v1", dir_hash)

    Context manager (automatic):
        with synchronized_inductor_cache("mymodel/v1"):
            self.model = torch.compile(self.model)
            self.warmup()  # Compilation is automatically synced after
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

LOCAL_INDUCTOR_CACHE_DIR = Path("/tmp/inductor-cache/")
GLOBAL_INDUCTOR_CACHES_DIR = Path("/data/inductor-caches/")
PERSISTENT_TMP_DIR = Path("/data/tmp/")


def get_gpu_type() -> str:
    """Detect the GPU type using nvidia-smi.

    Returns:
        The GPU model name (e.g., "H100", "A100", "H200") or "UNKNOWN"
        if detection fails.

    Example:
        >>> gpu_type = get_gpu_type()
        >>> print(f"Running on: {gpu_type}")
        Running on: H100
    """
    try:
        gpu_type_string = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        matches = re.search(r"NVIDIA [a-zA-Z0-9]*", gpu_type_string)
        # check for matches - if there are none, return "UNKNOWN"
        if matches:
            gpu_type = matches.group(0)
            return gpu_type[7:]  # remove `NVIDIA `
        else:
            return "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def _dir_hash(path: Path) -> str:
    """Compute a hash of all filenames in a directory (recursively).

    Args:
        path: Directory to hash.

    Returns:
        SHA256 hex digest of sorted filenames.
    """
    # Hash of all the filenames in the directory, recursively, sorted
    filenames = {str(file) for file in path.rglob("*") if file.is_file()}
    return hashlib.sha256("".join(sorted(filenames)).encode()).hexdigest()


def load_inductor_cache(cache_key: str) -> str:
    """Load PyTorch Inductor compilation cache from global storage.

    This function:
    1. Sets TORCHINDUCTOR_CACHE_DIR environment variable
    2. Looks for cached compiled kernels in GPU-specific global storage
    3. Unpacks the cache to local temporary directory
    4. Returns a hash of the unpacked directory for change detection

    Args:
        cache_key: Unique identifier for this cache (e.g., "flux/2", "mymodel/v1")

    Returns:
        Hash of the unpacked cache directory, or empty string if cache not found.

    Example:
        >>> dir_hash = load_inductor_cache("flux/2")
        Found compilation cache at /data/inductor-caches/H100/flux/2.zip, unpacking...
        Cache unpacked successfully.
    """
    gpu_type = get_gpu_type()

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(LOCAL_INDUCTOR_CACHE_DIR)

    cache_source_path = GLOBAL_INDUCTOR_CACHES_DIR / gpu_type / f"{cache_key}.zip"

    try:
        next(cache_source_path.parent.iterdir(), None)
    except Exception as e:
        # Check for cache without gpu_type in the path
        try:
            old_source_path = GLOBAL_INDUCTOR_CACHES_DIR / f"{cache_key}.zip"
            # Since old source exists, copy it over to global caches
            os.makedirs(cache_source_path.parent, exist_ok=True)
            shutil.copy(old_source_path, cache_source_path)
        except Exception:
            print(f"Failed to list: {e}")

    if not cache_source_path.exists():
        print(f"Couldn't find compilation cache at {cache_source_path}")
        return ""

    print(f"Found compilation cache at {cache_source_path}, unpacking...")
    try:
        shutil.unpack_archive(cache_source_path, LOCAL_INDUCTOR_CACHE_DIR)
    except Exception as e:
        print(f"Failed to unpack cache: {e}")
        return ""

    print("Cache unpacked successfully.")
    return _dir_hash(LOCAL_INDUCTOR_CACHE_DIR)


def sync_inductor_cache(cache_key: str, unpacked_dir_hash: str) -> None:
    """Sync updated PyTorch Inductor cache back to global storage.

    This function:
    1. Checks if the local cache has changed (by comparing hashes)
    2. If changed, creates a zip archive of the new cache
    3. Saves it to GPU-specific global storage

    Args:
        cache_key: Unique identifier for this cache (same as used in
            load_inductor_cache)
        unpacked_dir_hash: Hash returned from load_inductor_cache
            (for change detection)

    Example:
        >>> sync_inductor_cache("flux/2", dir_hash)
        No changes in the cache dir, skipping sync.
        # or
        Changes detected in the cache dir, syncing...
    """
    gpu_type = get_gpu_type()
    if not LOCAL_INDUCTOR_CACHE_DIR.exists():
        print(f"No cache to sync, {LOCAL_INDUCTOR_CACHE_DIR} doesn't exist.")
        return

    if not GLOBAL_INDUCTOR_CACHES_DIR.exists():
        GLOBAL_INDUCTOR_CACHES_DIR.mkdir(parents=True)

    # If we updated the cache (the hashes of LOCAL_INDUCTOR_CACHE_DIR and
    # unpacked_dir_hash differ), we pack the cache and move it to the
    # global cache directory.
    new_dir_hash = _dir_hash(LOCAL_INDUCTOR_CACHE_DIR)
    if new_dir_hash == unpacked_dir_hash:
        print("No changes in the cache dir, skipping sync.")
        return

    print("Changes detected in the cache dir, syncing...")
    os.makedirs(
        PERSISTENT_TMP_DIR, exist_ok=True
    )  # Non fal-ai users do not have this directory
    with tempfile.TemporaryDirectory(dir=PERSISTENT_TMP_DIR) as temp_dir:
        temp_dir_path = Path(temp_dir)
        cache_path = GLOBAL_INDUCTOR_CACHES_DIR / gpu_type / f"{cache_key}.zip"
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            zip_name = shutil.make_archive(
                str(temp_dir_path / "inductor_cache"),
                "zip",
                LOCAL_INDUCTOR_CACHE_DIR,
            )
            os.rename(
                zip_name,
                cache_path,
            )
        except Exception as e:
            print(f"Failed to sync cache: {e}")
            return


@contextmanager
def synchronized_inductor_cache(cache_key: str) -> Iterator[None]:
    """Context manager to automatically load and sync PyTorch Inductor cache.

    This wraps load_inductor_cache and sync_inductor_cache for convenience.
    The cache is loaded on entry and synced on exit (even if an exception occurs).

    Args:
        cache_key: Unique identifier for this cache (e.g., "flux/2", "mymodel/v1")

    Yields:
        None

    Example:
        >>> with synchronized_inductor_cache("mymodel/v1"):
        ...     self.model = torch.compile(self.model)
        ...     self.warmup()  # Compilation happens here
        # Cache is automatically synced after the with block
    """
    unpacked_dir_hash = load_inductor_cache(cache_key)
    try:
        yield
    finally:
        sync_inductor_cache(cache_key, unpacked_dir_hash)
