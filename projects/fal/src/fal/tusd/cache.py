"""Cache management for TUS uploads."""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional

# Cache directory for upload resumption
CACHE_DIR = Path.home() / ".fal" / "cache"
CACHE_FILE = CACHE_DIR / "_uploads.json"

# Lock to prevent concurrent cache access
_cache_lock = asyncio.Lock()


async def ensure_cache_dir() -> Path:
    """
    Ensure the cache directory exists.
    """
    await asyncio.to_thread(CACHE_DIR.mkdir, parents=True, exist_ok=True)
    return CACHE_DIR


async def _load_upload_cache_unlocked() -> Dict[str, Any]:
    """
    Load the upload cache from disk.
    """
    await ensure_cache_dir()

    def _load():
        if CACHE_FILE.exists():
            try:
                with CACHE_FILE.open("r") as f:
                    content = f.read()
                    # Check if file is empty or only whitespace
                    if not content.strip():
                        return {}
                    # Parse JSON from the content we already read
                    return json.loads(content)
            except (OSError, json.JSONDecodeError) as e:
                print(f"Could not load cache: {e}")
                return {}
        return {}

    return await asyncio.to_thread(_load)


async def _save_upload_cache_unlocked(cache: Dict[str, Any]):
    """
    Save the upload cache to disk (internal, assumes lock is held).
    """
    await ensure_cache_dir()

    def _save():
        try:
            # Write to a temporary file first, then rename atomically
            temp_file = CACHE_FILE.with_suffix(".tmp")
            with temp_file.open("w") as f:
                json.dump(cache, f, indent=2)
            # Atomic rename (on Unix systems, this is atomic)
            temp_file.replace(CACHE_FILE)
        except OSError as e:
            print(f"Could not save cache: {e}")
            # Clean up temp file if it exists
            temp_file = CACHE_FILE.with_suffix(".tmp")
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass

    await asyncio.to_thread(_save)


async def load_upload_cache() -> Dict[str, Any]:
    """
    Load the upload cache from disk.
    """
    async with _cache_lock:
        return await _load_upload_cache_unlocked()


async def save_upload_cache(cache: Dict[str, Any]):
    """
    Save the upload cache to disk.
    """
    async with _cache_lock:
        await _save_upload_cache_unlocked(cache)


def generate_upload_identifier(start_position: int, chunk_size: int) -> str:
    """
    Generate a unique identifier for an upload configuration.

    :param start_position: Starting position/offset in the file.
    :param chunk_size: The chunk size used for upload.
    :return: A unique identifier string.
    """
    return f"{start_position}_{chunk_size}"


async def get_cached_upload(
    file_hash: str, start_position: int, chunk_size: int
) -> Optional[str]:
    """
    Get cached upload URL for a file hash and upload configuration.

    :param file_hash: The hash of the file.
    :param start_position: Starting position in the file.
    :param chunk_size: The chunk size used for upload.
    :return: The cached URL or None if not found.
    """
    cache = await load_upload_cache()
    file_entry = cache.get(file_hash)

    if not file_entry:
        return None

    upload_id = generate_upload_identifier(start_position, chunk_size)
    url = file_entry.get(upload_id)

    return url


async def cache_upload(
    file_hash: str,
    url: str,
    start_position: int,
    chunk_size: int,
):
    """
    Cache an upload URL for a file hash and upload configuration.

    :param file_hash: The hash of the file.
    :param url: The upload URL to cache.
    :param start_position: Starting position in the file.
    :param chunk_size: The chunk size used for upload.
    """
    async with _cache_lock:
        # Load cache within the lock
        cache = await _load_upload_cache_unlocked()

        # Get or create file entry
        if file_hash not in cache:
            cache[file_hash] = {}

        # Generate unique identifier for this upload configuration
        upload_id = generate_upload_identifier(start_position, chunk_size)

        # Store the URL under this configuration
        cache[file_hash][upload_id] = url

        # Save cache within the same lock
        await _save_upload_cache_unlocked(cache)


async def remove_from_cache(file_hash: str):
    """
    Remove an entry from the cache.

    :param file_hash: The hash of the file to remove.
    """
    async with _cache_lock:
        # Load cache within the lock
        cache = await _load_upload_cache_unlocked()
        if file_hash in cache:
            del cache[file_hash]
            # Save cache within the same lock
            await _save_upload_cache_unlocked(cache)
