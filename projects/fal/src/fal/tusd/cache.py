"""Cache management for TUS uploads."""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional

CACHE_DIR = Path.home() / ".fal" / "cache"
CACHE_FILE = CACHE_DIR / "_uploads.json"

_cache_lock = asyncio.Lock()


async def ensure_cache_dir() -> Path:
    await asyncio.to_thread(CACHE_DIR.mkdir, parents=True, exist_ok=True)
    return CACHE_DIR


async def _load_upload_cache_unlocked() -> Dict[str, Any]:
    await ensure_cache_dir()

    def _load():
        if CACHE_FILE.exists():
            try:
                with CACHE_FILE.open("r") as f:
                    content = f.read()
                    if not content.strip():
                        return {}
                    return json.loads(content)
            except (OSError, json.JSONDecodeError) as e:
                print(f"Could not load cache: {e}")
                return {}
        return {}

    return await asyncio.to_thread(_load)


async def _save_upload_cache_unlocked(cache: Dict[str, Any]):
    await ensure_cache_dir()

    def _save():
        try:
            temp_file = CACHE_FILE.with_suffix(".tmp")
            with temp_file.open("w") as f:
                json.dump(cache, f, indent=2)
            temp_file.replace(CACHE_FILE)
        except OSError as e:
            print(f"Could not save cache: {e}")
            temp_file = CACHE_FILE.with_suffix(".tmp")
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass

    await asyncio.to_thread(_save)


async def load_upload_cache() -> Dict[str, Any]:
    async with _cache_lock:
        return await _load_upload_cache_unlocked()


async def save_upload_cache(cache: Dict[str, Any]):
    async with _cache_lock:
        await _save_upload_cache_unlocked(cache)


def generate_upload_identifier(start_position: int, chunk_size: int) -> str:
    return f"{start_position}_{chunk_size}"


async def get_cached_upload(
    file_hash: str, start_position: int, chunk_size: int
) -> Optional[str]:
    cache = await load_upload_cache()
    file_entry = cache.get(file_hash)

    if not file_entry:
        return None

    upload_id = generate_upload_identifier(start_position, chunk_size)
    return file_entry.get(upload_id)


async def cache_upload(
    file_hash: str,
    url: str,
    start_position: int,
    chunk_size: int,
):
    async with _cache_lock:
        cache = await _load_upload_cache_unlocked()

        if file_hash not in cache:
            cache[file_hash] = {}

        upload_id = generate_upload_identifier(start_position, chunk_size)
        cache[file_hash][upload_id] = url

        await _save_upload_cache_unlocked(cache)


async def remove_from_cache(file_hash: str):
    async with _cache_lock:
        cache = await _load_upload_cache_unlocked()
        if file_hash in cache:
            del cache[file_hash]
            await _save_upload_cache_unlocked(cache)
