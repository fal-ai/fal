import asyncio
import hashlib
import json
import os
from functools import cached_property
from pathlib import Path
from typing import Any

import httpx
from tusclient import client

USER_AGENT = "fal-sdk/1.14.0 (python)"


class FileSync:
    def __init__(self, local_file_path: str):
        self.local_file_path = local_file_path
        self.cache_dir = Path.home() / ".cache" / "fal"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "file_metadata.json"

    @cached_property
    def _client(self) -> httpx.AsyncClient:
        from fal.flags import REST_URL
        from fal.sdk import get_default_credentials

        creds = get_default_credentials()
        return httpx.AsyncClient(
            base_url=REST_URL,
            headers={
                **creds.to_headers(),
                "User-Agent": USER_AGENT,
            },
        )

    @cached_property
    def _tus_client(self) -> client.TusClient:
        from fal.flags import REST_URL
        from fal.sdk import get_default_credentials

        creds = get_default_credentials()
        return client.TusClient(
            f"{REST_URL}/files/tus",
            headers={
                **creds.to_headers(),
                "User-Agent": USER_AGENT,
            },
        )

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        from fal.exceptions import FalServerlessException

        response = await self._client.request(method, path, **kwargs)
        if response.status_code != 200:
            try:
                detail = response.json()["detail"]
            except Exception:
                detail = response.text
            raise FalServerlessException(detail)
        return response

    def compute_hash(self, file_path: str, mtime: float, mode: int) -> str:
        sha256 = hashlib.sha256()

        # Hash file contents
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)

        # Include metadata in hash
        metadata_string = f"{mtime}:{mode}"
        sha256.update(metadata_string.encode("utf-8"))

        return sha256.hexdigest()

    def get_file_metadata(self, file_path: str) -> dict[str, Any]:
        stat = os.stat(file_path)
        file_hash = self.compute_hash(file_path, stat.st_mtime, stat.st_mode)
        return {
            "size": str(stat.st_size),
            "mtime": str(stat.st_mtime),
            "mode": str(stat.st_mode),
            "hash": file_hash,
        }

    def normalize_path(self, path_str: str, base_path_str: str) -> tuple[str, str]:
        path = Path(path_str)
        base_path = Path(base_path_str)

        # Resolve relative paths against base directory
        if not path.is_absolute():
            if base_path.is_file():
                script_dir = base_path.parent
            else:
                script_dir = base_path
            absolute_path = (script_dir / path).resolve()
        else:
            absolute_path = path.resolve()

        # Create relative path, handling cross-drive scenarios
        try:
            relative_path = os.path.relpath(absolute_path, base_path.parent)
        except ValueError:
            relative_path = absolute_path.as_posix()

        # Remove parent directory traversals for security
        relative_path = relative_path.replace("../", "")

        return absolute_path.as_posix(), relative_path

    async def fetch_cache_from_server(self) -> list[dict[str, Any]]:
        try:
            response = await self._request("GET", "files/tus_file/cache")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print("Error fetching cache:", e)
            return []

    async def load_local_cache(self) -> dict[str, dict[str, Any]]:
        if not self.cache_file.exists():
            files = await self.fetch_cache_from_server()
            cache = {
                file["hash"]: {
                    "mtime": file["mtime"],
                    "mode": file["mode"],
                    "size": file["size_bytes"],
                    "hash": file["hash"],
                }
                for file in files
            }
            self.save_local_cache(cache)
            return cache

        with open(self.cache_file) as f:
            return json.load(f)

    def save_local_cache(self, cache_data: dict[str, dict[str, Any]]) -> None:
        try:
            with open(self.cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
        except OSError as e:
            print(f"Warning: Failed to save cache: {e}")

    async def check_local_cache(self, current_metadata: dict[str, Any]) -> bool:
        cache = await self.load_local_cache()
        cached_metadata = cache.get(current_metadata["hash"])

        if not cached_metadata:
            return False

        # Check if hash matches (hash includes content + mtime + mode)
        return cached_metadata.get("hash") == current_metadata.get("hash")

    async def update_local_cache(self, metadata: dict[str, Any]) -> None:
        cache = await self.load_local_cache()
        cache[metadata["hash"]] = {
            **metadata,
            "cached_at": str(asyncio.get_event_loop().time()),
        }
        self.save_local_cache(cache)

    def collect_files(self, paths: list[str]) -> list[dict[str, Any]]:
        collected_files = []

        for path in paths:
            abs_path, rel_path = self.normalize_path(path, self.local_file_path)
            if not os.path.exists(abs_path):
                continue

            if os.path.isfile(abs_path):
                metadata = self.get_file_metadata(abs_path)
                metadata["absolute_path"] = abs_path
                metadata["relative_path"] = rel_path
                collected_files.append(metadata)

            elif os.path.isdir(abs_path):
                # Recursively walk directory tree
                for root, _, files in os.walk(abs_path):
                    for file_name in files:
                        file_abs_path = os.path.join(root, file_name)
                        file_abs_path, file_rel_path = self.normalize_path(
                            file_abs_path, self.local_file_path
                        )

                        metadata = self.get_file_metadata(file_abs_path)
                        metadata["absolute_path"] = file_abs_path
                        metadata["relative_path"] = file_rel_path
                        collected_files.append(metadata)

        return collected_files

    async def check_hash_exists(self, file_hash: str) -> bool:
        try:
            response = await self._request(
                "HEAD", f"/files/tus_file/exists/{file_hash}"
            )
            print(response.status_code)
            return response.status_code == 200
        except Exception as e:
            print(f"Error checking hash {file_hash}: {e}")
            return False

    async def check_multiple_hashes_exist(
        self, file_hashes: list[str]
    ) -> dict[str, bool]:
        # Create concurrent hash check tasks
        tasks = [self.check_hash_exists(file_hash) for file_hash in file_hashes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build result dictionary, handling any exceptions
        hash_status = {}
        for file_hash, result in zip(file_hashes, results):
            if isinstance(result, Exception):
                print(f"Error checking hash {file_hash}: {result}")
                hash_status[file_hash] = False
            else:
                hash_status[file_hash] = result

        return hash_status

    def upload_file_tus(
        self,
        file_path: str,
        chunk_size: int = 5 * 1024 * 1024,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        uploader = self._tus_client.uploader(
            file_path, chunk_size=chunk_size, metadata=metadata or {}
        )
        uploader.upload()
        return uploader.url

    async def upload_multiple_files(
        self, files_to_upload: list[dict[str, Any]], chunk_size: int = 5 * 1024 * 1024
    ) -> list[dict[str, Any]]:
        loop = asyncio.get_event_loop()

        async def upload_single_file(metadata):
            try:
                # Run synchronous TUS upload in thread pool
                upload_url = await loop.run_in_executor(
                    None,
                    lambda: self.upload_file_tus(
                        metadata["absolute_path"],
                        chunk_size=chunk_size,
                        metadata=metadata,
                    ),
                )
                metadata["upload_url"] = upload_url
                return metadata
            except Exception as e:
                return {"relative_path": metadata["relative_path"], "error": str(e)}

        # Upload files concurrently
        tasks = [upload_single_file(metadata) for metadata in files_to_upload]
        return await asyncio.gather(*tasks)

    async def sync_files(
        self,
        paths: list[str],
        chunk_size: int = 5 * 1024 * 1024,
    ) -> dict[str, list[dict[str, Any]]]:
        files = self.collect_files(paths)
        results: dict[str, list[dict[str, Any]]] = {
            "existing_hashes": [],
            "uploaded_files": [],
            "errors": [],
        }

        # Remove duplicate files by absolute path
        unique_files = []
        seen_paths = set()
        for metadata in files:
            abs_path = metadata["absolute_path"]
            if abs_path not in seen_paths:
                seen_paths.add(abs_path)
                unique_files.append(metadata)

        # Check local cache first to avoid unnecessary server requests
        files_to_check = []
        for metadata in unique_files:
            abs_path = metadata["absolute_path"]
            if await self.check_local_cache(metadata):
                # File unchanged since last sync, skip
                results["existing_hashes"].append(metadata)
            else:
                files_to_check.append(metadata)

        if not files_to_check:
            return results

        # Check which hashes exist on server (parallel)
        hashes_to_check = [metadata["hash"] for metadata in files_to_check]
        hash_status = await self.check_multiple_hashes_exist(hashes_to_check)

        # Categorize files based on server hash status
        files_to_upload = []
        for metadata in files_to_check:
            file_hash = metadata["hash"]

            if hash_status.get(file_hash, False):
                # Hash exists on server
                results["existing_hashes"].append(metadata)
                # Update cache with hash exists status
                await self.update_local_cache(metadata)
            else:
                # Need to upload
                files_to_upload.append(metadata)

        # Upload files in parallel
        if files_to_upload:
            upload_results = await self.upload_multiple_files(
                files_to_upload, chunk_size
            )

            for result in upload_results:
                if "error" in result:
                    results["errors"].append(result)
                else:
                    results["uploaded_files"].append(result)
                    await self.update_local_cache(result)

        return results

    async def close(self):
        """Close async HTTP client."""
        await self._client.aclose()
