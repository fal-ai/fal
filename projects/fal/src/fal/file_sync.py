import concurrent.futures
import hashlib
import os
import re
from functools import cached_property
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Tuple

import httpx
from rich.tree import Tree
from tusclient import client

from fal._version import version_tuple
from fal.console import console
from fal.console.icons import CROSS_ICON
from fal.exceptions import FalServerlessException, FileTooLargeError

USER_AGENT = f"fal-sdk/{'.'.join(map(str, version_tuple))} (python)"
FILE_SIZE_LIMIT = 1024 * 1024 * 1024  # 1GB
DEFAULT_CONCURRENCY_UPLOADS = 10


def print_path_tree(file_paths):
    tree = Tree("/deploy-data")

    nodes = {"": tree}

    for file_path in sorted(file_paths):
        parts = Path(file_path).parts

        for i, part in enumerate(parts):
            current_path = str(Path(*parts[: i + 1]))

            if current_path not in nodes:
                parent_path = str(Path(*parts[:i])) if i > 0 else ""
                parent_node = nodes[parent_path]

                nodes[current_path] = parent_node.add(f"{part}")

    console.print(tree)


def sanitize_relative_path(rel_path: str) -> str:
    pure_path = PurePosixPath(rel_path)

    # Block files that are absolute or contain parent directory references
    if pure_path.is_absolute():
        raise FalServerlessException(f"Absolute Path is not allowed: {rel_path}")
    if ".." in pure_path.parts or "." in pure_path.parts:
        raise FalServerlessException(
            f"Parent directory reference is not allowed: {rel_path}"
        )

    return pure_path.as_posix()


class FileSync:
    def __init__(self, local_file_path: str):
        from fal.sdk import get_default_credentials

        self.creds = get_default_credentials()
        self.local_file_path = local_file_path

    @cached_property
    def _client(self) -> httpx.Client:
        from fal.flags import REST_URL

        return httpx.Client(
            base_url=REST_URL,
            headers={
                **self.creds.to_headers(),
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

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        response = self._client.request(method, path, **kwargs)
        if response.status_code == 404:
            raise FalServerlessException("Not Found")
        elif response.status_code != 200:
            try:
                detail = response.json()["detail"]
            except Exception:
                detail = response.text
            raise FalServerlessException(detail)
        return response

    def compute_hash(self, file_path: Path, mode: int) -> str:
        file_hash = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                file_hash.update(chunk)

        # Include metadata in hash
        metadata_string = f"{mode}"
        file_hash.update(metadata_string.encode("utf-8"))

        return file_hash.hexdigest()

    def get_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        stat = file_path.stat()
        # Limit allowed individual file size
        if stat.st_size > FILE_SIZE_LIMIT:
            raise FileTooLargeError(
                message=f"{file_path} is larger than {FILE_SIZE_LIMIT} bytes."
            )

        file_hash = self.compute_hash(file_path, stat.st_mode)
        return {
            "size": str(stat.st_size),
            "mtime": str(stat.st_mtime),
            "mode": str(stat.st_mode),
            "hash": file_hash,
        }

    def normalize_path(
        self, path_str: str, base_path_str: str, files_context_dir: str | None = None
    ) -> Tuple[str, str]:
        path = Path(path_str)
        base_path = Path(base_path_str).resolve()

        if base_path.is_dir():
            script_dir = base_path
        else:
            script_dir = base_path.parent

        if files_context_dir:
            context_path = Path(files_context_dir)
            if context_path.is_absolute():
                script_dir = context_path.resolve()
            else:
                script_dir = (script_dir / context_path).resolve()

        absolute_path = (
            path.resolve() if path.is_absolute() else (script_dir / path).resolve()
        )

        try:
            relative_path = os.path.relpath(absolute_path, script_dir)
            relative_path = sanitize_relative_path(relative_path)
        except ValueError:
            raise ValueError(f"Invalid relative path: {absolute_path}")

        return absolute_path.as_posix(), relative_path

    def collect_files(
        self, paths: List[str], files_context_dir: str | None = None
    ) -> List[Dict[str, Any]]:
        collected_files = []

        for path in paths:
            abs_path_str, rel_path = self.normalize_path(
                path, self.local_file_path, files_context_dir
            )
            abs_path = Path(abs_path_str)
            if not abs_path.exists():
                console.print(f"{abs_path} was not found, it will be skipped")
                continue

            if abs_path.is_file():
                metadata = self.get_file_metadata(abs_path)
                metadata["absolute_path"] = abs_path_str
                metadata["relative_path"] = rel_path
                collected_files.append(metadata)

            elif abs_path.is_dir():
                # Recursively walk directory tree
                for file_path in abs_path.rglob("*"):
                    if file_path.is_file():
                        file_abs_str, file_rel_path = self.normalize_path(
                            str(file_path), self.local_file_path, files_context_dir
                        )
                        metadata = self.get_file_metadata(Path(file_abs_str))
                        metadata["absolute_path"] = file_abs_str
                        metadata["relative_path"] = file_rel_path
                        collected_files.append(metadata)

        return collected_files

    def check_hashes_on_server(self, hashes: List[str]) -> List[str]:
        try:
            response = self._request(
                "POST", "/files/missing_hashes", json={"hashes": hashes}
            )
            response.raise_for_status()
            data = response.json()
            return data
        except Exception as e:
            console.print(f"{CROSS_ICON} Failed to check hashes on server: {e}")

            return hashes

    def upload_file_tus(
        self,
        file_path: str,
        chunk_size: int = 5 * 1024 * 1024,
        metadata: Dict[str, Any] = {},
    ) -> str:
        uploader = self._tus_client.uploader(
            file_path, chunk_size=chunk_size, metadata=metadata
        )
        uploader.upload()
        return uploader.url

    def _should_ignore_file(
        self, relative_path: str, ignore_patterns: List[str]
    ) -> bool:
        """Check if a file should be ignored based on regex patterns."""
        for pattern in ignore_patterns:
            try:
                if re.search(pattern, relative_path):
                    # Showing this to avoid confusion
                    console.print(f"Ignoring file: {relative_path}")
                    return True
            except re.error as e:
                console.print(f"Invalid regex pattern '{pattern}': {e}")
                continue
        return False

    def sync_files(
        self,
        paths: List[str],
        chunk_size: int = 5 * 1024 * 1024,
        max_concurrency_uploads: int = DEFAULT_CONCURRENCY_UPLOADS,
        files_ignore: List[str] = [],
        files_context_dir: str | None = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        files = self.collect_files(paths, files_context_dir)
        results: Dict[str, List[Dict[str, Any]]] = {
            "existing_hashes": [],
            "uploaded_files": [],
            "errors": [],
        }

        # Filter out ignored files
        if files_ignore:
            filtered_files = []
            for metadata in files:
                if not self._should_ignore_file(
                    metadata["relative_path"], files_ignore
                ):
                    filtered_files.append(metadata)
            files = filtered_files

        # Remove duplicate files by absolute path
        unique_files = []
        seen_paths = set()
        seen_relative_paths = set()
        for metadata in files:
            abs_path = metadata["absolute_path"]
            rel_path = metadata["relative_path"]
            if abs_path not in seen_paths:
                seen_paths.add(abs_path)
                if rel_path in seen_relative_paths:
                    raise Exception(
                        f"Duplicate relative path '{rel_path}' found for '{abs_path}'"
                    )
                seen_relative_paths.add(rel_path)
                unique_files.append(metadata)
            else:
                if rel_path not in seen_relative_paths:
                    seen_relative_paths.add(rel_path)

        files_to_check = []
        for metadata in unique_files:
            files_to_check.append(metadata)

        if not files_to_check:
            return results

        hashes_to_check = [metadata["hash"] for metadata in files_to_check]
        missing_hashes = self.check_hashes_on_server(hashes_to_check)

        # Categorize based on server response
        files_to_upload = []
        for file in files_to_check:
            if file["hash"] not in missing_hashes:
                results["existing_hashes"].append(file)
            else:
                files_to_upload.append(file)

        # Upload missing files in parallel with bounded concurrency
        if files_to_upload:

            def upload_single_file(metadata):
                try:
                    console.print(f"Uploading file: {metadata['relative_path']}")
                    upload_url = self.upload_file_tus(
                        metadata["absolute_path"],
                        chunk_size=chunk_size,
                        metadata=metadata,
                    )
                    metadata["upload_url"] = upload_url
                    return metadata
                except Exception as e:
                    return {"relative_path": metadata["relative_path"], "error": str(e)}

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_concurrency_uploads
            ) as executor:
                futures = [
                    executor.submit(upload_single_file, metadata)
                    for metadata in files_to_upload
                ]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if "error" in result:
                        results["errors"].append(result)
                    else:
                        results["uploaded_files"].append(result)

        relative_paths = [
            file["relative_path"]
            for file in results["uploaded_files"] + results["existing_hashes"]
        ]
        console.print("File Structure:")
        print_path_tree(relative_paths)
        return results

    def close(self):
        """Close HTTP client."""
        self._client.close()
