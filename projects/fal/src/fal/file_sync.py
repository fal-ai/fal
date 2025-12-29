import concurrent.futures
import hashlib
import os
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple

import httpx
from rich.tree import Tree

import fal.flags as flags
from fal._version import version_tuple
from fal.console import console
from fal.console.icons import CROSS_ICON
from fal.exceptions import (
    AppFileUploadException,
    FalServerlessException,
    FileTooLargeError,
)

USER_AGENT = f"fal-sdk/{'.'.join(map(str, version_tuple))} (python)"
FILE_SIZE_LIMIT = 1024 * 1024 * 1024  # 1GB
DEFAULT_CONCURRENCY_UPLOADS = 10


def print_path_tree(file_paths):
    tree = Tree("/app")

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


# This is the same function we use in the backend to compute file hashes
# It is important that this is the same
def compute_hash(file_path: Path, mode: int) -> str:
    file_hash = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            file_hash.update(chunk)

    # Include metadata in hash
    metadata_string = f"{mode}"
    file_hash.update(metadata_string.encode("utf-8"))

    return file_hash.hexdigest()


def normalize_path(
    path_str: str, base_path_str: str, files_context_dir: Optional[str] = None
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


@dataclass
class FileMetadata:
    size: int
    mtime: float
    mode: int
    hash: str
    relative_path: str
    absolute_path: str

    @classmethod
    def from_path(
        cls, file_path: Path, *, relative: str, absolute: str
    ) -> "FileMetadata":
        stat = file_path.stat()
        # Limit allowed individual file size
        if stat.st_size > FILE_SIZE_LIMIT:
            raise FileTooLargeError(
                message=f"{file_path} is larger than {FILE_SIZE_LIMIT} bytes."
            )

        file_hash = compute_hash(file_path, stat.st_mode)
        return FileMetadata(
            size=stat.st_size,
            mtime=stat.st_mtime,
            mode=stat.st_mode,
            hash=file_hash,
            relative_path=relative,
            absolute_path=absolute,
        )

    def to_dict(self) -> Dict[str, str]:
        return {
            "size": str(self.size),
            "mtime": str(self.mtime),
            "mode": str(self.mode),
            "hash": self.hash,
        }


class FileSync:
    def __init__(self, local_file_path: str):
        from fal.sdk import get_default_credentials  # noqa: PLC0415

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
    def _tus_client(self):
        # Import it here to avoid loading unless we use it
        from tusclient import client  # noqa: PLC0415

        from fal.flags import REST_URL  # noqa: PLC0415
        from fal.sdk import get_default_credentials  # noqa: PLC0415

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

    def collect_files(self, paths: List[str], files_context_dir: Optional[str] = None):
        collected_files: List[FileMetadata] = []

        for path in paths:
            abs_path_str, rel_path = normalize_path(
                path, self.local_file_path, files_context_dir
            )
            abs_path = Path(abs_path_str)
            if not abs_path.exists():
                console.print(f"{abs_path} was not found, it will be skipped")
                continue

            if abs_path.is_file():
                metadata = FileMetadata.from_path(
                    abs_path, relative=rel_path, absolute=abs_path_str
                )
                collected_files.append(metadata)

            elif abs_path.is_dir():
                # Recursively walk directory tree
                for file_path in abs_path.rglob("*"):
                    if file_path.is_file():
                        file_abs_str, file_rel_path = normalize_path(
                            str(file_path), self.local_file_path, files_context_dir
                        )
                        metadata = FileMetadata.from_path(
                            Path(file_abs_str),
                            relative=file_rel_path,
                            absolute=file_abs_str,
                        )
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
        metadata: FileMetadata,
        chunk_size: int = 5 * 1024 * 1024,
    ) -> str:
        uploader = self._tus_client.uploader(
            file_path, chunk_size=chunk_size, metadata=metadata.to_dict()
        )
        uploader.upload()
        if not uploader.url:
            raise AppFileUploadException("Upload failed, no URL returned", file_path)

        return uploader.url

    def _matches_patterns(self, relative_path: str, patterns: List[re.Pattern]) -> bool:
        """Check if a file matches any of the regex patterns."""
        return any(pattern.search(relative_path) for pattern in patterns)

    def sync_files(
        self,
        paths: List[str],
        chunk_size: int = 5 * 1024 * 1024,
        max_concurrency_uploads: int = DEFAULT_CONCURRENCY_UPLOADS,
        files_ignore: Optional[List[re.Pattern]] = None,
        files_context_dir: Optional[str] = None,
    ) -> Tuple[List[FileMetadata], List[AppFileUploadException]]:
        """Sync files to the server.

        Args:
            paths: List of file paths or glob patterns to sync
            chunk_size: Upload chunk size in bytes
            max_concurrency_uploads: Maximum concurrent uploads
            files_ignore: List of compiled regex patterns for ignoring files
            files_context_dir: Context directory for relative paths
        """
        files = self.collect_files(paths, files_context_dir)

        # Filter out ignored files using regex patterns (app_files)
        if files_ignore:
            filtered_files: List[FileMetadata] = []
            for metadata in files:
                if self._matches_patterns(metadata.relative_path, files_ignore):
                    if flags.DEBUG:
                        console.print(f"Ignoring file: {metadata.relative_path}")
                else:
                    filtered_files.append(metadata)

            # Update files list
            files = filtered_files

        # Remove duplicate files by absolute path
        unique_files: List[FileMetadata] = []
        seen_paths = set()
        seen_relative_paths = set()
        for metadata in files:
            console.print("metadata", metadata)
            abs_path = metadata.absolute_path
            rel_path = metadata.relative_path
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

        if not unique_files:
            return [], []

        hashes_to_check = list({metadata.hash for metadata in unique_files})
        missing_hashes = set(self.check_hashes_on_server(hashes_to_check))

        # Categorize based on server response
        files_to_upload: List[FileMetadata] = []
        for file in unique_files:
            if file.hash in missing_hashes:
                console.print("file", file)
                # No longer missing as we are uploading it
                # Removing it avoids duplicate uploads
                missing_hashes.remove(file.hash)
                files_to_upload.append(file)

        uploaded_files: List[Tuple[FileMetadata, str]] = []
        errors: List[AppFileUploadException] = []
        # Upload missing files in parallel with bounded concurrency
        if files_to_upload:
            # Embed it here to be able to pass it to the executor
            def upload_single_file(metadata: FileMetadata):
                console.print(f"Uploading file: {metadata.relative_path}")
                return self.upload_file_tus(
                    metadata.absolute_path,
                    chunk_size=chunk_size,
                    metadata=metadata,
                )

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_concurrency_uploads
            ) as executor:
                futures = [
                    executor.submit(upload_single_file, metadata)
                    for metadata in files_to_upload
                ]

                concurrent.futures.wait(futures)
                for metadata, future in zip(files_to_upload, futures):
                    if exc := future.exception():
                        errors.append(
                            AppFileUploadException(str(exc), metadata.relative_path)
                        )
                    else:
                        uploaded_files.append((metadata, future.result()))

        if flags.DEBUG:
            console.print("File Structure:")
            print_path_tree([m.relative_path for m in unique_files])

        return unique_files, errors

    def close(self):
        """Close HTTP client."""
        self._client.close()
