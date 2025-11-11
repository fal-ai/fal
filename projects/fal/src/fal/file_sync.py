import concurrent.futures
import hashlib
import math
import os
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple

import httpx
from rich.tree import Tree
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

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
# This is set to 1 to prevent thread explosion with nested ThreadPoolExecutors
DEFAULT_CONCURRENCY_UPLOADS = 1
MULTIPART_CHUNK_SIZE = 10 * 1024 * 1024  # 10MB

MULTIPART_WORKERS = 10


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


def compute_file_hashes(file_path: Path, mode: int) -> Tuple[str, str]:
    sha256_hash = hashlib.sha256()
    md5_hash = hashlib.md5()

    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
            md5_hash.update(chunk)

    metadata_string = f"{mode}"
    sha256_hash.update(metadata_string.encode("utf-8"))

    return sha256_hash.hexdigest(), md5_hash.hexdigest()


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
    md5: str
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

        file_hash, file_md5 = compute_file_hashes(file_path, stat.st_mode)
        return FileMetadata(
            size=stat.st_size,
            mtime=stat.st_mtime,
            mode=stat.st_mode,
            hash=file_hash,
            md5=file_md5,
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


def _retry_if_server_or_network_error(exception: BaseException) -> bool:
    if isinstance(exception, httpx.RequestError):
        return True
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code >= 500
    return False


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
            timeout=httpx.Timeout(
                connect=30,
                read=4 * 60,
                write=5 * 60,
                pool=30,
            ),
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
        @retry(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=_retry_if_server_or_network_error,
            reraise=True,
        )
        def _check_with_retry():
            try:
                response = self._request(
                    "POST", "/files/missing_hashes", json={"hashes": hashes}
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if flags.DEBUG:
                    console.print(f"{CROSS_ICON} Failed to check hashes on server: {e}")
                raise

        try:
            return _check_with_retry()
        except Exception as e:
            raise FalServerlessException(
                f"Failed to verify file hashes with the backend after 5 attempts. "
                f"Please check your network connection and try again. Error: {e}"
            ) from e

    def _put_file_part(self, file_hash, lpath, upload_id, part_number, chunk_size):
        offset = (part_number - 1) * chunk_size
        with open(lpath, "rb") as fobj:
            fobj.seek(offset)
            chunk = fobj.read(chunk_size)
            response = self._request(
                "PUT",
                f"/files/app/multipart/{file_hash}/{upload_id}/{part_number}",
                files={"file_upload": (os.path.basename(lpath), chunk)},
            )
            data = response.json()
            return {
                "part_number": data["part_number"],
                "etag": data["etag"],
            }

    def _get_upload_status(self, file_hash: str, upload_id: str):
        response = self._request(
            "GET",
            f"/files/app/multipart/{file_hash}/{upload_id}/status",
        )
        return response.json()

    def _put_file_multipart(self, lpath, rpath, metadata: FileMetadata):
        upload_id = None
        max_completion_retries = 3

        try:
            response = self._request(
                "POST",
                f"/files/app/multipart/{metadata.hash}/initiate",
                json={
                    "hash": metadata.hash,
                    "mode": str(metadata.mode),
                    "mtime": str(metadata.mtime),
                    "size": str(metadata.size),
                },
            )
            upload_id = response.json()["upload_id"]

            num_parts = math.ceil(metadata.size / MULTIPART_CHUNK_SIZE)

            for retry_attempt in range(max_completion_retries):
                if retry_attempt == 0:
                    parts_to_upload = list(range(1, num_parts + 1))
                else:
                    if flags.DEBUG:
                        console.print(
                            f"Querying upload status (attempt {retry_attempt + 1})"
                        )

                    status = self._get_upload_status(metadata.hash, upload_id)
                    uploaded_part_numbers = {
                        p["part_number"] for p in status["uploaded_parts"]
                    }

                    parts_to_upload = [
                        pnum
                        for pnum in range(1, num_parts + 1)
                        if pnum not in uploaded_part_numbers
                    ]

                    if not parts_to_upload:
                        if flags.DEBUG:
                            console.print("All parts uploaded, retrying completion")
                    else:
                        if flags.DEBUG:
                            console.print(
                                f"Re-uploading {len(parts_to_upload)} "
                                f"missing parts: {parts_to_upload[:10]}"
                            )

                parts = []
                failed_parts = []

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=MULTIPART_WORKERS
                ) as executor:
                    futures = {}
                    for part_number in parts_to_upload:
                        future = executor.submit(
                            self._put_file_part,
                            metadata.hash,
                            lpath,
                            upload_id,
                            part_number,
                            MULTIPART_CHUNK_SIZE,
                        )
                        futures[future] = part_number

                    for future in concurrent.futures.as_completed(futures):
                        part_number = futures[future]
                        try:
                            result = future.result()
                            parts.append(result)
                        except Exception as e:
                            failed_parts.append((part_number, str(e)))
                            if flags.DEBUG:
                                console.print(
                                    f"Failed to upload part {part_number}: {e}"
                                )

                if failed_parts:
                    if retry_attempt == max_completion_retries - 1:
                        raise RuntimeError(
                            f"Failed to upload {len(failed_parts)} parts "
                            f"after {max_completion_retries} attempts"
                        )
                    continue

                status = self._get_upload_status(metadata.hash, upload_id)
                all_parts = [
                    {"part_number": p["part_number"], "etag": p["etag"]}
                    for p in status["uploaded_parts"]
                ]

                all_parts.sort(key=lambda p: p["part_number"])

                try:
                    response = self._request(
                        "POST",
                        f"/files/app/multipart/{metadata.hash}/{upload_id}/complete",
                        json={"parts": all_parts},
                    )
                    data = response.json()

                    if data["etag"] != metadata.md5:
                        raise RuntimeError(
                            f"MD5 mismatch on {rpath}: {data['etag']} != {metadata.md5}"
                        )
                    return

                except FalServerlessException as e:
                    error_msg = str(e).lower()

                    if "already exists" in error_msg:
                        if flags.DEBUG:
                            console.print(f"File {rpath} already exists, skipping")
                        return

                    if (
                        "missing" in error_msg
                        or "duplicate" in error_msg
                        or "non-contiguous" in error_msg
                    ):
                        if retry_attempt == max_completion_retries - 1:
                            raise RuntimeError(
                                f"Upload completion failed "
                                f"after {max_completion_retries} attempts: {e}"
                            )
                        if flags.DEBUG:
                            console.print(
                                f"Completion validation failed: {e}, retrying..."
                            )
                        continue

                    raise

            raise RuntimeError(
                f"Failed to complete upload after {max_completion_retries} attempts"
            )

        except FalServerlessException as e:
            if "already exists" in str(e).lower():
                if flags.DEBUG:
                    console.print(f"File {rpath} already exists, skipping")
                return

            if upload_id:
                try:
                    self._request(
                        "POST",
                        f"/files/app/multipart/{metadata.hash}/{upload_id}/cancel",
                    )
                except Exception as cancel_error:
                    if flags.DEBUG:
                        console.print(f"Failed to cancel upload: {cancel_error}")
            raise
        except Exception:
            if upload_id:
                try:
                    self._request(
                        "POST",
                        f"/files/app/multipart/{metadata.hash}/{upload_id}/cancel",
                    )
                except Exception:
                    pass
            raise

    def upload_file(
        self,
        file_path: str,
        metadata: FileMetadata,
    ) -> str:
        rpath = metadata.relative_path
        self._put_file_multipart(file_path, rpath, metadata)
        return rpath

    def _matches_patterns(self, relative_path: str, patterns: List[re.Pattern]) -> bool:
        """Check if a file matches any of the patterns."""
        return any(pattern.search(relative_path) for pattern in patterns)

    def sync_files(
        self,
        paths: List[str],
        max_concurrency_uploads: int = DEFAULT_CONCURRENCY_UPLOADS,
        files_ignore: List[re.Pattern] = [],
        files_context_dir: Optional[str] = None,
    ) -> Tuple[List[FileMetadata], List[AppFileUploadException]]:
        files = self.collect_files(paths, files_context_dir)

        # Filter out ignored files
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
                return self.upload_file(
                    metadata.absolute_path,
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
