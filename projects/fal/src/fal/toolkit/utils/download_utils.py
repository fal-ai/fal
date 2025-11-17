from __future__ import annotations

import asyncio
import base64
import binascii
import errno
import fnmatch
import hashlib
import json
import math
import mimetypes
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import suppress
from pathlib import Path, PurePath
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import TYPE_CHECKING, Any, Literal, Optional
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    import tqdm

FAL_PERSISTENT_DIR = PurePath("/data")
FAL_REPOSITORY_DIR = FAL_PERSISTENT_DIR / ".fal" / "repos"
FAL_MODEL_WEIGHTS_DIR = FAL_PERSISTENT_DIR / ".fal" / "model_weights"


# TODO: how can we randomize the user agent to avoid being blocked?
TEMP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; "
        "Intel Mac OS X 10.8; rv:21.0) Gecko/20100101 Firefox/21.0"
    ),
}


class DownloadError(Exception):
    pass


def _hash_url(url: str) -> str:
    """Hashes a URL using SHA-256.

    Args:
        url: The URL to be hashed.

    Returns:
        A string representing the hashed URL.
    """
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _get_remote_file_properties(
    url: str,
    request_headers: dict[str, str] | None = None,
) -> tuple[str, int]:
    """Retrieves the file name and content length of a remote file.

    This function sends an HTTP request to the remote URL and retrieves the
    "Content-Disposition" header and the "Content-Length" header from the response.
    The "Content-Disposition" header contains the file name of the remote file, while
    the "Content-Length" header contains the expected content length of the remote
    file.

    If the "Content-Disposition" header is not available, the function attempts to
    extract the file name from the URL's path component. If the URL does not contain a
    path component, the function generates a hashed value of the URL using SHA-256 and
    uses it as the file name.

    If the "Content-Length" header is not available, the function returns -1 as the
    content length.

    Args:
        url: The URL of the remote file.
        request_headers: A dictionary containing additional headers to be included in
            the HTTP request.

    Returns:
        A tuple containing the file name and the content length of the remote file.
    """
    headers = {**TEMP_HEADERS, **(request_headers or {})}
    request = Request(url, headers=headers)

    with urlopen(request) as response:
        file_name = response.headers.get_filename()
        content_length = int(response.headers.get("Content-Length", -1))

    if not file_name:
        parsed_url = urlparse(url)

        if parsed_url.scheme == "data":
            file_name = _hash_url(url)
        else:
            url_path = parsed_url.path
            file_name = Path(url_path).name or _hash_url(url)

    # file name can still contain a forward slash if the server returns a relative path
    file_name = Path(file_name).name

    return file_name, content_length


def _file_content_length_matches(
    url: str, file_path: Path, request_headers: dict[str, str] | None = None
) -> bool:
    """Check if the local file's content length matches the expected remote
    file's content length.

    This function compares the content length of a local file to the expected content
    length of a remote file, which is determined by sending an HTTP request to the
    remote URL. If the content lengths match, the function returns `True`, indicating
    that the local file's content matches the expected remote content. If the content
    lengths do not match or information about the content length is unavailable, the
    function returns `False`.

    Args:
        url: The URL of the remote file.
        file_path: The local path to the file being compared.
        request_headers: A dictionary containing additional headers to be included in
            the HTTP request.

    Returns:
        bool: `True` if the local file's content length matches the remote file's
            content length, `False` otherwise.
    """
    local_file_content_length = file_path.stat().st_size
    remote_file_content_length = _get_remote_file_properties(url, request_headers)[1]

    return local_file_content_length == remote_file_content_length


def download_file(
    url: str,
    target_dir: str | Path,
    *,
    force: bool = False,
    request_headers: dict[str, str] | None = None,
    filesize_limit: int | None = None,
) -> Path:
    """Downloads a file from the specified URL to the target directory.

    The function downloads the file from the given URL and saves it in the specified
    target directory, provided it is below the given filesize limit.

    It also checks whether the local file already exists and whether its content length
    matches the expected content length from the remote file. If the local file already
    exists and its content length matches the expected content length from the remote
    file, the existing file is returned without re-downloading it.

    If the file needs to be downloaded or if an existing file's content length does not
    match the expected length, the function proceeds to download and save the file. It
    ensures that the target directory exists and handles any errors that may occur
    during the download process, raising a `DownloadError` if necessary.

    Parameters:
        url: The URL of the file to be downloaded.
        target_dir: The directory where the downloaded file will be saved. If it's not
            an absolute path, it's treated as a relative directory to "/data".
        force: If `True`, the file is downloaded even if it already exists locally and
            its content length matches the expected content length from the remote file.
            Defaults to `False`.
        request_headers: A dictionary containing additional headers to be included in
            the HTTP request. Defaults to `None`.
        filesize_limit: An integer specifying the maximum downloadable size,
            in megabytes. Defaults to `None`.


    Returns:
        A Path object representing the full path to the downloaded file.

    Raises:
        ValueError: If the provided `file_name` contains a forward slash ('/').
        DownloadError: If an error occurs during the download process.
    """
    ONE_MB = 1024**2

    try:
        file_name, expected_filesize = _get_remote_file_properties(url, request_headers)
    except Exception as e:
        print(f"Got error: {e}")
        raise DownloadError(f"Failed to get remote file properties for {url}") from e

    expected_filesize_mb = expected_filesize / ONE_MB

    if filesize_limit is not None and expected_filesize_mb > filesize_limit:
        raise DownloadError(
            f"""File to be downloaded is of size {expected_filesize_mb},
                which is over the limit of {filesize_limit}"""
        )

    if "/" in file_name:
        raise ValueError(f"File name '{file_name}' cannot contain a slash.")

    target_dir_path = Path(target_dir)

    # If target_dir is not an absolute path, use "/data" as the relative directory
    if not target_dir_path.is_absolute():
        target_dir_path = Path(FAL_PERSISTENT_DIR / target_dir_path)  # type: ignore[assignment]

    target_path = target_dir_path.resolve() / file_name

    if (
        target_path.exists()
        and _file_content_length_matches(url, target_path, request_headers)
        and not force
    ):
        return target_path

    if force:
        print(f"File already exists. Forcing download of {url} to {target_path}")
    else:
        print(f"Downloading {url} to {target_path}")

    # Make sure the directory exists
    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        _download_file_python(
            url=url,
            target_path=target_path,
            request_headers=request_headers,
            filesize_limit=filesize_limit,
        )
    except Exception as e:
        msg = f"Failed to download {url} to {target_path}"

        target_path.unlink(missing_ok=True)

        raise DownloadError(msg) from e

    return target_path


def _download_file_python(
    url: str,
    target_path: Path | str,
    request_headers: dict[str, str] | None = None,
    filesize_limit: int | None = None,
) -> Path:
    """Download a file from a given URL and save it to a specified path using a
    Python interface.

    Args:
        url: The URL of the file to be downloaded.
        target_path: The path where the downloaded file will be saved.
        request_headers: A dictionary containing additional headers to be included in
            the HTTP request. Defaults to `None`.
        filesize_limit: A integer value specifying how many megabytes can be
            downloaded at maximum. Defaults to `None`.

    Returns:
        The path where the downloaded file has been saved.
    """
    basename = os.path.basename(target_path)
    # NOTE: using the same directory to avoid potential copies across temp fs and target
    # fs, and also to be able to atomically rename a downloaded file into place.
    with NamedTemporaryFile(
        delete=False,
        dir=os.path.dirname(target_path),
        prefix=f"{basename}.tmp",
    ) as temp_file:
        try:
            file_path = temp_file.name

            for progress, total_size in _stream_url_data_to_file(
                url,
                temp_file.name,
                request_headers=request_headers,
                filesize_limit=filesize_limit,
            ):
                if total_size:
                    progress_msg = f"Downloading {url} ... {progress:.2%}"
                else:
                    progress_msg = f"Downloading {url} ... {progress:.2f} MB"

                print(progress_msg, end="\r\n")

            # NOTE: Atomically renaming the file into place when the file is downloaded
            # completely.
            #
            # Since the file used is temporary, in a case of an interruption, the
            # downloaded content will be lost. So, it is safe to redownload the file in
            # such cases.
            os.rename(file_path, target_path)

        finally:
            Path(temp_file.name).unlink(missing_ok=True)

    return Path(target_path)


def _stream_url_data_to_file(
    url: str,
    file_path: str,
    chunk_size_in_mb: int = 64,
    request_headers: dict[str, str] | None = None,
    filesize_limit: int | None = None,
):
    """Download data from a URL and stream it to a file.

    Note:
        - This function sets a User-Agent header to mimic a web browser to
            prevent issues with some websites.
        - It downloads the file in chunks to save memory and ensures the file
            is only moved when the download is complete.

    Args:
        request: The Request object representing the URL to download from.
        file_path: The path to the file where the downloaded data will be saved.
        chunk_size_in_mb: The size of each download chunk in megabytes.
            Defaults to 64.
        request_headers: A dictionary containing additional headers to be included in
            the HTTP request. Defaults to `None`.
        filesize_limit: An integer specifying how many megabytes can be
            downloaded at maximum. Defaults to `None`.

    Yields:
        A tuple containing two elements:
        - float: The progress of the download as a percentage (0.0 to 1.0) if
            the total size is known. Else, equals to the downloaded size in MB.
        - int: The total size of the downloaded content in bytes. If the total
            size is not known (e.g., the server doesn't provide a
            'content-length' header), the second element is 0.
    """
    ONE_MB = 1024**2

    headers = {**TEMP_HEADERS, **(request_headers or {})}
    request = Request(url, headers=headers)

    received_size = 0
    total_size = 0

    with urlopen(request) as response, open(file_path, "wb") as f_stream:
        total_size = int(response.headers.get("content-length", total_size))
        while data := response.read(chunk_size_in_mb * ONE_MB):
            f_stream.write(data)

            received_size = f_stream.tell()
            if filesize_limit is not None and received_size > filesize_limit:
                raise DownloadError(
                    f"""Attempted to download more data {received_size}
                        than the set limit of {filesize_limit}"""
                )

            if total_size:
                progress = received_size / total_size
            else:
                progress = received_size / ONE_MB
            yield progress, total_size

    # Check if received size matches the expected total size
    if total_size and received_size < total_size:
        raise DownloadError("Received less data than expected from the server.")


def _mark_used_dir(dir: Path):
    used_file = dir / ".fal_used"
    day_ago = time.time() - 86400
    if not used_file.exists() or used_file.stat().st_mtime < day_ago:
        # Touch a last-used file to indicate that the weights have been used
        used_file.touch()


def download_model_weights(
    url: str, force: bool = False, request_headers: dict[str, str] | None = None
) -> Path:
    """Downloads model weights from the specified URL and saves them to a
    predefined directory.

    This function is specifically designed for downloading model weights and stores
    them in a predefined directory.

    It calls the `download_file` function with the provided
    URL and the target directory set to a pre-defined location for model weights.
    The downloaded model weights are saved in this directory, and the function returns
    the full path to the downloaded weights file.

    Args:
        url: The URL from which the model weights will be downloaded.
        force: If `True`, the model weights are downloaded even if they already exist
            locally and their content length matches the expected content length from
            the remote file. Defaults to `False`.
        request_headers: A dictionary containing additional headers to be included in
            the HTTP request. Defaults to `None`.

    Returns:
        A Path object representing the full path to the downloaded model weights.
    """
    # This is not a protected path, so the user may change stuff internally
    weights_dir = Path(FAL_MODEL_WEIGHTS_DIR / _hash_url(url))

    if weights_dir.exists() and not force:
        try:
            # TODO: sometimes the directory can hold multiple files
            # Example:
            # .fal/model_weights/00155dc2d9579360d577d1a87d31b52c21135c14a5f44fcbab36fbb8352f3e0d  # noqa: E501
            # We need to either not allow multiple files in the directory or
            # find the one that is the most recently used.
            weights_path = next(
                # Ignore .fal dotfiles since they are metadata files
                f
                for f in weights_dir.glob("*")
                if not f.name.startswith(".fal")
            )
            _mark_used_dir(weights_dir)
            return weights_path
        # The model weights directory is empty, so we need to download the weights
        except StopIteration:
            pass

    path = download_file(
        url,
        target_dir=weights_dir,
        force=force,
        request_headers=request_headers,
    )

    _mark_used_dir(weights_dir)

    return path


def clone_repository(
    https_url: str,
    *,
    commit_hash: str | None = None,
    target_dir: str | Path | None = None,
    repo_name: str | None = None,
    force: bool = False,
    include_to_path: bool = False,
) -> Path:
    """Clones a Git repository from the specified HTTPS URL into a local
    directory.

    This function clones a Git repository from the specified HTTPS URL into a local
    directory. It can also checkout a specific commit if the `commit_hash` is provided.

    If a custom `target_dir` or `repo_name` is not specified, a predefined directory is
    used for the target directory, and the repository name is determined from the URL.

    Args:
        https_url: The HTTPS URL of the Git repository to be cloned.
        commit_hash: The commit hash to checkout after cloning.
        target_dir: The directory where the repository will be saved.
            If not provided, a predefined directory is used.
        repo_name: The name to be used for the cloned repository directory.
            If not provided, the repository's name from the URL is used.
        force: If `True`, the repository is cloned even if it already exists locally
            and its commit hash matches the provided commit hash. Defaults to `False`.
        include_to_path: If `True`, the cloned repository is added to the `sys.path`.
            Defaults to `False`.

    Returns:
        A Path object representing the full path to the cloned Git repository.
    """
    target_dir = target_dir or FAL_REPOSITORY_DIR  # type: ignore[assignment]

    if repo_name is None:
        repo_name = Path(https_url).stem
        if commit_hash:
            if len(commit_hash) < 8:
                raise ValueError(f"Commit hash '{commit_hash}' is too short.")
            repo_name += f"-{commit_hash[:8]}"

    local_repo_path = Path(target_dir) / repo_name  # type: ignore[arg-type]

    if local_repo_path.exists():
        local_repo_commit_hash = _git_rev_parse(local_repo_path, "HEAD")
        full_commit_hash = (
            _git_rev_parse(local_repo_path, commit_hash) if commit_hash else None
        )
        if (
            full_commit_hash
            and local_repo_commit_hash == full_commit_hash
            and not force
        ):
            if include_to_path:
                __add_local_path_to_sys_path(local_repo_path)
            return local_repo_path
        else:
            if local_repo_commit_hash != commit_hash:
                print(
                    f"Local repository '{local_repo_path}' has a different commit hash "
                    f"({local_repo_commit_hash}) than the one provided ({commit_hash})."
                )
            elif force:
                print(
                    f"Local repository '{local_repo_path}' already exists. "
                    f"Forcing re-download."
                )
            print(f"Removing the existing repository: {local_repo_path} ")
            with TemporaryDirectory(
                dir=target_dir, suffix=f"{local_repo_path.name}.tmp.old"
            ) as tmp_dir:
                with suppress(FileNotFoundError):
                    # repository might be already deleted by another worker
                    os.rename(local_repo_path, tmp_dir)
                    # sometimes seeing FileNotFoundError even here on juicefs
                    shutil.rmtree(tmp_dir, ignore_errors=True)

    # NOTE: using the target_dir to be able to avoid potential copies across temp fs
    # and target fs, and also to be able to atomically rename repo_name dir into place
    # when we are done setting it up.
    os.makedirs(target_dir, exist_ok=True)  # type: ignore[arg-type]
    with TemporaryDirectory(
        dir=target_dir,
        suffix=f"{local_repo_path.name}.tmp",
    ) as temp_dir:
        try:
            print(f"Cloning the repository '{https_url}' .")

            # Clone with disabling the logs and advices for detached HEAD state.
            clone_command = [
                "git",
                "clone",
                "--recursive",
                https_url,
                temp_dir,
            ]
            subprocess.check_call(clone_command)

            if commit_hash:
                checkout_command = ["git", "checkout", commit_hash]
                subprocess.check_call(checkout_command, cwd=temp_dir)
                subprocess.check_call(
                    ["git", "submodule", "update", "--init", "--recursive"],
                    cwd=temp_dir,
                )

            # NOTE: Atomically renaming the repository directory into place when the
            # clone and checkout are done.
            try:
                os.rename(temp_dir, local_repo_path)
            except OSError as error:
                shutil.rmtree(temp_dir, ignore_errors=True)

                # someone beat us to it, assume it's good
                if error.errno != errno.ENOTEMPTY:
                    raise
                print(f"{local_repo_path} already exists, skipping rename")

        except Exception as error:
            print(f"{error}\nFailed to clone repository '{https_url}' .")
            raise error

    if include_to_path:
        __add_local_path_to_sys_path(local_repo_path)

    return local_repo_path


def __add_local_path_to_sys_path(local_path: Path | str):
    local_path_str = str(local_path)

    if local_path_str not in sys.path:
        sys.path.insert(0, local_path_str)


def _git_rev_parse(repo_path: Path, ref: str) -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", ref],
            cwd=repo_path,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as error:
        if "not a git repository" in error.output:
            print(f"Repository '{repo_path}' is not a git repository.")
            return ""

        print(
            f"{error}\nFailed to get the commit hash of the repository '{repo_path}' ."
        )
        raise error


# ============================================================================
# Async Download Functions
# ============================================================================


def _debug_print(*messages: Any) -> None:
    """Print debug messages with [debug] prefix if running in deployed fal app."""
    prefix = ""
    if os.environ.get("FAL_APP_NAME"):
        prefix = "[debug]"
    message_str = " ".join(str(message) for message in messages)
    if prefix:
        print(f"{prefix} {message_str}")
    else:
        print(message_str)


def _get_retry_delay(
    num_retry: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_type: Literal["exponential", "fixed"] = "exponential",
    jitter: bool = False,
) -> float:
    """Calculate retry delay with exponential backoff and optional jitter."""
    if backoff_type == "exponential":
        delay = min(base_delay * (2 ** (num_retry - 1)), max_delay)
    else:
        delay = min(base_delay, max_delay)

    if jitter:
        delay *= random.uniform(0.5, 1.5)

    return min(delay, max_delay)


async def download_file_async(
    url: str,
    output_path: Path | str | None = None,
    timeout: int | float = 60,
    max_retries: int = 3,
    max_size: int | None = None,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    headers: dict | None = None,
    follow_redirects: bool = True,
) -> bytes | Path:
    """
    Download a file asynchronously from a URL.
    
    This is a lightweight async downloader that can either return the file content
    as bytes (for in-memory processing) or save it to a specific file path.

    Args:
        url: The URL to download the file from
        output_path: Optional path to save the file to. If None, returns the file content as bytes
        timeout: Timeout for the request in seconds
        max_retries: Maximum number of retries for failed requests
        max_size: Maximum allowed file size in bytes. None means no limit
        base_delay: Base delay for retry attempts
        max_delay: Maximum delay for retry attempts
        headers: Optional additional headers for the request
        follow_redirects: Whether to follow redirects

    Returns:
        Path object pointing to the downloaded file if output_path is provided,
        otherwise returns the file content as bytes
        
    Raises:
        DownloadError: If the download fails after all retries
    """
    import httpx

    if url.startswith("data:"):
        try:
            content_str = url.split(",")[1]
            content = base64.b64decode(content_str)
        except Exception as e:
            raise DownloadError(f"Error decoding base64 data: {e.__class__.__name__} {str(e)}")

        if max_size and len(content) > max_size:
            raise DownloadError(
                f"File size ({len(content)} bytes) exceeds maximum allowed size ({max_size} bytes)"
            )

        if output_path:
            output_path = Path(output_path)
            # Create parent directories if they don't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save content to file
            with open(output_path, "wb") as f:
                f.write(content)
            return output_path

        return content

    if headers is None:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:21.0) "
                "Gecko/20100101 Firefox/21.0"
            ),
        }

    _debug_print(f"Downloading file from {url}")
    start_time = time.perf_counter()

    for num_retry in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(
                    url, headers=headers, follow_redirects=follow_redirects
                )
                response.raise_for_status()
                break

        except httpx.HTTPError as e:
            if num_retry < max_retries:
                delay = _get_retry_delay(
                    num_retry + 1,
                    base_delay,
                    max_delay,
                    backoff_type="exponential",
                    jitter=True,
                )

                if isinstance(e, httpx.HTTPStatusError):
                    status_code = e.response.status_code
                    # Only retry on 429 (too many requests) or 5xx (server errors)
                    if status_code != 429 and not (500 <= status_code < 600):
                        raise DownloadError(
                            f"Error downloading file from {url}"
                        ) from e

                _debug_print(
                    f"Error downloading file from {url}. Retrying attempt {num_retry + 1}/{max_retries} after {delay:.2f}s...\n",
                    traceback.format_exc(),
                )
                await asyncio.sleep(delay)
                continue

            raise DownloadError(f"Error downloading file from {url}") from e

        except httpx.InvalidURL as e:
            raise DownloadError(
                f"Error downloading file from {url}, URL too long"
            ) from e

    # Check file size before processing
    content = response.content
    if max_size and len(content) > max_size:
        raise DownloadError(
            f"File size ({len(content)} bytes) exceeds maximum allowed size ({max_size} bytes)"
        )

    if output_path:
        output_path = Path(output_path)
        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save content to file
        with open(output_path, "wb") as f:
            f.write(content)

        end_time = time.perf_counter()
        _debug_print(
            f"Time taken to download {url} to {output_path}: {end_time - start_time:.2f} seconds"
        )
        return output_path

    end_time = time.perf_counter()
    _debug_print(f"Time taken to download {url}: {end_time - start_time:.2f} seconds")
    # Return content as bytes
    return content


async def download_file_to_dir_async(
    url: str,
    target_dir: str | Path,
    file_name: str | None = None,
    timeout: int | None = 120,
    max_size: int | None = None,
    force: bool = False,
    headers: dict[str, str] | None = None,
    use_tqdm: bool = False,
    num_retries: int = 3,
    retry_delay: float = 0.0,
    disallowed_file_types: set[str] | None = None,
    validate_extension: bool = True,
) -> str:
    """
    Download a file from a URL and save it to a directory.
    
    This is the production-grade async downloader with automatic filename detection,
    MIME validation, token injection for HuggingFace/Civitai, and comprehensive
    error handling.
    
    :param url: The URL to download the file from.
    :param target_dir: The directory to save the downloaded file.
    :param file_name: Optional filename. If None, auto-detected from headers or URL.
    :param timeout: The timeout for the download request.
    :param max_size: The maximum size of the file to download in bytes.
    :param force: Whether to force download the file even if it already exists.
    :param headers: Additional headers to include in the download request.
    :param use_tqdm: Whether to show a progress bar during download.
    :param num_retries: Number of retry attempts on failure.
    :param retry_delay: Delay between retries in seconds.
    :param disallowed_file_types: Set of MIME types that are not allowed (e.g., text/html).
    :param validate_extension: Whether to validate and correct file extension based on MIME type.
    :return: The path to the downloaded file.
    
    Raises:
        DownloadError: If the download fails after all retries
    """
    import httpx

    if disallowed_file_types is None:
        disallowed_file_types = {"text/html"}

    if headers is None:
        headers = {}
    else:
        # Standardize headers to lowercase
        headers = {header.lower(): value for header, value in headers.items()}

    if not (url.startswith("http://") or url.startswith("https://")):
        if url.startswith("data:"):
            # Support data URLs - these are not downloaded but parsed directly
            # Get the content type and data from the URL
            data_parts = url.split(",", 1)
            if len(data_parts) != 2:
                raise DownloadError("Invalid data URL format.")

            content_type = data_parts[0].split(":", 1)[1].split(";")[0]
            data = data_parts[1]

            if content_type in disallowed_file_types:
                raise DownloadError(
                    f"File type {content_type} is not allowed for download."
                )

            if not file_name:
                # Make a default file name based on the content type
                extension = mimetypes.guess_extension(content_type)

                if not extension:
                    extension = ".bin"

                # If MIME-type is of format media/ext, use the media type as the file name
                if "/" in content_type:
                    media_type = content_type.split("/")[0]
                    file_name = f"{media_type}{extension}"
                else:
                    file_name = f"file{extension}"

            # Decode the data part of the URL
            try:
                file_data = base64.b64decode(data)
            except binascii.Error as e:
                raise DownloadError(f"Invalid base64 data in URL: {e}")

            # Ensure the target directory exists
            target_dir = Path(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)

            # Write the data to the file
            target_file_path = target_dir / file_name
            with open(target_file_path, "wb") as f:
                f.write(file_data)

            return str(target_file_path)

        # Unsupported URL scheme (e.g., blob:)
        raise DownloadError("Unsupported URL scheme.")

    parsed_url = urlparse(url)
    parsed_domain = parsed_url.netloc.split(":")[0]
    parsed_qs = parse_qs(parsed_url.query)

    if "huggingface.co" in parsed_domain:
        url = url.replace("/blob/", "/resolve/")
        token = os.getenv("HF_TOKEN", None)
        if token:
            headers["authorization"] = f"Bearer {token}"
    elif "civitai.com" in parsed_domain:
        token = os.getenv("CIVITAI_TOKEN", None)
        if token:
            headers["authorization"] = f"Bearer {token}"

    if "user-agent" not in headers:
        headers[
            "user-agent"
        ] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"

    # Get details of the file to be downloaded
    if file_name is None:
        if "response-content-disposition" in parsed_qs:
            # This is commonly used for Civitai
            disposition_parts = parsed_qs["response-content-disposition"][0].split(";")
            for part in disposition_parts:
                part_data = part.strip("'\" ").split("=")
                if len(part_data) >= 2 and part_data[0] == "filename":
                    file_name = "=".join(part_data[1:]).strip("'\" ")
                    break

    file_bytes = 0
    file_type = "application/octet-stream"

    def get_file_name_from_response(response: httpx.Response) -> str | None:
        """
        Given a response, attempt to extract a file name from it.
        """
        if response.headers.get("Content-Disposition", None) is not None:
            file_name_parts = response.headers["Content-Disposition"].split(";")
            for file_name_part in file_name_parts:
                if "=" in file_name_part:
                    key, _, value = file_name_part.partition("=")
                    if key.lower().strip() == "filename":
                        return value.strip("\"' ")
        return None

    # Try to get details about the file before downloading it
    async with httpx.AsyncClient() as client:
        response: httpx.Response | None = None
        try:
            if (
                "civitai.com" in parsed_domain
                or "cloudflarestorage.com" in parsed_domain
            ):
                # these don't support HEAD requests, go straight to fallback
                raise DownloadError("HEAD request not supported, falling back to GET")

            response = await client.head(
                url, headers=headers, timeout=timeout, follow_redirects=True
            )
            response.raise_for_status()

        except (httpx.HTTPStatusError, DownloadError) as e:
            # Couldn't get HEAD, possible the web server does not support it. Check the code.
            status_code = 500
            if isinstance(e, httpx.HTTPStatusError):
                status_code = e.response.status_code
            else:
                status_code = getattr(e, "status_code", 500)
            if status_code == 404:
                # The server supports HEAD _and_ the file is not found. Raise here.
                raise DownloadError(f"File not found at {url}.") from e

        is_default_file_name = False
        if response is not None:
            file_bytes = int(response.headers.get("Content-Length", 0))
            file_type = response.headers.get("Content-Type", "application/octet-stream")
            header_file_name = get_file_name_from_response(response)
            if header_file_name and not file_name:
                file_name = header_file_name

        # If the filename is missing or explicitly set to "attachment" (a common default in HTTP headers),
        # use the last part of the URL as the default filename.
        if not file_name:
            file_name = os.path.basename(parsed_url.path)
            is_default_file_name = True

        # Determine the target file path
        target_file = os.path.join(str(target_dir), file_name)

        # Check if the file already exists
        if os.path.exists(target_file) and file_bytes > 0 and not force:
            # Check if the file size matches
            if os.path.getsize(target_file) == file_bytes:
                # File already exists and size matches, no need to download
                return target_file

        # The file doesn't exist, or the size doesn't match, or force is True
        # If we know the file size and have a limit, check if it exceeds the limit
        if file_bytes > 0 and max_size is not None and file_bytes > max_size:
            raise DownloadError(
                f"File size {file_bytes} exceeds the maximum limit of {max_size} bytes."
            )

        # Check if the file type is known bad
        if file_type in disallowed_file_types:
            raise DownloadError(f"File type {file_type} is not allowed for download.")

        try:
            # Now we can proceed to download the file
            async with client.stream(
                "GET",
                url,
                headers=headers,
                timeout=timeout,
                follow_redirects=True,
            ) as response:
                response.raise_for_status()
                # Do another check on size and location before writing the file now that we've done a GET
                file_bytes = int(response.headers.get("Content-Length", 0))
                file_type = response.headers.get(
                    "Content-Type", "application/octet-stream"
                )
                header_file_name = get_file_name_from_response(response)
                if header_file_name is not None and is_default_file_name:
                    file_name = header_file_name

                    # The file name might have changed, so we may need to update the target file path
                    this_target_file = os.path.join(str(target_dir), file_name)  # type: ignore[arg-type]
                    if this_target_file != target_file:
                        # Do size check again
                        if (
                            os.path.exists(this_target_file)
                            and file_bytes > 0
                            and not force
                        ):
                            if os.path.getsize(this_target_file) == file_bytes:
                                # File already exists and size matches, no need to download
                                return this_target_file
                        elif (
                            file_bytes > 0
                            and max_size is not None
                            and file_bytes > max_size
                        ):
                            raise DownloadError(
                                f"File size {file_bytes} exceeds the maximum limit of {max_size} bytes."
                            )
                        # Check if the file type is known bad
                        if file_type in disallowed_file_types:
                            raise DownloadError(
                                f"File type {file_type} is not allowed for download."
                            )

                        # Checks passed, update target_file
                        target_file = this_target_file

                # Ensure the target directory exists
                os.makedirs(os.path.dirname(target_file), exist_ok=True)

                # Ensure the file has a proper extension if needed
                file_type_parts = file_type.lower().split("/")
                if (
                    file_type_parts[0] in ["audio", "image", "video"]
                    and len(file_type_parts) > 1
                    and validate_extension
                ):
                    # Check for extension, strip out any extra parameters
                    # For example, "image/png; charset=utf-8" should become ".png"
                    valid_extensions = mimetypes.guess_all_extensions(file_type.lower())
                    base_name, ext = os.path.splitext(target_file)

                    if len(valid_extensions) > 0 and ext not in valid_extensions:
                        _debug_print(
                            f"Incorrect extension '{ext}' found for content type {file_type}, replacing with '{valid_extensions[0]}'"
                        )
                        target_file = f"{base_name}{valid_extensions[0]}"

                # Write the file to the target location
                with open(target_file, "wb") as f:
                    if use_tqdm:
                        try:
                            import tqdm as tqdm_module

                            progress_bar = tqdm_module.tqdm(
                                total=file_bytes,
                                unit="B",
                                unit_scale=True,
                                desc=f"Downloading {file_name}",
                                mininterval=1.0,
                            )
                        except ImportError:
                            progress_bar = None
                    else:
                        progress_bar = None

                    async for chunk in response.aiter_bytes():
                        f.write(chunk)
                        if progress_bar:
                            progress_bar.update(len(chunk))

                    if progress_bar:
                        progress_bar.close()

                return target_file

        except Exception as e:
            if (
                isinstance(e, httpx.HTTPStatusError)
                and response is not None
                and response.status_code == 404
            ):
                raise DownloadError(f"File not found at {url}.") from e

            elif num_retries > 0:
                _debug_print(
                    f"Download failed with status {response.status_code if response else 'UNK'}. Retrying..."
                )
                return await download_file_to_dir_async(
                    url,
                    target_dir,
                    file_name=file_name,
                    timeout=timeout,
                    max_size=max_size,
                    force=force,
                    headers=headers,
                    use_tqdm=use_tqdm,
                    num_retries=num_retries - 1,
                    retry_delay=retry_delay,
                    disallowed_file_types=disallowed_file_types,
                    validate_extension=validate_extension,
                )

            raise DownloadError(
                f"Failed to download file from {url}. Status code: {response.status_code if response else 'UNK'}."
            ) from e


async def download_model_weights_async(
    url: str,
    force: bool = False,
    timeout: int | None = 120,
    max_size: int | None = None,
    headers: dict[str, str] | None = None,
    disallowed_file_types: set[str] | None = None,
    lora: bool = False,
    use_tqdm: bool = False,
    num_retries: int = 3,
    retry_delay: float = 0.0,
) -> str:
    """
    Download model weights from a URL and save them to shared storage.

    This will try to be intelligent about whether or not it needs to download the file,
    and will check file size when possible to detect changes in a remote resource.

    :param url: The URL to download the model weights from.
    :param force: Whether to force download the model weights even if they already exist.
    :param timeout: The timeout for the download request.
    :param max_size: The maximum size of the file to download in bytes.
    :param headers: Additional headers to include in the download request.
    :param disallowed_file_types: A set of file types that are not allowed to be downloaded.
    :param lora: Whether the model is a LoRA model. This is kept for legacy reasons.
    :param use_tqdm: Whether to show a progress bar during download.
    :param num_retries: Number of retry attempts on failure.
    :param retry_delay: Delay between retries in seconds.
    :return: The path to the downloaded file.
    
    Raises:
        DownloadError: If the download fails after all retries
    """
    if disallowed_file_types is None:
        disallowed_file_types = {"text/html"}
    
    if lora and max_size is None:
        max_size = 1750 * 1024 * 1024  # 1.75 GB, the historical maximum size for LoRA

    target_file = await download_file_to_dir_async(
        url=url,
        target_dir=FAL_MODEL_WEIGHTS_DIR / _hash_url(url),  # type: ignore[arg-type]
        file_name=None,  # Let the function determine the file name
        timeout=timeout,
        max_size=max_size,
        force=force,
        headers=headers,
        use_tqdm=use_tqdm,
        disallowed_file_types=disallowed_file_types,
        num_retries=num_retries,
        retry_delay=retry_delay,
    )
    return target_file


async def _download_file_parallel_async(
    url: str,
    output_path: str | Path,
    max_workers: int = 8,
    target_chunk_size: int = 128 * 1024 * 1024,  # 128 MB default
    timeout: int | None = 120,
    headers: dict[str, str] | None = None,
    num_retries: int = 3,
    retry_delay: float = 0.0,
    use_tqdm: bool = False,
    progress_bar: Optional["tqdm.tqdm"] = None,
) -> str:
    """
    Download a file in parallel using range requests (internal helper).

    This function splits the download into multiple chunks and downloads them concurrently
    using HTTP range requests. It first gets the file size, then calculates optimal chunk
    sizes and worker counts based on the target chunk size and max workers parameters.

    :param url: The URL to download the file from.
    :param output_path: The path where the file should be saved.
    :param max_workers: Maximum number of concurrent workers.
    :param target_chunk_size: Target size per worker in bytes. Default 128 MB.
    :param timeout: The timeout for each download request.
    :param headers: Additional headers to include in the download request.
    :param num_retries: Number of retries for failed chunks.
    :param retry_delay: Delay between retries in seconds.
    :param use_tqdm: Whether to show progress bar.
    :param progress_bar: Optional existing progress bar to update.
    :return: The path to the downloaded file.
    """
    import httpx

    if headers is None:
        headers = {}
    else:
        # Standardize headers to lowercase
        headers = {header.lower(): value for header, value in headers.items()}

    parsed_url = urlparse(url)
    parsed_domain = parsed_url.netloc.split(":")[0]

    # Handle special domains
    if "huggingface.co" in parsed_domain:
        url = url.replace("/blob/", "/resolve/")
        token = os.getenv("HF_TOKEN", None)
        if token:
            headers["authorization"] = f"Bearer {token}"
    elif "civitai.com" in parsed_domain:
        token = os.getenv("CIVITAI_TOKEN", None)
        if token:
            headers["authorization"] = f"Bearer {token}"

    if "user-agent" not in headers:
        headers[
            "user-agent"
        ] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"

    # Get file size first
    async with httpx.AsyncClient() as client:
        try:
            response = await client.head(
                url, headers=headers, timeout=timeout, follow_redirects=True
            )
            response.raise_for_status()
            file_size = int(response.headers.get("Content-Length", 0))

            if file_size == 0:
                # Can't chunk. Fall back to regular download
                with tempfile.TemporaryDirectory() as temp_dir:
                    downloaded_path = await download_file_to_dir_async(
                        url,
                        temp_dir,
                        timeout=timeout,
                        headers=headers,
                        num_retries=num_retries,
                    )
                    shutil.move(downloaded_path, output_path)

                if progress_bar:
                    progress_bar.update(os.path.getsize(str(output_path)))
                    progress_bar.close()

                return str(output_path)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise DownloadError(f"File not found at {url}.")
            raise DownloadError(f"Failed to get file size: {e}") from e

    # Calculate optimal number of workers and chunk size
    file_chunks = math.ceil(file_size / target_chunk_size)
    num_workers = min(max_workers, file_chunks)

    semaphore = asyncio.Semaphore(num_workers)

    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-allocate the file with the correct size
    with open(output_path, "wb") as f:
        f.seek(file_size - 1)
        f.write(b"\0")

    # Create a lock for thread-safe file writing
    file_lock = asyncio.Lock()

    # Define chunk download function
    async def download_chunk(start_byte: int, end_byte: int, chunk_index: int) -> int:
        """Download a specific byte range from the URL and write it to disk."""
        range_header = f"bytes={start_byte}-{end_byte}"
        chunk_headers = {**headers, "range": range_header}

        for attempt in range(num_retries + 1):
            try:
                async with semaphore:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(
                            url,
                            headers=chunk_headers,
                            timeout=timeout,
                            follow_redirects=True,
                        )
                        response.raise_for_status()

                        chunk_data = response.content
                        expected_size = end_byte - start_byte + 1

                        if len(chunk_data) != expected_size:
                            raise DownloadError(
                                f"Chunk {chunk_index} size mismatch: expected {expected_size}, got {len(chunk_data)}"
                            )

                        # Write chunk to disk at the correct position
                        async with file_lock:
                            with open(output_path, "r+b") as f:
                                f.seek(start_byte)
                                f.write(chunk_data)

                        return len(chunk_data)

            except Exception as e:
                if attempt == num_retries:
                    raise DownloadError(
                        f"Failed to download chunk {chunk_index} after {num_retries} retries: {e}"
                    ) from e

                if retry_delay > 0:
                    await asyncio.sleep(retry_delay)
                continue

        return 0

    # Create download tasks
    tasks = []
    for i in range(file_chunks):
        start_byte = i * target_chunk_size
        end_byte = start_byte + target_chunk_size - 1

        # Adjust the last chunk to include any remaining bytes
        if i == file_chunks - 1:
            end_byte = file_size - 1

        task = download_chunk(start_byte, end_byte, i)
        tasks.append(asyncio.create_task(task))

    # Execute all downloads concurrently
    if use_tqdm and not progress_bar:
        try:
            import tqdm as tqdm_module

            progress_bar = tqdm_module.tqdm(
                total=file_size,
                desc=f"Downloading {os.path.basename(str(output_path))}",
                unit="B",
                unit_scale=True,
            )
        except ImportError:
            progress_bar = None

    try:
        # Process downloads as they complete and update progress
        for result in asyncio.as_completed(tasks):
            chunk_size = await result
            if progress_bar:
                progress_bar.update(chunk_size)

        if progress_bar:
            progress_bar.close()

        return str(output_path)

    except Exception as e:
        # Clean up partial file if it exists
        if output_path.exists():
            output_path.unlink()
        raise DownloadError(f"Failed to download file: {e}") from e


async def snapshot_download_async(
    repo_id: str,
    local_dir: str | Path,
    revision: str | None = None,
    max_jobs: int = 4,
    max_workers: int = 8,
    target_chunk_size: int = 128 * 1024 * 1024,  # 128 MB default
    timeout: int | None = 120,
    headers: dict[str, str] | None = None,
    num_retries: int = 3,
    retry_delay: float = 0.0,
    use_tqdm: bool = False,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> str:
    """
    Download a snapshot from a HuggingFace repository asynchronously.
    
    This function downloads all files from a HuggingFace repository, using parallel
    downloads for efficiency. Each file is downloaded with parallel chunk downloading
    for maximum speed.

    :param repo_id: The HuggingFace repository ID (e.g., "black-forest-labs/FLUX.1-dev")
    :param local_dir: The local directory to save the repository files
    :param revision: Optional git revision (branch, tag, or commit hash)
    :param max_jobs: Maximum number of files to download concurrently
    :param max_workers: Maximum number of parallel chunks per file
    :param target_chunk_size: Target size of each chunk in bytes
    :param timeout: Timeout for each request
    :param headers: Optional additional headers
    :param num_retries: Number of retry attempts
    :param retry_delay: Delay between retries
    :param use_tqdm: Whether to show progress bars
    :param allow_patterns: List of file patterns to include (e.g., ["*.safetensors"])
    :param ignore_patterns: List of file patterns to exclude (e.g., ["*.md"])
    :return: Path to the local directory containing the downloaded files
    
    Raises:
        DownloadError: If the download fails after all retries
    """
    from huggingface_hub import get_paths_info, hf_hub_url, list_repo_files

    all_files = list_repo_files(repo_id, revision=revision)
    target_files = []

    for filename in all_files:
        if ignore_patterns is not None:
            skip = False
            for pattern in ignore_patterns:
                if fnmatch.fnmatch(filename, pattern):
                    skip = True
                    break
            if skip:
                continue

        if allow_patterns is not None:
            is_allowed = False
            for pattern in allow_patterns:
                if fnmatch.fnmatch(filename, pattern):
                    is_allowed = True
                    break
            if not is_allowed:
                continue
        target_files.append(filename)

    path_infos = get_paths_info(repo_id, target_files, revision=revision)
    download_files = []

    local_dir_path = Path(local_dir)
    local_dir_path.mkdir(parents=True, exist_ok=True)
    local_dir_sha256_file = local_dir_path / ".map.sha256.json"

    if local_dir_sha256_file.exists():
        with open(local_dir_sha256_file) as f:
            sha256_map = json.load(f)
    else:
        sha256_map = {}

    for filename, path_info in zip(target_files, path_infos):
        path = local_dir_path / filename
        expected_size = path_info.size

        try:
            expected_sha256 = path_info.lfs.sha256
        except AttributeError:
            expected_sha256 = None  # not LFS

        if os.path.exists(path):
            if expected_sha256 is not None:
                if expected_sha256 == sha256_map.get(str(path), None):
                    continue
            elif expected_size == os.path.getsize(path):
                continue
            os.remove(path)

        sha256_map[str(path)] = expected_sha256
        url = hf_hub_url(repo_id, filename, revision=revision)
        download_files.append((filename, url, path, expected_size))

    if not download_files:
        # Nothing to download
        return str(local_dir)

    if use_tqdm:
        try:
            import tqdm as tqdm_module

            progress_bars = [
                tqdm_module.tqdm(
                    total=expected_size,
                    desc=f"Downloading {filename}",
                    unit="B",
                    unit_scale=True,
                )
                for filename, url, path, expected_size in download_files
            ]
        except ImportError:
            progress_bars = None
    else:
        progress_bars = None

    semaphore = asyncio.Semaphore(max_jobs)

    async def download_file(
        url: str, path: str, progress_bar: Optional["tqdm.tqdm"]
    ) -> None:
        async with semaphore:
            await _download_file_parallel_async(
                url,
                path,
                max_workers=max_workers,
                target_chunk_size=target_chunk_size,
                timeout=timeout,
                headers=headers,
                num_retries=num_retries,
                retry_delay=retry_delay,
                use_tqdm=use_tqdm,
                progress_bar=progress_bar,
            )

    tasks = []
    for i, (filename, url, path, expected_size) in enumerate(download_files):
        progress_bar_item = progress_bars[i] if progress_bars else None
        coro = download_file(url, str(path), progress_bar_item)
        tasks.append(asyncio.create_task(coro))

    for task in asyncio.as_completed(tasks):
        await task

    with open(local_dir_sha256_file, "w") as f:
        json.dump(sha256_map, f)

    return str(local_dir)
