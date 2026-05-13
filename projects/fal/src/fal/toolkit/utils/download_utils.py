from __future__ import annotations

import errno
import hashlib
import os
import shutil
import subprocess
import sys
import time
from contextlib import suppress
from email.message import Message
from pathlib import Path, PurePath
from tempfile import TemporaryDirectory, mkstemp
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from fal.toolkit.utils.ssrf import (
    SafeResponse,
    SSRFError,
    SSRFHTTPStatusError,
    _ssrf_safe_get_to_file,
)

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


def _headers(request_headers: dict[str, str] | None = None) -> dict[str, str]:
    return {**TEMP_HEADERS, **(request_headers or {})}


def _filename_from_response(url: str, response: SafeResponse) -> str:
    content_disposition = response.headers.get("content-disposition", "")
    if content_disposition:
        message = Message()
        message["content-disposition"] = content_disposition
        filename = message.get_filename()
        if filename:
            return Path(filename).name

    parsed_url = urlparse(url)
    if parsed_url.scheme == "data":
        return _hash_url(url)

    return Path(parsed_url.path).name or _hash_url(url)


def _content_length_from_response(response: SafeResponse) -> int:
    try:
        return int(response.headers.get("content-length", -1))
    except ValueError:
        return -1


def _content_length_from_headers(headers: dict[str, str]) -> int:
    try:
        return int(headers.get("content-length", -1))
    except ValueError:
        return -1


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

    If the downloaded response resolves to an existing local file whose content length
    matches the downloaded file, the existing file is kept.

    If the file needs to be downloaded or if an existing file's content length does not
    match the expected length, the function proceeds to download and save the file. It
    ensures that the target directory exists and handles any errors that may occur
    during the download process, raising a `DownloadError` if necessary.

    Parameters:
        url: The URL of the file to be downloaded.
        target_dir: The directory where the downloaded file will be saved. If it's not
            an absolute path, it's treated as a relative directory to "/data".
        force: If `True`, the file is downloaded even if it already exists locally and
            its content length matches the downloaded response. Defaults to `False`.
        request_headers: A dictionary containing additional headers to be included in
            the HTTP request. Defaults to `None`.
        filesize_limit: An integer specifying the maximum downloadable size,
            in megabytes. Defaults to `None`.


    Returns:
        A Path object representing the full path to the downloaded file.

    Raises:
        DownloadError: If an error occurs during the download process.
    """
    ONE_MB = 1024**2
    parsed_url = urlparse(url)
    limit_bytes = filesize_limit * ONE_MB if filesize_limit is not None else None

    def raise_if_declared_size_exceeds_limit(headers: dict[str, str]) -> None:
        expected_filesize = _content_length_from_headers(headers)
        expected_filesize_mb = expected_filesize / ONE_MB

        if filesize_limit is not None and expected_filesize_mb > filesize_limit:
            raise DownloadError(
                f"""File to be downloaded is of size {expected_filesize_mb},
                    which is over the limit of {filesize_limit}"""
            )

    target_dir_path = Path(target_dir)

    # If target_dir is not an absolute path, use "/data" as the relative directory
    if not target_dir_path.is_absolute():
        target_dir_path = Path(FAL_PERSISTENT_DIR / target_dir_path)  # type: ignore[assignment]

    target_dir_path = target_dir_path.resolve()
    target_dir_path.mkdir(parents=True, exist_ok=True)
    target_path: Path | None = None

    fd, temp_file_path = mkstemp(dir=target_dir_path, prefix=".fal_download.tmp.")
    os.close(fd)
    temp_path = Path(temp_file_path)

    try:
        if parsed_url.scheme == "data":
            target_path = target_dir_path / _hash_url(url)
            _download_file_python(
                url=url,
                target_path=temp_path,
                request_headers=request_headers,
                filesize_limit=filesize_limit,
            )
            response = SafeResponse(
                200,
                headers={"content-length": str(temp_path.stat().st_size)},
            )
        else:
            response = _ssrf_safe_get_to_file(
                url,
                temp_path,
                headers=_headers(request_headers),
                max_size=limit_bytes,
                on_response_headers=raise_if_declared_size_exceeds_limit,
            )

        file_name = _filename_from_response(url, response)
        expected_filesize = _content_length_from_response(response)
        target_path = target_dir_path / file_name

        if (
            target_path.exists()
            and expected_filesize >= 0
            and target_path.stat().st_size == expected_filesize
            and not force
        ):
            temp_path.unlink(missing_ok=True)
            return target_path

        if force:
            print(f"File already exists. Forcing download of {url} to {target_path}")
        else:
            print(f"Downloading {url} to {target_path}")

        os.replace(temp_path, target_path)
    except DownloadError:
        temp_path.unlink(missing_ok=True)
        raise
    except SSRFHTTPStatusError as e:
        temp_path.unlink(missing_ok=True)
        raise DownloadError(f"Failed to get remote file properties for {url}") from e
    except SSRFError as e:
        temp_path.unlink(missing_ok=True)
        if str(e).startswith("File body exceeded"):
            error_target = target_path or target_dir_path
            raise DownloadError(f"Failed to download {url} to {error_target}") from e
        raise DownloadError(str(e)) from e
    except Exception as e:
        temp_path.unlink(missing_ok=True)
        error_target = target_path or target_dir_path
        raise DownloadError(f"Failed to download {url} to {error_target}") from e

    return target_path


def _download_file_python(
    url: str,
    target_path: Path | str,
    request_headers: dict[str, str] | None = None,
    filesize_limit: int | None = None,
) -> Path:
    """Download a non-HTTP URL and save it to a specified path.

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
    ONE_MB = 1024**2
    if urlparse(url).scheme in {"http", "https"}:
        raise DownloadError("HTTP(S) downloads must use ssrf_safe_get_to_file")

    target_path = Path(target_path)
    limit_bytes = filesize_limit * ONE_MB if filesize_limit is not None else None
    basename = target_path.name

    # NOTE: using the same directory to avoid potential copies across temp fs and target
    # fs, and also to be able to atomically rename a downloaded file into place.
    fd, temp_file_path = mkstemp(dir=target_path.parent, prefix=f"{basename}.tmp")
    os.close(fd)
    temp_path = Path(temp_file_path)
    try:
        request = Request(url, headers=_headers(request_headers))
        received_size = 0
        total_size = 0

        with urlopen(request, timeout=30) as response, open(
            temp_path, "wb"
        ) as temp_file:
            total_size = int(response.headers.get("content-length", total_size))
            while data := response.read(64 * ONE_MB):
                temp_file.write(data)

                received_size = temp_file.tell()
                if limit_bytes is not None and received_size > limit_bytes:
                    raise DownloadError(
                        f"""Attempted to download more data {received_size}
                            than the set limit of {limit_bytes}"""
                    )

                if total_size:
                    progress_msg = (
                        f"Downloading {url} ... {received_size / total_size:.2%}"
                    )
                else:
                    progress_msg = (
                        f"Downloading {url} ... {received_size / ONE_MB:.2f} MB"
                    )
                print(progress_msg, end="\r\n")

        if total_size and received_size < total_size:
            raise DownloadError("Received less data than expected from the server.")

        # NOTE: Atomically replacing the file when the download is complete.
        #
        # Since the file used is temporary, in a case of an interruption, the
        # downloaded content will be lost. So, it is safe to redownload the file in
        # such cases.
        os.replace(temp_path, target_path)

    finally:
        temp_path.unlink(missing_ok=True)

    return target_path


def _mark_used_dir(dir: Path):
    used_file = dir / ".fal_used"
    day_ago = time.time() - 86400
    if not used_file.exists() or used_file.stat().st_mtime < day_ago:
        # Touch a last-used file to indicate that the weights have been used
        used_file.touch()


def download_model_weights(
    url: str,
    force: bool = False,
    request_headers: dict[str, str] | None = None,
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
