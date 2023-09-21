from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

import openapi_fal_rest.api.files.file_exists as file_exists_api
import openapi_fal_rest.models.file_spec as file_spec_model
from fal.rest_client import REST_CLIENT
from fal.toolkit import mainify

if sys.version_info >= (3, 11):
    from typing import Concatenate
else:
    from typing_extensions import Concatenate

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

import fal.flags as flags
from fal.api import FalServerlessError, InternalFalServerlessError, function

ArgsT = ParamSpec("ArgsT")
ReturnT = TypeVar("ReturnT")


class DownloadError(Exception):
    pass


def file_exists(
    file_path: Path | str, calculate_checksum: bool
) -> file_spec_model.FileSpec | None:
    file_str = str(file_path)
    response = file_exists_api.sync_detailed(
        file_str,
        client=REST_CLIENT,
        calculate_checksum=calculate_checksum,
    )

    if response.status_code == 200:
        return response.parsed  # type: ignore
    elif response.status_code == 404:
        return None
    else:
        raise InternalFalServerlessError(
            f"Failed to check if file exists: {response.status_code} {response.parsed}"
        )


@mainify
@dataclass
class FileSpec:
    path: Path
    size: int
    is_file: bool
    checksum_sha256: str | None = None
    checksum_md5: str | None = None

    @classmethod
    def from_model(cls, file_spec: file_spec_model.FileSpec) -> FileSpec:
        return cls(
            path=Path(file_spec.path),
            size=file_spec.size,
            is_file=file_spec.is_file,
            checksum_sha256=file_spec.checksum_sha256
            if file_spec.checksum_sha256
            else None,
            checksum_md5=file_spec.checksum_md5 if file_spec.checksum_md5 else None,
        )


def setup(
    file_path: Path | str,
    checksum_sha256: str | None = None,
    checksum_md5: str | None = None,
    force: bool = False,
    *,
    _func_name: str | None = None,
    **isolated_config: Any,
):
    file_path = Path(file_path)
    target_path = file_path.relative_to("/data")

    def wrapper(
        func: Callable[Concatenate[ArgsT], ReturnT]
    ) -> Callable[Concatenate[ArgsT], Path]:
        name = _func_name or func.__name__

        @wraps(func)
        def internal_wrapper(
            *args: ArgsT.args,
            **kwargs: ArgsT.kwargs,
        ) -> Path:
            checksum = bool(checksum_sha256 or checksum_md5)

            file = file_exists(target_path, calculate_checksum=checksum)

            if not file or force or flags.FORCE_SETUP:
                config = {
                    "machine_type": "S",
                    **isolated_config,
                }
                function(**config)(func)(*args, **kwargs)  # type: ignore

                file = file_exists(target_path, calculate_checksum=checksum)

            if not file:
                raise FalServerlessError(
                    f"Setup function {name} did not create expected path."
                )

            if checksum:
                if not file.is_file:
                    raise FalServerlessError(
                        f"Setup function {name} created a directory instead of a file."
                    )

                if checksum_sha256 and file.checksum_sha256 != checksum_sha256:
                    raise FalServerlessError(
                        f"Setup function {name} created file with unexpected SHA256 checksum.\n"
                        f"Expected {checksum_sha256} but got {file.checksum_sha256}"
                    )

                if checksum_md5 and file.checksum_md5 != checksum_md5:
                    raise FalServerlessError(
                        f"Setup function {name} created file with unexpected MD5 checksum.\n"
                        f"Expected {checksum_md5} but got {file.checksum_md5}"
                    )

            return Path(file.path)

        return internal_wrapper

    return wrapper


def download_file(
    url: str,
    target_location: str | Path,
    *,
    checksum_sha256: str | None = None,
    checksum_md5: str | None = None,
    force: bool = False,
    _func_name: str = "download_file",
):
    """
    Download a file from a given URL using specified download tool.

    Args:
        url: The URL of the file to be downloaded.
        check_location: The location to save the downloaded file, either as a
            string path or a Path object.
        checksum_sha256: SHA-256 checksum value to verify the downloaded file's
            integrity. Defaults to None.
        checksum_md5: MD5 checksum value to verify the downloaded file's
            integrity. Defaults to None.
        force: If True, force re-download even if the file already exists.
            Defaults to False.
        _func_name: Name of the function to use for debugging purposes.
            Defaults to 'download_file'.

    Raises:
        ValueError: If an unsupported download tool is specified.
        Exception: If any error occurs during the download process,
            including checksum validation failure.

    Returns:
        The path where the downloaded file has been saved.
    """
    target_path = Path(target_location)

    @setup(
        target_path,
        checksum_sha256,
        checksum_md5,
        force=force,
        # isolated configs
        requirements=["urllib3"],
        _func_name=_func_name,
    )
    def download():
        print(f"Downloading {url} to {target_path}")

        # Make sure the directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            _download_file_python(url=url, download_path=target_path)
        except Exception as e:
            msg = f"Failed to download {url} to {target_path}"

            target_path.unlink(missing_ok=True)

            raise DownloadError(msg) from e

    return download()


def _download_file_python(url: str, download_path: Path) -> Path:
    """Download a file from a given URL and save it to a specified path using a
    Python interface.

    Args:
        url: The URL of the file to be downloaded.
        download_path: The path where the downloaded file will be saved.

    Returns:
        The path where the downloaded file has been saved.
    """
    import shutil
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        try:
            file_path = temp_file.name

            for (progress, total_size) in _stream_url_data_to_file(url, temp_file.name):
                if total_size:
                    progress_msg = f"Downloading {url}... {progress:.2%}"
                else:
                    progress_msg = f"Downloading {url}... {progress:.2f} MB"
                print(progress_msg, end="\r\n")

            # Move the file when the file is downloaded completely. Since the
            # file used is temporary, in a case of an interruption, the downloaded
            # content will be lost. So, it is safe to redownload the file in such cases.
            shutil.move(file_path, download_path)

        except:
            Path(temp_file.name).unlink(missing_ok=True)
            raise

    return download_path


def _stream_url_data_to_file(url: str, file_path: str, chunk_size_in_mb: int = 64):
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

    Yields:
        A tuple containing two elements:
        - float: The progress of the download as a percentage (0.0 to 1.0) if
            the total size is known. Else, equals to the downloaded size in MB.
        - int: The total size of the downloaded content in bytes. If the total
            size is not known (e.g., the server doesn't provide a
            'content-length' header), the second element is 0.
    """
    from urllib.request import Request, urlopen

    # TODO: how can we randomize the user agent to avoid being blocked?
    TEMP_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:21.0) Gecko/20100101 Firefox/21.0"
    }
    ONE_MB = 1024**2

    request = Request(url, headers=TEMP_HEADERS)

    received_size = 0
    total_size = 0

    with urlopen(request) as response, open(file_path, "wb") as f_stream:
        total_size = int(response.headers.get("content-length", total_size))
        while data := response.read(chunk_size_in_mb * ONE_MB):
            f_stream.write(data)

            received_size = f_stream.tell()

            if total_size:
                progress = received_size / total_size
            else:
                progress = received_size / ONE_MB
            yield progress, total_size

    # Check if received size matches the expected total size
    if total_size and received_size < total_size:
        raise DownloadError("Received less data than expected from the server.")


def download_weights(
    url: str,
    *,
    checksum_sha256: str | None = None,
    checksum_md5: str | None = None,
    force: bool = False,
):
    """Download model weights from a given URL using a specified download tool
    and store them in a predefined persistent directory.

    Args:
        url: The URL from which to download the model weights.
        checksum_sha256: SHA-256 checksum value to verify the downloaded file's
            integrity. Defaults to None.
        checksum_md5: MD5 checksum value to verify the downloaded file's
            integrity. Defaults to None.
        force: If True, force re-download even if the file already exists.
            Defaults to False.

    Returns:
        The path to the downloaded model weights file in the temporary directory.

    """
    import hashlib

    url_id = hashlib.sha256(url.encode("utf-8")).hexdigest()
    # This is not a protected path, so the user may change stuff internally
    url_path = Path(f"/data/.fal/downloads/{url_id}")

    return download_file(
        url,
        url_path,
        checksum_sha256=checksum_sha256,
        checksum_md5=checksum_md5,
        force=force,
        _func_name="download_weights",
    )


def download_repo(
    https_url: str,
    *,
    commit_hash: str | None = None,
    local_repo_location: str | Path | None = None,
    checksum_sha256: str | None = None,
    checksum_md5: str | None = None,
    force: bool = False,
):
    """
    Download a repository from a given HTTPS URL and optionally a specific
    commit hash.

    Args:
        https_url: The HTTPS URL of the repository to be downloaded.
        commit_hash: Optional commit hash or reference to checkout after
            download. Defaults to None, which means the latest commit will be
            used.
        local_repo_location: The local directory where the repository will be
            cloned or saved. This can be specified as a string path or a Path
            object. Defaults to None, which will use the current working
            directory.
        checksum_sha256: SHA-256 checksum value to verify the integrity of the
            downloaded repository. Defaults to None, skipping checksum
            validation.
        checksum_md5: MD5 checksum value to verify the integrity of the
            downloaded repository. Defaults to None, skipping checksum
            validation.
        force: If True, force re-download or re-clone even if the repository
            already exists in the specified local directory. Defaults to False.

    Returns:
        The path where the downloaded repository has been cloned or saved.
    """

    if local_repo_location is None:
        repo_name = Path(https_url).stem
        local_repos_dir = Path("/data/repos")
        local_repo_location = local_repos_dir / repo_name

    local_repo_path = Path(local_repo_location)
    local_repo_path_str = str(local_repo_path)

    @setup(
        local_repo_path,
        checksum_sha256,
        checksum_md5,
        force=force,
        _func_name="download_repo",
    )
    def download():
        try:
            if local_repo_path.exists():
                print("Removing existing repository.")
                remove_command = ["rm", "-rf", local_repo_path_str]
                subprocess.run(remove_command, check=True)

            print(f"Downloading repository '{https_url}' to {local_repo_path}")
            clone_command = ["git", "clone", https_url, local_repo_path_str]
            subprocess.run(clone_command, check=True)

            if commit_hash:
                checkout_command = ["git", "checkout", commit_hash]
                subprocess.run(checkout_command, cwd=local_repo_path, check=True)

        except Exception as e:
            print(
                f"Failed to download repository '{https_url}' to '{local_repo_path}' .",
                e,
                sep="\n",
            )

            subprocess.run(["rm", "-rf", local_repo_path_str])

    output_path = download()
    return output_path
