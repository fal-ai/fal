from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path, PurePath
from tempfile import TemporaryDirectory
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from fal.toolkit.mainify import mainify

FAL_PERSISTENT_DIR = PurePath("/data")
FAL_REPOSITORY_DIR = FAL_PERSISTENT_DIR / ".fal" / "repos"
FAL_MODEL_WEIGHTS_DIR = FAL_PERSISTENT_DIR / ".fal" / "model_weights"


# TODO: how can we randomize the user agent to avoid being blocked?
TEMP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:21.0) Gecko/20100101 Firefox/21.0"
}


@mainify
class DownloadError(Exception):
    pass


@mainify
def _hash_url(url: str) -> str:
    """Hashes a URL using SHA-256.

    Args:
        url: The URL to be hashed.

    Returns:
        A string representing the hashed URL.
    """
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


@mainify
@lru_cache()
def _get_remote_file_properties(url: str) -> tuple[str, int]:
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

    Returns:
        A tuple containing the file name and the content length of the remote file.
    """
    request = Request(url, headers=TEMP_HEADERS)
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

    return file_name, content_length


@mainify
def _file_content_length_matches(url: str, file_path: Path) -> bool:
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

    Returns:
        bool:`True` if the local file's content length matches the remote file's content length,
              `False` otherwise.
    """
    local_file_content_length = file_path.stat().st_size
    remote_file_content_length = _get_remote_file_properties(url)[1]

    return local_file_content_length == remote_file_content_length


@mainify
def download_file(
    url: str,
    target_dir: str | Path,
    *,
    force: bool = False,
) -> Path:
    """Downloads a file from the specified URL to the target directory.

    The function downloads the file from the given URL and saves it in the specified
    target directory.

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

    Returns:
        A Path object representing the full path to the downloaded file.

    Raises:
        ValueError: If the provided `file_name` contains a forward slash ('/').
        DownloadError: If an error occurs during the download process.
    """
    file_name = _get_remote_file_properties(url)[0]
    if "/" in file_name:
        raise ValueError(f"File name '{file_name}' cannot contain a slash.")

    target_dir_path = Path(target_dir)

    # If target_dir is not an absolute path, use "/data" as the relative directory
    if not target_dir_path.is_absolute():
        target_dir_path = FAL_PERSISTENT_DIR / target_dir_path  # type: ignore[assignment]

    target_path = target_dir_path.resolve() / file_name

    if (
        target_path.exists()
        and _file_content_length_matches(url, target_path)
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
        _download_file_python(url=url, target_path=target_path)
    except Exception as e:
        msg = f"Failed to download {url} to {target_path}"

        target_path.unlink(missing_ok=True)

        raise DownloadError(msg) from e

    return target_path


@mainify
def _download_file_python(url: str, target_path: Path | str) -> Path:
    """Download a file from a given URL and save it to a specified path using a
    Python interface.

    Args:
        url: The URL of the file to be downloaded.
        target_path: The path where the downloaded file will be saved.

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
                    progress_msg = f"Downloading {url} ... {progress:.2%}"
                else:
                    progress_msg = f"Downloading {url} ... {progress:.2f} MB"
                print(progress_msg, end="\r\n")

            # Move the file when the file is downloaded completely. Since the
            # file used is temporary, in a case of an interruption, the downloaded
            # content will be lost. So, it is safe to redownload the file in such cases.
            shutil.move(file_path, target_path)

        except Exception as error:
            raise error
        finally:
            Path(temp_file.name).unlink(missing_ok=True)

    return Path(target_path)


@mainify
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


@mainify
def download_model_weights(url: str, force: bool = False):
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

    Returns:
        A Path object representing the full path to the downloaded model weights.
    """
    # This is not a protected path, so the user may change stuff internally
    weights_dir = Path(FAL_MODEL_WEIGHTS_DIR / _hash_url(url))

    if weights_dir.exists() and not force:
        try:
            weights_path = next(weights_dir.glob("*"))
            return weights_path
        # The model weights directory is empty, so we need to download the weights
        except StopIteration:
            pass

    return download_file(
        url,
        target_dir=weights_dir,
        force=force,
    )


@mainify
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
    repo_name = repo_name or Path(https_url).stem

    local_repo_path = Path(target_dir) / repo_name  # type: ignore[arg-type]

    if local_repo_path.exists():
        local_repo_commit_hash = _get_git_revision_hash(local_repo_path)
        if local_repo_commit_hash == commit_hash and not force:
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
            shutil.rmtree(local_repo_path)

    # Temporary directory to download the repo into.
    temp_dir = TemporaryDirectory()
    temp_dir_path = temp_dir.name

    try:
        print(f"Cloning the repository '{https_url}' .")

        # Clone with disabling the logs and advices for detached HEAD state.
        clone_command = [
            "git",
            "clone",
            "--recursive",
            https_url,
            temp_dir_path,
        ]
        subprocess.check_call(clone_command)

        if commit_hash:
            checkout_command = ["git", "checkout", commit_hash]
            subprocess.check_call(checkout_command, cwd=temp_dir_path)

        # Move the repository when the clone and checkout is finished.
        shutil.move(temp_dir_path, local_repo_path)

    except Exception as error:
        print(f"{error}\nFailed to clone repository '{https_url}' .")
        temp_dir.cleanup()

        raise error

    if include_to_path:
        __add_local_path_to_sys_path(local_repo_path)

    return local_repo_path


@mainify
def __add_local_path_to_sys_path(local_path: Path | str):
    local_path_str = str(local_path)

    if local_path_str not in sys.path:
        sys.path.insert(0, local_path_str)


@mainify
def _get_git_revision_hash(repo_path: Path) -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
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
