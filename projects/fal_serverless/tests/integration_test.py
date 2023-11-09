from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fal import FalServerlessHost, FalServerlessKeyCredentials, local, sync_dir
from fal.api import FalServerlessError
from fal.toolkit import (
    FAL_MODEL_WEIGHTS_DIR,
    FAL_PERSISTENT_DIR,
    FAL_REPOSITORY_DIR,
    clone_repository,
    download_file,
    download_model_weights,
)
from fal.toolkit.utils.download_utils import _get_git_revision_hash, _hash_url


def test_isolated(isolated_client):
    @isolated_client("virtualenv", requirements=["pyjokes==0.5.0"])
    def get_pyjokes_version():
        import pyjokes

        return pyjokes.__version__

    result = get_pyjokes_version()
    assert result == "0.5.0"

    @isolated_client("virtualenv")
    def get_hostname() -> str:
        import socket

        return socket.gethostname()

    first = get_hostname()
    assert first.startswith("worker")

    get_hostname_local = get_hostname.on(local)
    second = get_hostname_local()
    assert not second.startswith("worker-")

    get_hostname_m = get_hostname.on(machine_type="L")
    third = get_hostname_m()
    assert third.startswith("worker")
    assert third != first

    # The machine_type should be dropped when using local
    get_hostname_m_local = get_hostname_m.on(local)
    fourth = get_hostname_m_local()
    assert not fourth.startswith("worker-")


def test_isolate_setup_funcs(isolated_client):
    def setup_function():
        import math

        return math.pi

    @isolated_client(setup_function=setup_function)
    def is_tau(setup, by_factor) -> str:
        import math

        return setup * by_factor == math.tau

    assert is_tau(2)
    assert not is_tau(by_factor=3)


def test_isolate_setup_func_order(isolated_client):
    def setup_function():
        return "one "

    @isolated_client(setup_function=setup_function)
    def one_and(setup, num) -> str:
        return setup + num

    assert one_and("two") == "one two"
    assert one_and("three") == "one three"


def test_isolate_error_handling(isolated_client):
    requirements = ["pyjokes", "requests"]

    def setup():
        print("hello")

    @isolated_client(requirements=requirements, keep_alive=20, setup=setup)
    def raises_value_error():
        import pyjokes

        return pyjokes.get_joke()

    with pytest.raises(ValueError):
        raises_value_error()

    creds = FalServerlessKeyCredentials(key_id="fake", key_secret="fake")
    host = FalServerlessHost(url="not.there", credentials=creds)

    @isolated_client(host=host)
    def raises_grpc_error():
        return True

    with pytest.raises(FalServerlessError):
        raises_grpc_error()


@pytest.mark.skip(reason="No way to test this yet")
def test_sync(isolated_client):
    import os
    import random
    import string
    import tempfile

    def create_random_file(file_path, size):
        with open(file_path, "wb") as f:
            f.write(os.urandom(size))

    def create_test_directory():
        temp_dir = tempfile.mkdtemp()
        for i in range(3):
            file_name = "".join(random.choices(string.ascii_letters, k=10)) + ".txt"
            file_path = os.path.join(temp_dir, file_name)
            create_random_file(file_path, 1024)
        return temp_dir

    @isolated_client()
    def list_remote_directory(remote_dir):
        import os

        contents = os.listdir(remote_dir)
        return contents

    @isolated_client()
    def remove_remote_directory(remote_dir):
        import shutil

        shutil.rmtree(remote_dir)

    local_dir = create_test_directory()
    # Copy the basename of the local directory to keep it unique
    remote_dir_name = "/data/" + os.path.basename(local_dir)
    remote_dir_ref = sync_dir(local_dir, remote_dir_name)

    remote_contents = list_remote_directory(remote_dir_ref)
    local_contents = os.listdir(local_dir)

    local_contents.sort()
    remote_contents.sort()

    assert (
        local_contents == remote_contents
    ), "Local and remote directory contents do not match"

    print("Test passed: Local and remote directory contents match")

    remove_remote_directory(remote_dir_ref)


def test_download_file(isolated_client):
    EXAMPLE_FILE_URL = "https://raw.githubusercontent.com/fal-ai/isolate/d553f927348206530208442556f481f39b161732/README.md"

    output_directory = FAL_PERSISTENT_DIR / "test"

    @isolated_client()
    def absolute_path_persistent_dir():
        downloaded_path = download_file(
            EXAMPLE_FILE_URL,
            target_dir=output_directory,
        )

        downloaded_path.unlink()
        return downloaded_path

    expected_path = output_directory / "README.md"
    assert str(expected_path) == str(
        absolute_path_persistent_dir()
    ), f"Path should be the target location sent '{expected_path!r}'"

    output_directory = Path("/test")

    @isolated_client()
    def absolute_path_non_persistent_dir():
        downloaded_path = download_file(
            EXAMPLE_FILE_URL,
            target_dir=output_directory,
        )

        downloaded_path.unlink()
        return downloaded_path

    expected_path = output_directory / "README.md"
    assert str(expected_path) == str(
        absolute_path_non_persistent_dir()
    ), f"Path should be the target location sent '{expected_path!r}'"

    output_directory = Path("test")

    @isolated_client()
    def relative_path():
        downloaded_path = download_file(
            EXAMPLE_FILE_URL,
            target_dir=output_directory,
        )

        downloaded_path.unlink()
        return downloaded_path

    expected_path = FAL_PERSISTENT_DIR / output_directory / "README.md"
    assert str(expected_path) == str(
        relative_path()
    ), f"Path should be the target location sent '{expected_path!r}'"

    @isolated_client
    def remove_downloaded_file(path: Path):
        path.unlink()

    @isolated_client()
    def test_with_force(force: bool = False):
        downloaded_path = download_file(
            EXAMPLE_FILE_URL,
            target_dir=output_directory,
            force=force,
        )

        return downloaded_path, downloaded_path.stat()

    initial_path, initial_stat = test_with_force(force=False)
    second_path, second_stat = test_with_force(force=True)

    assert initial_path == second_path, "The path should be the same"
    assert (
        initial_stat.st_mtime < second_stat.st_mtime
    ), "The file should be redownloaded"

    # Remove the downloaded file before the `force=True` test
    remove_downloaded_file(expected_path)


def test_download_model_weights(isolated_client):
    EXAMPLE_FILE_URL = "https://raw.githubusercontent.com/fal-ai/isolate/d553f927348206530208442556f481f39b161732/README.md"
    expected_path = FAL_MODEL_WEIGHTS_DIR / _hash_url(EXAMPLE_FILE_URL) / "README.md"

    @isolated_client()
    def download_weights(force: bool = False):
        model_weights_path = download_model_weights(EXAMPLE_FILE_URL, force=force)

        return model_weights_path, model_weights_path.stat()

    @isolated_client()
    def remove_model_weights(path: Path):
        path.unlink()

    initial_weights_path, initial_weights_stat = download_weights(force=False)
    assert str(initial_weights_path) == str(
        expected_path
    ), "Path should be the target location"

    second_weights_path, second_weights_stat = download_weights(force=True)
    assert str(initial_weights_path) == str(
        second_weights_path
    ), "The path should be the same"

    # Check for file last modified time, the weights should be re-downloaded
    # (and thus, modified) since `force` parameter is set to `True`.
    assert (
        initial_weights_stat.st_mtime < second_weights_stat.st_mtime
    ), "The weights should be redownloaded"

    remove_model_weights(initial_weights_path)


def test_clone_repository(isolated_client):
    # https://github.com/comfyanonymous/ComfyUI/tree/0793eb926933034997cc2383adc414d080643e77
    EXAMPLE_REPO_URL = "https://github.com/comfyanonymous/ComfyUI.git"
    EXAMPLE_REPO_COMMIT = "0793eb926933034997cc2383adc414d080643e77"
    expected_path = FAL_REPOSITORY_DIR / "ComfyUI"

    @isolated_client()
    def remove_repo(repo_path: Path):
        shutil.rmtree(repo_path)

    @isolated_client()
    def clone_without_commit_hash():
        repo_path = clone_repository(EXAMPLE_REPO_URL)

        return repo_path

    repo_path = clone_without_commit_hash()

    assert str(repo_path) == str(expected_path), "Path should be the target location"

    @isolated_client()
    def clone_with_commit_hash(force: bool = False):
        repo_path = clone_repository(
            EXAMPLE_REPO_URL, commit_hash=EXAMPLE_REPO_COMMIT, force=force
        )
        repo_commit_hash = _get_git_revision_hash(repo_path)

        return repo_path, repo_commit_hash, repo_path.stat()

    (
        initial_repo_path,
        initial_repo_commit_hash,
        initial_repo_stat,
    ) = clone_with_commit_hash(force=False)

    assert str(initial_repo_path) == str(
        expected_path
    ), "Path should be the target location"
    assert (
        initial_repo_commit_hash == EXAMPLE_REPO_COMMIT
    ), "The commit hash of the cloned repository must match the provided commit hash argument."

    (
        second_repo_path,
        second_repo_commit_hash,
        second_repo_stat,
    ) = clone_with_commit_hash(force=True)
    assert str(initial_repo_path) == str(
        second_repo_path
    ), "The path should be the same"
    assert (
        initial_repo_commit_hash == second_repo_commit_hash
    ), "The commit hash should be the same"

    # Check for repository last modified time, the repository should be re-downloaded
    # (and thus, modified) since `force` parameter is set to `True`.
    assert (
        initial_repo_stat.st_mtime < second_repo_stat.st_mtime
    ), "The repository should be redownloaded"

    remove_repo(initial_repo_path)
