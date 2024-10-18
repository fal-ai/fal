from __future__ import annotations

import tempfile
import unittest.mock
from pathlib import Path
from typing import Callable
from uuid import uuid4

import fal
import pytest
from fal import FalServerlessHost, FalServerlessKeyCredentials, local, sync_dir
from fal.api import FalServerlessError, IsolatedFunction
from fal.toolkit import (
    File,
    clone_repository,
    download_file,
    download_model_weights,
)
from fal.toolkit import (
    Image as FalImage,
)
from fal.toolkit.file.file import CompressedFile
from fal.toolkit.utils.download_utils import _get_git_revision_hash, _hash_url
from pydantic import BaseModel, Field
from pydantic import __version__ as pydantic_version

EXAMPLE_FILE_URL = "https://raw.githubusercontent.com/fal-ai/fal/main/projects/fal/tests/assets/cat.png"


@pytest.mark.flaky(max_runs=3)
def test_isolated(isolated_client: Callable[..., Callable[..., IsolatedFunction]]):
    @isolated_client("virtualenv", requirements=["pyjokes==0.5.0"])
    def get_pyjokes_version():
        import pyjokes

        return pyjokes.__version__

    result = get_pyjokes_version()
    assert result == "0.5.0"

    @isolated_client("virtualenv")
    def get_hostname() -> str:
        import socket

        hostname = socket.gethostname()
        return hostname

    import socket

    local_hostname = socket.gethostname()

    first = get_hostname()
    assert local_hostname != first

    get_hostname_local = get_hostname.on(local)
    second = get_hostname_local()
    assert local_hostname == second

    get_hostname_m = get_hostname.on(machine_type="L")
    third = get_hostname_m()
    assert local_hostname != third


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

    # Clean the directory after tests finished
    remove_remote_directory(isolated_client, remote_dir_ref)


@pytest.fixture
def mock_fal_persistent_dirs(monkeypatch):
    """Mock fal persistent directories.
    It is NOT enough to mock only `FAL_PERSISTENT_DIR`, since the other two are
    might already be evaluated based on the `FAL_PERSISTENT_DIR` during imports.
    """
    temp_data_dir = Path(f"/tmp/data/fal-tests-{uuid4().hex}")
    temp_repository_dir = temp_data_dir / ".fal" / "repos"
    temp_model_weights_dir = temp_data_dir / ".fal" / "model_weights"

    monkeypatch.setattr(
        fal.toolkit.utils.download_utils,
        "FAL_PERSISTENT_DIR",
        temp_data_dir,
    )

    monkeypatch.setattr(
        fal.toolkit.utils.download_utils,
        "FAL_REPOSITORY_DIR",
        temp_repository_dir,
    )

    monkeypatch.setattr(
        fal.toolkit.utils.download_utils,
        "FAL_MODEL_WEIGHTS_DIR",
        temp_model_weights_dir,
    )


def test_download_file(isolated_client, mock_fal_persistent_dirs):
    from fal.toolkit.utils.download_utils import FAL_PERSISTENT_DIR

    relative_directory = "test"
    output_directory = FAL_PERSISTENT_DIR / relative_directory
    expected_path = output_directory / "cat.png"

    @isolated_client()
    def absolute_path_persistent_dir():
        downloaded_path = download_file(
            EXAMPLE_FILE_URL,
            target_dir=output_directory,
        )

        return downloaded_path

    assert str(expected_path) == str(
        absolute_path_persistent_dir()
    ), f"Path should be the target location sent '{expected_path!r}'"

    @isolated_client()
    def absolute_path_non_persistent_dir():
        downloaded_path = download_file(
            EXAMPLE_FILE_URL,
            target_dir=output_directory,
        )

        return downloaded_path

    assert str(expected_path) == str(
        absolute_path_non_persistent_dir()
    ), f"Path should be the target location sent '{expected_path!r}'"

    @isolated_client()
    def relative_path():
        downloaded_path = download_file(
            EXAMPLE_FILE_URL,
            target_dir=relative_directory,
        )

        return downloaded_path

    assert str(expected_path) == str(
        relative_path()
    ), f"Path should be the target location sent '{expected_path!r}'"

    @isolated_client()
    def test_with_force():
        first_path = download_file(
            EXAMPLE_FILE_URL,
            target_dir=output_directory,
            force=False,
        )
        first_path_stat = first_path.stat()

        second_path = download_file(
            EXAMPLE_FILE_URL,
            target_dir=output_directory,
            force=False,
        )
        second_path_stat = second_path.stat()

        third_path = download_file(
            EXAMPLE_FILE_URL,
            target_dir=output_directory,
            force=True,
        )
        third_path_stat = third_path.stat()

        return (
            first_path,
            first_path_stat,
            second_path,
            second_path_stat,
            third_path,
            third_path_stat,
        )

    (
        first_path,
        first_path_stat,
        second_path,
        second_path_stat,
        third_path,
        third_path_stat,
    ) = test_with_force()

    assert str(expected_path) == str(first_path), "Path should be the target location"
    assert str(expected_path) == str(second_path), "Path should be the target location"
    assert str(expected_path) == str(third_path), "Path should be the target location"

    assert (
        first_path_stat.st_mtime == second_path_stat.st_mtime
    ), "The file should not be redownloaded"

    assert (
        second_path_stat.st_mtime < third_path_stat.st_mtime
    ), "The file should be redownloaded with force=True"


def test_download_model_weights(isolated_client, mock_fal_persistent_dirs):
    from fal.toolkit.utils.download_utils import FAL_MODEL_WEIGHTS_DIR

    print(FAL_MODEL_WEIGHTS_DIR)

    expected_path = FAL_MODEL_WEIGHTS_DIR / _hash_url(EXAMPLE_FILE_URL) / "cat.png"

    @isolated_client()
    def download_weights():
        first_path = download_model_weights(EXAMPLE_FILE_URL, force=False)
        first_path_stat = first_path.stat()

        second_path = download_model_weights(EXAMPLE_FILE_URL, force=False)
        second_path_stat = second_path.stat()

        third_path = download_model_weights(EXAMPLE_FILE_URL, force=True)
        third_path_stat = third_path.stat()

        return (
            first_path,
            first_path_stat,
            second_path,
            second_path_stat,
            third_path,
            third_path_stat,
        )

    (
        first_path,
        first_path_stat,
        second_path,
        second_path_stat,
        third_path,
        third_path_stat,
    ) = download_weights()

    assert str(expected_path) == str(first_path), "Path should be the target location"
    assert str(expected_path) == str(second_path), "Path should be the target location"
    assert str(expected_path) == str(third_path), "Path should be the target location"

    assert (
        first_path_stat.st_mtime == second_path_stat.st_mtime
    ), "The model weights should not be redownloaded"

    assert (
        second_path_stat.st_mtime < third_path_stat.st_mtime
    ), "The model weights should be redownloaded with force=True"


def test_clone_repository(isolated_client, mock_fal_persistent_dirs):
    from fal.toolkit.utils.download_utils import FAL_REPOSITORY_DIR

    # https://github.com/fal-ai/isolate/tree/64b0a89c8391bd2cb3ca23cdeae01779e11aee05
    EXAMPLE_REPO_URL = "https://github.com/fal-ai/isolate.git"
    EXAMPLE_REPO_FIRST_COMMIT = "64b0a89c8391bd2cb3ca23cdeae01779e11aee05"
    EXAMPLE_REPO_SECOND_COMMIT = "34ecbca8cc7b64719d2a5c40dd3272f8d13bc1d2"
    expected_path = FAL_REPOSITORY_DIR / "isolate"

    @isolated_client()
    def clone_without_commit_hash():
        repo_path = clone_repository(EXAMPLE_REPO_URL)

        return repo_path

    repo_path = clone_without_commit_hash()
    assert str(repo_path) == str(expected_path), "Path should be the target location"

    @isolated_client()
    def clone_with_commit_hash():
        first_path = clone_repository(
            EXAMPLE_REPO_URL, commit_hash=EXAMPLE_REPO_FIRST_COMMIT
        )
        first_repo_hash = _get_git_revision_hash(first_path)

        second_path = clone_repository(
            EXAMPLE_REPO_URL, commit_hash=EXAMPLE_REPO_SECOND_COMMIT
        )

        second_repo_hash = _get_git_revision_hash(repo_path)

        return first_path, first_repo_hash, second_path, second_repo_hash

    (
        first_path,
        first_repo_hash,
        second_path,
        second_repo_hash,
    ) = clone_with_commit_hash()

    assert str(expected_path) == str(first_path), "Path should be the target location"
    assert str(expected_path) == str(second_path), "Path should be the target location"

    assert (
        first_repo_hash == EXAMPLE_REPO_FIRST_COMMIT
    ), "The commit hash should be the same"
    assert (
        second_repo_hash == EXAMPLE_REPO_SECOND_COMMIT
    ), "The commit hash should be the same"

    @isolated_client()
    def clone_with_force():
        first_path = clone_repository(
            EXAMPLE_REPO_URL, commit_hash=EXAMPLE_REPO_FIRST_COMMIT, force=False
        )
        first_repo_stat = first_path.stat()

        second_path = clone_repository(
            EXAMPLE_REPO_URL, commit_hash=EXAMPLE_REPO_FIRST_COMMIT, force=False
        )
        second_repo_stat = second_path.stat()

        third_path = clone_repository(
            EXAMPLE_REPO_URL, commit_hash=EXAMPLE_REPO_FIRST_COMMIT, force=True
        )
        third_repo_stat = third_path.stat()

        return (
            first_path,
            first_repo_stat,
            second_path,
            second_repo_stat,
            third_path,
            third_repo_stat,
        )

    (
        first_path,
        first_repo_stat,
        second_path,
        second_repo_stat,
        third_path,
        third_repo_stat,
    ) = clone_with_force()

    assert str(expected_path) == str(first_path), "Path should be the target location"
    assert str(expected_path) == str(second_path), "Path should be the target location"
    assert str(expected_path) == str(third_path), "Path should be the target location"

    assert (
        first_repo_stat.st_mtime == second_repo_stat.st_mtime
    ), "The repository should not be cloned again"

    assert (
        first_repo_stat.st_mtime < third_repo_stat.st_mtime
    ), "The repository should be cloned again with force=True"


def fal_file_downloaded(file: File):
    return file.file_size is not None


def fal_file_url_matches(file: File, url: str):
    return file.url == url


def fal_file_content_matches(file: File, content: str):
    return file.as_bytes().decode() == content


def test_fal_file_from_path(isolated_client):
    @isolated_client(requirements=[f"pydantic=={pydantic_version}"])
    def fal_file_from_temp(content: str):
        with tempfile.NamedTemporaryFile() as temp_file:
            file_path = temp_file.name

            with open(file_path, "w") as fp:
                fp.write(content)

            return File.from_path(file_path, repository="in_memory")

    file_content = "file-test"
    file = fal_file_from_temp(file_content)

    assert fal_file_content_matches(file, file_content)


def test_fal_file_from_bytes(isolated_client):
    @isolated_client(requirements=[f"pydantic=={pydantic_version}"])
    def fal_file_from_bytes(content: str):
        return File.from_bytes(content.encode(), repository="in_memory")

    file_content = "file-test"
    file = fal_file_from_bytes(file_content)

    assert fal_file_content_matches(file, file_content)


def test_fal_file_save(isolated_client):
    @isolated_client(requirements=[f"pydantic=={pydantic_version}"])
    def fal_file_to_local_file(content: str):
        file = File.from_bytes(content.encode(), repository="in_memory")

        with tempfile.NamedTemporaryFile() as temp_file:
            file_name = temp_file.name
            # We need to overwrite the file, since the file is already created
            # by the tempfile.NamedTemporaryFile
            file.save(file_name, overwrite=True)

            with open(file_name) as fp:
                file_content = fp.read()

        return file_content

    file_content = "file-test"
    saved_file_content = fal_file_to_local_file(file_content)

    assert file_content == saved_file_content


@pytest.mark.parametrize(
    "file_url, expected_content",
    [
        (
            EXAMPLE_FILE_URL,
            "projects/fal/cat.png",
        ),
        ("data:text/plain;charset=UTF-8,fal", "fal"),
    ],
)
def test_fal_file_input(isolated_client, file_url: str, expected_content: str):
    class TestInput(BaseModel):
        file: File = Field()

    @isolated_client(requirements=[f"pydantic=={pydantic_version}"])
    def init_file_on_fal(input: TestInput) -> File:
        return input.file

    test_input = TestInput(file=file_url)
    file = init_file_on_fal(test_input)

    # File is not downloaded until it is needed
    assert not fal_file_downloaded(file)

    assert fal_file_url_matches(file, file_url)

    # Expect value error if we try to access the file content for input file
    with pytest.raises(ValueError):
        fal_file_content_matches(file, expected_content)


def test_fal_compressed_file(isolated_client):
    class TestInput(BaseModel):
        files: CompressedFile

    @isolated_client(requirements=[f"pydantic=={pydantic_version}"])
    def init_compressed_file_on_fal(input: TestInput) -> int:
        extracted_file_paths = [file for file in input.files]
        return extracted_file_paths

    archive_url = "https://storage.googleapis.com/falserverless/sdk_tests/compressed_file_test.zip"
    test_input = TestInput(files=archive_url)

    extracted_file_paths = init_compressed_file_on_fal(test_input)

    assert all(isinstance(file, Path) for file in extracted_file_paths)
    assert len(extracted_file_paths) == 3


def test_fal_cdn(isolated_client):
    @isolated_client(requirements=[f"pydantic=={pydantic_version}"])
    def upload_to_fal_cdn() -> FalImage:
        return FalImage.from_bytes(b"0", "jpeg", repository="cdn")

    uploaded_image = upload_to_fal_cdn()

    assert uploaded_image


def test_download_file_with_slash_in_filename():
    from fal.toolkit.utils.download_utils import _get_remote_file_properties

    test_url = "https://example.com/file/with/slash.txt"

    with unittest.mock.patch(
        "fal.toolkit.utils.download_utils.urlopen"
    ) as mock_urlopen:
        # urlopen is a context manager, so we need to mock the __enter__ method
        mock_response = mock_urlopen.return_value.__enter__.return_value
        mock_response.headers.get_filename.return_value = "file/with/slash.txt"
        mock_response.headers.get.return_value = "100"

        file_name, _ = _get_remote_file_properties(test_url)

    assert "/" not in file_name
