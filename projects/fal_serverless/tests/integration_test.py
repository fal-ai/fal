from __future__ import annotations

from uuid import uuid4

import pytest
from fal import (
    FalServerlessHost,
    FalServerlessKeyCredentials,
    download_file,
    download_repo,
    download_weights,
    local,
    sync_dir,
)
from fal.api import FalServerlessError


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


EXAMPLE_FILE_URL = "https://raw.githubusercontent.com/fal-ai/isolate/d553f927348206530208442556f481f39b161732/README.md"
EXAMPLE_FILE_CHECKSUM = (
    "81ff101c9f6f5b4af6868a3004d43d55fc546716d40ca12c678684a77e76c67c"
)

EXAMPLE_REPO_URL = "https://github.com/comfyanonymous/ComfyUI.git"
EXAMPLE_REPO_COMMIT = "0793eb926933034997cc2383adc414d080643e77"  # https://github.com/comfyanonymous/ComfyUI/tree/0793eb926933034997cc2383adc414d080643e77


def test_download_file_invalid_location():
    try:
        download_file(
            EXAMPLE_FILE_URL,
            target_dir="/some",
            checksum_sha256=EXAMPLE_FILE_CHECKSUM,
        )
        assert False, "Should have raised an exception"
    except Exception as e:
        msg = "Should have raised an exception about an invalid directory: " + str(e)
        assert "invalid dir" in str(e).lower(), msg


@pytest.mark.skip(
    "Generates 'Error while serializing the given object' in test 'projects/fal_serverless/tests/test_stability.py::test_missing_dependencies_nested_server_error'"
)
def test_download_file():
    file_name = uuid4().hex + ".md"

    example_path = download_file(
        EXAMPLE_FILE_URL,
        target_dir="/data/test",
        file_name=file_name,
        checksum_sha256=EXAMPLE_FILE_CHECKSUM,
    )
    example_path_str = str(example_path)
    assert (
        example_path_str == f"/data/test/{file_name}"
    ), f"Path should be the target location sent '{example_path!r}'"

    example_path = download_file(
        EXAMPLE_FILE_URL,
        target_dir="test",
        file_name=file_name,
        checksum_sha256=EXAMPLE_FILE_CHECKSUM,
    )
    example_path_str = str(example_path)
    assert (
        example_path_str == f"/data/test/{file_name}"
    ), f"Path should be the target location sent '{example_path!r}'"


def test_download_weights():
    example_path = download_weights(
        EXAMPLE_FILE_URL,
        checksum_sha256=EXAMPLE_FILE_CHECKSUM,
    )
    example_path_str = str(example_path)
    empty, data, *other = example_path_str.split("/")
    assert empty == "", "Path should start with a slash"
    assert data == "data", "Path should start with the data directory"
    assert other, "Path should contain the rest of the path"


def test_download_repo():
    example_path = download_repo(EXAMPLE_REPO_URL)
    example_path_str = str(example_path)

    empty, data, fal_prefix, repos_dir, *other = example_path_str.split("/")

    assert empty == "", "Path should start with a slash"
    assert data == "data", "Path should start with the data directory"
    assert fal_prefix == ".fal", "Path should start with '.fal'"
    assert repos_dir == "repos", "Path should start with the repos directory"
    assert other, "Path should contain the rest of the path"

    example_path = download_repo(
        EXAMPLE_REPO_URL,
        commit_hash=EXAMPLE_REPO_COMMIT,
    )
    assert example_path, "Should have returned a path"
    # TODO: check that the commit hash is correct
