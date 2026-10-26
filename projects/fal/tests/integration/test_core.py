from __future__ import annotations

import sys
import tempfile
from typing import Callable

import pytest
from pydantic import BaseModel, Field
from pydantic import __version__ as pydantic_version

from fal import local
from fal.api import IsolatedFunction
from fal.toolkit import File
from fal.toolkit.file.file import CompressedFile

# Each test deploys and runs an isolated function inline, so the deploy
# counts against the timer and needs more than the unit default.
pytestmark = pytest.mark.timeout(180)

EXAMPLE_FILE_URL = (
    "https://storage.googleapis.com/falserverless/sdk_tests/compressed_file_test.zip"
)
EXAMPLE_FILE_NAME = "compressed_file_test.zip"


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

    get_hostname_m = get_hostname.on(machine_type="L")
    third = get_hostname_m()
    assert local_hostname != third


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="isolate's local virtualenv backend does not work on Windows",
)
def test_isolated_on_local_host(isolated_client):
    @isolated_client("virtualenv")
    def get_hostname() -> str:
        import socket

        return socket.gethostname()

    import socket

    assert get_hostname.on(local)() == socket.gethostname()


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


def fal_file_downloaded(file: File):
    return file.file_size is not None


def fal_file_url_matches(file: File, url: str):
    return file.url == url


def fal_file_content_matches(file: File, content: str):
    return file.as_bytes().decode() == content


@pytest.mark.flaky(max_runs=3)
def test_fal_file_from_path(isolated_client):
    @isolated_client(requirements=[f"pydantic=={pydantic_version}", "tomli"])
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
    @isolated_client(requirements=[f"pydantic=={pydantic_version}", "tomli"])
    def fal_file_from_bytes(content: str):
        return File.from_bytes(content.encode(), repository="in_memory")

    file_content = "file-test"
    file = fal_file_from_bytes(file_content)

    assert fal_file_content_matches(file, file_content)


def test_fal_file_save(isolated_client):
    @isolated_client(requirements=[f"pydantic=={pydantic_version}", "tomli"])
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
        (EXAMPLE_FILE_URL, EXAMPLE_FILE_NAME),
        ("data:text/plain;charset=UTF-8,fal", "fal"),
    ],
)
def test_fal_file_input(isolated_client, file_url: str, expected_content: str):
    class TestInput(BaseModel):
        file: File = Field()

    @isolated_client(requirements=[f"pydantic=={pydantic_version}", "tomli"])
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

    @isolated_client(requirements=[f"pydantic=={pydantic_version}", "tomli"])
    def init_compressed_file_on_fal(input: TestInput) -> list[str]:
        return [file.name for file in input.files]

    test_input = TestInput(files=EXAMPLE_FILE_URL)

    extracted_file_names = init_compressed_file_on_fal(test_input)

    assert len(set(extracted_file_names)) == 3
