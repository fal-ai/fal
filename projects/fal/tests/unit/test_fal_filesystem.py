import pytest

from fal.files import FalFileSystem


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/data", "."),
        ("/data/tmp/file.txt", "tmp/file.txt"),
        ("tmp/file.txt", "tmp/file.txt"),
        (".", "."),
    ],
)
def test_endpoint_path_is_relative_to_data(path, expected):
    fs = FalFileSystem(skip_instance_cache=True)

    assert fs._endpoint_path(path) == expected
