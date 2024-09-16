from unittest.mock import MagicMock, patch

import pytest
from fal.cli.main import parse_args
from fal.cli.run import _run
from fal.files import find_project_root


def test_run():
    args = parse_args(["run", "/my/path.py::myfunc"])
    assert args.func == _run
    assert args.func_ref == ("/my/path.py", "myfunc")


@pytest.fixture
def mock_parse_pyproject_toml():
    return {
        "apps": {
            "my-app": {
                "ref": "src/my_app/inference.py::MyApp",
                "auth": "shared",
            },
            # auth is not provided for another-app
            "another-app": {
                "ref": "src/another_app/inference.py::AnotherApp",
            },
        }
    }


def mocked_fal_serverless_host(host):
    mock = MagicMock()
    mock.host = host
    return mock


def mock_args(host, func_ref: tuple[str]):
    args = MagicMock()
    args.host = host
    args.func_ref = func_ref
    return args


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.api.FalServerlessHost")
@patch("fal.utils.load_function_from")
def test_run_with_toml_success(
    mock_load_function_from,
    mock_fal_serverless_host,
    mock_parse_toml,
    mock_find_toml,
    mock_parse_pyproject_toml,
):
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    host = mocked_fal_serverless_host("my-host")
    mock_fal_serverless_host.return_value = host

    args = mock_args(func_ref=("my-app", None), host=host)

    _run(args)

    project_root, _ = find_project_root(None)

    # Ensure the correct app is ran
    mock_load_function_from.assert_called_once_with(
        host, f"{project_root / 'src/my_app/inference.py'}", "MyApp"
    )


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.utils.load_function_from")
def test_run_with_toml_app_not_found(
    mock_load_function_from, mock_parse_toml, mock_find_toml, mock_parse_pyproject_toml
):
    mock_parse_toml.return_value = mock_parse_pyproject_toml
    args = mock_args(func_ref=("non-existent-app", None), host="my-host")

    with pytest.raises(
        ValueError, match="App non-existent-app not found in pyproject.toml"
    ):
        _run(args)

    # Ensure _run_from_reference was not called
    mock_load_function_from.assert_not_called()


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.utils.load_function_from")
def test_run_with_toml_missing_ref_key(
    mock_load_function_from, mock_parse_toml, mock_find_toml
):
    mock_parse_toml.return_value = {
        "apps": {
            "my-app": {
                "auth": "shared",
            }
        }
    }

    args = mock_args(func_ref=("my-app", None), host="my-host")

    with pytest.raises(
        ValueError, match="App my-app does not have a ref key in pyproject.toml"
    ):
        _run(args)

    # Ensure _run_from_reference was not called
    mock_load_function_from.assert_not_called()
