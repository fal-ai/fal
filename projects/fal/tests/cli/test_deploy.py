from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from fal.cli.deploy import _deploy
from fal.cli.main import parse_args
from fal.files import find_project_root


def test_deploy():
    args = parse_args(["deploy", "myfile.py::MyApp"])
    assert args.func == _deploy
    assert args.app_ref == ("myfile.py", "MyApp")


@pytest.fixture
def mock_parse_pyproject_toml():
    return {
        "apps": {
            "my-app": {
                "ref": "src/my_app/inference.py::MyApp",
                "auth": "shared",
                "deployment_strategy": "rolling",
            },
            "another-app": {
                "ref": "src/another_app/inference.py::AnotherApp",
            },
        }
    }


def mock_args(
    app_ref: tuple[str],
    app_name: Optional[str] = None,
    auth: Optional[str] = None,
    strategy: Optional[str] = None,
):
    args = MagicMock()

    args.app_ref = app_ref
    args.app_name = app_name
    args.auth = auth
    args.strategy = strategy

    return args


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.cli.deploy._deploy_from_reference")
def test_deploy_with_toml_success(
    mock_deploy_ref, mock_parse_toml, mock_find_toml, mock_parse_pyproject_toml
):
    # Mocking the parse_pyproject_toml function to return a predefined dict
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    args = mock_args(app_ref=("my-app", None))

    _deploy(args)

    project_root, _ = find_project_root(None)

    # Ensure the correct app is deployed
    mock_deploy_ref.assert_called_once_with(
        (f"{project_root / 'src/my_app/inference.py'}", "MyApp"),
        "my-app",
        args,
        "shared",
        "rolling",
    )


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.cli.deploy._deploy_from_reference")
def test_deploy_with_toml_no_auth(
    mock_deploy_ref, mock_parse_toml, mock_find_toml, mock_parse_pyproject_toml
):
    # Mocking the parse_pyproject_toml function to return a predefined dict
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    args = mock_args(app_ref=("another-app", None))

    _deploy(args)

    project_root, _ = find_project_root(None)

    # Since auth is not provided for "another-app", it should default to "private"
    mock_deploy_ref.assert_called_once_with(
        (f"{project_root / 'src/another_app/inference.py'}", "AnotherApp"),
        "another-app",
        args,
        "private",
        "recreate",
    )


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.cli.deploy._deploy_from_reference")
def test_deploy_with_toml_app_not_found(
    mock_deploy_ref, mock_parse_toml, mock_find_toml, mock_parse_pyproject_toml
):
    # Mocking the parse_pyproject_toml function to return a predefined dict
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    args = mock_args(app_ref=("non-existent-app", None))

    # Expect a ValueError since "non-existent-app" is not in the toml
    with pytest.raises(
        ValueError, match="App non-existent-app not found in pyproject.toml"
    ):
        _deploy(args)


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.cli.deploy._deploy_from_reference")
def test_deploy_with_toml_missing_ref_key(
    mock_deploy_ref, mock_parse_toml, mock_find_toml
):
    # Mocking a toml structure without a "ref" key for a certain app
    mock_parse_toml.return_value = {
        "apps": {
            "my-app": {
                "auth": "shared",
            }
        }
    }

    args = mock_args(app_ref=("my-app", None))

    # Expect a ValueError since "ref" key is missing
    with pytest.raises(
        ValueError, match="App my-app does not have a ref key in pyproject.toml"
    ):
        _deploy(args)


@patch("fal.cli._utils.find_pyproject_toml", return_value=None)
def test_deploy_with_toml_file_not_found(mock_find_toml):
    args = mock_args(app_ref=("my-app", None))

    # Expect a ValueError since no pyproject.toml file is found
    with pytest.raises(ValueError, match="No pyproject.toml file found."):
        _deploy(args)


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_deploy_with_toml_only_app_name_is_provided(
    mock_parse_toml, mock_find_toml, mock_parse_pyproject_toml
):
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    args = mock_args(
        app_ref=("my-app", None), app_name="custom-app-name", auth="public"
    )

    # Expect a ValueError since --app-name and --auth cannot be used with just
    # the app name reference
    with pytest.raises(
        ValueError, match="Cannot use --app-name or --auth with app name reference."
    ):
        _deploy(args)


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.cli.deploy._deploy_from_reference")
def test_deploy_with_toml_deployment_strategy(
    mock_deploy_ref, mock_parse_toml, mock_find_toml, mock_parse_pyproject_toml
):
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    args = mock_args(app_ref=("my-app", None), strategy="rolling")

    _deploy(args)

    project_root, _ = find_project_root(None)

    mock_deploy_ref.assert_called_once_with(
        (f"{project_root / 'src/my_app/inference.py'}", "MyApp"),
        "my-app",
        args,
        "shared",
        "rolling",
    )


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.cli.deploy._deploy_from_reference")
def test_deploy_with_toml_default_deployment_strategy(
    mock_deploy_ref, mock_parse_toml, mock_find_toml, mock_parse_pyproject_toml
):
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    args = mock_args(app_ref=("another-app", None))

    _deploy(args)

    project_root, _ = find_project_root(None)

    mock_deploy_ref.assert_called_once_with(
        (f"{project_root / 'src/another_app/inference.py'}", "AnotherApp"),
        "another-app",
        args,
        "private",
        "recreate",
    )


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.cli.deploy._deploy_from_reference")
def test_deploy_with_cli_auth(
    mock_deploy_ref, mock_parse_toml, mock_find_toml, mock_parse_pyproject_toml
):
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    args = mock_args(app_ref=("src/my_app/inference.py", "MyApp"), auth="shared")

    _deploy(args)

    mock_deploy_ref.assert_called_once_with(
        ("src/my_app/inference.py", "MyApp"),
        None,
        args,
        "shared",
        None,
    )


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.cli.deploy._deploy_from_reference")
def test_deploy_with_cli_deployment_strategy(
    mock_deploy_ref, mock_parse_toml, mock_find_toml, mock_parse_pyproject_toml
):
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    args = mock_args(app_ref=("src/my_app/inference.py", "MyApp"), strategy="rolling")

    _deploy(args)

    mock_deploy_ref.assert_called_once_with(
        ("src/my_app/inference.py", "MyApp"),
        None,
        args,
        None,
        "rolling",
    )
