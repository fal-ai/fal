from typing import Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

from fal.api import Options
from fal.cli._utils import AppData
from fal.cli.deploy import _deploy
from fal.cli.main import parse_args
from fal.project import find_project_root


def test_deploy():
    args = parse_args(["deploy", "myfile.py::MyApp"])
    assert args.func == _deploy
    assert args.app_ref == ("myfile.py", "MyApp")


def test_deploy_with_env():
    args = parse_args(["deploy", "myfile.py::MyApp", "--env", "dev"])
    assert args.func == _deploy
    assert args.app_ref == ("myfile.py", "MyApp")
    assert args.env == "dev"


def test_deploy_with_env_and_other_options():
    args = parse_args(
        [
            "deploy",
            "myfile.py::MyApp",
            "--app-name",
            "my-app",
            "--auth",
            "public",
            "--env",
            "staging",
        ]
    )
    assert args.func == _deploy
    assert args.app_ref == ("myfile.py", "MyApp")
    assert args.app_name == "my-app"
    assert args.auth == "public"
    assert args.env == "staging"


@pytest.fixture
def mock_parse_pyproject_toml():
    return {
        "apps": {
            "my-app": {
                "ref": "src/my_app/inference.py::MyApp",
                "auth": "shared",
                "deployment_strategy": "rolling",
            },
            "override-app": {
                "ref": "src/override_app/inference.py::OverrideApp",
                "name": "override-name",
                "auth": "private",
                "requirements": ["numpy==1.26.4"],
                "min_concurrency": 2,
                "regions": ["us-east"],
            },
            "another-app": {
                "ref": "src/another_app/inference.py::AnotherApp",
            },
            "app-with-extras": {
                "ref": "src/app_with_extras/inference.py::AppWithExtras",
                "extra_key": "extra_value",
            },
            "team-app": {
                "ref": "src/team_app/inference.py::TeamApp",
                "team": "my-team",
                "auth": "shared",
            },
        }
    }


def mock_args(
    app_ref: Tuple[str, Optional[str]],
    app_name: Optional[str] = None,
    auth: Optional[str] = None,
    strategy: Optional[str] = None,
    reset_scale: bool = False,
    team: Optional[str] = None,
    no_cache: bool = False,
    env: Optional[str] = None,
):
    args = MagicMock()

    args.app_ref = app_ref
    args.app_name = app_name
    args.auth = auth
    args.strategy = strategy
    args.app_scale_settings = reset_scale
    args.output = "pretty"
    args.team = team
    args.no_cache = no_cache
    args.force_env_build = False
    args.env = env

    return args


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.api.deploy._deploy_from_reference")
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
        mock_deploy_ref.call_args[0][0],
        (f"{project_root / 'src/my_app/inference.py'}", "MyApp"),
        AppData(
            ref=f"{project_root / 'src/my_app/inference.py'}::MyApp",
            auth="shared",
            deployment_strategy="rolling",
            reset_scale=False,
            team=None,
            name="my-app",
        ),
        force_env_build=False,
        environment_name=None,
    )


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.api.deploy._deploy_from_reference")
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
        mock_deploy_ref.call_args[0][0],
        (f"{project_root / 'src/another_app/inference.py'}", "AnotherApp"),
        AppData(
            ref=f"{project_root / 'src/another_app/inference.py'}::AnotherApp",
            auth=None,
            deployment_strategy=None,
            reset_scale=False,
            team=None,
            name="another-app",
        ),
        force_env_build=False,
        environment_name=None,
    )


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.api.deploy._deploy_from_reference")
def test_deploy_with_toml_overrides_applied(
    mock_deploy_ref, mock_parse_toml, mock_find_toml, mock_parse_pyproject_toml
):
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    args = mock_args(app_ref=("override-app", None))

    _deploy(args)

    project_root, _ = find_project_root(None)
    mock_deploy_ref.assert_called_once_with(
        mock_deploy_ref.call_args[0][0],
        (f"{project_root / 'src/override_app/inference.py'}", "OverrideApp"),
        AppData(
            ref=f"{project_root / 'src/override_app/inference.py'}::OverrideApp",
            auth="private",
            deployment_strategy=None,
            reset_scale=False,
            team=None,
            name="override-name",
            options=Options(
                host={"min_concurrency": 2, "regions": ["us-east"]},
                environment={"requirements": ["numpy==1.26.4"]},
            ),
        ),
        force_env_build=False,
        environment_name=None,
    )


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.api.deploy._deploy_from_reference")
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
@patch("fal.api.deploy._deploy_from_reference")
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


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.api.deploy._deploy_from_reference")
def test_deploy_with_toml_extra_keys_in_toml(
    mock_deploy_ref, mock_parse_toml, mock_find_toml, mock_parse_pyproject_toml
):
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    args = mock_args(app_ref=("app-with-extras", None))

    with pytest.raises(
        ValueError,
        match="Found unexpected keys in pyproject.toml: {'extra_key': 'extra_value'}",
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
@patch("fal.api.deploy._deploy_from_reference")
def test_deploy_with_toml_deployment_strategy(
    mock_deploy_ref, mock_parse_toml, mock_find_toml, mock_parse_pyproject_toml
):
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    args = mock_args(app_ref=("my-app", None), strategy="rolling")

    _deploy(args)

    project_root, _ = find_project_root(None)

    mock_deploy_ref.assert_called_once_with(
        mock_deploy_ref.call_args[0][0],
        (f"{project_root / 'src/my_app/inference.py'}", "MyApp"),
        AppData(
            ref=f"{project_root / 'src/my_app/inference.py'}::MyApp",
            auth="shared",
            deployment_strategy="rolling",
            reset_scale=False,
            team=None,
            name="my-app",
        ),
        force_env_build=False,
        environment_name=None,
    )


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.api.deploy._deploy_from_reference")
def test_deploy_with_toml_default_deployment_strategy(
    mock_deploy_ref, mock_parse_toml, mock_find_toml, mock_parse_pyproject_toml
):
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    args = mock_args(app_ref=("another-app", None))

    _deploy(args)

    project_root, _ = find_project_root(None)

    mock_deploy_ref.assert_called_once_with(
        mock_deploy_ref.call_args[0][0],
        (f"{project_root / 'src/another_app/inference.py'}", "AnotherApp"),
        AppData(
            ref=f"{project_root / 'src/another_app/inference.py'}::AnotherApp",
            auth=None,
            deployment_strategy=None,
            reset_scale=False,
            team=None,
            name="another-app",
        ),
        force_env_build=False,
        environment_name=None,
    )


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.api.deploy._deploy_from_reference")
def test_deploy_with_cli_auth(
    mock_deploy_ref, mock_parse_toml, mock_find_toml, mock_parse_pyproject_toml
):
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    args = mock_args(app_ref=("src/my_app/inference.py", "MyApp"), auth="shared")

    _deploy(args)

    project_root, _ = find_project_root(None)

    mock_deploy_ref.assert_called_once_with(
        mock_deploy_ref.call_args[0][0],
        (f"{project_root / 'src/my_app/inference.py'}", "MyApp"),
        AppData(
            ref=f"{project_root / 'src/my_app/inference.py'}::MyApp",
            auth="shared",
            deployment_strategy=None,
            reset_scale=False,
            team=None,
            name=None,
        ),
        force_env_build=False,
        environment_name=None,
    )


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.api.deploy._deploy_from_reference")
def test_deploy_with_cli_app_name(
    mock_deploy_ref, mock_parse_toml, mock_find_toml, mock_parse_pyproject_toml
):
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    args = mock_args(
        app_ref=("src/my_app/inference.py", "MyApp"),
        app_name="cli-app",
    )

    _deploy(args)

    project_root, _ = find_project_root(None)

    mock_deploy_ref.assert_called_once_with(
        mock_deploy_ref.call_args[0][0],
        (f"{project_root / 'src/my_app/inference.py'}", "MyApp"),
        AppData(
            ref=f"{project_root / 'src/my_app/inference.py'}::MyApp",
            auth=None,
            deployment_strategy=None,
            reset_scale=False,
            team=None,
            name="cli-app",
        ),
        force_env_build=False,
        environment_name=None,
    )


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.api.deploy._deploy_from_reference")
def test_deploy_with_cli_deployment_strategy(
    mock_deploy_ref, mock_parse_toml, mock_find_toml, mock_parse_pyproject_toml
):
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    args = mock_args(app_ref=("src/my_app/inference.py", "MyApp"), strategy="rolling")

    _deploy(args)

    project_root, _ = find_project_root(None)

    mock_deploy_ref.assert_called_once_with(
        mock_deploy_ref.call_args[0][0],
        (f"{project_root / 'src/my_app/inference.py'}", "MyApp"),
        AppData(
            ref=f"{project_root / 'src/my_app/inference.py'}::MyApp",
            auth=None,
            deployment_strategy="rolling",
            reset_scale=False,
            team=None,
            name=None,
        ),
        force_env_build=False,
        environment_name=None,
    )


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.api.deploy._deploy_from_reference")
def test_deploy_with_cli_reset_scale(
    mock_deploy_ref, mock_parse_toml, mock_find_toml, mock_parse_pyproject_toml
):
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    args = mock_args(app_ref=("src/my_app/inference.py", "MyApp"), reset_scale=True)

    _deploy(args)

    project_root, _ = find_project_root(None)

    mock_deploy_ref.assert_called_once_with(
        mock_deploy_ref.call_args[0][0],
        (f"{project_root / 'src/my_app/inference.py'}", "MyApp"),
        AppData(
            ref=f"{project_root / 'src/my_app/inference.py'}::MyApp",
            auth=None,
            deployment_strategy=None,
            reset_scale=True,
            team=None,
            name=None,
        ),
        force_env_build=False,
        environment_name=None,
    )


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.api.deploy._deploy_from_reference")
def test_deploy_with_cli_scale(
    mock_deploy_ref, mock_parse_toml, mock_find_toml, mock_parse_pyproject_toml
):
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    args = mock_args(app_ref=("src/my_app/inference.py", "MyApp"))

    _deploy(args)

    project_root, _ = find_project_root(None)

    mock_deploy_ref.assert_called_once_with(
        mock_deploy_ref.call_args[0][0],
        (f"{project_root / 'src/my_app/inference.py'}", "MyApp"),
        AppData(
            ref=f"{project_root / 'src/my_app/inference.py'}::MyApp",
            auth=None,
            deployment_strategy=None,
            reset_scale=False,
            team=None,
            name=None,
        ),
        force_env_build=False,
        environment_name=None,
    )


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.api.deploy._deploy_from_reference")
def test_deploy_with_cli_no_cache(
    mock_deploy_ref, mock_parse_toml, mock_find_toml, mock_parse_pyproject_toml
):
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    args = mock_args(app_ref=("src/my_app/inference.py", "MyApp"), no_cache=True)

    _deploy(args)

    project_root, _ = find_project_root(None)

    mock_deploy_ref.assert_called_once_with(
        mock_deploy_ref.call_args[0][0],
        (f"{project_root / 'src/my_app/inference.py'}", "MyApp"),
        AppData(
            ref=f"{project_root / 'src/my_app/inference.py'}::MyApp",
            auth=None,
            deployment_strategy=None,
            reset_scale=False,
            team=None,
            name=None,
        ),
        force_env_build=True,
        environment_name=None,
    )


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.cli.deploy.SyncServerlessClient")
def test_deploy_with_team_from_toml(
    mock_client,
    mock_parse_toml,
    mock_find_toml,
    mock_parse_pyproject_toml,
):
    """Test that team is read from pyproject.toml and passed to SyncServerlessClient"""
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    # Mock the client instance
    mock_client_instance = MagicMock()
    mock_client.return_value = mock_client_instance
    mock_client_instance.deploy.return_value = MagicMock(
        revision="rev-123",
        app_name="team-app",
        urls={"playground": {}, "sync": {}, "async": {}},
    )

    args = mock_args(app_ref=("team-app", None))
    args.host = "my-host"

    _deploy(args)

    # Ensure the client was initialized with the correct team
    mock_client.assert_called_once_with(host="my-host", team="my-team")


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.cli.deploy.SyncServerlessClient")
def test_deploy_with_team_from_toml_cli_team_override(
    mock_client,
    mock_parse_toml,
    mock_find_toml,
    mock_parse_pyproject_toml,
):
    """Test that team is read from pyproject.toml and passed to SyncServerlessClient"""
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    # Mock the client instance
    mock_client_instance = MagicMock()
    mock_client.return_value = mock_client_instance
    mock_client_instance.deploy.return_value = MagicMock(
        revision="rev-123",
        app_name="team-app",
        urls={"playground": {}, "sync": {}, "async": {}},
    )

    args = mock_args(app_ref=("team-app", None), team="my-cli-team")
    args.host = "my-host"

    _deploy(args)

    # Ensure the client was initialized with the correct team
    mock_client.assert_called_once_with(host="my-host", team="my-cli-team")


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.cli.deploy.SyncServerlessClient")
def test_deploy_without_team_in_toml(
    mock_client,
    mock_parse_toml,
    mock_find_toml,
    mock_parse_pyproject_toml,
):
    """Test that team defaults to None when not specified in pyproject.toml"""
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    # Mock the client instance
    mock_client_instance = MagicMock()
    mock_client.return_value = mock_client_instance
    mock_client_instance.deploy.return_value = MagicMock(
        revision="rev-123",
        app_name="my-app",
        urls={"playground": {}, "sync": {}, "async": {}},
    )

    args = mock_args(app_ref=("my-app", None))
    args.host = "my-host"

    _deploy(args)

    # Ensure the client was initialized with team=None
    mock_client.assert_called_once_with(host="my-host", team=None)


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_with_team(
    mock_parse_toml, mock_find_toml, mock_parse_pyproject_toml
):
    """Test that get_app_data_from_toml returns the team field"""
    from fal.cli._utils import get_app_data_from_toml

    mock_parse_toml.return_value = mock_parse_pyproject_toml

    toml_data = get_app_data_from_toml("team-app")

    project_root, _ = find_project_root(None)
    assert toml_data.ref == f"{project_root / 'src/team_app/inference.py'}::TeamApp"
    assert toml_data.auth == "shared"
    assert toml_data.deployment_strategy is None
    assert toml_data.reset_scale is False
    assert toml_data.team == "my-team"
    assert toml_data.name == "team-app"
    assert toml_data.options.host == {}
    assert toml_data.options.environment == {}


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_without_team(
    mock_parse_toml, mock_find_toml, mock_parse_pyproject_toml
):
    """Test that get_app_data_from_toml returns None for team when not specified"""
    from fal.cli._utils import get_app_data_from_toml

    mock_parse_toml.return_value = mock_parse_pyproject_toml

    toml_data = get_app_data_from_toml("my-app")

    project_root, _ = find_project_root(None)
    assert toml_data.ref == f"{project_root / 'src/my_app/inference.py'}::MyApp"
    assert toml_data.auth == "shared"
    assert toml_data.deployment_strategy == "rolling"
    assert toml_data.reset_scale is False
    assert toml_data.team is None
    assert toml_data.name == "my-app"
    assert toml_data.options.host == {}
    assert toml_data.options.environment == {}
