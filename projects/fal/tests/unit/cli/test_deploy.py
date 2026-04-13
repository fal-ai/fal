from types import SimpleNamespace
from typing import Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from fal.api import Options
from fal.cli._utils import AppData
from fal.cli.deploy import (
    _build_deployment_check_summary,
    _deploy,
    _diff_table,
    _payload_requires_deploy_check,
    _render_auth_line,
    _render_deployment_check_summary,
    _render_deployment_strategy_line,
    _render_environment_build_cache_line,
    _resolve_deploy_check_source,
)
from fal.cli.main import parse_args
from fal.project import find_project_root
from fal.sdk import AliasInfo


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


def test_deploy_with_check_and_yes():
    args = parse_args(["deploy", "myfile.py::MyApp", "--check", "--yes"])
    assert args.check is True
    assert args.yes is True


@patch.dict("os.environ", {"FAL_ENV": "from-env-var"})
def test_deploy_uses_fal_env_variable():
    args = parse_args(["deploy", "myfile.py::MyApp"])
    assert args.env == "from-env-var"


@patch.dict("os.environ", {"FAL_ENV": "from-env-var"})
def test_deploy_cli_env_overrides_fal_env_variable():
    args = parse_args(["deploy", "myfile.py::MyApp", "--env", "cli-env"])
    assert args.env == "cli-env"


@pytest.fixture(autouse=True)
def disable_admin_deploy_check_lookup():
    with patch("fal.cli.deploy._admin_requires_deploy_check", return_value=False):
        yield


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
            "app-with-files": {
                "ref": "src/app_with_files/inference.py::AppWithFiles",
                "app_files": ["assets", "config.yaml"],
                "app_files_ignore": [r"\\.venv/"],
                "app_files_context_dir": ".",
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
            "no-scale-app": {
                "ref": "src/no_scale_app/inference.py::NoScaleApp",
                "team": "my-team",
                "no_scale": True,
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
    args.check = False
    args.yes = False
    args.host = "my-host"

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


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_with_app_files(
    mock_parse_toml, mock_find_toml, mock_parse_pyproject_toml
):
    from fal.cli._utils import get_app_data_from_toml

    mock_parse_toml.return_value = mock_parse_pyproject_toml

    toml_data = get_app_data_from_toml("app-with-files")

    project_root, _ = find_project_root(None)
    assert toml_data.ref == (
        f"{project_root / 'src/app_with_files/inference.py'}::AppWithFiles"
    )
    assert toml_data.options.host == {
        "app_files": ["assets", "config.yaml"],
        "app_files_ignore": [r"\\.venv/"],
        "app_files_context_dir": ".",
    }
    assert toml_data.options.environment == {}


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_rejects_app_files_context_dir_without_app_files(
    mock_parse_toml, mock_find_toml
):
    from fal.cli._utils import get_app_data_from_toml

    mock_parse_toml.return_value = {
        "apps": {
            "my-app": {
                "ref": "src/my_app/inference.py::MyApp",
                "app_files_context_dir": ".",
            }
        }
    }

    with pytest.raises(
        ValueError,
        match="app_files_context_dir is only supported when app_files is provided.",
    ):
        get_app_data_from_toml("my-app")


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_no_scale_warning_can_be_suppressed(
    mock_parse_toml, mock_find_toml, mock_parse_pyproject_toml, capsys
):
    from fal.cli._utils import get_app_data_from_toml

    mock_parse_toml.return_value = mock_parse_pyproject_toml

    get_app_data_from_toml("no-scale-app", emit_deprecation_warnings=False)
    captured = capsys.readouterr()

    assert (
        "[WARNING] no_scale is deprecated, use app_scale_settings instead"
        not in captured.out
    )


@patch("fal.cli._utils.get_app_data_from_toml")
@patch("fal.cli.deploy.SyncServerlessClient")
def test_deploy_team_lookup_uses_silent_toml_read(mock_client, mock_get_app_data):
    mock_get_app_data.return_value = AppData(team="my-team")

    # Mock the client instance
    mock_client_instance = MagicMock()
    mock_client.return_value = mock_client_instance
    mock_client_instance.deploy.return_value = MagicMock(
        revision="rev-123",
        app_name="no-scale-app",
        auth_mode="private",
        urls={"playground": {}, "sync": {}, "async": {}},
        log_url="https://fal.ai/logs/123",
    )

    args = mock_args(app_ref=("no-scale-app", None))
    args.host = "my-host"

    _deploy(args)

    mock_get_app_data.assert_called_once_with(
        "no-scale-app", emit_deprecation_warnings=False
    )


def _prepared_deployment(
    *,
    reset_scale: bool,
    auth: str = "public",
    host_options: Optional[dict] = None,
):
    return SimpleNamespace(
        loaded=SimpleNamespace(
            app_name="my-app",
            app_auth=auth,
            function=SimpleNamespace(
                options=Options(
                    host=host_options
                    or {
                        "machine_type": ["GPU-H100"],
                        "keep_alive": 10,
                        "max_concurrency": 2,
                        "min_concurrency": 0,
                        "concurrency_buffer": 0,
                        "concurrency_buffer_perc": 0,
                        "scaling_delay": 0,
                        "max_multiplexing": 4,
                        "request_timeout": 3600,
                        "startup_timeout": 900,
                        "regions": ["us-east"],
                    },
                    environment={},
                )
            ),
        ),
        display_name="MyApp",
        environment_name="staging",
        app_data=AppData(
            reset_scale=reset_scale,
            deployment_strategy="rolling",
            name="my-app",
        ),
    )


def _production_alias(**overrides) -> AliasInfo:
    defaults = dict(
        alias="my-app--staging",
        revision="rev-prod",
        auth_mode="private",
        keep_alive=300,
        max_concurrency=8,
        max_multiplexing=1,
        active_runners=1,
        min_concurrency=1,
        concurrency_buffer=2,
        concurrency_buffer_perc=0,
        scaling_delay=30,
        machine_types=["GPU-A100"],
        request_timeout=120,
        startup_timeout=600,
        valid_regions=["eu-west"],
        environment_name="staging",
    )
    defaults.update(overrides)
    return AliasInfo(**defaults)


def _render_summary(summary) -> str:
    console = Console(record=True, width=120)
    _render_deployment_check_summary(console, summary)
    return console.export_text()


@patch("fal.cli.deploy._admin_requires_deploy_check")
@patch.dict("os.environ", {"FAL_DEPLOY_CHECK": "true"})
def test_resolve_deploy_check_source_env_can_be_skipped_with_yes(mock_admin):
    args = mock_args(app_ref=("src/my_app/inference.py", "MyApp"))
    args.yes = True

    assert _resolve_deploy_check_source(args, MagicMock()) is None
    mock_admin.assert_not_called()


@patch("fal.cli.deploy._admin_requires_deploy_check", return_value=True)
def test_resolve_deploy_check_source_uses_admin_trigger(mock_admin):
    args = mock_args(app_ref=("src/my_app/inference.py", "MyApp"))

    assert _resolve_deploy_check_source(args, MagicMock()) == "admin"
    mock_admin.assert_called_once()


def test_payload_requires_deploy_check_supports_nested_org_config():
    payload = SimpleNamespace(
        additional_properties={
            "org_config": {
                "deploy_check": True,
            }
        }
    )

    assert _payload_requires_deploy_check(payload) is True


def test_build_deployment_check_summary_inherits_production_scale_without_reset_scale():
    summary = _build_deployment_check_summary(
        _prepared_deployment(reset_scale=False),
        _production_alias(),
        source="flag",
        force_env_build=False,
    )

    effective_changes = {row.label: row for row in summary.effective_changes}

    assert effective_changes["Machine Types"].production == "GPU-A100"
    assert effective_changes["Machine Types"].after_deploy == "GPU-H100"
    assert effective_changes["Machine Types"].note is None
    assert effective_changes["Max Multiplexing"].production == "1"
    assert effective_changes["Max Multiplexing"].after_deploy == "4"
    assert effective_changes["Startup Timeout"].production == "600"
    assert effective_changes["Startup Timeout"].after_deploy == "900"

    assert effective_changes["Keep Alive"].production == "300"
    assert effective_changes["Keep Alive"].after_deploy == "10"
    assert (
        effective_changes["Keep Alive"].note
        == "Code value will not apply without --reset-scale"
    )
    assert effective_changes["Max Concurrency"].production == "8"
    assert effective_changes["Max Concurrency"].after_deploy == "2"
    assert effective_changes["Regions"].production == "eu-west"
    assert effective_changes["Regions"].after_deploy == "us-east"


def test_diff_table_colors_unapplied_after_deploy_values_yellow():
    summary = _build_deployment_check_summary(
        _prepared_deployment(reset_scale=False),
        _production_alias(),
        source="flag",
        force_env_build=False,
    )

    table = _diff_table("Effective Production Diff", summary.effective_changes)
    labels = list(table.columns[0]._cells)
    after_deploy_cells = table.columns[2]._cells

    keep_alive_index = labels.index("Keep Alive")
    machine_types_index = labels.index("Machine Types")

    assert after_deploy_cells[keep_alive_index].plain == "10"
    assert after_deploy_cells[keep_alive_index].style == "yellow"
    assert after_deploy_cells[machine_types_index].plain == "GPU-H100"
    assert after_deploy_cells[machine_types_index].style == ""


def test_render_auth_line_is_red_when_auth_changes():
    auth_line = _render_auth_line("private", "public")

    assert auth_line.plain == "Auth: private -> public"
    assert any(span.style == "red" for span in auth_line.spans)


def test_render_auth_line_is_not_red_when_auth_is_unchanged():
    auth_line = _render_auth_line("private", "private")

    assert auth_line.plain == "Auth: private -> private"
    assert all(span.style != "red" for span in auth_line.spans)


def test_render_deployment_strategy_line_colors_rolling_green():
    strategy_line = _render_deployment_strategy_line("rolling")

    assert strategy_line.plain == "Deployment strategy: rolling (this deployment only)"
    assert any(span.style == "green" for span in strategy_line.spans)


def test_render_deployment_strategy_line_colors_non_rolling_red():
    strategy_line = _render_deployment_strategy_line("recreate")

    assert strategy_line.plain == "Deployment strategy: recreate (this deployment only)"
    assert any(span.style == "red" for span in strategy_line.spans)


def test_render_environment_build_cache_line_colors_enabled_green():
    cache_line = _render_environment_build_cache_line(False)

    assert cache_line.plain == "Environment build cache: enabled"
    assert any(span.style == "green" for span in cache_line.spans)


def test_render_environment_build_cache_line_colors_disabled_orange():
    cache_line = _render_environment_build_cache_line(True)

    assert cache_line.plain == "Environment build cache: disabled (--no-cache)"
    assert any(span.style == "#ff8800" for span in cache_line.spans)


def test_build_deployment_check_summary_reset_scale_applies_code_scale():
    summary = _build_deployment_check_summary(
        _prepared_deployment(reset_scale=True),
        _production_alias(),
        source="flag",
        force_env_build=True,
    )

    effective_changes = {row.label: row for row in summary.effective_changes}

    assert all(row.note is None for row in summary.effective_changes)
    assert effective_changes["Keep Alive"].production == "300"
    assert effective_changes["Keep Alive"].after_deploy == "10"
    assert effective_changes["Max Concurrency"].production == "8"
    assert effective_changes["Max Concurrency"].after_deploy == "2"
    assert effective_changes["Regions"].production == "eu-west"
    assert effective_changes["Regions"].after_deploy == "us-east"


def test_build_deployment_check_summary_handles_none_regions():
    summary = _build_deployment_check_summary(
        _prepared_deployment(
            reset_scale=True,
            host_options={
                "machine_type": ["GPU-H100"],
                "keep_alive": 10,
                "max_concurrency": 2,
                "min_concurrency": 0,
                "concurrency_buffer": 0,
                "concurrency_buffer_perc": 0,
                "scaling_delay": 0,
                "max_multiplexing": 4,
                "request_timeout": 3600,
                "startup_timeout": 900,
                "regions": None,
            },
        ),
        _production_alias(valid_regions=["eu-west"]),
        source="flag",
        force_env_build=False,
    )

    effective_changes = {row.label: row for row in summary.effective_changes}

    assert effective_changes["Regions"].production == "eu-west"
    assert effective_changes["Regions"].after_deploy == "any"


def test_render_deployment_check_summary_removes_triggered_by_and_scale_behavior():
    summary = _build_deployment_check_summary(
        _prepared_deployment(reset_scale=False),
        _production_alias(),
        source="flag",
        force_env_build=False,
    )

    rendered = _render_summary(summary)

    assert "Triggered by:" not in rendered
    assert "Scale behavior:" not in rendered
    assert "Code Values Not Applied" not in rendered
    assert "Code value will not apply without --reset-scale" in rendered
    assert rendered.index("Target:") < rendered.index("Current revision:")
    assert "Deployment strategy:" in rendered
    assert "Environment build cache:" in rendered


@patch("fal.api.deploy.execute_prepared_deployment")
@patch("fal.api.deploy.prepare_deployment")
@patch("fal.cli.deploy._get_production_alias", return_value=None)
@patch("sys.stdin.isatty", return_value=True)
@patch("builtins.input", return_value="ConFiRm")
def test_deploy_with_check_prepares_and_executes(
    mock_input,
    mock_isatty,
    mock_get_production_alias,
    mock_prepare_deployment,
    mock_execute_prepared_deployment,
):
    mock_prepare_deployment.return_value = _prepared_deployment(reset_scale=False)
    mock_execute_prepared_deployment.return_value = MagicMock(
        revision="rev-checked",
        app_name="my-app",
        auth_mode="public",
        urls={"playground": {}, "sync": {}, "async": {}},
        log_url="https://fal.ai/logs/checked",
    )

    args = mock_args(app_ref=("src/my_app/inference.py", "MyApp"))
    args.check = True

    _deploy(args)

    mock_prepare_deployment.assert_called_once()
    mock_execute_prepared_deployment.assert_called_once()
    mock_input.assert_called_once_with("Type 'confirm' to confirm deployment: ")
