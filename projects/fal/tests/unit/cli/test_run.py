from typing import Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

from fal.cli.main import parse_args
from fal.cli.run import _run
from fal.project import find_project_root


def test_run():
    args = parse_args(["run", "/my/path.py::myfunc"])
    assert args.func == _run
    assert args.func_ref == ("/my/path.py", "myfunc")
    assert args.machine_type is None
    assert args.limit_max_requests is None
    assert args.no_pickle is False


def test_run_with_env():
    args = parse_args(["run", "/my/path.py::myfunc", "--env", "dev"])
    assert args.func == _run
    assert args.func_ref == ("/my/path.py", "myfunc")
    assert args.env == "dev"
    assert args.auth == "public"


@patch.dict("os.environ", {"FAL_ENV": "from-env-var"})
def test_run_uses_fal_env_variable():
    args = parse_args(["run", "/my/path.py::myfunc"])
    assert args.env == "from-env-var"


@patch.dict("os.environ", {"FAL_ENV": "from-env-var"})
def test_run_cli_env_overrides_fal_env_variable():
    args = parse_args(["run", "/my/path.py::myfunc", "--env", "cli-env"])
    assert args.env == "cli-env"


def test_run_with_machine_type():
    args = parse_args(["run", "/my/path.py::myfunc", "--machine-type", "GPU-H100"])
    assert args.func == _run
    assert args.func_ref == ("/my/path.py", "myfunc")
    assert args.machine_type == "GPU-H100"


def test_run_with_limit_max_requests():
    args = parse_args(["run", "/my/path.py::myfunc", "--limit-max-requests", "1"])
    assert args.func == _run
    assert args.func_ref == ("/my/path.py", "myfunc")
    assert args.limit_max_requests == 1


def test_run_with_no_pickle():
    args = parse_args(["run", "/my/path.py::myfunc", "--no-pickle"])
    assert args.func == _run
    assert args.func_ref == ("/my/path.py", "myfunc")
    assert args.no_pickle is True


@patch("fal.api.client.SyncServerlessClient._create_host")
@patch("fal.utils.load_function_from")
def test_run_uses_cli_auth_when_provided(mock_load_function_from, mock_create_host):
    host = mocked_fal_serverless_host("my-host")
    mock_create_host.return_value = host

    isolated_function = MagicMock()
    loaded = MagicMock()
    loaded.function = isolated_function
    loaded.app_name = None
    loaded.app_auth = "private"
    mock_load_function_from.return_value = loaded

    args = mock_args(
        func_ref=("/my/path.py", "myfunc"),
        host="my-host",
        auth="public",
    )

    _run(args)

    _, call_kwargs = mock_load_function_from.call_args
    assert call_kwargs["app_auth"] == "public"


@pytest.fixture
def mock_parse_pyproject_toml():
    return {
        "apps": {
            "my-app": {
                "ref": "src/my_app/inference.py::MyApp",
                "auth": "shared",
            },
            "override-app": {
                "ref": "src/override_app/inference.py::OverrideApp",
                "name": "override-name",
                "auth": "private",
                "requirements": ["numpy==1.26.4"],
                "min_concurrency": 2,
                "regions": ["us-east"],
            },
            # auth is not provided for another-app
            "another-app": {
                "ref": "src/another_app/inference.py::AnotherApp",
            },
            "team-app": {
                "ref": "src/team_app/inference.py::TeamApp",
                "team": "my-team",
                "auth": "shared",
            },
        }
    }


def mocked_fal_serverless_host(host):
    mock = MagicMock()
    mock.host = host
    return mock


def mock_args(
    host,
    func_ref: Tuple[str, Optional[str]],
    team: Optional[str] = None,
    no_cache: bool = False,
    auth: Optional[str] = "public",
    machine_type: Optional[str] = None,
    limit_max_requests: Optional[int] = None,
    no_pickle: bool = False,
    local: bool = False,
):
    args = MagicMock()
    args.host = host
    args.func_ref = func_ref
    args.team = team
    args.no_cache = no_cache
    args.force_env_build = False
    args.auth = auth
    args.console = MagicMock()
    args.app_name = None
    args.env = None
    args.machine_type = machine_type
    args.limit_max_requests = limit_max_requests
    args.no_pickle = no_pickle
    args.local = local
    return args


@patch("fal.api.client.SyncServerlessClient._create_host")
@patch("fal.utils.load_no_pickle_function_from")
@patch("fal.utils.load_function_from")
def test_run_uses_no_pickle_loader_when_enabled(
    mock_load_function_from,
    mock_load_no_pickle_function_from,
    mock_create_host,
):
    host = mocked_fal_serverless_host("my-host")
    mock_create_host.return_value = host

    isolated_function = MagicMock()
    isolated_function.options = MagicMock()
    isolated_function.options.host = {}
    loaded = MagicMock()
    loaded.function = isolated_function
    loaded.app_name = None
    loaded.app_auth = None
    mock_load_no_pickle_function_from.return_value = loaded

    args = mock_args(
        func_ref=("/my/path.py", "myfunc"),
        host="my-host",
        no_pickle=True,
    )

    _run(args)

    mock_load_no_pickle_function_from.assert_called_once()
    mock_load_function_from.assert_not_called()
    mock_create_host.assert_called_once_with(environment_name=None)


@patch("fal.api.client.SyncServerlessClient._create_host")
@patch("fal.utils.load_no_pickle_function_from")
def test_run_with_no_pickle_and_local_runs_wrapper_locally(
    mock_load_no_pickle_function_from,
    mock_create_host,
):
    host = mocked_fal_serverless_host("my-host")
    mock_create_host.return_value = host

    isolated_function = MagicMock()
    isolated_function.options = MagicMock()
    isolated_function.options.host = {}
    loaded = MagicMock()
    loaded.function = isolated_function
    loaded.app_name = None
    loaded.app_auth = None
    mock_load_no_pickle_function_from.return_value = loaded

    args = mock_args(
        func_ref=("/my/path.py", "myfunc"),
        host="my-host",
        no_pickle=True,
        local=True,
    )

    _run(args)

    isolated_function.run_local.assert_called_once_with()
    isolated_function.assert_not_called()


@patch("fal.api.client.SyncServerlessClient._create_host")
@patch("fal.utils.load_function_from")
def test_run_forwards_limit_max_requests_to_load_function_from(
    mock_load_function_from, mock_create_host
):
    host = mocked_fal_serverless_host("my-host")
    mock_create_host.return_value = host

    isolated_function = MagicMock()
    loaded = MagicMock()
    loaded.function = isolated_function
    loaded.app_name = None
    loaded.app_auth = "private"
    mock_load_function_from.return_value = loaded

    args = mock_args(
        func_ref=("/my/path.py", "myfunc"),
        host="my-host",
        limit_max_requests=1,
    )

    _run(args)

    _, call_kwargs = mock_load_function_from.call_args
    assert call_kwargs["limit_max_requests"] == 1


@patch("fal.api.client.SyncServerlessClient._create_host")
@patch("fal.utils.load_function_from")
def test_run_applies_machine_type_override(mock_load_function_from, mock_create_host):
    host = mocked_fal_serverless_host("my-host")
    mock_create_host.return_value = host

    isolated_function = MagicMock()
    isolated_function.options = MagicMock()
    isolated_function.options.host = {"machine_type": "GPU-A10G"}
    loaded = MagicMock()
    loaded.function = isolated_function
    loaded.app_name = None
    loaded.app_auth = None
    mock_load_function_from.return_value = loaded

    args = mock_args(
        func_ref=("/my/path.py", "myfunc"),
        host="my-host",
        machine_type="GPU-H100",
    )

    _run(args)

    assert isolated_function.options.host["machine_type"] == "GPU-H100"


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.api.client.SyncServerlessClient._create_host")
@patch("fal.utils.load_function_from")
def test_run_with_toml_success(
    mock_load_function_from,
    mock_create_host,
    mock_parse_toml,
    mock_find_toml,
    mock_parse_pyproject_toml,
):
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    host = mocked_fal_serverless_host("my-host")
    mock_create_host.return_value = host
    loaded = MagicMock()
    loaded.function = MagicMock()
    loaded.function.options = MagicMock()
    loaded.function.options.host = {}
    loaded.app_name = None
    loaded.app_auth = None
    mock_load_function_from.return_value = loaded

    args = mock_args(func_ref=("my-app", None), host="my-host")

    _run(args)

    project_root, _ = find_project_root(None)

    # Ensure the correct app is ran
    mock_load_function_from.assert_called_once()
    _, call_kwargs = mock_load_function_from.call_args
    assert call_kwargs["force_env_build"] is False
    assert call_kwargs["options"].host == {}
    assert call_kwargs["options"].environment == {}
    assert call_kwargs["options"].gateway == {}

    assert call_kwargs["app_auth"] == "public"
    assert call_kwargs["app_name"] is None


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.api.client.SyncServerlessClient._create_host")
@patch("fal.utils.load_function_from")
def test_run_with_toml_cli_name_override(
    mock_load_function_from,
    mock_create_host,
    mock_parse_toml,
    mock_find_toml,
    mock_parse_pyproject_toml,
):
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    host = mocked_fal_serverless_host("my-host")
    mock_create_host.return_value = host

    loaded = MagicMock()
    loaded.function = MagicMock()
    loaded.function.options = MagicMock()
    loaded.function.options.host = {}
    loaded.app_name = None
    loaded.app_auth = None
    mock_load_function_from.return_value = loaded

    args = mock_args(func_ref=("override-app", None), host="my-host")
    args.app_name = "cli-app"

    _run(args)

    _, call_kwargs = mock_load_function_from.call_args
    assert call_kwargs["app_name"] == "cli-app"
    assert call_kwargs["app_auth"] == "public"
    assert call_kwargs["options"].environment["requirements"] == ["numpy==1.26.4"]
    assert call_kwargs["options"].host["min_concurrency"] == 2
    assert call_kwargs["options"].host["regions"] == ["us-east"]


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.api.client.SyncServerlessClient._create_host")
@patch("fal.utils.load_function_from")
def test_run_with_toml_overrides_applied(
    mock_load_function_from,
    mock_create_host,
    mock_parse_toml,
    mock_find_toml,
    mock_parse_pyproject_toml,
):
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    host = mocked_fal_serverless_host("my-host")
    mock_create_host.return_value = host

    isolated_function = MagicMock()
    isolated_function.options = MagicMock()
    isolated_function.options.host = {}
    isolated_function.options.add_requirements = MagicMock()
    loaded = MagicMock()
    loaded.function = isolated_function
    loaded.app_name = None
    loaded.app_auth = None
    mock_load_function_from.return_value = loaded

    args = mock_args(func_ref=("override-app", None), host="my-host")

    _run(args)

    _, call_kwargs = mock_load_function_from.call_args
    assert call_kwargs["app_name"] is None
    assert call_kwargs["app_auth"] == "public"
    assert call_kwargs["options"].environment["requirements"] == ["numpy==1.26.4"]
    assert call_kwargs["options"].host["min_concurrency"] == 2
    assert call_kwargs["options"].host["regions"] == ["us-east"]


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.api.client.SyncServerlessClient._create_host")
@patch("fal.utils.load_function_from")
def test_run_with_toml_cli_auth_override(
    mock_load_function_from,
    mock_create_host,
    mock_parse_toml,
    mock_find_toml,
    mock_parse_pyproject_toml,
):
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    host = mocked_fal_serverless_host("my-host")
    mock_create_host.return_value = host

    loaded = MagicMock()
    loaded.function = MagicMock()
    loaded.function.options = MagicMock()
    loaded.function.options.host = {}
    loaded.app_name = None
    loaded.app_auth = None
    mock_load_function_from.return_value = loaded

    args = mock_args(func_ref=("team-app", None), host="my-host", auth="private")

    _run(args)

    _, call_kwargs = mock_load_function_from.call_args
    assert call_kwargs["app_auth"] == "private"


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.api.client.SyncServerlessClient._create_host")
@patch("fal.utils.load_no_pickle_function_from")
@patch("fal.utils.load_function_from")
def test_run_with_no_pickle_and_toml_app(
    mock_load_function_from,
    mock_load_no_pickle_function_from,
    mock_create_host,
    mock_parse_toml,
    mock_find_toml,
    mock_parse_pyproject_toml,
):
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    host = mocked_fal_serverless_host("my-host")
    mock_create_host.return_value = host

    isolated_function = MagicMock()
    isolated_function.options = MagicMock()
    isolated_function.options.host = {}
    loaded = MagicMock()
    loaded.function = isolated_function
    loaded.app_name = None
    loaded.app_auth = None
    mock_load_no_pickle_function_from.return_value = loaded

    args = mock_args(func_ref=("override-app", None), host="my-host", no_pickle=True)

    _run(args)

    mock_load_function_from.assert_not_called()
    _, call_kwargs = mock_load_no_pickle_function_from.call_args
    assert call_kwargs["app_name"] is None
    assert call_kwargs["app_auth"] == "public"
    assert call_kwargs["options"].environment["requirements"] == ["numpy==1.26.4"]
    assert call_kwargs["options"].host["min_concurrency"] == 2
    assert call_kwargs["options"].host["regions"] == ["us-east"]


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


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.api.client.SyncServerlessClient")
@patch("fal.utils.load_function_from")
def test_run_with_team_from_toml(
    mock_load_function_from,
    mock_client,
    mock_parse_toml,
    mock_find_toml,
    mock_parse_pyproject_toml,
):
    """Test that team is read from pyproject.toml and passed to SyncServerlessClient"""
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    host = mocked_fal_serverless_host("my-host")

    # Mock the client instance
    mock_client_instance = MagicMock()
    mock_client_instance._create_host.return_value = host
    mock_client.return_value = mock_client_instance

    args = mock_args(func_ref=("team-app", None), host="my-host")

    _run(args)

    # Ensure the client was initialized with the correct team
    mock_client.assert_called_once_with(host="my-host", team="my-team")

    project_root, _ = find_project_root(None)

    # Ensure the correct app is ran
    mock_load_function_from.assert_called_once()
    call_args, call_kwargs = mock_load_function_from.call_args
    assert call_args[0] == host
    assert call_args[1] == f"{project_root / 'src/team_app/inference.py'}"
    assert call_args[2] == "TeamApp"
    assert call_kwargs["force_env_build"] is False
    assert call_kwargs["options"].host == {}
    assert call_kwargs["options"].environment == {}
    assert call_kwargs["options"].gateway == {}


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.api.client.SyncServerlessClient")
@patch("fal.utils.load_function_from")
def test_run_with_team_from_toml_cli_team_override(
    mock_load_function_from,
    mock_client,
    mock_parse_toml,
    mock_find_toml,
    mock_parse_pyproject_toml,
):
    """Test that team is read from pyproject.toml and passed to SyncServerlessClient"""
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    host = mocked_fal_serverless_host("my-host")

    # Mock the client instance
    mock_client_instance = MagicMock()
    mock_client_instance._create_host.return_value = host
    mock_client.return_value = mock_client_instance

    args = mock_args(func_ref=("team-app", None), host="my-host", team="my-cli-team")

    _run(args)

    # Ensure the client was initialized with the correct team
    mock_client.assert_called_once_with(host="my-host", team="my-cli-team")

    project_root, _ = find_project_root(None)

    # Ensure the correct app is ran
    mock_load_function_from.assert_called_once()
    call_args, call_kwargs = mock_load_function_from.call_args
    assert call_args[0] == host
    assert call_args[1] == f"{project_root / 'src/team_app/inference.py'}"
    assert call_args[2] == "TeamApp"
    assert call_kwargs["force_env_build"] is False
    assert call_kwargs["options"].host == {}
    assert call_kwargs["options"].environment == {}
    assert call_kwargs["options"].gateway == {}


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.api.client.SyncServerlessClient")
@patch("fal.utils.load_function_from")
def test_run_without_team_in_toml(
    mock_load_function_from,
    mock_client,
    mock_parse_toml,
    mock_find_toml,
    mock_parse_pyproject_toml,
):
    """Test that team defaults to None when not specified in pyproject.toml"""
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    host = mocked_fal_serverless_host("my-host")

    # Mock the client instance
    mock_client_instance = MagicMock()
    mock_client_instance._create_host.return_value = host
    mock_client.return_value = mock_client_instance

    args = mock_args(func_ref=("my-app", None), host="my-host")

    _run(args)

    # Ensure the client was initialized with team=None
    mock_client.assert_called_once_with(host="my-host", team=None)
