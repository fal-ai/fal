from pathlib import Path
from typing import Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

from fal.api import Options
from fal.cli._result_handlers import PrepareRequirementsCallback
from fal.cli.main import parse_args
from fal.cli.run import _run
from fal.project import find_project_root


def test_run():
    args = parse_args(["run", "/my/path.py::myfunc"])
    assert args.func == _run
    assert args.func_ref == ("/my/path.py", "myfunc")
    assert args.machine_type is None
    assert args.limit_max_requests is None
    assert args.exposed_port is None
    assert args.exposed_metrics_port is None


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


def test_run_with_exposed_port_options():
    args = parse_args(
        [
            "run",
            "/my/path.py::myfunc",
            "--local",
            "--exposed-port",
            "3000",
            "--exposed-metrics-port",
            "3001",
        ]
    )
    assert args.func == _run
    assert args.func_ref == ("/my/path.py", "myfunc")
    assert args.local is True
    assert args.exposed_port == 3000
    assert args.exposed_metrics_port == 3001


@patch("fal.api.client.SyncServerlessClient._create_host")
@patch("fal.utils.load_function_from")
def test_run_uses_cli_auth_when_provided(mock_load_function_from, mock_create_host):
    host = mocked_fal_serverless_host("my-host")
    mock_create_host.return_value = host

    loaded = mocked_loaded_function(app_auth="private")
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
                "machine_type": "GPU-H100",
                "num_gpus": 2,
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
            "entrypoint-app": {
                "python_entry_point": "simple.app:SimpleApp",
                "python_version": "3.12",
                "requirements": ["fal"],
                "exposed_port": 9000,
            },
        }
    }


def mocked_fal_serverless_host(host):
    mock = MagicMock()
    mock.host = host
    mock.local_project_root = ""
    mock.prepare_options.side_effect = lambda options, **_: options
    return mock


_DEFAULT_FUNC = object()


def mocked_loaded_function(
    *,
    options: Optional[Options] = None,
    func=_DEFAULT_FUNC,
    app_name=None,
    app_auth=None,
):
    isolated_function = MagicMock()
    isolated_function.options = options or Options()
    isolated_function.func = (lambda: None) if func is _DEFAULT_FUNC else func
    isolated_function.run_entrypoint = None
    isolated_function.endpoints = ["/"]

    loaded = MagicMock()
    loaded.function = isolated_function
    loaded.app_name = app_name
    loaded.app_auth = app_auth
    return loaded


def mock_args(
    host,
    func_ref: Tuple[str, Optional[str]],
    team: Optional[str] = None,
    no_cache: bool = False,
    auth: Optional[str] = "public",
    machine_type: Optional[str] = None,
    limit_max_requests: Optional[int] = None,
    local: bool = False,
    exposed_port: Optional[int] = None,
    exposed_metrics_port: Optional[int] = None,
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
    args.local = local
    args.exposed_port = exposed_port
    args.exposed_metrics_port = exposed_metrics_port
    return args


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.api.client.SyncServerlessClient._create_host")
@patch("fal.utils.load_function_from")
def test_run_forwards_python_entry_point_to_loader(
    mock_load_function_from,
    mock_create_host,
    mock_parse_toml,
    mock_find_toml,
    mock_parse_pyproject_toml,
):
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    host = mocked_fal_serverless_host("my-host")
    mock_create_host.return_value = host

    loaded = mocked_loaded_function(
        options=Options(
            environment={"python_version": "3.12", "requirements": ["fal"]},
            gateway={"exposed_port": 9000},
        ),
        func=None,
    )
    mock_load_function_from.return_value = loaded

    args = mock_args(func_ref=("entrypoint-app", None), host="my-host")

    _run(args)

    mock_create_host.assert_called_once_with(
        local_file_path="",
        local_project_root=str(Path("pyproject.toml").parent),
        environment_name=None,
    )
    mock_load_function_from.assert_called_once()
    _, call_kwargs = mock_load_function_from.call_args
    assert call_kwargs["python_entry_point"] == "simple.app:SimpleApp"
    assert call_kwargs["options"].environment["python_version"] == "3.12"
    assert call_kwargs["options"].environment["requirements"] == ["fal"]
    assert call_kwargs["options"].gateway["exposed_port"] == 9000


@patch("fal.cli._utils.find_pyproject_toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.api.client.SyncServerlessClient._create_host")
@patch("fal.utils.load_function_from")
def test_run_image_only_forwards_no_ref_to_loader(
    mock_load_function_from,
    mock_create_host,
    mock_parse_toml,
    mock_find_toml,
    tmp_path,
):
    dockerfile = "FROM debian:bookworm-slim\n"
    (tmp_path / "Dockerfile").write_text(dockerfile)
    mock_find_toml.return_value = str(tmp_path / "pyproject.toml")
    mock_parse_toml.return_value = {
        "apps": {
            "container-app": {
                "image": {"dockerfile": "Dockerfile"},
                "exposed_port": 8080,
            }
        }
    }

    host = mocked_fal_serverless_host("my-host")
    mock_create_host.return_value = host

    loaded = mocked_loaded_function(
        options=Options(
            environment={
                "kind": "container",
                "image": {
                    "dockerfile_str": dockerfile,
                    "use_isolate": False,
                },
            },
            gateway={"exposed_port": 8080},
        ),
        func=None,
        app_name="container-app",
    )
    mock_load_function_from.return_value = loaded

    args = mock_args(func_ref=("container-app", None), host="my-host")

    _run(args)

    mock_create_host.assert_called_once_with(
        local_file_path="",
        local_project_root=str(tmp_path),
        environment_name=None,
    )
    mock_load_function_from.assert_called_once()
    call_args, call_kwargs = mock_load_function_from.call_args
    assert call_args[1] is None
    assert call_args[2] is None
    assert call_kwargs["python_entry_point"] is None
    assert call_kwargs["options"].environment["kind"] == "container"
    assert call_kwargs["options"].environment["image"]["dockerfile_str"] == dockerfile
    assert call_kwargs["options"].environment["image"]["use_isolate"] is False
    assert call_kwargs["options"].gateway["exposed_port"] == 8080


@patch("fal.api.run.run")
@patch("fal.cli._utils.find_pyproject_toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.api.client.SyncServerlessClient._create_host")
@patch("fal.utils.load_function_from")
def test_run_image_only_builds_no_isolate_container(
    mock_load_function_from,
    mock_create_host,
    mock_parse_toml,
    mock_find_toml,
    mock_run_api,
    tmp_path,
):
    from fal.api.api import IsolatedFunction
    from fal.utils import LoadedFunction

    (tmp_path / "Dockerfile").write_text("FROM debian:bookworm-slim\n")
    mock_find_toml.return_value = str(tmp_path / "pyproject.toml")
    mock_parse_toml.return_value = {
        "apps": {"container-app": {"image": {"dockerfile": "Dockerfile"}}}
    }
    host = mocked_fal_serverless_host("my-host")
    mock_create_host.return_value = host

    def fake_load_function_from(
        host_arg,
        _file_path,
        _func_name,
        *,
        options,
        app_name,
        app_auth,
        **_kwargs,
    ):
        isolated_function = IsolatedFunction(
            host=host_arg,
            options=options,
            app_name=app_name,
            app_auth=app_auth,
        )
        return LoadedFunction(
            function=isolated_function,
            app_name=app_name,
            app_auth=app_auth,
            source_code=None,
        )

    mock_load_function_from.side_effect = fake_load_function_from

    args = mock_args(func_ref=("container-app", None), host="my-host")
    _run(args)

    host.build_environment.assert_called_once()
    _, run_kwargs = mock_run_api.call_args
    assert run_kwargs["build_environment"] is False


@patch("fal.api.client.SyncServerlessClient._create_host")
@patch("fal.utils.load_function_from")
def test_run_forwards_limit_max_requests_to_load_function_from(
    mock_load_function_from, mock_create_host
):
    host = mocked_fal_serverless_host("my-host")
    mock_create_host.return_value = host

    loaded = mocked_loaded_function(app_auth="private")
    mock_load_function_from.return_value = loaded

    args = mock_args(
        func_ref=("/my/path.py", "myfunc"),
        host="my-host",
        limit_max_requests=1,
    )

    _run(args)

    _, call_kwargs = mock_load_function_from.call_args
    assert call_kwargs["limit_max_requests"] == 1
    assert "local" not in call_kwargs
    host.prepare_options.assert_called_once()
    (options,) = host.prepare_options.call_args.args
    progress = host.prepare_options.call_args.kwargs["on_progress"]

    assert options is loaded.function.options
    assert host.prepare_options.call_args.kwargs["func"] is loaded.function.func
    assert isinstance(progress, PrepareRequirementsCallback)
    assert progress.console is args.console


@patch("fal.api.run.run")
@patch("fal.api.client.SyncServerlessClient._create_host")
@patch("fal.utils.load_function_from")
def test_run_forwards_exposed_port_options_to_run_api(
    mock_load_function_from,
    mock_create_host,
    mock_run_api,
):
    host = mocked_fal_serverless_host("my-host")
    mock_create_host.return_value = host

    loaded = mocked_loaded_function(app_auth="private")
    mock_load_function_from.return_value = loaded

    args = mock_args(
        func_ref=("/my/path.py", "myfunc"),
        host="my-host",
        local=True,
        exposed_port=3000,
        exposed_metrics_port=3001,
    )

    _run(args)

    host.prepare_options.assert_not_called()
    _, call_kwargs = mock_run_api.call_args
    assert call_kwargs["local"] is True
    assert call_kwargs["exposed_port"] == 3000
    assert call_kwargs["exposed_metrics_port"] == 3001


@patch("fal.api.client.SyncServerlessClient._create_host")
@patch("fal.utils.load_function_from")
def test_run_rejects_exposed_port_options_without_local(
    mock_load_function_from, mock_create_host
):
    args = mock_args(
        func_ref=("/my/path.py", "myfunc"),
        host="my-host",
        exposed_port=3000,
    )

    with pytest.raises(
        ValueError,
        match="--exposed-port and --exposed-metrics-port can only be used with --local",
    ):
        _run(args)

    mock_create_host.assert_not_called()
    mock_load_function_from.assert_not_called()


@patch("fal.api.client.SyncServerlessClient._create_host")
@patch("fal.utils.load_function_from")
def test_run_applies_machine_type_override(mock_load_function_from, mock_create_host):
    host = mocked_fal_serverless_host("my-host")
    mock_create_host.return_value = host

    loaded = mocked_loaded_function(options=Options(host={"machine_type": "GPU-A10G"}))
    mock_load_function_from.return_value = loaded

    args = mock_args(
        func_ref=("/my/path.py", "myfunc"),
        host="my-host",
        machine_type="GPU-H100",
    )

    _run(args)

    assert loaded.function.options.host["machine_type"] == "GPU-H100"


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
    loaded = mocked_loaded_function()
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

    loaded = mocked_loaded_function(
        options=Options(
            environment={"requirements": ["numpy==1.26.4"]},
            host={
                "machine_type": "GPU-H100",
                "num_gpus": 2,
                "min_concurrency": 2,
                "regions": ["us-east"],
            },
        )
    )
    mock_load_function_from.return_value = loaded

    args = mock_args(func_ref=("override-app", None), host="my-host")
    args.app_name = "cli-app"

    _run(args)

    _, call_kwargs = mock_load_function_from.call_args
    assert call_kwargs["app_name"] == "cli-app"
    assert call_kwargs["app_auth"] == "public"
    assert call_kwargs["options"].environment["requirements"] == ["numpy==1.26.4"]
    assert call_kwargs["options"].host["machine_type"] == "GPU-H100"
    assert call_kwargs["options"].host["num_gpus"] == 2
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

    loaded = mocked_loaded_function(
        options=Options(
            environment={"requirements": ["numpy==1.26.4"]},
            host={
                "machine_type": "GPU-H100",
                "num_gpus": 2,
                "min_concurrency": 2,
                "regions": ["us-east"],
            },
        )
    )
    mock_load_function_from.return_value = loaded

    args = mock_args(func_ref=("override-app", None), host="my-host")

    _run(args)

    _, call_kwargs = mock_load_function_from.call_args
    assert call_kwargs["app_name"] is None
    assert call_kwargs["app_auth"] == "public"
    assert call_kwargs["options"].environment["requirements"] == ["numpy==1.26.4"]
    assert call_kwargs["options"].host["machine_type"] == "GPU-H100"
    assert call_kwargs["options"].host["num_gpus"] == 2
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

    loaded = mocked_loaded_function()
    mock_load_function_from.return_value = loaded

    args = mock_args(func_ref=("team-app", None), host="my-host", auth="private")

    _run(args)

    _, call_kwargs = mock_load_function_from.call_args
    assert call_kwargs["app_auth"] == "private"


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
    mock_load_function_from.return_value = mocked_loaded_function()

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
    mock_load_function_from.return_value = mocked_loaded_function()

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
    mock_load_function_from.return_value = mocked_loaded_function()

    args = mock_args(func_ref=("my-app", None), host="my-host")

    _run(args)

    # Ensure the client was initialized with team=None
    mock_client.assert_called_once_with(host="my-host", team=None)
