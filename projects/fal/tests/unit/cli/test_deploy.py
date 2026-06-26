from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from fal.api import Options
from fal.api.api import IsolatedFunction
from fal.cli._utils import AppData
from fal.cli.deploy import _deploy
from fal.cli.deploy_check import (
    _build_deployment_check_summary,
    _diff_table,
    _is_truthy,
    _payload_requires_deploy_check,
    _render_auth_line,
    _render_deployment_check_summary,
    _render_deployment_strategy_line,
    _render_environment_build_cache_line,
    _resolve_deploy_check_source,
)
from fal.cli.main import parse_args
from fal.project import find_project_root
from fal.sdk import AliasInfo, ApplicationHealthCheckConfig


def test_deploy():
    args = parse_args(["deploy", "myfile.py::MyApp"])
    assert args.func == _deploy
    assert args.app_ref == ("myfile.py", "MyApp")


def test_deploy_with_env():
    args = parse_args(["deploy", "myfile.py::MyApp", "--env", "dev"])
    assert args.func == _deploy
    assert args.app_ref == ("myfile.py", "MyApp")
    assert args.env == "dev"


def test_execute_prepared_deployment_reuses_result_handler_for_build_by_default():
    from fal.api.deploy import PreparedDeployment, execute_prepared_deployment
    from fal.sdk import (
        RegisterApplicationResult,
        RegisterApplicationResultType,
        ServiceURLs,
    )

    host = MagicMock()
    options = Options(environment={"requirements": ["."]})
    prepared_options = Options(environment={"requirements": ["simple @ https://cdn"]})
    isolated_function = IsolatedFunction(
        host=host,
        raw_func=lambda: None,
        options=options,
        app_name="my-app",
        app_auth="public",
    )

    host.register.return_value = RegisterApplicationResult(
        result=RegisterApplicationResultType(application_id="app-id"),
        service_urls=ServiceURLs(
            playground="https://playground.example",
            run="https://run.example",
            queue="https://queue.example",
            ws="wss://ws.example",
            log="https://log.example",
        ),
    )
    host.prepare_options.return_value = prepared_options
    prepared = PreparedDeployment(
        host=host,
        loaded=SimpleNamespace(
            function=isolated_function,
            app_name="my-app",
            app_auth="public",
            source_code=None,
        ),
        app_data=AppData(deployment_strategy="rolling"),
        display_name="MyApp",
    )

    result_handler = MagicMock()
    prepare_options_handler = MagicMock()
    execute_prepared_deployment(
        prepared,
        result_handler=result_handler,
        prepare_options_handler=prepare_options_handler,
    )

    _, build_kwargs = host.build_environment.call_args
    assert build_kwargs["result_handler"] is result_handler
    assert host.build_environment.call_args.args[0] is prepared_options
    host.prepare_options.assert_called_once_with(
        options,
        func=isolated_function.func,
        on_progress=prepare_options_handler,
    )
    _, register_kwargs = host.register.call_args
    assert register_kwargs["options"] is prepared_options
    assert isolated_function.options is options
    assert isolated_function.options.environment["requirements"] == ["."]


def test_execute_prepared_deployment_builds_no_isolate_container():
    from fal.api.deploy import PreparedDeployment, execute_prepared_deployment
    from fal.sdk import (
        RegisterApplicationResult,
        RegisterApplicationResultType,
        ServiceURLs,
    )

    host = MagicMock()
    options = Options(
        environment={"kind": "container", "image": {"use_isolate": False}}
    )
    isolated_function = IsolatedFunction(
        host=host,
        options=options,
        app_name="container-app",
        app_auth="public",
    )
    host.register.return_value = RegisterApplicationResult(
        result=RegisterApplicationResultType(application_id="app-id"),
        service_urls=ServiceURLs(
            playground="https://playground.example",
            run="https://run.example",
            queue="https://queue.example",
            ws="wss://ws.example",
            log="https://log.example",
        ),
    )
    host.prepare_options.return_value = options
    prepared = PreparedDeployment(
        host=host,
        loaded=SimpleNamespace(
            function=isolated_function,
            app_name="container-app",
            app_auth="public",
            source_code=None,
            class_name=None,
        ),
        app_data=AppData(deployment_strategy="rolling"),
        display_name="container-app",
    )

    execute_prepared_deployment(prepared)

    host.build_environment.assert_called_once()
    _, register_kwargs = host.register.call_args
    assert register_kwargs["build_environment"] is False


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
    with patch(
        "fal.cli.deploy_check._admin_requires_deploy_check", return_value=False
    ) as mock_admin:
        yield mock_admin


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
                "machine_type": "GPU-H100",
                "num_gpus": 2,
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
            "advanced-options-app": {
                "ref": "src/advanced_options/inference.py::AdvancedOptionsApp",
                "keep_alive": 300,
                "private_logs": True,
                "_scheduler": "nomad",
                "_scheduler_options": {"storage_region": "us-east"},
                "skip_retry_conditions": [
                    "timeout",
                    "server_error",
                    "connection_error",
                ],
                "retry_config": {"server_error": {"retries": 3}},
                "termination_grace_period_seconds": 30,
                "secrets": ["OPENAI_API_KEY", "HF_TOKEN"],
                "data_mounts": ["/data", "/data/.cache"],
                "health_check": {
                    "path": "/health",
                    "start_period_seconds": 30,
                    "timeout_seconds": 5,
                    "failure_threshold": 3,
                    "call_regularly": True,
                },
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
            "entrypoint-app": {
                "python_entry_point": "simple.app:SimpleApp",
                "python_version": "3.12",
                "requirements": ["fal"],
            },
        }
    }


def _default_requirements_context_dir() -> str:
    return str(Path("pyproject.toml").parent.resolve())


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
@patch("fal.api.deploy.execute_prepared_deployment")
@patch("fal.api.deploy._prepare_deployment_from_reference")
def test_deploy_with_toml_success(
    mock_prepare_ref,
    mock_execute,
    mock_parse_toml,
    mock_find_toml,
    mock_parse_pyproject_toml,
):
    # Mocking the parse_pyproject_toml function to return a predefined dict
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    args = mock_args(app_ref=("my-app", None))

    _deploy(args)

    project_root, _ = find_project_root(None)

    # Ensure the correct app is deployed
    mock_prepare_ref.assert_called_once_with(
        mock_prepare_ref.call_args[0][0],
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
@patch("fal.api.deploy.execute_prepared_deployment")
@patch("fal.api.client.SyncServerlessClient._create_host")
@patch("fal.utils.load_function_from")
def test_deploy_python_entry_point_forwards_to_loader(
    mock_load_function_from,
    mock_create_host,
    mock_execute,
    mock_parse_toml,
    mock_find_toml,
    mock_parse_pyproject_toml,
):
    """End-to-end: deploying a python_entry_point app passes the entrypoint
    through to ``load_function_from`` and skips the cwd ``*.py`` glob.
    """
    mock_parse_toml.return_value = mock_parse_pyproject_toml
    mock_create_host.return_value = MagicMock()
    loaded = MagicMock()
    loaded.app_name = "entrypoint-app"
    loaded.app_auth = "public"
    loaded.class_name = "SimpleApp"
    loaded.function = MagicMock(spec=["entrypoint"])
    mock_load_function_from.return_value = loaded

    args = mock_args(app_ref=("entrypoint-app", None))
    _deploy(args)

    mock_create_host.assert_called_once()
    _, host_kwargs = mock_create_host.call_args
    assert host_kwargs["local_file_path"] == ""

    mock_load_function_from.assert_called_once()
    _, call_kwargs = mock_load_function_from.call_args
    assert call_kwargs["python_entry_point"] == "simple.app:SimpleApp"
    assert call_kwargs["options"].environment["python_version"] == "3.12"
    assert call_kwargs["options"].environment["requirements"] == ["fal"]
    assert (
        call_kwargs["options"].host["requirements_context_dir"]
        == _default_requirements_context_dir()
    )


@patch("fal.cli._utils.find_pyproject_toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.api.deploy.execute_prepared_deployment")
@patch("fal.api.client.SyncServerlessClient._create_host")
@patch("fal.utils.load_function_from")
def test_deploy_image_only_forwards_no_ref_to_loader(
    mock_load_function_from,
    mock_create_host,
    mock_execute,
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
                "auth": "public",
                "image": {"dockerfile": "Dockerfile"},
            }
        }
    }
    mock_create_host.return_value = MagicMock()
    loaded = MagicMock()
    loaded.app_name = "container-app"
    loaded.app_auth = "public"
    loaded.class_name = None
    loaded.function = MagicMock()
    mock_load_function_from.return_value = loaded

    args = mock_args(app_ref=("container-app", None))
    _deploy(args)

    mock_create_host.assert_called_once()
    _, host_kwargs = mock_create_host.call_args
    assert host_kwargs["local_file_path"] == ""

    mock_load_function_from.assert_called_once()
    call_args, call_kwargs = mock_load_function_from.call_args
    assert call_args[1] is None
    assert call_args[2] is None
    assert call_kwargs["python_entry_point"] is None
    assert call_kwargs["app_name"] == "container-app"
    assert call_kwargs["app_auth"] == "public"
    assert call_kwargs["options"].environment["kind"] == "container"
    assert call_kwargs["options"].environment["image"]["dockerfile_str"] == dockerfile
    assert call_kwargs["options"].environment["image"]["use_isolate"] is False


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.api.deploy.execute_prepared_deployment")
@patch("fal.api.deploy._prepare_deployment_from_reference")
def test_deploy_with_toml_python_entry_point(
    mock_prepare_ref,
    mock_execute,
    mock_parse_toml,
    mock_find_toml,
    mock_parse_pyproject_toml,
):
    """``fal deploy <app>`` with python_entry_point in pyproject.toml resolves
    cleanly without crashing on the absent ``ref``.
    """
    mock_parse_toml.return_value = mock_parse_pyproject_toml
    options = Options(
        host={"requirements_context_dir": _default_requirements_context_dir()},
        environment={"requirements": ["fal"], "python_version": "3.12"},
    )

    args = mock_args(app_ref=("entrypoint-app", None))

    _deploy(args)

    mock_prepare_ref.assert_called_once_with(
        mock_prepare_ref.call_args[0][0],
        (None, None),
        AppData(
            ref=None,
            python_entry_point="simple.app:SimpleApp",
            auth=None,
            deployment_strategy=None,
            reset_scale=False,
            team=None,
            name="entrypoint-app",
            options=options,
        ),
        force_env_build=False,
        environment_name=None,
    )


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.api.deploy.execute_prepared_deployment")
@patch("fal.api.deploy._prepare_deployment_from_reference")
def test_deploy_with_toml_no_auth(
    mock_prepare_ref,
    mock_execute,
    mock_parse_toml,
    mock_find_toml,
    mock_parse_pyproject_toml,
):
    # Mocking the parse_pyproject_toml function to return a predefined dict
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    args = mock_args(app_ref=("another-app", None))

    _deploy(args)

    project_root, _ = find_project_root(None)

    # Since auth is not provided for "another-app", it should default to "private"
    mock_prepare_ref.assert_called_once_with(
        mock_prepare_ref.call_args[0][0],
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
@patch("fal.api.deploy.execute_prepared_deployment")
@patch("fal.api.deploy._prepare_deployment_from_reference")
def test_deploy_with_toml_overrides_applied(
    mock_prepare_ref,
    mock_execute,
    mock_parse_toml,
    mock_find_toml,
    mock_parse_pyproject_toml,
):
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    args = mock_args(app_ref=("override-app", None))

    _deploy(args)

    project_root, _ = find_project_root(None)
    mock_prepare_ref.assert_called_once_with(
        mock_prepare_ref.call_args[0][0],
        (f"{project_root / 'src/override_app/inference.py'}", "OverrideApp"),
        AppData(
            ref=f"{project_root / 'src/override_app/inference.py'}::OverrideApp",
            auth="private",
            deployment_strategy=None,
            reset_scale=False,
            team=None,
            name="override-name",
            options=Options(
                host={
                    "min_concurrency": 2,
                    "machine_type": "GPU-H100",
                    "num_gpus": 2,
                    "regions": ["us-east"],
                    "requirements_context_dir": _default_requirements_context_dir(),
                },
                environment={"requirements": ["numpy==1.26.4"]},
            ),
        ),
        force_env_build=False,
        environment_name=None,
    )


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.api.deploy.execute_prepared_deployment")
@patch("fal.api.deploy._prepare_deployment_from_reference")
def test_deploy_with_toml_app_not_found(
    mock_prepare_ref,
    mock_execute,
    mock_parse_toml,
    mock_find_toml,
    mock_parse_pyproject_toml,
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
@patch("fal.api.deploy.execute_prepared_deployment")
@patch("fal.api.deploy._prepare_deployment_from_reference")
def test_deploy_with_toml_missing_ref_key(
    mock_prepare_ref, mock_execute, mock_parse_toml, mock_find_toml
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
@patch("fal.api.deploy.execute_prepared_deployment")
@patch("fal.api.deploy._prepare_deployment_from_reference")
def test_deploy_with_toml_extra_keys_in_toml(
    mock_prepare_ref,
    mock_execute,
    mock_parse_toml,
    mock_find_toml,
    mock_parse_pyproject_toml,
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
@patch("fal.api.deploy.execute_prepared_deployment")
@patch("fal.api.deploy._prepare_deployment_from_reference")
def test_deploy_with_toml_deployment_strategy(
    mock_prepare_ref,
    mock_execute,
    mock_parse_toml,
    mock_find_toml,
    mock_parse_pyproject_toml,
):
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    args = mock_args(app_ref=("my-app", None), strategy="rolling")

    _deploy(args)

    project_root, _ = find_project_root(None)

    mock_prepare_ref.assert_called_once_with(
        mock_prepare_ref.call_args[0][0],
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
@patch("fal.api.deploy.execute_prepared_deployment")
@patch("fal.api.deploy._prepare_deployment_from_reference")
def test_deploy_cli_strategy_overrides_toml_for_app_name(
    mock_prepare_ref,
    mock_execute,
    mock_parse_toml,
    mock_find_toml,
    mock_parse_pyproject_toml,
):
    # Regression test: an explicit --strategy on the CLI must take precedence
    # over the deployment_strategy set in pyproject.toml when deploying by app
    # name. Previously the TOML value silently won and --strategy was dropped.
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    # my-app has deployment_strategy="rolling" in the TOML fixture.
    args = mock_args(app_ref=("my-app", None), strategy="recreate")

    _deploy(args)

    project_root, _ = find_project_root(None)

    mock_prepare_ref.assert_called_once_with(
        mock_prepare_ref.call_args[0][0],
        (f"{project_root / 'src/my_app/inference.py'}", "MyApp"),
        AppData(
            ref=f"{project_root / 'src/my_app/inference.py'}::MyApp",
            auth="shared",
            deployment_strategy="recreate",
            reset_scale=False,
            team=None,
            name="my-app",
            local_project_root=".",
        ),
        force_env_build=False,
        environment_name=None,
    )


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.api.deploy.execute_prepared_deployment")
@patch("fal.api.deploy._prepare_deployment_from_reference")
def test_deploy_with_toml_default_deployment_strategy(
    mock_prepare_ref,
    mock_execute,
    mock_parse_toml,
    mock_find_toml,
    mock_parse_pyproject_toml,
):
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    args = mock_args(app_ref=("another-app", None))

    _deploy(args)

    project_root, _ = find_project_root(None)

    mock_prepare_ref.assert_called_once_with(
        mock_prepare_ref.call_args[0][0],
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
@patch("fal.api.deploy.execute_prepared_deployment")
@patch("fal.api.deploy._prepare_deployment_from_reference")
def test_deploy_with_cli_auth(
    mock_prepare_ref,
    mock_execute,
    mock_parse_toml,
    mock_find_toml,
    mock_parse_pyproject_toml,
):
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    args = mock_args(app_ref=("src/my_app/inference.py", "MyApp"), auth="shared")

    _deploy(args)

    project_root, _ = find_project_root(None)

    mock_prepare_ref.assert_called_once_with(
        mock_prepare_ref.call_args[0][0],
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
@patch("fal.api.deploy.execute_prepared_deployment")
@patch("fal.api.deploy._prepare_deployment_from_reference")
def test_deploy_with_cli_app_name(
    mock_prepare_ref,
    mock_execute,
    mock_parse_toml,
    mock_find_toml,
    mock_parse_pyproject_toml,
):
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    args = mock_args(
        app_ref=("src/my_app/inference.py", "MyApp"),
        app_name="cli-app",
    )

    _deploy(args)

    project_root, _ = find_project_root(None)

    mock_prepare_ref.assert_called_once_with(
        mock_prepare_ref.call_args[0][0],
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
@patch("fal.api.deploy.execute_prepared_deployment")
@patch("fal.api.deploy._prepare_deployment_from_reference")
def test_deploy_with_cli_deployment_strategy(
    mock_prepare_ref,
    mock_execute,
    mock_parse_toml,
    mock_find_toml,
    mock_parse_pyproject_toml,
):
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    args = mock_args(app_ref=("src/my_app/inference.py", "MyApp"), strategy="rolling")

    _deploy(args)

    project_root, _ = find_project_root(None)

    mock_prepare_ref.assert_called_once_with(
        mock_prepare_ref.call_args[0][0],
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
@patch("fal.api.deploy.execute_prepared_deployment")
@patch("fal.api.deploy._prepare_deployment_from_reference")
def test_deploy_with_cli_reset_scale(
    mock_prepare_ref,
    mock_execute,
    mock_parse_toml,
    mock_find_toml,
    mock_parse_pyproject_toml,
):
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    args = mock_args(app_ref=("src/my_app/inference.py", "MyApp"), reset_scale=True)

    _deploy(args)

    project_root, _ = find_project_root(None)

    mock_prepare_ref.assert_called_once_with(
        mock_prepare_ref.call_args[0][0],
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
@patch("fal.api.deploy.execute_prepared_deployment")
@patch("fal.api.deploy._prepare_deployment_from_reference")
def test_deploy_with_cli_scale(
    mock_prepare_ref,
    mock_execute,
    mock_parse_toml,
    mock_find_toml,
    mock_parse_pyproject_toml,
):
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    args = mock_args(app_ref=("src/my_app/inference.py", "MyApp"))

    _deploy(args)

    project_root, _ = find_project_root(None)

    mock_prepare_ref.assert_called_once_with(
        mock_prepare_ref.call_args[0][0],
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
@patch("fal.api.deploy.execute_prepared_deployment")
@patch("fal.api.deploy._prepare_deployment_from_reference")
def test_deploy_with_cli_no_cache(
    mock_prepare_ref,
    mock_execute,
    mock_parse_toml,
    mock_find_toml,
    mock_parse_pyproject_toml,
):
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    args = mock_args(app_ref=("src/my_app/inference.py", "MyApp"), no_cache=True)

    _deploy(args)

    project_root, _ = find_project_root(None)

    mock_prepare_ref.assert_called_once_with(
        mock_prepare_ref.call_args[0][0],
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
@patch("fal.api.deploy.execute_prepared_deployment")
@patch("fal.api.deploy.prepare_deployment")
@patch("fal.cli.deploy.SyncServerlessClient")
def test_deploy_with_team_from_toml(
    mock_client,
    mock_prepare,
    mock_execute,
    mock_parse_toml,
    mock_find_toml,
    mock_parse_pyproject_toml,
):
    """Test that team is read from pyproject.toml and passed to SyncServerlessClient"""
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    # Mock the client instance
    mock_client_instance = MagicMock()
    mock_client.return_value = mock_client_instance

    args = mock_args(app_ref=("team-app", None))
    args.host = "my-host"

    _deploy(args)

    # Ensure the client was initialized with the correct team
    mock_client.assert_called_once_with(host="my-host", team="my-team")


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.api.deploy.execute_prepared_deployment")
@patch("fal.api.deploy.prepare_deployment")
@patch("fal.cli.deploy.SyncServerlessClient")
def test_deploy_with_team_from_toml_cli_team_override(
    mock_client,
    mock_prepare,
    mock_execute,
    mock_parse_toml,
    mock_find_toml,
    mock_parse_pyproject_toml,
):
    """Test that team is read from pyproject.toml and passed to SyncServerlessClient"""
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    # Mock the client instance
    mock_client_instance = MagicMock()
    mock_client.return_value = mock_client_instance

    args = mock_args(app_ref=("team-app", None), team="my-cli-team")
    args.host = "my-host"

    _deploy(args)

    # Ensure the client was initialized with the correct team
    mock_client.assert_called_once_with(host="my-host", team="my-cli-team")


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
@patch("fal.api.deploy.execute_prepared_deployment")
@patch("fal.api.deploy.prepare_deployment")
@patch("fal.cli.deploy.SyncServerlessClient")
def test_deploy_without_team_in_toml(
    mock_client,
    mock_prepare,
    mock_execute,
    mock_parse_toml,
    mock_find_toml,
    mock_parse_pyproject_toml,
):
    """Test that team defaults to None when not specified in pyproject.toml"""
    mock_parse_toml.return_value = mock_parse_pyproject_toml

    # Mock the client instance
    mock_client_instance = MagicMock()
    mock_client.return_value = mock_client_instance

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
def test_get_app_data_from_toml_with_python_version(mock_parse_toml, mock_find_toml):
    from fal.cli._utils import get_app_data_from_toml

    mock_parse_toml.return_value = {
        "apps": {
            "python-version-app": {
                "ref": "src/my_app/inference.py::MyApp",
                "python_version": "3.12",
            }
        }
    }

    toml_data = get_app_data_from_toml("python-version-app")

    assert toml_data.options.environment == {"python_version": "3.12"}


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_with_machine_type(
    mock_parse_toml, mock_find_toml, mock_parse_pyproject_toml
):
    from fal.cli._utils import get_app_data_from_toml

    mock_parse_toml.return_value = mock_parse_pyproject_toml

    toml_data = get_app_data_from_toml("override-app")

    assert toml_data.options.host["machine_type"] == "GPU-H100"


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_with_num_gpus(
    mock_parse_toml, mock_find_toml, mock_parse_pyproject_toml
):
    from fal.cli._utils import get_app_data_from_toml

    mock_parse_toml.return_value = mock_parse_pyproject_toml

    toml_data = get_app_data_from_toml("override-app")

    assert toml_data.options.host["num_gpus"] == 2


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_with_machine_type_fallbacks(
    mock_parse_toml, mock_find_toml
):
    from fal.cli._utils import get_app_data_from_toml

    mock_parse_toml.return_value = {
        "apps": {
            "gpu-app": {
                "ref": "src/gpu_app/inference.py::GpuApp",
                "machine_type": ["GPU-H100", "GPU-A100"],
            }
        }
    }

    toml_data = get_app_data_from_toml("gpu-app")

    assert toml_data.options.host["machine_type"] == ["GPU-H100", "GPU-A100"]


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_rejects_non_string_python_version(
    mock_parse_toml, mock_find_toml
):
    from fal.cli._utils import get_app_data_from_toml

    mock_parse_toml.return_value = {
        "apps": {
            "python-version-app": {
                "ref": "src/my_app/inference.py::MyApp",
                "python_version": 3.12,
            }
        }
    }

    with pytest.raises(
        ValueError,
        match=(
            "App python-version-app python_version must be a string in pyproject.toml"
        ),
    ):
        get_app_data_from_toml("python-version-app")


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
def test_get_app_data_from_toml_with_advanced_runtime_options(
    mock_parse_toml, mock_find_toml, mock_parse_pyproject_toml
):
    from fal.cli._utils import get_app_data_from_toml

    mock_parse_toml.return_value = mock_parse_pyproject_toml

    toml_data = get_app_data_from_toml("advanced-options-app")

    assert toml_data.options.host == {
        "keep_alive": 300,
        "private_logs": True,
        "_scheduler": "nomad",
        "_scheduler_options": {"storage_region": "us-east"},
        "skip_retry_conditions": [
            "timeout",
            "server_error",
            "connection_error",
        ],
        "retry_config": {"server_error": {"retries": 3}},
        "termination_grace_period_seconds": 30,
        "secrets": ["OPENAI_API_KEY", "HF_TOKEN"],
        "data_mounts": ["/data", "/data/.cache"],
        "health_check_config": ApplicationHealthCheckConfig(
            path="/health",
            start_period_seconds=30,
            timeout_seconds=5,
            failure_threshold=3,
            call_regularly=True,
        ),
    }


@patch("fal.cli._utils.find_pyproject_toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_resolves_app_files_context_dir_from_pyproject(
    mock_parse_toml, mock_find_toml, tmp_path
):
    from fal.cli._utils import get_app_data_from_toml

    mock_find_toml.return_value = str(tmp_path / "pyproject.toml")
    mock_parse_toml.return_value = {
        "apps": {
            "app-with-files": {
                "ref": "src/app.py::App",
                "app_files": ["assets"],
                "app_files_context_dir": ".",
            }
        }
    }

    toml_data = get_app_data_from_toml("app-with-files")

    assert toml_data.options.host["app_files_context_dir"] == str(tmp_path)


@patch("fal.cli._utils.find_pyproject_toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_resolves_requirements_context_dir_from_pyproject(
    mock_parse_toml, mock_find_toml, tmp_path
):
    from fal.cli._utils import get_app_data_from_toml

    mock_find_toml.return_value = str(tmp_path / "pyproject.toml")
    mock_parse_toml.return_value = {
        "apps": {
            "app-with-local-req": {
                "ref": "src/app.py::App",
                "requirements": [".[worker]"],
                "requirements_context_dir": "apps/../packages/my_package",
            }
        }
    }

    toml_data = get_app_data_from_toml("app-with-local-req")

    assert toml_data.options.host["requirements_context_dir"] == str(
        tmp_path / "packages" / "my_package"
    )


@patch("fal.cli._utils.find_pyproject_toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_preserves_absolute_requirements_context_dir(
    mock_parse_toml, mock_find_toml, tmp_path
):
    from fal.cli._utils import get_app_data_from_toml

    context_dir = tmp_path / "packages" / "my_package"
    mock_find_toml.return_value = str(tmp_path / "pyproject.toml")
    mock_parse_toml.return_value = {
        "apps": {
            "app-with-local-req": {
                "ref": "src/app.py::App",
                "requirements": [".[worker]"],
                "requirements_context_dir": str(context_dir),
            }
        }
    }

    toml_data = get_app_data_from_toml("app-with-local-req")

    assert toml_data.options.host["requirements_context_dir"] == str(context_dir)


@patch("fal.cli._utils.find_pyproject_toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_rejects_invalid_requirements_context_dir(
    mock_parse_toml, mock_find_toml, tmp_path
):
    from fal.cli._utils import get_app_data_from_toml

    mock_find_toml.return_value = str(tmp_path / "pyproject.toml")
    mock_parse_toml.return_value = {
        "apps": {
            "app-with-local-req": {
                "ref": "src/app.py::App",
                "requirements": [".[worker]"],
                "requirements_context_dir": ["packages/my_package"],
            }
        }
    }

    with pytest.raises(
        ValueError, match=r"requirements_context_dir must be a string\."
    ):
        get_app_data_from_toml("app-with-local-req")


@patch("fal.cli._utils.find_pyproject_toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_preserves_empty_app_files_context_dir(
    mock_parse_toml, mock_find_toml, tmp_path
):
    from fal.cli._utils import get_app_data_from_toml

    mock_find_toml.return_value = str(tmp_path / "pyproject.toml")
    mock_parse_toml.return_value = {
        "apps": {
            "app-with-files": {
                "ref": "src/app.py::App",
                "app_files": ["assets"],
                "app_files_context_dir": "",
            }
        }
    }

    toml_data = get_app_data_from_toml("app-with-files")

    assert toml_data.options.host["app_files_context_dir"] == ""


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


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("keep_alive", True, "keep_alive must be an integer."),
        ("private_logs", "yes", "private_logs must be a boolean."),
        ("_scheduler", "", "_scheduler must be a non-empty string."),
        ("_scheduler_options", "nomad", "_scheduler_options must be a table"),
        ("secrets", "OPENAI_API_KEY", "secrets must be a list of strings."),
        ("data_mounts", ["/data", 1], "data_mounts must be a list of strings."),
        (
            "skip_retry_conditions",
            ["client_error"],
            "Invalid skip_retry_conditions: client_error.",
        ),
        (
            "termination_grace_period_seconds",
            "30",
            "termination_grace_period_seconds must be an integer.",
        ),
    ],
)
@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_rejects_invalid_advanced_runtime_options(
    mock_parse_toml, mock_find_toml, field_name, value, message
):
    from fal.cli._utils import get_app_data_from_toml

    mock_parse_toml.return_value = {
        "apps": {
            "my-app": {
                "ref": "src/my_app/inference.py::MyApp",
                field_name: value,
            }
        }
    }

    with pytest.raises(ValueError, match=message):
        get_app_data_from_toml("my-app")


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_rejects_invalid_health_check(
    mock_parse_toml, mock_find_toml
):
    from fal.cli._utils import get_app_data_from_toml

    mock_parse_toml.return_value = {
        "apps": {
            "my-app": {
                "ref": "src/my_app/inference.py::MyApp",
                "health_check": {
                    "path": "/health",
                    "timeout_seconds": "5",
                },
            }
        }
    }

    with pytest.raises(
        ValueError,
        match="health_check.timeout_seconds must be an integer.",
    ):
        get_app_data_from_toml("my-app")


@patch("fal.cli._utils.find_pyproject_toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_with_image(mock_parse_toml, mock_find_toml, tmp_path):
    from fal.cli._utils import get_app_data_from_toml

    dockerfile = "FROM python:3.12-slim\n"
    (tmp_path / "Dockerfile").write_text(dockerfile)
    mock_find_toml.return_value = str(tmp_path / "pyproject.toml")
    mock_parse_toml.return_value = {
        "apps": {
            "container-app": {
                "ref": "src/my_app/inference.py::MyApp",
                "image": {
                    "dockerfile": "Dockerfile",
                    "build_args": {"FOO": "bar"},
                    "registries": {
                        "myregistry.com": {"username": "u", "password": "p"}
                    },
                    "secrets": {"TOKEN": "shh"},
                    "entrypoint": ["python", "-m", "server"],
                    "cmd": ["--host", "0.0.0.0", "--port", "8080"],
                },
            }
        }
    }

    toml_data = get_app_data_from_toml("container-app")

    env = toml_data.options.environment
    assert env["kind"] == "container"
    assert env["image"]["dockerfile_str"] == dockerfile
    assert env["image"]["build_args"] == {"FOO": "bar"}
    assert env["image"]["registries"] == {
        "myregistry.com": {"username": "u", "password": "p"}
    }
    assert env["image"]["secrets"] == {"TOKEN": "shh"}
    assert env["image"]["entrypoint"] == ["python", "-m", "server"]
    assert env["image"]["cmd"] == ["--host", "0.0.0.0", "--port", "8080"]
    assert "use_isolate" not in env["image"]


@patch("fal.cli._utils.find_pyproject_toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_with_image_reference(
    mock_parse_toml, mock_find_toml, tmp_path
):
    from fal.cli._utils import get_app_data_from_toml

    mock_find_toml.return_value = str(tmp_path / "pyproject.toml")
    mock_parse_toml.return_value = {
        "apps": {
            "container-app": {
                "ref": "src/my_app/inference.py::MyApp",
                "image": {
                    "image": "ghcr.io/fal-ai/container-app:latest",
                    "registries": {"ghcr.io": {"username": "u", "password": "p"}},
                    "entrypoint": ["python", "-m", "server"],
                    "cmd": ["--host", "0.0.0.0", "--port", "8080"],
                },
            }
        }
    }

    toml_data = get_app_data_from_toml("container-app")

    env = toml_data.options.environment
    assert env["kind"] == "container"
    assert env["image"]["image"] == "ghcr.io/fal-ai/container-app:latest"
    assert env["image"]["registries"] == {"ghcr.io": {"username": "u", "password": "p"}}
    assert env["image"]["entrypoint"] == ["python", "-m", "server"]
    assert env["image"]["cmd"] == ["--host", "0.0.0.0", "--port", "8080"]
    assert "dockerfile_str" not in env["image"]
    assert "use_isolate" not in env["image"]


@patch("fal.cli._utils.find_pyproject_toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_allows_python_entry_point_with_image(
    mock_parse_toml, mock_find_toml, tmp_path
):
    from fal.cli._utils import get_app_data_from_toml

    dockerfile = "FROM python:3.12-slim\n"
    (tmp_path / "Dockerfile").write_text(dockerfile)
    mock_find_toml.return_value = str(tmp_path / "pyproject.toml")
    mock_parse_toml.return_value = {
        "apps": {
            "container-app": {
                "python_entry_point": "simple.app:SimpleApp",
                "image": {"dockerfile": "Dockerfile"},
            }
        }
    }

    toml_data = get_app_data_from_toml("container-app")

    assert toml_data.ref is None
    assert toml_data.python_entry_point == "simple.app:SimpleApp"
    assert toml_data.options.environment["kind"] == "container"
    assert toml_data.options.environment["image"]["dockerfile_str"] == dockerfile
    assert "use_isolate" not in toml_data.options.environment["image"]


@patch("fal.cli._utils.find_pyproject_toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_allows_python_entry_point_with_image_reference(
    mock_parse_toml, mock_find_toml, tmp_path
):
    from fal.cli._utils import get_app_data_from_toml

    mock_find_toml.return_value = str(tmp_path / "pyproject.toml")
    mock_parse_toml.return_value = {
        "apps": {
            "container-app": {
                "python_entry_point": "simple.app:SimpleApp",
                "image": {"image": "ghcr.io/fal-ai/container-app:latest"},
            }
        }
    }

    toml_data = get_app_data_from_toml("container-app")

    assert toml_data.ref is None
    assert toml_data.python_entry_point == "simple.app:SimpleApp"
    assert toml_data.options.environment["kind"] == "container"
    assert (
        toml_data.options.environment["image"]["image"]
        == "ghcr.io/fal-ai/container-app:latest"
    )
    assert "use_isolate" not in toml_data.options.environment["image"]


@patch("fal.cli._utils.find_pyproject_toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_with_image_without_ref(
    mock_parse_toml, mock_find_toml, tmp_path
):
    from fal.cli._utils import get_app_data_from_toml

    dockerfile = "FROM debian:bookworm-slim\n"
    (tmp_path / "Dockerfile").write_text(dockerfile)
    mock_find_toml.return_value = str(tmp_path / "pyproject.toml")
    mock_parse_toml.return_value = {
        "apps": {
            "container-app": {
                "auth": "public",
                "machine_type": "GPU-H100",
                "image": {"dockerfile": "Dockerfile"},
            }
        }
    }

    toml_data = get_app_data_from_toml("container-app")

    assert toml_data.ref is None
    assert toml_data.python_entry_point is None
    assert toml_data.auth == "public"
    assert toml_data.options.host["machine_type"] == "GPU-H100"
    assert toml_data.options.environment["kind"] == "container"
    assert toml_data.options.environment["image"]["dockerfile_str"] == dockerfile
    assert toml_data.options.environment["image"]["use_isolate"] is False


@patch("fal.cli._utils.find_pyproject_toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_with_image_reference_without_ref(
    mock_parse_toml, mock_find_toml, tmp_path
):
    from fal.cli._utils import get_app_data_from_toml

    mock_find_toml.return_value = str(tmp_path / "pyproject.toml")
    mock_parse_toml.return_value = {
        "apps": {
            "container-app": {
                "auth": "public",
                "machine_type": "GPU-H100",
                "image": {"image": "ghcr.io/fal-ai/container-app:latest"},
            }
        }
    }

    toml_data = get_app_data_from_toml("container-app")

    assert toml_data.ref is None
    assert toml_data.python_entry_point is None
    assert toml_data.auth == "public"
    assert toml_data.options.host["machine_type"] == "GPU-H100"
    assert toml_data.options.environment["kind"] == "container"
    assert (
        toml_data.options.environment["image"]["image"]
        == "ghcr.io/fal-ai/container-app:latest"
    )
    assert toml_data.options.environment["image"]["use_isolate"] is False


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_rejects_image_without_source(
    mock_parse_toml, mock_find_toml
):
    from fal.cli._utils import get_app_data_from_toml

    mock_parse_toml.return_value = {
        "apps": {
            "container-app": {
                "ref": "src/my_app/inference.py::MyApp",
                "image": {"build_args": {"FOO": "bar"}},
            }
        }
    }

    with pytest.raises(
        ValueError,
        match=(
            "App container-app image must specify either 'dockerfile' "
            r"\(path\) or 'image' \(container image\) in pyproject.toml"
        ),
    ):
        get_app_data_from_toml("container-app")


@patch("fal.cli._utils.find_pyproject_toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_rejects_image_with_multiple_sources(
    mock_parse_toml, mock_find_toml, tmp_path
):
    from fal.cli._utils import get_app_data_from_toml

    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim\n")
    mock_find_toml.return_value = str(tmp_path / "pyproject.toml")
    mock_parse_toml.return_value = {
        "apps": {
            "container-app": {
                "ref": "src/my_app/inference.py::MyApp",
                "image": {
                    "dockerfile": "Dockerfile",
                    "image": "ghcr.io/fal-ai/container-app:latest",
                },
            }
        }
    }

    with pytest.raises(
        ValueError,
        match=(
            "App container-app image must specify only one of "
            "'dockerfile' or 'image'"
        ),
    ):
        get_app_data_from_toml("container-app")


@pytest.mark.parametrize("image_ref", ["", "   ", 123])
@patch("fal.cli._utils.find_pyproject_toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_rejects_invalid_image_reference(
    mock_parse_toml, mock_find_toml, tmp_path, image_ref
):
    from fal.cli._utils import get_app_data_from_toml

    mock_find_toml.return_value = str(tmp_path / "pyproject.toml")
    mock_parse_toml.return_value = {
        "apps": {
            "container-app": {
                "ref": "src/my_app/inference.py::MyApp",
                "image": {"image": image_ref},
            }
        }
    }

    with pytest.raises(
        ValueError,
        match="App container-app image.image must be a non-empty string",
    ):
        get_app_data_from_toml("container-app")


@patch("fal.cli._utils.find_pyproject_toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_rejects_unknown_image_keys(
    mock_parse_toml, mock_find_toml, tmp_path
):
    from fal.cli._utils import get_app_data_from_toml

    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim\n")
    mock_find_toml.return_value = str(tmp_path / "pyproject.toml")
    mock_parse_toml.return_value = {
        "apps": {
            "container-app": {
                "ref": "src/my_app/inference.py::MyApp",
                "image": {
                    "dockerfile": "Dockerfile",
                    "bogus": "value",
                },
            }
        }
    }

    with pytest.raises(
        ValueError,
        match=r"Found unexpected keys in app container-app image",
    ):
        get_app_data_from_toml("container-app")


@patch("fal.cli._utils.find_pyproject_toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_rejects_unknown_image_reference_keys(
    mock_parse_toml, mock_find_toml, tmp_path
):
    from fal.cli._utils import get_app_data_from_toml

    mock_find_toml.return_value = str(tmp_path / "pyproject.toml")
    mock_parse_toml.return_value = {
        "apps": {
            "container-app": {
                "ref": "src/my_app/inference.py::MyApp",
                "image": {
                    "image": "ghcr.io/fal-ai/container-app:latest",
                    "bogus": "value",
                },
            }
        }
    }

    with pytest.raises(
        ValueError,
        match=r"Found unexpected keys in app container-app image",
    ):
        get_app_data_from_toml("container-app")


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("build_args", {"FOO": "bar"}),
        ("secrets", {"TOKEN": "shh"}),
    ],
)
@patch("fal.cli._utils.find_pyproject_toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_rejects_image_reference_build_keys(
    mock_parse_toml, mock_find_toml, tmp_path, field_name, value
):
    from fal.cli._utils import get_app_data_from_toml

    mock_find_toml.return_value = str(tmp_path / "pyproject.toml")
    mock_parse_toml.return_value = {
        "apps": {
            "container-app": {
                "image": {
                    "image": "ghcr.io/fal-ai/container-app:latest",
                    field_name: value,
                },
            }
        }
    }

    with pytest.raises(
        ValueError,
        match=f"{field_name!r}.*only supported with image.dockerfile",
    ):
        get_app_data_from_toml("container-app")


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("entrypoint", 123),
        ("entrypoint", ["python", 123]),
        ("cmd", 123),
        ("cmd", ["--port", 8080]),
    ],
)
@patch("fal.cli._utils.find_pyproject_toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_rejects_invalid_image_command_overrides(
    mock_parse_toml, mock_find_toml, tmp_path, field_name, value
):
    from fal.cli._utils import get_app_data_from_toml

    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim\n")
    mock_find_toml.return_value = str(tmp_path / "pyproject.toml")
    mock_parse_toml.return_value = {
        "apps": {
            "container-app": {
                "image": {
                    "dockerfile": "Dockerfile",
                    field_name: value,
                },
            }
        }
    }

    with pytest.raises(
        ValueError,
        match=f"{field_name} must be a string or list of strings",
    ):
        get_app_data_from_toml("container-app")


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("entrypoint", 123),
        ("entrypoint", ["python", 123]),
        ("cmd", 123),
        ("cmd", ["--port", 8080]),
    ],
)
@patch("fal.cli._utils.find_pyproject_toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_rejects_invalid_image_reference_command_overrides(
    mock_parse_toml, mock_find_toml, tmp_path, field_name, value
):
    from fal.cli._utils import get_app_data_from_toml

    mock_find_toml.return_value = str(tmp_path / "pyproject.toml")
    mock_parse_toml.return_value = {
        "apps": {
            "container-app": {
                "image": {
                    "image": "ghcr.io/fal-ai/container-app:latest",
                    field_name: value,
                },
            }
        }
    }

    with pytest.raises(
        ValueError,
        match=f"{field_name} must be a string or list of strings",
    ):
        get_app_data_from_toml("container-app")


@pytest.mark.parametrize(
    "registries",
    [
        {"ghcr.io": {"username": "u"}},
        {"ghcr.io": {"password": "p"}},
    ],
)
@patch("fal.cli._utils.find_pyproject_toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_rejects_invalid_image_reference_registries(
    mock_parse_toml, mock_find_toml, tmp_path, registries
):
    from fal.cli._utils import get_app_data_from_toml

    mock_find_toml.return_value = str(tmp_path / "pyproject.toml")
    mock_parse_toml.return_value = {
        "apps": {
            "container-app": {
                "image": {
                    "image": "ghcr.io/fal-ai/container-app:latest",
                    "registries": registries,
                },
            }
        }
    }

    with pytest.raises(ValueError, match="Username and password"):
        get_app_data_from_toml("container-app")


@pytest.mark.parametrize(
    ("registries", "message"),
    [
        ([], "registries must be a table"),
        ("ghcr.io", "registries must be a table"),
        ({"ghcr.io": "token"}, "Each registry must be a table"),
    ],
)
@patch("fal.cli._utils.find_pyproject_toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_rejects_malformed_image_reference_registries(
    mock_parse_toml, mock_find_toml, tmp_path, registries, message
):
    from fal.cli._utils import get_app_data_from_toml

    mock_find_toml.return_value = str(tmp_path / "pyproject.toml")
    mock_parse_toml.return_value = {
        "apps": {
            "container-app": {
                "image": {
                    "image": "ghcr.io/fal-ai/container-app:latest",
                    "registries": registries,
                },
            }
        }
    }

    with pytest.raises(ValueError, match=message):
        get_app_data_from_toml("container-app")


@patch("fal.cli._utils.find_pyproject_toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_rejects_image_with_app_files(
    mock_parse_toml, mock_find_toml, tmp_path
):
    from fal.cli._utils import get_app_data_from_toml

    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim\n")
    mock_find_toml.return_value = str(tmp_path / "pyproject.toml")
    mock_parse_toml.return_value = {
        "apps": {
            "container-app": {
                "ref": "src/my_app/inference.py::MyApp",
                "image": {"dockerfile": "Dockerfile"},
                "app_files": ["assets"],
            }
        }
    }

    with pytest.raises(
        ValueError,
        match="app_files is not supported for container apps.",
    ):
        get_app_data_from_toml("container-app")


@patch("fal.cli._utils.find_pyproject_toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_allows_image_with_empty_app_files(
    mock_parse_toml, mock_find_toml, tmp_path
):
    from fal.cli._utils import get_app_data_from_toml

    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim\n")
    mock_find_toml.return_value = str(tmp_path / "pyproject.toml")
    mock_parse_toml.return_value = {
        "apps": {
            "container-app": {
                "ref": "src/my_app/inference.py::MyApp",
                "image": {"dockerfile": "Dockerfile"},
                "app_files": [],
            }
        }
    }

    toml_data = get_app_data_from_toml("container-app")
    assert toml_data.options.environment["kind"] == "container"


@patch("fal.cli._utils.find_pyproject_toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_get_app_data_from_toml_rejects_missing_dockerfile(
    mock_parse_toml, mock_find_toml, tmp_path
):
    from fal.cli._utils import get_app_data_from_toml

    mock_find_toml.return_value = str(tmp_path / "pyproject.toml")
    mock_parse_toml.return_value = {
        "apps": {
            "container-app": {
                "ref": "src/my_app/inference.py::MyApp",
                "image": {"dockerfile": "missing.Dockerfile"},
            }
        }
    }

    with pytest.raises(
        ValueError,
        match=r"App container-app image.dockerfile not found:",
    ):
        get_app_data_from_toml("container-app")


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
@patch("fal.api.deploy.execute_prepared_deployment")
@patch("fal.api.deploy.prepare_deployment")
@patch("fal.cli.deploy.SyncServerlessClient")
def test_deploy_team_lookup_uses_silent_toml_read(
    mock_client, mock_prepare, mock_execute, mock_get_app_data
):
    mock_get_app_data.return_value = AppData(team="my-team")

    # Mock the client instance
    mock_client_instance = MagicMock()
    mock_client.return_value = mock_client_instance

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


@patch.dict("os.environ", {"FAL_DEPLOY_CHECK": "true"})
def test_resolve_deploy_check_source_env_still_applies_with_yes(
    disable_admin_deploy_check_lookup,
):
    args = mock_args(app_ref=("src/my_app/inference.py", "MyApp"))
    args.yes = True

    assert _resolve_deploy_check_source(args, MagicMock()) == "env"
    disable_admin_deploy_check_lookup.assert_not_called()


@patch("fal.cli.deploy_check._admin_requires_deploy_check", return_value=True)
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


def test_is_truthy_supports_integer_flag_values():
    assert _is_truthy(1) is True
    assert _is_truthy(-1) is True
    assert _is_truthy(0) is False


def test_payload_requires_deploy_check_supports_integer_flags():
    payload = SimpleNamespace(
        additional_properties={
            "org_config": {
                "deploy_check": 1,
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


def test_render_deployment_check_summary_includes_triggered_by():
    summary = _build_deployment_check_summary(
        _prepared_deployment(reset_scale=False),
        _production_alias(),
        source="flag",
        force_env_build=False,
    )

    rendered = _render_summary(summary)

    assert "Triggered by: flag" in rendered
    assert "Scale behavior:" not in rendered
    assert "Code Values Not Applied" not in rendered
    assert "Code value will not apply without --reset-scale" in rendered
    assert rendered.index("Target:") < rendered.index("Current revision:")
    assert "Deployment strategy:" in rendered
    assert "Environment build cache:" in rendered


@patch("fal.api.deploy.execute_prepared_deployment")
@patch("fal.api.deploy.prepare_deployment")
@patch("fal.cli.deploy_check._get_production_alias", return_value=None)
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


@patch("fal.api.deploy.execute_prepared_deployment")
@patch("fal.api.deploy.prepare_deployment")
@patch("fal.cli.deploy_check._get_production_alias", return_value=None)
@patch("sys.stdin.isatty", return_value=False)
@patch("builtins.input")
def test_deploy_with_check_and_yes_skips_prompt(
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
    args.yes = True

    _deploy(args)

    mock_prepare_deployment.assert_called_once()
    mock_execute_prepared_deployment.assert_called_once()
    mock_input.assert_not_called()


@patch.dict("os.environ", {"FAL_DEPLOY_CHECK": "true"})
@patch("fal.api.deploy.execute_prepared_deployment")
@patch("fal.api.deploy.prepare_deployment")
@patch("fal.cli.deploy_check._get_production_alias", return_value=None)
@patch("sys.stdin.isatty", return_value=False)
@patch("builtins.input")
def test_deploy_with_env_check_and_yes_renders_summary_without_prompt(
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
    args.console = Console(record=True, width=120)
    args.yes = True

    _deploy(args)

    rendered = args.console.export_text()
    assert "Deployment Check: my-app" in rendered
    assert "Target: my-app" in rendered
    mock_prepare_deployment.assert_called_once()
    mock_execute_prepared_deployment.assert_called_once()
    mock_input.assert_not_called()
