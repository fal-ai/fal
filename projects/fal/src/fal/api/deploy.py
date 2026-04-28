from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from fal.sdk import AuthModeLiteral, DeploymentStrategyLiteral

if TYPE_CHECKING:
    from openapi_fal_rest.client import Client

    from fal.cli._utils import AppData
    from fal.utils import LoadedFunction

    from .api import FalServerlessHost
    from .client import SyncServerlessClient

import json
from collections import namedtuple
from typing import cast

User = namedtuple("User", ["user_id", "username"])


@dataclass
class DeploymentResult:
    revision: str
    app_name: str
    urls: dict[str, dict[str, str]]
    log_url: str
    auth_mode: str


@dataclass
class PreparedDeployment:
    host: FalServerlessHost
    loaded: LoadedFunction
    app_data: AppData
    display_name: str
    environment_name: str | None = None


def _deployment_display_name(
    display_name: str | None,
    loaded: LoadedFunction,
) -> str:
    if display_name:
        return display_name
    if loaded.class_name:
        return loaded.class_name

    assert loaded.app_name
    return loaded.app_name


def _remove_http_and_port_from_url(url):
    # Remove http://
    if "http://" in url:
        url = url.replace("http://", "")

    # Remove https://
    if "https://" in url:
        url = url.replace("https://", "")

    # Remove port information
    url_parts = url.split(":")
    if len(url_parts) > 1:
        url = url_parts[0]

    return url


def _get_user(client: Client) -> User:
    from http import HTTPStatus

    import openapi_fal_rest.api.users.get_current_user as get_current_user

    from fal.api import FalServerlessError

    try:
        user_details_response = get_current_user.sync_detailed(
            client=client,
        )
    except Exception as e:
        raise FalServerlessError(f"Error fetching user details: {str(e)}")

    if user_details_response.status_code != HTTPStatus.OK:
        try:
            content = json.loads(user_details_response.content.decode("utf8"))
        except Exception:
            raise FalServerlessError(
                f"Error fetching user details: {user_details_response}"
            )
        else:
            raise FalServerlessError(content["detail"])
    try:
        full_user_id = user_details_response.parsed.user_id
        _provider, _, user_id = full_user_id.partition("|")
        if not user_id:
            user_id = full_user_id

        return User(user_id=user_id, username=user_details_response.parsed.nickname)
    except Exception as e:
        raise FalServerlessError(f"Could not parse the user data: {e}")


def _resolve_deployment_reference(
    app_ref: str | tuple[str | None, str | None] | None = None,
    *,
    app_name: str | None = None,
    auth: AuthModeLiteral | None = None,
    strategy: DeploymentStrategyLiteral = "rolling",
    reset_scale: bool = False,
) -> tuple[tuple[str | None, str | None], AppData]:
    from fal.cli._utils import AppData, get_app_data_from_toml, is_app_name
    from fal.cli.parser import RefAction

    if isinstance(app_ref, tuple):
        app_ref_tuple = app_ref
    elif app_ref:
        app_ref_tuple = RefAction.split_ref(app_ref)
    else:
        raise ValueError("Invalid app reference")

    app_data = AppData(
        auth=auth,
        deployment_strategy=cast(DeploymentStrategyLiteral, strategy),
        reset_scale=cast(bool, reset_scale),
        name=app_name,
    )

    app_ref_candidate = cast(tuple[str, str | None], app_ref_tuple)
    if is_app_name(app_ref_candidate):
        if app_name or auth:
            raise ValueError("Cannot use --app-name or --auth with app name reference.")

        resolved_app_name, _unused_func_name = app_ref_candidate
        assert resolved_app_name is not None

        app_data = get_app_data_from_toml(resolved_app_name)
        assert app_data.ref is not None
        file_path, func_name = RefAction.split_ref(app_data.ref)
        return (file_path, func_name), app_data

    file_path, func_name = app_ref_tuple
    if file_path is not None:
        file_path = str(Path(file_path).absolute())

    ref = f"{file_path}::{func_name}" if func_name else file_path
    return (file_path, func_name), replace(app_data, ref=ref)


def _prepare_deployment_from_reference(
    client: SyncServerlessClient,
    app_ref: tuple[str | Path | None, str | None],
    app_data: AppData,
    force_env_build: bool,
    environment_name: Optional[str] = None,
) -> PreparedDeployment:
    from fal.api import FalServerlessError
    from fal.utils import load_function_from

    file_path, func_name = app_ref
    if file_path is None:
        # Try to find a python file in the current directory
        options = list(Path(".").glob("*.py"))
        if len(options) == 0:
            raise FalServerlessError("No python files found in the current directory")
        elif len(options) > 1:
            raise FalServerlessError(
                "Multiple python files found in the current directory. "
                "Please specify the file path of the app you want to deploy."
            )

        [resolved_file_path] = options
        file_path = str(resolved_file_path)

    local_file_path = str(file_path)
    host = client._create_host(
        local_file_path=local_file_path,
        environment_name=environment_name,
    )
    loaded = load_function_from(
        host,
        local_file_path,
        func_name,
        force_env_build=force_env_build,
        options=app_data.options,
        app_name=app_data.name,
        app_auth=app_data.auth,
    )

    return PreparedDeployment(
        host=host,
        loaded=loaded,
        app_data=app_data,
        display_name=_deployment_display_name(func_name, loaded),
        environment_name=environment_name,
    )


def _deploy_from_reference(
    client: SyncServerlessClient,
    app_ref: tuple[str | Path | None, str | None],
    app_data: AppData,
    force_env_build: bool,
    environment_name: Optional[str] = None,
) -> DeploymentResult:
    prepared = _prepare_deployment_from_reference(
        client,
        app_ref,
        app_data,
        force_env_build=force_env_build,
        environment_name=environment_name,
    )
    return execute_prepared_deployment(prepared)


def _execute_loaded_deployment(
    *,
    host: FalServerlessHost,
    loaded: LoadedFunction,
    app_data: AppData,
    display_name: str | None,
    environment_name: str | None = None,
) -> DeploymentResult:
    from fal.api import FalServerlessError

    isolated_function = loaded.function
    strategy = app_data.deployment_strategy or "rolling"

    from fal.console import console

    resolved_display_name = _deployment_display_name(display_name, loaded)

    # Show what app name will be used
    console.print(
        f"Deploying '{resolved_display_name}' as app '{loaded.app_name}'",
        style="bold",
    )
    console.print("")

    result = host.register(
        func=isolated_function.func,
        options=isolated_function.options,
        application_name=loaded.app_name,
        application_auth_mode=loaded.app_auth,  # type: ignore
        source_code=loaded.source_code,
        metadata=isolated_function.options.host.get("metadata", {}),
        deployment_strategy=strategy,
        scale=app_data.reset_scale,
        environment_name=environment_name,
    )

    if not result or not result.result:
        raise FalServerlessError(
            "Deployment failed: The server did not confirm the deployment. "
            "This may indicate a network issue or server error. "
            "Please try again."
        )
    if not result.service_urls:
        raise FalServerlessError(
            "Deployment failed: Could not generate app endpoints. "
            "The app name may be invalid - try using --app-name with a simple "
            "kebab-case name (e.g., --app-name my-app)."
        )

    urls: dict[str, dict[str, str]] = {
        "playground": {},
        "sync": {},
        "async": {},
    }
    for endpoint in loaded.endpoints:
        urls["playground"][endpoint] = f"{result.service_urls.playground}{endpoint}"
        urls["sync"][endpoint] = f"{result.service_urls.run}{endpoint}"
        urls["async"][endpoint] = f"{result.service_urls.queue}{endpoint}"

    assert loaded.app_name
    return DeploymentResult(
        revision=result.result.application_id,
        app_name=loaded.app_name,
        urls=urls,
        log_url=result.service_urls.log,
        auth_mode=loaded.app_auth or "private",
    )


def prepare_deployment(
    client: SyncServerlessClient,
    app_ref: str | tuple[str | None, str | None] | None = None,
    *,
    app_name: str | None = None,
    auth: AuthModeLiteral | None = None,
    strategy: DeploymentStrategyLiteral = "rolling",
    reset_scale: bool = False,
    force_env_build: bool = False,
    environment_name: str | None = None,
) -> PreparedDeployment:
    resolved_app_ref, app_data = _resolve_deployment_reference(
        app_ref,
        app_name=app_name,
        auth=auth,
        strategy=strategy,
        reset_scale=reset_scale,
    )
    return _prepare_deployment_from_reference(
        client,
        resolved_app_ref,
        app_data,
        force_env_build=force_env_build,
        environment_name=environment_name,
    )


def execute_prepared_deployment(prepared: PreparedDeployment) -> DeploymentResult:
    return _execute_loaded_deployment(
        host=prepared.host,
        loaded=prepared.loaded,
        app_data=prepared.app_data,
        display_name=prepared.display_name,
        environment_name=prepared.environment_name,
    )


def deploy(
    client: SyncServerlessClient,
    app_ref: str | tuple[str, str] | None = None,
    *,
    app_name: str | None = None,
    auth: AuthModeLiteral | None = None,
    strategy: DeploymentStrategyLiteral = "rolling",
    reset_scale: bool = False,
    force_env_build: bool = False,
    environment_name: str | None = None,
) -> DeploymentResult:
    resolved_app_ref, app_data = _resolve_deployment_reference(
        app_ref,
        app_name=app_name,
        auth=auth,
        strategy=strategy,
        reset_scale=reset_scale,
    )
    return _deploy_from_reference(
        client,
        resolved_app_ref,
        app_data,
        force_env_build=force_env_build,
        environment_name=environment_name,
    )
