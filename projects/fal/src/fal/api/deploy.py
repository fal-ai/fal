from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from fal.sdk import AuthModeLiteral, DeploymentStrategyLiteral

if TYPE_CHECKING:
    from .client import SyncServerlessClient


import json
from collections import namedtuple
from typing import TYPE_CHECKING, Tuple, Union, cast

if TYPE_CHECKING:
    from openapi_fal_rest.client import Client

User = namedtuple("User", ["user_id", "username"])


@dataclass
class DeploymentResult:
    revision: str
    app_name: str
    urls: dict[str, dict[str, str]]
    log_url: str


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


def _deploy_from_reference(
    client: SyncServerlessClient,
    app_ref: Tuple[Optional[Union[Path, str]], ...],
    app_name: str,
    auth: Optional[AuthModeLiteral],
    strategy: Optional[DeploymentStrategyLiteral],
    scale: bool,
    force_env_build: bool,
    environment_name: Optional[str] = None,
) -> DeploymentResult:
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

        [file_path] = options
        file_path = str(file_path)  # type: ignore

    host = client._create_host(
        local_file_path=str(file_path),
        environment_name=environment_name,
    )
    loaded = load_function_from(
        host,
        file_path,  # type: ignore
        func_name,  # type: ignore
        force_env_build=force_env_build,
    )
    isolated_function = loaded.function
    app_name = app_name or loaded.app_name  # type: ignore
    app_auth = auth or loaded.app_auth
    strategy = strategy or "rolling"

    # Get the actual function/class name that was loaded
    display_name = func_name or loaded.class_name

    # Show what app name will be used
    from fal.console import console

    console.print(
        f"==> Deploying class '{display_name}' as app '{app_name}'", style="bold blue"
    )

    result = host.register(
        func=isolated_function.func,
        options=isolated_function.options,
        application_name=app_name,
        application_auth_mode=app_auth,  # type: ignore
        source_code=loaded.source_code,
        metadata=isolated_function.options.host.get("metadata", {}),
        deployment_strategy=strategy,
        scale=scale,
        environment_name=environment_name,
    )

    if not result or not result.result:
        raise FalServerlessError(
            "Deployment failed: The server did not confirm the deployment. "
            "This may indicate a network issue or server error. "
            "Please try again, or check 'fal doctor' for connection issues."
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

    return DeploymentResult(
        revision=result.result.application_id,
        app_name=app_name,
        urls=urls,
        log_url=result.service_urls.log,
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
    from fal.cli._utils import get_app_data_from_toml, is_app_name
    from fal.cli.parser import RefAction

    if isinstance(app_ref, tuple):
        app_ref_tuple = app_ref
    elif app_ref:
        app_ref_tuple = RefAction.split_ref(app_ref)
    else:
        raise ValueError("Invalid app reference")

    # my-app
    if is_app_name(app_ref_tuple):
        # we do not allow --app-name and --auth to be used with app name
        if app_name or auth:
            raise ValueError("Cannot use --app-name or --auth with app name reference.")

        app_name = app_ref_tuple[0]
        app_ref, app_auth, app_strategy, app_scale_settings, _ = get_app_data_from_toml(
            app_name
        )

        file_path, func_name = RefAction.split_ref(app_ref)

    # path/to/myfile.py::MyApp
    else:
        file_path, func_name = app_ref_tuple
        app_name = cast(str, app_name)
        # default to be set in the backend
        app_auth = cast(Optional[AuthModeLiteral], auth)
        # default comes from the CLI
        app_strategy = cast(DeploymentStrategyLiteral, strategy)
        app_scale_settings = cast(bool, reset_scale)
        file_path = str(Path(file_path).absolute())

    return _deploy_from_reference(
        client,
        (file_path, func_name),
        app_name,  # type: ignore
        app_auth,
        strategy=app_strategy,
        scale=app_scale_settings,
        force_env_build=force_env_build,
        environment_name=environment_name,
    )
