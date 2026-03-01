from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Optional

from fal.api import Options
from fal.project import find_project_root, find_pyproject_toml, parse_pyproject_toml
from fal.sdk import AuthModeLiteral, DeploymentStrategyLiteral


@dataclass(frozen=True)
class AppData:
    ref: Optional[str] = None
    auth: Optional[AuthModeLiteral] = None
    deployment_strategy: Optional[DeploymentStrategyLiteral] = None
    reset_scale: bool = False
    team: Optional[str] = None
    name: Optional[str] = None
    options: Options = field(default_factory=Options)


def get_client(host: str, team: str | None = None):
    from fal.sdk import FalServerlessClient, get_default_credentials  # noqa: PLC0415

    credentials = get_default_credentials(team=team)
    return FalServerlessClient(host, credentials)


def is_app_name(app_ref: tuple[str, str | None]) -> bool:
    is_single_file = app_ref[1] is None
    is_python_file = app_ref[0] is None or app_ref[0].endswith(".py")

    return is_single_file and not is_python_file


def get_app_data_from_toml(
    app_name: str, *, emit_deprecation_warnings: bool = True
) -> AppData:
    toml_path = find_pyproject_toml()

    if toml_path is None:
        raise ValueError("No pyproject.toml file found.")

    fal_data = parse_pyproject_toml(toml_path)
    apps = fal_data.get("apps", {})

    try:
        app_data: dict[str, Any] = copy.deepcopy(apps[app_name])
    except KeyError:
        raise ValueError(f"App {app_name} not found in pyproject.toml")

    try:
        app_ref: str = app_data.pop("ref")
    except KeyError:
        raise ValueError(f"App {app_name} does not have a ref key in pyproject.toml")

    # Convert the app_ref to a path relative to the project root
    project_root, _ = find_project_root(None)
    app_ref = str(project_root / app_ref)

    app_auth: Optional[AuthModeLiteral] = app_data.pop("auth", None)
    app_deployment_strategy: Optional[DeploymentStrategyLiteral] = app_data.pop(
        "deployment_strategy", None
    )
    app_team: Optional[str] = app_data.pop("team", None)
    app_name_value: Optional[str] = app_data.pop("name", None)
    if app_name_value is None:
        app_name_value = app_name

    requirements = app_data.pop("requirements", None)
    if requirements is not None:
        _validate_requirements(requirements)
    options = Options()
    min_concurrency = app_data.pop("min_concurrency", None)
    max_concurrency = app_data.pop("max_concurrency", None)
    max_multiplexing = app_data.pop("max_multiplexing", None)
    concurrency_buffer = app_data.pop("concurrency_buffer", None)
    concurrency_buffer_perc = app_data.pop("concurrency_buffer_perc", None)
    scaling_delay = app_data.pop("scaling_delay", None)
    request_timeout = app_data.pop("request_timeout", None)
    startup_timeout = app_data.pop("startup_timeout", None)
    regions = app_data.pop("regions", None)
    app_files = app_data.pop("app_files", None)
    app_files_ignore = app_data.pop("app_files_ignore", None)
    app_files_context_dir = app_data.pop("app_files_context_dir", None)

    if regions is not None:
        _validate_str_list("regions", regions)
    if app_files is not None:
        _validate_str_list("app_files", app_files)
    if app_files_ignore is not None:
        _validate_str_list("app_files_ignore", app_files_ignore)
    if app_files_context_dir is not None and not isinstance(app_files_context_dir, str):
        raise ValueError("app_files_context_dir must be a string.")
    if app_files_context_dir is not None and app_files is None:
        raise ValueError(
            "app_files_context_dir is only supported when app_files is provided."
        )

    if min_concurrency is not None:
        options.host["min_concurrency"] = min_concurrency
    if max_concurrency is not None:
        options.host["max_concurrency"] = max_concurrency
    if max_multiplexing is not None:
        options.host["max_multiplexing"] = max_multiplexing
    if concurrency_buffer is not None:
        options.host["concurrency_buffer"] = concurrency_buffer
    if concurrency_buffer_perc is not None:
        options.host["concurrency_buffer_perc"] = concurrency_buffer_perc
    if scaling_delay is not None:
        options.host["scaling_delay"] = scaling_delay
    if request_timeout is not None:
        options.host["request_timeout"] = request_timeout
    if startup_timeout is not None:
        options.host["startup_timeout"] = startup_timeout
    if regions is not None:
        options.host["regions"] = regions
    if app_files is not None:
        options.host["app_files"] = app_files
    if app_files_ignore is not None:
        options.host["app_files_ignore"] = app_files_ignore
    if app_files_context_dir is not None:
        options.host["app_files_context_dir"] = app_files_context_dir
    if requirements is not None:
        options.environment["requirements"] = requirements

    app_reset_scale: bool
    if "no_scale" in app_data:
        # Deprecated
        app_no_scale: bool = app_data.pop("no_scale")
        if emit_deprecation_warnings:
            print("[WARNING] no_scale is deprecated, use app_scale_settings instead")
        app_reset_scale = not app_no_scale
    else:
        app_reset_scale = app_data.pop("app_scale_settings", False)

    if len(app_data) > 0:
        raise ValueError(f"Found unexpected keys in pyproject.toml: {app_data}")

    return AppData(
        ref=app_ref,
        auth=app_auth,
        deployment_strategy=app_deployment_strategy,
        reset_scale=app_reset_scale,
        team=app_team,
        name=app_name_value,
        options=options,
    )


def _validate_requirements(requirements: Any) -> None:
    is_str_list = isinstance(requirements, list) and all(
        isinstance(item, str) for item in requirements
    )
    is_str_list_list = isinstance(requirements, list) and all(
        isinstance(item, list) and all(isinstance(req, str) for req in item)
        for item in requirements
    )
    if not is_str_list and not is_str_list_list:
        raise ValueError(
            "requirements must be a list of strings or a list of lists of strings."
        )


def _validate_str_list(field_name: str, value: Any) -> None:
    if not (isinstance(value, list) and all(isinstance(item, str) for item in value)):
        raise ValueError(f"{field_name} must be a list of strings.")
