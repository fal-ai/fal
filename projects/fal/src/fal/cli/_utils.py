from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, get_args

from fal.api import Options
from fal.container import ContainerImage
from fal.project import find_project_root, find_pyproject_toml, parse_pyproject_toml
from fal.sdk import (
    ApplicationHealthCheckConfig,
    AuthModeLiteral,
    DeploymentStrategyLiteral,
    RetryConditionLiteral,
)

VALID_REGIONS = {
    "us-west",
    "us-central",
    "us-east",
    "eu-north",
    "eu-west",
}
VALID_RETRY_CONDITIONS = set(get_args(RetryConditionLiteral))


@dataclass(frozen=True)
class AppData:
    ref: Optional[str] = None
    python_entry_point: Optional[str] = None
    auth: Optional[AuthModeLiteral] = None
    deployment_strategy: Optional[DeploymentStrategyLiteral] = None
    reset_scale: bool = False
    team: Optional[str] = None
    name: Optional[str] = None
    options: Options = field(default_factory=Options)
    # Directory of the pyproject.toml this app was loaded from. Used by
    # ``FalServerlessHost`` to materialize ``.``/``.[extras]`` requirements
    # into uploaded sdists. ``None`` when the app wasn't loaded from a
    # pyproject (e.g. file::func ref directly).
    local_project_root: Optional[str] = None


def get_client(host: str, team: str | None = None):
    from fal.sdk import FalServerlessClient, get_credentials  # noqa: PLC0415

    credentials = get_credentials(team=team)
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

    python_entry_point: str | None = app_data.pop("python_entry_point", None)
    if python_entry_point is not None:
        if not isinstance(python_entry_point, str):
            raise ValueError(
                f"App {app_name} python_entry_point must be a string in pyproject.toml"
            )
        app_ref: str | None = app_data.pop("ref", None)
        if app_ref is not None:
            raise ValueError(
                f"App {app_name} cannot have both ref "
                "and python_entry_point keys in pyproject.toml"
            )
    else:
        try:
            app_ref = app_data.pop("ref")
        except KeyError:
            raise ValueError(
                f"App {app_name} does not have a ref key in pyproject.toml"
            )
        if not isinstance(app_ref, str):
            raise ValueError(f"App {app_name} ref must be a string in pyproject.toml")
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
    python_version = app_data.pop("python_version", None)
    if python_version is not None and not isinstance(python_version, str):
        raise ValueError(
            f"App {app_name} python_version must be a string in pyproject.toml"
        )
    options = Options()
    min_concurrency = app_data.pop("min_concurrency", None)
    max_concurrency = app_data.pop("max_concurrency", None)
    max_multiplexing = app_data.pop("max_multiplexing", None)
    concurrency_buffer = app_data.pop("concurrency_buffer", None)
    concurrency_buffer_perc = app_data.pop("concurrency_buffer_perc", None)
    scaling_delay = app_data.pop("scaling_delay", None)
    request_timeout = app_data.pop("request_timeout", None)
    startup_timeout = app_data.pop("startup_timeout", None)
    keep_alive = app_data.pop("keep_alive", None)
    private_logs = app_data.pop("private_logs", None)
    machine_type = app_data.pop("machine_type", None)
    num_gpus = app_data.pop("num_gpus", None)
    regions = app_data.pop("regions", None)
    app_files = app_data.pop("app_files", None)
    app_files_ignore = app_data.pop("app_files_ignore", None)
    app_files_context_dir = app_data.pop("app_files_context_dir", None)
    exposed_port = app_data.pop("exposed_port", None)
    scheduler = app_data.pop("_scheduler", None)
    scheduler_options = app_data.pop("_scheduler_options", None)
    skip_retry_conditions = app_data.pop("skip_retry_conditions", None)
    termination_grace_period_seconds = app_data.pop(
        "termination_grace_period_seconds", None
    )
    secrets = app_data.pop("secrets", None)
    data_mounts = app_data.pop("data_mounts", None)
    health_check = app_data.pop("health_check", None)

    image_config = app_data.pop("image", None)

    if keep_alive is not None:
        _validate_int("keep_alive", keep_alive)
    if private_logs is not None:
        _validate_bool("private_logs", private_logs)
    if regions is not None:
        _validate_regions(regions)
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
    if app_files_context_dir:
        context_path = Path(app_files_context_dir)
        if not context_path.is_absolute():
            app_files_context_dir = str(Path(toml_path).parent / context_path)
    if exposed_port is not None:
        _validate_port("exposed_port", exposed_port)
    if image_config is not None and app_files:
        raise ValueError("app_files is not supported for container apps.")
    if scheduler is not None and not isinstance(scheduler, str):
        raise ValueError("_scheduler must be a string.")
    if scheduler_options is not None and not isinstance(scheduler_options, dict):
        raise ValueError("_scheduler_options must be a table in pyproject.toml.")
    if skip_retry_conditions is not None:
        _validate_skip_retry_conditions(skip_retry_conditions)
    if termination_grace_period_seconds is not None:
        _validate_int(
            "termination_grace_period_seconds", termination_grace_period_seconds
        )
    if secrets is not None:
        _validate_str_list("secrets", secrets)
    if data_mounts is not None:
        _validate_str_list("data_mounts", data_mounts)
    health_check_config = None
    if health_check is not None:
        health_check_config = _build_health_check_config_from_toml(
            app_name, health_check
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
    if keep_alive is not None:
        options.host["keep_alive"] = keep_alive
    if private_logs is not None:
        options.host["private_logs"] = private_logs
    if machine_type is not None:
        options.host["machine_type"] = machine_type
    if num_gpus is not None:
        options.host["num_gpus"] = num_gpus
    if regions is not None:
        options.host["regions"] = regions
    if app_files is not None:
        options.host["app_files"] = app_files
    if app_files_ignore is not None:
        options.host["app_files_ignore"] = app_files_ignore
    if app_files_context_dir is not None:
        options.host["app_files_context_dir"] = app_files_context_dir
    if scheduler is not None:
        options.host["_scheduler"] = scheduler
    if scheduler_options is not None:
        options.host["_scheduler_options"] = scheduler_options
    if skip_retry_conditions is not None:
        options.host["skip_retry_conditions"] = skip_retry_conditions
    if termination_grace_period_seconds is not None:
        options.host["termination_grace_period_seconds"] = (
            termination_grace_period_seconds
        )
    if secrets is not None:
        options.host["secrets"] = secrets
    if data_mounts is not None:
        options.host["data_mounts"] = data_mounts
    if health_check_config is not None:
        options.host["health_check_config"] = health_check_config
    if exposed_port is not None:
        options.gateway["exposed_port"] = exposed_port
    if requirements is not None:
        options.environment["requirements"] = requirements
    if python_version is not None:
        options.environment["python_version"] = python_version
    if image_config is not None:
        container_image = _build_container_image_from_toml(
            app_name, image_config, Path(toml_path).parent
        )
        options.environment["kind"] = "container"
        options.environment["image"] = container_image.to_dict()

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
        python_entry_point=python_entry_point,
        auth=app_auth,
        deployment_strategy=app_deployment_strategy,
        reset_scale=app_reset_scale,
        team=app_team,
        name=app_name_value,
        options=options,
        local_project_root=str(Path(toml_path).parent),
    )


def _validate_regions(regions: Any) -> None:
    """Validate that regions is a list of valid region strings."""
    if not (
        isinstance(regions, list) and all(isinstance(item, str) for item in regions)
    ):
        raise ValueError("regions must be a list of strings.")

    invalid_regions = set(regions) - VALID_REGIONS
    if invalid_regions:
        raise ValueError(
            f"Invalid regions: {', '.join(sorted(invalid_regions))}. "
            f"Valid regions are: {', '.join(sorted(VALID_REGIONS))}"
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


def _validate_int(field_name: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer.")


def _validate_bool(field_name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean.")


def _validate_skip_retry_conditions(value: Any) -> None:
    _validate_str_list("skip_retry_conditions", value)
    invalid_conditions = set(value) - VALID_RETRY_CONDITIONS
    if invalid_conditions:
        raise ValueError(
            "Invalid skip_retry_conditions: "
            f"{', '.join(sorted(invalid_conditions))}. "
            "Valid conditions are: "
            f"{', '.join(sorted(VALID_RETRY_CONDITIONS))}"
        )


def _validate_port(field_name: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer.")
    if value < 0 or value > 65535:
        raise ValueError(f"{field_name} must be between 0 and 65535.")


def _build_health_check_config_from_toml(
    app_name: str, health_check: Any
) -> ApplicationHealthCheckConfig:
    if not isinstance(health_check, dict):
        raise ValueError(
            f"App {app_name} health_check must be a table in pyproject.toml"
        )

    health_check = dict(health_check)
    path = health_check.pop("path", None)
    if path is None:
        raise ValueError(
            f"App {app_name} health_check must specify 'path' in pyproject.toml"
        )
    if not isinstance(path, str):
        raise ValueError(
            f"App {app_name} health_check.path must be a string in pyproject.toml"
        )

    start_period_seconds = health_check.pop("start_period_seconds", None)
    timeout_seconds = health_check.pop("timeout_seconds", None)
    failure_threshold = health_check.pop("failure_threshold", None)
    call_regularly = health_check.pop("call_regularly", None)

    if start_period_seconds is not None:
        _validate_int("health_check.start_period_seconds", start_period_seconds)
    if timeout_seconds is not None:
        _validate_int("health_check.timeout_seconds", timeout_seconds)
    if failure_threshold is not None:
        _validate_int("health_check.failure_threshold", failure_threshold)
    if call_regularly is not None:
        _validate_bool("health_check.call_regularly", call_regularly)

    if health_check:
        raise ValueError(
            f"Found unexpected keys in app {app_name} health_check: {health_check}"
        )

    return ApplicationHealthCheckConfig(
        path=path,
        start_period_seconds=start_period_seconds,
        timeout_seconds=timeout_seconds,
        failure_threshold=failure_threshold,
        call_regularly=call_regularly,
    )


_IMAGE_PASSTHROUGH_KEYS = (
    "build_args",
    "registries",
    "secrets",
)


def _build_container_image_from_toml(
    app_name: str, image_config: Any, project_root: Path
) -> ContainerImage:
    if not isinstance(image_config, dict):
        raise ValueError(f"App {app_name} image must be a table in pyproject.toml")

    image_config = dict(image_config)

    dockerfile_path = image_config.pop("dockerfile", None)
    if dockerfile_path is None:
        raise ValueError(
            f"App {app_name} image must specify 'dockerfile' (path) in pyproject.toml"
        )
    if not isinstance(dockerfile_path, str):
        raise ValueError(
            f"App {app_name} image.dockerfile must be a string in pyproject.toml"
        )

    resolved_path = Path(dockerfile_path)
    if not resolved_path.is_absolute():
        resolved_path = project_root / resolved_path
    try:
        with open(resolved_path) as f:
            dockerfile_str = f.read()
    except FileNotFoundError:
        raise ValueError(f"App {app_name} image.dockerfile not found: {resolved_path}")

    kwargs: dict[str, Any] = {
        "dockerfile_str": dockerfile_str,
        "context_dir": project_root,
    }

    for key in _IMAGE_PASSTHROUGH_KEYS:
        if key in image_config:
            kwargs[key] = image_config.pop(key)

    if image_config:
        raise ValueError(
            f"Found unexpected keys in app {app_name} image: {image_config}"
        )

    return ContainerImage(**kwargs)
