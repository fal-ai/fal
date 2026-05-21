from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn, Optional, TypeVar, get_args

import pydantic
from pydantic import BaseModel, Field, StrictBool, StrictInt, StrictStr
from pydantic import ValidationError as PydanticValidationError

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

if not hasattr(pydantic, "__version__") or pydantic.__version__.startswith("1."):
    IS_PYDANTIC_V2 = False
else:
    IS_PYDANTIC_V2 = True

_PydanticModelT = TypeVar("_PydanticModelT", bound=BaseModel)


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


class _TomlBaseModel(BaseModel):
    if IS_PYDANTIC_V2:
        model_config = {"extra": "forbid", "populate_by_name": True}
    else:

        class Config:
            extra = "forbid"
            allow_population_by_field_name = True


class _TomlHealthCheckConfig(_TomlBaseModel):
    path: StrictStr
    start_period_seconds: StrictInt | None = None
    timeout_seconds: StrictInt | None = None
    failure_threshold: StrictInt | None = None
    call_regularly: StrictBool | None = None


class _TomlContainerImageConfig(_TomlBaseModel):
    dockerfile: StrictStr
    build_args: Any = None
    registries: Any = None
    # Image build-time secrets, distinct from app-level runtime `secrets`.
    secrets: Any = None


class _TomlAppConfig(_TomlBaseModel):
    ref: StrictStr | None = None
    python_entry_point: StrictStr | None = None
    auth: AuthModeLiteral | None = None
    deployment_strategy: DeploymentStrategyLiteral | None = None
    team: StrictStr | None = None
    name: StrictStr | None = None

    requirements: list[StrictStr] | list[list[StrictStr]] | None = None
    python_version: StrictStr | None = None
    min_concurrency: StrictInt | None = None
    max_concurrency: StrictInt | None = None
    max_multiplexing: StrictInt | None = None
    concurrency_buffer: StrictInt | None = None
    concurrency_buffer_perc: StrictInt | None = None
    scaling_delay: StrictInt | None = None
    request_timeout: StrictInt | None = None
    startup_timeout: StrictInt | None = None
    keep_alive: StrictInt | None = None
    private_logs: StrictBool | None = None
    machine_type: StrictStr | list[StrictStr] | None = None
    num_gpus: StrictInt | None = None
    regions: list[StrictStr] | None = None
    app_files: list[StrictStr] | None = None
    app_files_ignore: list[StrictStr] | None = None
    app_files_context_dir: StrictStr | None = None
    exposed_port: StrictInt | None = None
    scheduler: StrictStr | None = Field(default=None, alias="_scheduler")
    scheduler_options: dict[str, Any] | None = Field(
        default=None, alias="_scheduler_options"
    )
    skip_retry_conditions: list[StrictStr] | None = None
    termination_grace_period_seconds: StrictInt | None = None
    secrets: list[StrictStr] | None = None
    data_mounts: list[StrictStr] | None = None
    health_check: _TomlHealthCheckConfig | None = None
    image: _TomlContainerImageConfig | None = None

    no_scale: StrictBool | None = None
    app_scale_settings: StrictBool = False


_FIELD_ALIASES = {
    "scheduler": "_scheduler",
    "scheduler_options": "_scheduler_options",
}

_APP_STRING_FIELDS = {
    "ref",
    "python_entry_point",
    "python_version",
    "team",
    "name",
}
_APP_INTEGER_FIELDS = {
    "min_concurrency",
    "max_concurrency",
    "max_multiplexing",
    "concurrency_buffer",
    "concurrency_buffer_perc",
    "scaling_delay",
    "request_timeout",
    "startup_timeout",
    "keep_alive",
    "num_gpus",
    "exposed_port",
    "termination_grace_period_seconds",
}
_APP_BOOLEAN_FIELDS = {
    "private_logs",
    "no_scale",
    "app_scale_settings",
}
_APP_STRING_LIST_FIELDS = {
    "regions",
    "app_files",
    "app_files_ignore",
    "skip_retry_conditions",
    "secrets",
    "data_mounts",
}
_HEALTH_CHECK_INTEGER_FIELDS = {
    "start_period_seconds",
    "timeout_seconds",
    "failure_threshold",
}


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
    if not isinstance(apps, dict):
        raise ValueError("apps must be a table in pyproject.toml")

    try:
        raw_app_data = copy.deepcopy(apps[app_name])
    except KeyError:
        raise ValueError(f"App {app_name} not found in pyproject.toml")

    app_config = _parse_toml_app_config(app_name, raw_app_data)

    python_entry_point = app_config.python_entry_point
    if python_entry_point is not None:
        if app_config.ref is not None:
            raise ValueError(
                f"App {app_name} cannot have both ref "
                "and python_entry_point keys in pyproject.toml"
            )
        app_ref: str | None = None
    else:
        app_ref_value = app_config.ref
        if app_ref_value is None:
            raise ValueError(
                f"App {app_name} does not have a ref key in pyproject.toml"
            )
        # Convert the app_ref to a path relative to the project root
        project_root, _ = find_project_root(None)
        app_ref = str(project_root / app_ref_value)

    app_auth = app_config.auth
    app_deployment_strategy = app_config.deployment_strategy
    app_team = app_config.team
    app_name_value = app_config.name
    if app_name_value is None:
        app_name_value = app_name

    requirements = app_config.requirements
    python_version = app_config.python_version
    options = Options()
    min_concurrency = app_config.min_concurrency
    max_concurrency = app_config.max_concurrency
    max_multiplexing = app_config.max_multiplexing
    concurrency_buffer = app_config.concurrency_buffer
    concurrency_buffer_perc = app_config.concurrency_buffer_perc
    scaling_delay = app_config.scaling_delay
    request_timeout = app_config.request_timeout
    startup_timeout = app_config.startup_timeout
    keep_alive = app_config.keep_alive
    private_logs = app_config.private_logs
    machine_type = app_config.machine_type
    num_gpus = app_config.num_gpus
    regions = app_config.regions
    app_files = app_config.app_files
    app_files_ignore = app_config.app_files_ignore
    app_files_context_dir = app_config.app_files_context_dir
    exposed_port = app_config.exposed_port
    scheduler = app_config.scheduler
    scheduler_options = app_config.scheduler_options
    skip_retry_conditions = app_config.skip_retry_conditions
    termination_grace_period_seconds = app_config.termination_grace_period_seconds
    secrets = app_config.secrets
    data_mounts = app_config.data_mounts
    health_check = app_config.health_check

    image_config = app_config.image

    if regions is not None:
        _validate_regions(regions)
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
    if scheduler is not None and not scheduler:
        raise ValueError("_scheduler must be a non-empty string.")
    if skip_retry_conditions is not None:
        _validate_skip_retry_conditions(skip_retry_conditions)
    health_check_config = None
    if health_check is not None:
        health_check_config = _build_health_check_config_from_toml(health_check)

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
    if app_config.no_scale is not None:
        # Deprecated
        if emit_deprecation_warnings:
            print("[WARNING] no_scale is deprecated, use app_scale_settings instead")
        app_reset_scale = not app_config.no_scale
    else:
        app_reset_scale = app_config.app_scale_settings

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


def _parse_toml_app_config(app_name: str, raw_app_data: Any) -> _TomlAppConfig:
    if not isinstance(raw_app_data, dict):
        raise ValueError(f"App {app_name} must be a table in pyproject.toml")

    try:
        app_config = _model_validate(_TomlAppConfig, raw_app_data)
    except PydanticValidationError as exc:
        _raise_toml_app_config_error(app_name, raw_app_data, exc)

    fields_set = _model_fields_set(app_config)
    if app_config.no_scale is not None and "app_scale_settings" in fields_set:
        raise ValueError(
            "Found unexpected keys in pyproject.toml: "
            f"{{'app_scale_settings': {raw_app_data['app_scale_settings']!r}}}"
        )

    return app_config


def _model_validate(model: type[_PydanticModelT], data: Any) -> _PydanticModelT:
    if IS_PYDANTIC_V2:
        return model.model_validate(data)  # type: ignore[attr-defined]
    return model.parse_obj(data)


def _model_dump(model: BaseModel, *, exclude_none: bool = True) -> dict[str, Any]:
    if IS_PYDANTIC_V2:
        return model.model_dump(exclude_none=exclude_none)  # type: ignore[attr-defined]
    return model.dict(exclude_none=exclude_none)


def _model_fields_set(model: BaseModel) -> set[str]:
    fields_set: Any = getattr(model, "model_fields_set", None)
    if fields_set is None:
        fields_set = getattr(model, "__fields_set__", set())
    return set(fields_set)


def _raise_toml_app_config_error(
    app_name: str, raw_app_data: dict[str, Any], exc: PydanticValidationError
) -> NoReturn:
    errors = exc.errors()
    extra_keys = [
        _error_loc(error)[0]
        for error in errors
        if _is_extra_error(error) and len(_error_loc(error)) == 1
    ]
    if extra_keys:
        unexpected = {key: raw_app_data[key] for key in extra_keys}
        raise ValueError(f"Found unexpected keys in pyproject.toml: {unexpected}")

    for section in ("health_check", "image"):
        section_extra_keys = [
            _error_loc(error)[1]
            for error in errors
            if (
                _is_extra_error(error)
                and len(_error_loc(error)) >= 2
                and _error_loc(error)[0] == section
            )
        ]
        if section_extra_keys:
            section_data = raw_app_data.get(section, {})
            unexpected = {
                key: section_data[key]
                for key in section_extra_keys
                if isinstance(section_data, dict)
            }
            raise ValueError(
                f"Found unexpected keys in app {app_name} {section}: {unexpected}"
            )

    for error in errors:
        loc = _error_loc(error)
        if not loc:
            continue

        field_name = _field_display_name(loc[0])

        if field_name in _APP_STRING_LIST_FIELDS:
            raise ValueError(f"{field_name} must be a list of strings.")
        if field_name == "requirements":
            raise ValueError(
                "requirements must be a list of strings or a list of lists "
                "of strings."
            )
        if field_name == "machine_type":
            raise ValueError("machine_type must be a string or list of strings.")

        if len(loc) == 1:
            if field_name in _APP_INTEGER_FIELDS:
                raise ValueError(f"{field_name} must be an integer.")
            if field_name in _APP_BOOLEAN_FIELDS:
                raise ValueError(f"{field_name} must be a boolean.")
            if field_name == "_scheduler":
                raise ValueError("_scheduler must be a non-empty string.")
            if field_name == "_scheduler_options":
                raise ValueError(
                    "_scheduler_options must be a table in pyproject.toml."
                )
            if field_name == "health_check":
                raise ValueError(
                    f"App {app_name} health_check must be a table in pyproject.toml"
                )
            if field_name == "image":
                raise ValueError(
                    f"App {app_name} image must be a table in pyproject.toml"
                )
            if field_name in _APP_STRING_FIELDS:
                raise ValueError(
                    f"App {app_name} {field_name} must be a string in pyproject.toml"
                )

        if loc[0] == "health_check" and len(loc) >= 2:
            health_check_field = loc[1]
            if _is_missing_error(error) and health_check_field == "path":
                raise ValueError(
                    f"App {app_name} health_check must specify 'path' in pyproject.toml"
                )
            if health_check_field == "path":
                raise ValueError(
                    f"App {app_name} health_check.path must be a string "
                    "in pyproject.toml"
                )
            if health_check_field in _HEALTH_CHECK_INTEGER_FIELDS:
                raise ValueError(
                    f"health_check.{health_check_field} must be an integer."
                )
            if health_check_field == "call_regularly":
                raise ValueError("health_check.call_regularly must be a boolean.")

        if loc[0] == "image" and len(loc) >= 2:
            image_field = loc[1]
            if _is_missing_error(error) and image_field == "dockerfile":
                raise ValueError(
                    f"App {app_name} image must specify 'dockerfile' (path) "
                    "in pyproject.toml"
                )
            if image_field == "dockerfile":
                raise ValueError(
                    f"App {app_name} image.dockerfile must be a string "
                    "in pyproject.toml"
                )

    raise ValueError(str(exc))


def _error_loc(error: dict[str, Any]) -> tuple[str, ...]:
    loc = error.get("loc", ())
    if isinstance(loc, str):
        return (loc,)
    return tuple(str(item) for item in loc)


def _is_extra_error(error: dict[str, Any]) -> bool:
    error_type = str(error.get("type", ""))
    return "extra" in error_type


def _is_missing_error(error: dict[str, Any]) -> bool:
    error_type = str(error.get("type", ""))
    return error_type in {"missing", "value_error.missing"}


def _field_display_name(field_name: str) -> str:
    return _FIELD_ALIASES.get(field_name, field_name)


def _validate_regions(regions: list[str]) -> None:
    """Validate that regions is a list of valid region strings."""
    invalid_regions = set(regions) - VALID_REGIONS
    if invalid_regions:
        raise ValueError(
            f"Invalid regions: {', '.join(sorted(invalid_regions))}. "
            f"Valid regions are: {', '.join(sorted(VALID_REGIONS))}"
        )


def _validate_int(field_name: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer.")


def _validate_skip_retry_conditions(value: list[str]) -> None:
    invalid_conditions = set(value) - VALID_RETRY_CONDITIONS
    if invalid_conditions:
        raise ValueError(
            "Invalid skip_retry_conditions: "
            f"{', '.join(sorted(invalid_conditions))}. "
            "Valid conditions are: "
            f"{', '.join(sorted(VALID_RETRY_CONDITIONS))}"
        )


def _validate_port(field_name: str, value: Any) -> None:
    _validate_int(field_name, value)
    if value < 0 or value > 65535:
        raise ValueError(f"{field_name} must be between 0 and 65535.")


def _build_health_check_config_from_toml(
    health_check: _TomlHealthCheckConfig,
) -> ApplicationHealthCheckConfig:
    return ApplicationHealthCheckConfig(
        path=health_check.path,
        start_period_seconds=health_check.start_period_seconds,
        timeout_seconds=health_check.timeout_seconds,
        failure_threshold=health_check.failure_threshold,
        call_regularly=health_check.call_regularly,
    )


_IMAGE_PASSTHROUGH_KEYS = (
    "build_args",
    "registries",
    # Image build-time secrets, distinct from app-level runtime `secrets`.
    "secrets",
)


def _build_container_image_from_toml(
    app_name: str, image_config: _TomlContainerImageConfig, project_root: Path
) -> ContainerImage:
    image_data = _model_dump(image_config)
    dockerfile_path = image_data.pop("dockerfile")

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
        if key in image_data:
            kwargs[key] = image_data.pop(key)

    return ContainerImage(**kwargs)
