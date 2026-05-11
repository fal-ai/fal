from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from fal.api import Options
from fal.container import ContainerImage
from fal.project import find_project_root, find_pyproject_toml, parse_pyproject_toml
from fal.sdk import AuthModeLiteral, DeploymentStrategyLiteral

VALID_REGIONS = {
    "us-west",
    "us-central",
    "us-east",
    "eu-north",
    "eu-west",
}


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
    regions = app_data.pop("regions", None)
    app_files = app_data.pop("app_files", None)
    app_files_ignore = app_data.pop("app_files_ignore", None)
    app_files_context_dir = app_data.pop("app_files_context_dir", None)

    image_config = app_data.pop("image", None)

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
    if image_config is not None and app_files is not None:
        raise ValueError("app_files is not supported for container apps.")

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


_IMAGE_PASSTHROUGH_KEYS = (
    "build_args",
    "registries",
    "builder",
    "compression",
    "force_compression",
    "secrets",
    "dockerignore",
)


def _build_container_image_from_toml(
    app_name: str, image_config: Any, project_root: Path
) -> ContainerImage:
    if not isinstance(image_config, dict):
        raise ValueError(f"App {app_name} image must be a table in pyproject.toml")

    image_config = dict(image_config)

    dockerfile_path = image_config.pop("dockerfile", None)
    dockerfile_str = image_config.pop("dockerfile_str", None)

    if dockerfile_path is None and dockerfile_str is None:
        raise ValueError(
            f"App {app_name} image must specify either 'dockerfile' (path) "
            "or 'dockerfile_str' in pyproject.toml"
        )
    if dockerfile_path is not None and dockerfile_str is not None:
        raise ValueError(
            f"App {app_name} image cannot specify both 'dockerfile' and "
            "'dockerfile_str' in pyproject.toml"
        )

    if dockerfile_path is not None:
        if not isinstance(dockerfile_path, str):
            raise ValueError(
                f"App {app_name} image.dockerfile must be a string in pyproject.toml"
            )
        resolved_path = Path(dockerfile_path)
        if not resolved_path.is_absolute():
            resolved_path = project_root / resolved_path
        with open(resolved_path) as f:
            dockerfile_str = f.read()
    elif not isinstance(dockerfile_str, str):
        raise ValueError(
            f"App {app_name} image.dockerfile_str must be a string in pyproject.toml"
        )

    kwargs: dict[str, Any] = {"dockerfile_str": dockerfile_str}

    for key in _IMAGE_PASSTHROUGH_KEYS:
        if key in image_config:
            kwargs[key] = image_config.pop(key)

    context_dir = image_config.pop("context_dir", None)
    if context_dir is not None:
        if not isinstance(context_dir, str):
            raise ValueError(
                f"App {app_name} image.context_dir must be a string in pyproject.toml"
            )
        context_path = Path(context_dir)
        if not context_path.is_absolute():
            context_path = project_root / context_path
        kwargs["context_dir"] = context_path
    else:
        kwargs["context_dir"] = project_root

    dockerignore_path = image_config.pop("dockerignore_path", None)
    if dockerignore_path is not None:
        if not isinstance(dockerignore_path, str):
            raise ValueError(
                f"App {app_name} image.dockerignore_path must be a string "
                "in pyproject.toml"
            )
        dockerignore_resolved = Path(dockerignore_path)
        if not dockerignore_resolved.is_absolute():
            dockerignore_resolved = project_root / dockerignore_resolved
        kwargs["dockerignore_path"] = dockerignore_resolved

    if image_config:
        raise ValueError(
            f"Found unexpected keys in app {app_name} image: {image_config}"
        )

    return ContainerImage(**kwargs)
