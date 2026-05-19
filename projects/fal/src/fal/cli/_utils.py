from __future__ import annotations

import atexit
import copy
import json
import re
import shutil
import sys
import tempfile
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

    image_config = app_data.pop("image", None)

    generated_docker_entrypoint = False
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
        app_ref = app_data.pop("ref", None)
        if app_ref is None:
            if image_config is None:
                raise ValueError(
                    f"App {app_name} does not have a ref key in pyproject.toml"
                )
            python_entry_point = _GENERATED_DOCKER_ENTRYPOINT
            generated_docker_entrypoint = True
        else:
            if not isinstance(app_ref, str):
                raise ValueError(
                    f"App {app_name} ref must be a string in pyproject.toml"
                )
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
    machine_type = app_data.pop("machine_type", None)
    num_gpus = app_data.pop("num_gpus", None)
    regions = app_data.pop("regions", None)
    app_files = app_data.pop("app_files", None)
    app_files_ignore = app_data.pop("app_files_ignore", None)
    app_files_context_dir = app_data.pop("app_files_context_dir", None)
    exposed_port = app_data.pop("exposed_port", None)

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
    if exposed_port is not None:
        options.gateway["exposed_port"] = exposed_port
    if requirements is not None:
        options.environment["requirements"] = requirements
    if python_version is not None:
        options.environment["python_version"] = python_version
    if image_config is not None:
        container_image = _build_container_image_from_toml(
            app_name,
            image_config,
            Path(toml_path).parent,
            inject_generated_entrypoint=generated_docker_entrypoint,
            python_version=python_version,
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


def _validate_port(field_name: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer.")
    if value < 0 or value > 65535:
        raise ValueError(f"{field_name} must be between 0 and 65535.")


_IMAGE_PASSTHROUGH_KEYS = (
    "build_args",
    "registries",
    "secrets",
)


_GENERATED_DOCKER_ENTRYPOINT = "fal_entrypoint:fal_entry_point"
_PYTHON_BUILD_STANDALONE_RELEASE = "20260510"
_PYTHON_BUILD_STANDALONE_ARCHIVES = {
    "3.10": (
        "cpython-3.10.20+20260510-x86_64-unknown-linux-gnu-"
        "install_only_stripped.tar.gz",
        "sha256:dc734bdd388975c0b093fe730b272af741a2e192475d38bc6845a687b6405922",
    ),
    "3.11": (
        "cpython-3.11.15+20260510-x86_64-unknown-linux-gnu-"
        "install_only_stripped.tar.gz",
        "sha256:171dffd8c0f66e8a0725364a7428015b22fc18dd298b24f541392e17dd0e561f",
    ),
    "3.12": (
        "cpython-3.12.13+20260510-x86_64-unknown-linux-gnu-"
        "install_only_stripped.tar.gz",
        "sha256:d480f5d5878910ecbae212bf23bd7c25d7b209eb8cf5e98823c977384d272e88",
    ),
    "3.13": (
        "cpython-3.13.13+20260510-x86_64-unknown-linux-gnu-"
        "install_only_stripped.tar.gz",
        "sha256:bbe27549e475fe5f22d42a8e0d553dc79d80d8a00e05712599637857d287360e",
    ),
    "3.14": (
        "cpython-3.14.5+20260510-x86_64-unknown-linux-gnu-"
        "install_only_stripped.tar.gz",
        "sha256:dc10977b0db3bef1ee2275107fde6fe9c148135b556fa352e83c6baa67d17ed6",
    ),
}


def _build_container_image_from_toml(
    app_name: str,
    image_config: Any,
    project_root: Path,
    *,
    inject_generated_entrypoint: bool = False,
    python_version: str | None = None,
) -> ContainerImage:
    if not isinstance(image_config, dict):
        raise ValueError(f"App {app_name} image must be a table in pyproject.toml")

    project_root = project_root.resolve()
    image_config = dict(image_config)

    wrapper_entrypoint = image_config.pop("entrypoint", None)
    wrapper_cmd = image_config.pop("cmd", None)
    if inject_generated_entrypoint:
        wrapper_entrypoint = _validate_docker_command_list(
            app_name, "entrypoint", wrapper_entrypoint
        )
        wrapper_cmd = _validate_docker_command_list(app_name, "cmd", wrapper_cmd)
        if wrapper_entrypoint is None and wrapper_cmd is None:
            raise ValueError(
                f"App {app_name} image-only deployment must specify "
                "image.entrypoint or image.cmd in pyproject.toml"
            )
    elif wrapper_entrypoint is not None or wrapper_cmd is not None:
        raise ValueError(
            f"App {app_name} image.entrypoint and image.cmd are only supported "
            "for image-only deployments in pyproject.toml"
        )

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

    if inject_generated_entrypoint:
        assert wrapper_entrypoint is not None or wrapper_cmd is not None
        generated_source = _write_generated_docker_entrypoint(
            project_root,
            entrypoint=wrapper_entrypoint,
            cmd=wrapper_cmd,
        )
        source_path = generated_source.relative_to(project_root).as_posix()
        dockerfile_str = _append_generated_entrypoint_copy(
            dockerfile_str,
            source_path=source_path,
            python_version=python_version,
        )

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


def _validate_docker_command_list(
    app_name: str,
    field_name: str,
    value: Any,
) -> list[str] | None:
    if value is None:
        return None
    if not (isinstance(value, list) and all(isinstance(item, str) for item in value)):
        raise ValueError(
            f"App {app_name} image.{field_name} must be a list of strings "
            "in pyproject.toml"
        )
    if len(value) == 0:
        raise ValueError(
            f"App {app_name} image.{field_name} must not be empty in pyproject.toml"
        )
    return value


def _write_generated_docker_entrypoint(
    project_root: Path,
    *,
    entrypoint: list[str] | None,
    cmd: list[str] | None,
) -> Path:
    generated_dir = Path(
        tempfile.mkdtemp(prefix="fal-generated-", dir=project_root)
    ).resolve()
    atexit.register(shutil.rmtree, generated_dir, ignore_errors=True)
    generated_path = generated_dir / "fal_entrypoint"
    generated_path.write_text(
        _render_generated_docker_entrypoint(entrypoint=entrypoint, cmd=cmd),
        encoding="utf-8",
    )
    return generated_path


def _render_generated_docker_entrypoint(
    *,
    entrypoint: list[str] | None,
    cmd: list[str] | None,
) -> str:
    return f"""\
import os
import signal
import subprocess

_ENTRYPOINT = {json.dumps(entrypoint or [])}
_CMD = {json.dumps(cmd or [])}


def _argv():
    if _ENTRYPOINT:
        return [*_ENTRYPOINT, *_CMD]
    return [*_CMD]


class fal_entry_point:
    @staticmethod
    def build_metadata():
        return {{}}

    @staticmethod
    def run_local():
        popen_kwargs = {{}}
        if os.name != "nt":
            popen_kwargs["start_new_session"] = True

        proc = subprocess.Popen(
            _argv(),
            stdin=subprocess.DEVNULL,
            **popen_kwargs,
        )

        def _terminate(signum, frame):
            del frame
            try:
                if os.name != "nt":
                    os.killpg(proc.pid, signum)
                else:
                    proc.send_signal(signum)
            except ProcessLookupError:
                pass

        try:
            signal.signal(signal.SIGTERM, _terminate)
            signal.signal(signal.SIGINT, _terminate)
        except ValueError:
            pass

        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"Process exited with code {{code}}")
"""


def _append_generated_entrypoint_copy(
    dockerfile_str: str,
    *,
    source_path: str,
    python_version: str | None,
) -> str:
    if not re.match(r"^[A-Za-z0-9._/-]+$", source_path):
        raise ValueError(f"Invalid generated entrypoint path: {source_path}")

    python_archive_url, python_archive_checksum = _python_runtime_archive(
        python_version
    )

    return (
        dockerfile_str.rstrip()
        + "\n\n"
        + "# fal generated Python runtime for wrapper entrypoint\n"
        + f"ADD --checksum={python_archive_checksum} {python_archive_url} "
        + "/tmp/fal-python.tar.gz\n"
        + "RUN mkdir -p /opt/fal && "
        + "tar -xzf /tmp/fal-python.tar.gz -C /opt/fal && "
        + "rm /tmp/fal-python.tar.gz\n"
        + 'ENV PATH="${PATH}:/opt/fal/python/bin"\n\n'
        + "# fal generated Docker wrapper entrypoint\n"
        + f"COPY {source_path} /app/fal_entrypoint.py\n"
        + 'ENV PYTHONPATH="/app:${PYTHONPATH}"\n'
    )


def _python_runtime_archive(python_version: str | None) -> tuple[str, str]:
    if python_version is None:
        normalized_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    else:
        match = re.match(r"^(\d+\.\d+)(?:\.\d+)?$", python_version)
        if match is None:
            raise ValueError(
                "python_version must be in '<major>.<minor>' or "
                "'<major>.<minor>.<patch>' format for image-only deployments."
            )
        normalized_version = match.group(1)

    try:
        filename, checksum = _PYTHON_BUILD_STANDALONE_ARCHIVES[normalized_version]
    except KeyError:
        supported = ", ".join(sorted(_PYTHON_BUILD_STANDALONE_ARCHIVES))
        raise ValueError(
            "image-only deployments currently support generated Python runtime "
            f"injection for Python versions: {supported}."
        )

    escaped_filename = filename.replace("+", "%2B")
    return (
        "https://github.com/astral-sh/python-build-standalone/releases/download/"
        f"{_PYTHON_BUILD_STANDALONE_RELEASE}/{escaped_filename}",
        checksum,
    )
