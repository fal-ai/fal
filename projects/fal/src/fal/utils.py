from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from fal.api.api import merge_basic_config
from fal.sdk import AuthModeLiteral

if TYPE_CHECKING:
    from fal import App

    from .api import FalServerlessHost, IsolatedFunction, Options


@dataclass
class LoadedFunction:
    function: IsolatedFunction
    endpoints: list[str]
    app_name: str | None
    app_auth: AuthModeLiteral | None
    source_code: str | None
    class_name: str | None = None


def _find_target(
    module: dict[str, object], function_name: str | None = None
) -> tuple[object, str | None, AuthModeLiteral | None, str | None]:
    """Find the target function/class in the module.

    Returns:
        A tuple of (target, app_name, app_auth, class_name) where:
        - target: The fal.App class or IsolatedFunction
        - app_name: The deployment name (e.g., "mock-model")
        - app_auth: The auth mode
        - class_name: The actual class/function name (e.g., "MockModel")
    """
    import fal
    from fal.api import FalServerlessError, IsolatedFunction

    if function_name is not None:
        if function_name not in module:
            raise FalServerlessError(f"Function '{function_name}' not found in module")

        target = module[function_name]

        if isinstance(target, type) and issubclass(target, fal.App):
            return target, target.app_name, target.app_auth, function_name

        if isinstance(target, IsolatedFunction):
            return target, function_name, None, function_name

        raise FalServerlessError(
            f"Function '{function_name}' is not a fal.App or a fal.function"
        )

    fal_apps = {
        obj_name: obj
        for obj_name, obj in module.items()
        if isinstance(obj, type) and issubclass(obj, fal.App) and obj is not fal.App
    }

    if len(fal_apps) == 1:
        [(class_name, target)] = fal_apps.items()
        return target, target.app_name, target.app_auth, class_name
    elif len(fal_apps) > 1:
        raise FalServerlessError(
            f"Multiple fal.Apps found in the module: {list(fal_apps.keys())}. "
            "Please specify the name of the app."
        )

    fal_functions = {
        obj_name: obj
        for obj_name, obj in module.items()
        if isinstance(obj, IsolatedFunction)
    }

    if len(fal_functions) == 0:
        raise FalServerlessError("No fal.App or fal.function found in the module.")
    elif len(fal_functions) > 1:
        raise FalServerlessError(
            "Multiple fal.functions found in the module: "
            f"{list(fal_functions.keys())}. "
            "Please specify the name of the function."
        )

    [(function_name, target)] = fal_functions.items()
    return target, function_name, None, function_name


def _apply_toml_app_file_options(
    app_cls: type[App], options: Optional[Options]
) -> None:
    if options is None:
        return

    host_options = options.host
    app_files = host_options.get("app_files")

    # Preserve App-defined app_files; TOML app_files are only a fallback.
    if not app_files or app_cls.app_files:
        return

    app_cls.app_files = app_files
    app_cls.host_kwargs["app_files"] = app_files

    app_files_ignore = host_options.get("app_files_ignore")
    if app_files_ignore is not None:
        app_cls.app_files_ignore = app_files_ignore
        app_cls.host_kwargs["app_files_ignore"] = app_files_ignore

    app_files_context_dir = host_options.get("app_files_context_dir")
    if app_files_context_dir is not None:
        app_cls.app_files_context_dir = app_files_context_dir
        app_cls.host_kwargs["app_files_context_dir"] = app_files_context_dir


def load_function_from(
    host: FalServerlessHost,
    file_path: str | None,
    function_name: str | None = None,
    *,
    force_env_build: bool = False,
    options: Optional[Options] = None,
    app_name: str | None = None,
    app_auth: AuthModeLiteral | None = None,
    limit_max_requests: int | None = None,
    python_entry_point: str | None = None,
    project_root: Path | None = None,
) -> LoadedFunction:
    import runpy
    import sys

    import fal._serialization
    from fal import App, wrap_app

    from .api import FalServerlessError, IsolatedFunction

    if python_entry_point is not None:
        return _load_from_python_entry_point(
            host,
            python_entry_point,
            project_root=project_root,
            options=options,
            app_name=app_name,
            app_auth=app_auth,
        )

    if file_path is None:
        raise ValueError("App ref must resolve to a file path.")

    sys.path.append(os.getcwd())
    module = runpy.run_path(file_path)
    target, found_app_name, found_app_auth, class_name = _find_target(
        module, function_name
    )
    app_name = app_name or found_app_name
    app_auth = app_auth or found_app_auth

    # The module for the function is set to <run_path> when runpy is used, in which
    # case we want to manually include the package it is defined in.
    fal._serialization.include_package_from_path(file_path)

    with open(file_path) as f:
        source_code = f.read()

    endpoints = ["/"]
    if isinstance(target, type) and issubclass(target, App):
        _apply_toml_app_file_options(target, options)
        endpoints = target.get_endpoints() or ["/"]
        target = wrap_app(
            target,
            host=host,
            force_env_build=force_env_build,
            limit_max_requests=limit_max_requests,
        )

    if not isinstance(target, IsolatedFunction):
        raise FalServerlessError(
            f"Function '{function_name}' is not a fal.function or a fal.App"
        )
    if options is not None:
        _merge_options(target.options, options)

    # Override the host so CLI-provided context (e.g. team) is applied.
    # For @fal.function, the host was set at import time without CLI context.
    target.host = host

    target.app_name = app_name
    target.app_auth = app_auth
    return LoadedFunction(
        target,
        endpoints,
        app_name=app_name,
        app_auth=app_auth,
        source_code=source_code,
        class_name=class_name,
    )


def _replace_local_project_requirement(
    requirement: str, *, package_name: str, package_url: str
) -> str | None:
    req = requirement.strip()
    if req == "." or req.startswith("./"):
        return package_url
    if not req.startswith(".[") or not req.endswith("]"):
        return None

    extras = [item.strip() for item in req[2:-1].split(",") if item.strip()]
    if not extras:
        return package_url

    extras_spec = ",".join(sorted(set(extras)))
    return f"{package_name}[{extras_spec}] @ {package_url}"


def _build_project_sdist(project_root: Path) -> Path:
    from build import ProjectBuilder
    from build.env import DefaultIsolatedEnv

    sdist_dir = project_root / "dist"
    sdist_dir.mkdir(parents=True, exist_ok=True)
    with DefaultIsolatedEnv() as env:
        builder = ProjectBuilder.from_isolated_env(env, str(project_root))
        env.install(builder.build_system_requires)
        env.install(builder.get_requires_for_build("sdist"))
        sdist_filename = builder.build("sdist", str(sdist_dir))

    return sdist_dir / sdist_filename


_FAL_ENTRYPOINT_TEMPLATE = """\
# Auto-generated by fal SDK. Do not edit.
USER_ENTRYPOINT = {user_entrypoint!r}


def _resolve():
    import importlib

    module_name, _, attr = USER_ENTRYPOINT.partition(":")
    obj = importlib.import_module(module_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def get_metadata():
    import fal
    from fal.api import IsolatedFunction

    target = _resolve()
    if (
        isinstance(target, type)
        and issubclass(target, fal.App)
        and target is not fal.App
    ):
        app = target(_allow_init=True)
        return {{
            "openapi": app.openapi(),
            "endpoints": target.get_endpoints() or ["/"],
            "exposed_port": 8080,
        }}
    if isinstance(target, IsolatedFunction):
        return {{
            "openapi": {{}},
            "endpoints": getattr(target.raw_func, "_routes", ["/"]),
            "exposed_port": target.options.get_exposed_port() or 8080,
        }}
    raise RuntimeError(
        f"Resolved entrypoint {{USER_ENTRYPOINT!r}} is not a fal.App or fal.function."
    )


def run_local():
    target = _resolve()
    target.run_local()
"""


def _build_sdist_with_helper(
    project_root: Path, python_entry_point: str
) -> tuple[Path, str]:
    """Build sdist of the user's project with a generated _fal_entrypoint.py
    helper injected into the top-level package. Returns (sdist_path, helper_module).
    """
    import shutil
    import tempfile

    from fal.api import FalServerlessError

    module_name, _ = _parse_python_entry_point(python_entry_point)
    top_package = module_name.split(".", 1)[0]

    project_root = project_root.resolve()
    tmp_dir = Path(tempfile.mkdtemp(prefix="fal-sdist-"))
    try:
        project_copy = tmp_dir / project_root.name
        shutil.copytree(
            project_root,
            project_copy,
            ignore=shutil.ignore_patterns(
                "dist", "build", "*.egg-info", ".venv", "venv", "__pycache__"
            ),
        )

        package_dir: Optional[Path] = None
        for candidate in (
            project_copy / top_package,
            project_copy / "src" / top_package,
        ):
            if candidate.is_dir():
                package_dir = candidate
                break
        if package_dir is None:
            raise FalServerlessError(
                f"Cannot find package directory for {top_package!r} in "
                f"{project_root}. Expected {top_package}/ or src/{top_package}/."
            )

        helper_path = package_dir / "_fal_entrypoint.py"
        helper_path.write_text(
            _FAL_ENTRYPOINT_TEMPLATE.format(user_entrypoint=python_entry_point)
        )

        sdist_path = _build_project_sdist(project_copy)
        # Copy the sdist out of tmp_dir before it's cleaned up.
        final_sdist_dir = project_root / "dist"
        final_sdist_dir.mkdir(parents=True, exist_ok=True)
        final_sdist = final_sdist_dir / sdist_path.name
        shutil.copy2(sdist_path, final_sdist)

        return final_sdist, f"{top_package}._fal_entrypoint"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _prepare_no_pickle_entry_point_options(
    project_root: Path | None,
    options: Optional[Options],
    python_entry_point: str,
) -> tuple[Options, str]:
    from fal.api import FalServerlessError, Options
    from fal.toolkit import File

    if project_root is None:
        raise FalServerlessError(
            "python_entry_point runs require project root metadata from pyproject.toml."
        )

    project_root_path = Path(project_root).resolve()
    merged_options = deepcopy(options) if options is not None else Options()

    requirements = merged_options.environment.get("requirements")
    sdist_path, helper_module = _build_sdist_with_helper(
        project_root_path, python_entry_point
    )
    uploaded_sdist = File.from_path(sdist_path)
    default_packaged_requirement = uploaded_sdist.url
    distribution_name = sdist_path.name.split("-", 1)[0].replace("_", "-")

    if requirements is None:
        merged_options.environment["requirements"] = [default_packaged_requirement]
    elif all(isinstance(item, str) for item in requirements):
        replaced_any = False
        replaced_requirements: list[str] = []
        for raw_req in requirements:
            req_value = str(raw_req)
            replacement = _replace_local_project_requirement(
                req_value,
                package_name=distribution_name,
                package_url=uploaded_sdist.url,
            )
            if replacement is not None:
                replaced_any = True
                replaced_requirements.append(replacement)
            else:
                replaced_requirements.append(req_value)
        if not replaced_any:
            replaced_requirements.append(default_packaged_requirement)
        merged_options.environment["requirements"] = replaced_requirements
    elif all(isinstance(item, list) for item in requirements):
        replaced_any = False
        replaced_layers: list[list[str]] = []
        for layer in requirements:
            replaced_layer: list[str] = []
            for raw_req in layer:
                req_value = str(raw_req)
                replacement = _replace_local_project_requirement(
                    req_value,
                    package_name=distribution_name,
                    package_url=uploaded_sdist.url,
                )
                if replacement is not None:
                    replaced_any = True
                    replaced_layer.append(replacement)
                else:
                    replaced_layer.append(req_value)
            replaced_layers.append(replaced_layer)
        if replaced_layers:
            tail_layer = replaced_layers[-1]
            if not replaced_any:
                tail_layer.append(default_packaged_requirement)
        else:
            replaced_layers = [[default_packaged_requirement]]
        merged_options.environment["requirements"] = replaced_layers
    else:
        merged_options.environment["requirements"] = [default_packaged_requirement]

    return merged_options, helper_module


def _parse_python_entry_point(python_entry_point: str) -> tuple[str, str]:
    from fal.api import FalServerlessError

    if ":" not in python_entry_point:
        raise FalServerlessError(
            "python_entry_point must be in '<module>:<symbol>' format."
        )
    module_name, symbol_name = python_entry_point.split(":", 1)
    if not module_name or not symbol_name:
        raise FalServerlessError(
            "python_entry_point must be in '<module>:<symbol>' format."
        )
    return module_name, symbol_name


def _validate_probe_payload(payload: object) -> tuple[list[str], dict, int]:
    from fal.api import FalServerlessError

    if not isinstance(payload, dict):
        raise FalServerlessError(
            "python_entry_point metadata probe returned invalid payload."
        )

    openapi = payload.get("openapi")
    endpoints = payload.get("endpoints")
    exposed_port = payload.get("exposed_port")
    if not isinstance(openapi, dict):
        raise FalServerlessError(
            "python_entry_point metadata probe is missing valid openapi spec."
        )
    if (
        not isinstance(endpoints, list)
        or not endpoints
        or not all(isinstance(e, str) for e in endpoints)
    ):
        raise FalServerlessError(
            "python_entry_point metadata probe is missing valid endpoints list."
        )
    if not isinstance(exposed_port, int):
        raise FalServerlessError(
            "python_entry_point metadata probe is missing valid exposed_port."
        )
    return endpoints, openapi, exposed_port


def _probe_python_entry_point_metadata(
    host: FalServerlessHost,
    helper_module: str,
    options: Options,
) -> tuple[list[str], dict, int]:
    """Run a one-shot probe on the worker via ``<helper>:get_metadata`` and
    return ``(endpoints, openapi, exposed_port)``.

    The result is a plain dict, so it round-trips through pickle safely across
    Python versions.
    """
    from fal.api import ResultHandler

    payload: object = host.run(
        None,
        options,
        args=(),
        kwargs={},
        entrypoint=f"{helper_module}:get_metadata",
        result_handler=ResultHandler(),
    )
    return _validate_probe_payload(payload)


def _load_from_python_entry_point(
    host: FalServerlessHost,
    python_entry_point: str,
    *,
    project_root: Path | None,
    options: Optional[Options] = None,
    app_name: str | None = None,
    app_auth: AuthModeLiteral | None = None,
) -> LoadedFunction:
    import fal

    _parse_python_entry_point(python_entry_point)

    merged_options, helper_module = _prepare_no_pickle_entry_point_options(
        project_root, options, python_entry_point
    )

    @fal.function()
    def _placeholder() -> None: ...

    isolated_function = _placeholder.on(host=host)
    _merge_options(isolated_function.options, merged_options)
    endpoints, openapi, exposed_port = _probe_python_entry_point_metadata(
        host, helper_module, isolated_function.options
    )
    isolated_function.options.host["metadata"] = {"openapi": openapi}
    isolated_function.options.gateway["exposed_port"] = exposed_port
    isolated_function.app_name = app_name
    isolated_function.app_auth = app_auth
    isolated_function.entrypoint = f"{helper_module}:run_local"

    return LoadedFunction(
        isolated_function,
        endpoints=endpoints,
        app_name=app_name,
        app_auth=app_auth,
        source_code=None,
        class_name=python_entry_point,
    )


def _merge_options(target: Options, incoming: Options) -> None:
    merge_basic_config(target.host, incoming.host)
    merge_basic_config(target.environment, incoming.environment)
    merge_basic_config(target.gateway, incoming.gateway)
