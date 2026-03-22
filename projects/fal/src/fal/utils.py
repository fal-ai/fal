from __future__ import annotations

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
    import os
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


def _extract_local_project_extras(requirement: str) -> list[str]:
    req = requirement.strip()
    if req.startswith("-e "):
        req = req[3:].strip()
    if not req.startswith(".[") or not req.endswith("]"):
        return []
    extras = req[2:-1]
    return [item.strip() for item in extras.split(",") if item.strip()]


def _is_local_project_requirement(requirement: str) -> bool:
    req = requirement.strip()
    if req.startswith("-e "):
        req = req[3:].strip()
    return req == "." or req.startswith("./") or req.startswith(".[")


def _strip_local_project_requirements(
    requirements: list[str] | list[list[str]] | None,
) -> tuple[list[str] | list[list[str]] | None, list[str]]:
    if requirements is None:
        return None, []

    extras: list[str] = []
    if all(isinstance(item, str) for item in requirements):
        filtered: list[str] = []
        for raw_req in requirements:
            req_value = str(raw_req)
            if _is_local_project_requirement(req_value):
                extras.extend(_extract_local_project_extras(req_value))
                continue
            filtered.append(req_value)
        return filtered, extras

    if all(isinstance(item, list) for item in requirements):
        filtered_layers: list[list[str]] = []
        for layer in requirements:
            filtered_layer: list[str] = []
            for raw_req in layer:
                req_value = str(raw_req)
                if _is_local_project_requirement(req_value):
                    extras.extend(_extract_local_project_extras(req_value))
                    continue
                filtered_layer.append(req_value)
            if filtered_layer:
                filtered_layers.append(filtered_layer)
        return filtered_layers, extras

    return requirements, extras


def _is_fal_requirement(requirement: str) -> bool:
    req = requirement.strip()
    if not req or req.startswith("-"):
        return False

    package_name = req
    for separator in ("[", "<", ">", "=", "!", "~", ";", " "):
        package_name = package_name.split(separator, 1)[0]
    return package_name.lower() == "fal"


def _ensure_requirements_include_fal(
    requirements: list[str] | list[list[str]] | None,
) -> list[str] | list[list[str]]:
    if requirements is None:
        return ["fal"]

    if all(isinstance(item, str) for item in requirements):
        normalized_requirements = [str(item) for item in requirements]
        if any(_is_fal_requirement(req) for req in normalized_requirements):
            return normalized_requirements
        return [*normalized_requirements, "fal"]

    if all(isinstance(item, list) for item in requirements):
        normalized_layers = [[str(item) for item in layer] for layer in requirements]
        if any(
            _is_fal_requirement(req) for layer in normalized_layers for req in layer
        ):
            return normalized_layers
        if not normalized_layers:
            return [["fal"]]
        normalized_layers[-1] = [*normalized_layers[-1], "fal"]
        return normalized_layers

    return ["fal"]


def _materialize_runtime_wheel_path(
    wheel_local_path: Path, wheel_file_name: str
) -> Path:
    import shutil
    import tempfile

    resolved_wheel_path = wheel_local_path.resolve()
    if (
        resolved_wheel_path.suffix == ".whl"
        and resolved_wheel_path.name == wheel_file_name
    ):
        return resolved_wheel_path

    runtime_wheel_dir = Path(tempfile.mkdtemp(prefix="fal-no-pickle-runtime-wheel-"))
    runtime_wheel_path = runtime_wheel_dir / wheel_file_name
    shutil.copy2(resolved_wheel_path, runtime_wheel_path)
    return runtime_wheel_path


def _build_local_wheel_install_target(
    wheel_local_path: Path, wheel_file_name: str, extras: list[str]
) -> str:
    if not extras:
        return str(wheel_local_path.resolve())

    distribution_name = wheel_file_name.split("-", 1)[0].replace("_", "-")
    extras_spec = ",".join(sorted(set(extras)))
    return (
        f"{distribution_name}[{extras_spec}]"
        f" @ {wheel_local_path.resolve().as_uri()}"
    )


def _build_runtime_pip_install_command(
    install_target: str, target_dir: Path
) -> list[str]:
    import sys

    return [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--target",
        str(target_dir),
        install_target,
    ]


def _build_project_wheel(project_root: Path) -> Path:
    import subprocess
    import sys
    import tempfile

    wheel_dir = Path(tempfile.mkdtemp(prefix="fal-no-pickle-wheel-"))
    command = [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--wheel-dir",
        str(wheel_dir),
        str(project_root),
    ]
    process = subprocess.run(command, check=False, capture_output=True, text=True)
    if process.returncode != 0:
        stderr = process.stderr.strip()
        stdout = process.stdout.strip()
        details = stderr or stdout or "unknown pip wheel error"
        raise RuntimeError(f"Failed to build project wheel for --no-pickle: {details}")

    wheel_files = sorted(wheel_dir.glob("*.whl"))
    if not wheel_files:
        raise RuntimeError(
            "Failed to build project wheel for --no-pickle: no wheel found."
        )
    return wheel_files[0]


def _prepare_no_pickle_entry_point_options(
    project_root: Path | None, options: Optional[Options]
) -> tuple[Options, str, str, list[str]]:
    import shutil

    from fal.api import FalServerlessError, Options

    if project_root is None:
        raise FalServerlessError(
            "python_entry_point runs require project root metadata from pyproject.toml."
        )

    project_root_path = Path(project_root).resolve()
    merged_options = deepcopy(options) if options is not None else Options()

    requirements = merged_options.environment.get("requirements")
    filtered_requirements, extras = _strip_local_project_requirements(requirements)
    merged_options.environment["requirements"] = _ensure_requirements_include_fal(
        filtered_requirements
    )

    wheel_path = _build_project_wheel(project_root_path)

    host_options = merged_options.host
    context_dir_raw = host_options.get("app_files_context_dir")
    if context_dir_raw is None:
        context_dir = project_root_path
        host_options["app_files_context_dir"] = str(context_dir)
    else:
        context_dir = Path(context_dir_raw).resolve()

    try:
        wheel_relative_path = str(wheel_path.relative_to(context_dir))
    except ValueError:
        # pip wheel outputs to a temp dir by default. For --no-pickle runs we need
        # the wheel to be part of app_files_context_dir so it is synced and
        # installable remotely.
        staged_wheel_path = context_dir / "fal-no-pickle-wheels" / wheel_path.name
        staged_wheel_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wheel_path, staged_wheel_path)
        wheel_path = staged_wheel_path
        try:
            wheel_relative_path = str(wheel_path.relative_to(context_dir))
        except ValueError as exc:
            raise FalServerlessError(
                "Built project wheel must be inside "
                "app_files_context_dir for --no-pickle."
            ) from exc

    app_files = host_options.get("app_files")
    if app_files is None:
        host_options["app_files"] = [wheel_relative_path]
    elif wheel_relative_path not in app_files:
        host_options["app_files"] = [*app_files, wheel_relative_path]

    return merged_options, wheel_relative_path, wheel_path.name, extras


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


def _install_packaged_project_wheel(
    wheel_relative_path: str, wheel_file_name: str, extras: list[str]
) -> None:
    import subprocess
    import sys
    import tempfile

    from fal.api import FalServerlessError

    wheel_local_path = (Path.cwd() / wheel_relative_path).resolve()
    if not wheel_local_path.exists():
        raise FalServerlessError(f"Wheel file not found at runtime: {wheel_local_path}")

    runtime_wheel_path = _materialize_runtime_wheel_path(
        wheel_local_path, wheel_file_name
    )
    install_target = _build_local_wheel_install_target(
        runtime_wheel_path, wheel_file_name, extras
    )
    runtime_site_packages = Path(
        tempfile.mkdtemp(prefix="fal-no-pickle-site-packages-")
    )
    install_command = _build_runtime_pip_install_command(
        install_target, runtime_site_packages
    )

    process = subprocess.run(
        install_command,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        message = process.stderr.strip() or process.stdout.strip()
        if not message:
            message = "unknown pip install error"
        raise FalServerlessError(
            f"Failed to install packaged project wheel " f"{wheel_file_name}: {message}"
        )
    sys.path.insert(0, str(runtime_site_packages))


def _resolve_python_entry_target(python_entry_point: str) -> object:
    import importlib

    from fal.api import FalServerlessError

    module_name, symbol_name = _parse_python_entry_point(python_entry_point)
    module = importlib.import_module(module_name)
    if not hasattr(module, symbol_name):
        raise FalServerlessError(
            f"Symbol '{symbol_name}' not found in module '{module_name}'."
        )
    return getattr(module, symbol_name)


def _run_python_entry_target(target: object) -> None:
    from typing import cast

    import fal
    from fal.api import FalServerlessError, IsolatedFunction

    if (
        isinstance(target, type)
        and issubclass(target, fal.App)
        and target is not fal.App
    ):
        wrapped_target = fal.wrap_app(cast(type[fal.App], target))
        wrapped_target.run_local()
        return

    if isinstance(target, IsolatedFunction):
        target.run_local()
        return

    run_local = getattr(target, "run_local", None)
    if callable(run_local) and hasattr(target, "options"):
        run_local()
        return

    raise FalServerlessError(
        "Resolved python_entry_point symbol is not a fal.App or fal.function."
    )


def _build_python_entry_point_wrapper(
    python_entry_point: str,
    wheel_relative_path: str,
    wheel_file_name: str,
    extras: list[str],
):
    import fal

    @fal.function()
    def no_pickle_entry_point_wrapper():
        _python_entry_point = python_entry_point
        _wheel_relative_path = wheel_relative_path
        _wheel_file_name = wheel_file_name
        _extras = extras

        _install_packaged_project_wheel(
            _wheel_relative_path,
            _wheel_file_name,
            _extras,
        )
        target = _resolve_python_entry_target(_python_entry_point)
        _run_python_entry_target(target)

    return no_pickle_entry_point_wrapper


def _load_from_python_entry_point(
    host: FalServerlessHost,
    python_entry_point: str,
    *,
    project_root: Path | None,
    options: Optional[Options] = None,
    app_name: str | None = None,
    app_auth: AuthModeLiteral | None = None,
) -> LoadedFunction:
    merged_options, wheel_relative_path, wheel_file_name, extras = (
        _prepare_no_pickle_entry_point_options(project_root, options)
    )
    no_pickle_entry_point_wrapper = _build_python_entry_point_wrapper(
        python_entry_point,
        wheel_relative_path,
        wheel_file_name,
        extras,
    )
    isolated_function = no_pickle_entry_point_wrapper.on(host=host)
    _merge_options(isolated_function.options, merged_options)
    isolated_function.app_name = app_name
    isolated_function.app_auth = app_auth

    return LoadedFunction(
        isolated_function,
        ["/"],
        app_name=app_name,
        app_auth=app_auth,
        source_code=None,
        class_name=python_entry_point,
    )


def _merge_options(target: Options, incoming: Options) -> None:
    merge_basic_config(target.host, incoming.host)
    merge_basic_config(target.environment, incoming.environment)
    merge_basic_config(target.gateway, incoming.gateway)
