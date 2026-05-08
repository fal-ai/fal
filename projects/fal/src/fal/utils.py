from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from fal.api.api import merge_basic_config
from fal.sdk import AuthModeLiteral

if TYPE_CHECKING:
    from fal import App

    from .api import FalServerlessHost, IsolatedFunction, Options


@dataclass
class LoadedFunction:
    function: IsolatedFunction
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
            options=options,
            app_name=app_name,
            app_auth=app_auth,
        )

    if file_path is None:
        raise FalServerlessError("App ref must resolve to a file path.")

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

    if isinstance(target, type) and issubclass(target, App):
        _apply_toml_app_file_options(target, options)
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
        app_name=app_name,
        app_auth=app_auth,
        source_code=source_code,
        class_name=class_name,
    )


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


def _load_from_python_entry_point(
    host: FalServerlessHost,
    python_entry_point: str,
    *,
    options: Optional[Options] = None,
    app_name: str | None = None,
    app_auth: AuthModeLiteral | None = None,
) -> LoadedFunction:
    from copy import deepcopy

    from fal.api import IsolatedFunction
    from fal.api import Options as ApiOptions

    # Validate the format up-front; the return value is unused but the call
    # raises FalServerlessError on a malformed entrypoint string so we fail
    # fast instead of leaking an obscure error at dispatch time.
    _parse_python_entry_point(python_entry_point)

    merged_options = deepcopy(options) if options is not None else ApiOptions()
    merged_options.gateway.setdefault("exposed_port", 8080)

    isolated_function: IsolatedFunction = IsolatedFunction(
        host=host,
        options=merged_options,
        app_name=app_name,
        app_auth=app_auth,
        entrypoint=python_entry_point,
    )

    return LoadedFunction(
        isolated_function,
        app_name=app_name,
        app_auth=app_auth,
        source_code=None,
        class_name=python_entry_point,
    )


def _merge_options(target: Options, incoming: Options) -> None:
    merge_basic_config(target.host, incoming.host)
    merge_basic_config(target.environment, incoming.environment)
    merge_basic_config(target.gateway, incoming.gateway)
