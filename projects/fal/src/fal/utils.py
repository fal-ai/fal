from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .api import FalServerlessHost, IsolatedFunction


@dataclass
class LoadedFunction:
    function: IsolatedFunction
    endpoints: list[str]
    app_name: str | None
    app_auth: str | None
    source_code: str | None


def _find_target(
    module: dict[str, object], function_name: str | None = None
) -> tuple[object, str | None, str | None]:
    import fal
    from fal.api import FalServerlessError, IsolatedFunction

    if function_name is not None:
        if function_name not in module:
            raise FalServerlessError(f"Function '{function_name}' not found in module")

        target = module[function_name]

        if isinstance(target, type) and issubclass(target, fal.App):
            return target, target.app_name, target.app_auth

        if isinstance(target, IsolatedFunction):
            return target, function_name, None

        raise FalServerlessError(
            f"Function '{function_name}' is not a fal.App or a fal.function"
        )

    fal_apps = {
        obj_name: obj
        for obj_name, obj in module.items()
        if isinstance(obj, type) and issubclass(obj, fal.App) and obj is not fal.App
    }

    if len(fal_apps) == 1:
        [(function_name, target)] = fal_apps.items()
        return target, target.app_name, target.app_auth
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
    return target, function_name, None


def load_function_from(
    host: FalServerlessHost,
    file_path: str,
    function_name: str | None = None,
    *,
    force_env_build: bool = False,
) -> LoadedFunction:
    import os
    import runpy
    import sys

    import fal._serialization
    from fal import App, wrap_app

    from .api import FalServerlessError, IsolatedFunction

    sys.path.append(os.getcwd())
    module = runpy.run_path(file_path)
    target, app_name, app_auth = _find_target(module, function_name)

    # The module for the function is set to <run_path> when runpy is used, in which
    # case we want to manually include the package it is defined in.
    fal._serialization.include_package_from_path(file_path)

    with open(file_path) as f:
        source_code = f.read()

    endpoints = ["/"]
    if isinstance(target, type) and issubclass(target, App):
        endpoints = target.get_endpoints() or ["/"]
        target = wrap_app(target, host=host, force_env_build=force_env_build)

    if not isinstance(target, IsolatedFunction):
        raise FalServerlessError(
            f"Function '{function_name}' is not a fal.function or a fal.App"
        )
    return LoadedFunction(
        target, endpoints, app_name=app_name, app_auth=app_auth, source_code=source_code
    )
