from __future__ import annotations

import fal._serialization
from fal import App, wrap_app

from .api import FalServerlessError, FalServerlessHost, IsolatedFunction


def load_function_from(
    host: FalServerlessHost,
    file_path: str,
    function_name: str | None = None,
) -> tuple[IsolatedFunction, str | None]:
    import runpy

    module = runpy.run_path(file_path)
    if function_name is None:
        fal_objects = {
            obj.app_name: obj_name
            for obj_name, obj in module.items()
            if isinstance(obj, type)
            and issubclass(obj, fal.App)
            and hasattr(obj, "app_name")
        }
        if len(fal_objects) == 0:
            raise FalServerlessError("No fal.App found in the module.")
        elif len(fal_objects) > 1:
            raise FalServerlessError(
                "Multiple fal.Apps found in the module. "
                "Please specify the name of the app."
            )

        [(app_name, function_name)] = fal_objects.items()
    else:
        app_name = None

    if function_name not in module:
        raise FalServerlessError(f"Function '{function_name}' not found in module")

    # The module for the function is set to <run_path> when runpy is used, in which
    # case we want to manually include the package it is defined in.
    fal._serialization.include_package_from_path(file_path)

    target = module[function_name]
    if isinstance(target, type) and issubclass(target, App):
        app_name = target.app_name
        target = wrap_app(target, host=host)

    if not isinstance(target, IsolatedFunction):
        raise FalServerlessError(
            f"Function '{function_name}' is not a fal.function or a fal.App"
        )
    return target, app_name

