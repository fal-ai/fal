from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .api import (
    FalMissingDependencyError,
    FalServerlessError,
    UserFunctionException,
    find_missing_dependencies,
)

if TYPE_CHECKING:
    from .api import IsolatedFunction, ResultHandler


def run(
    isolated_function: IsolatedFunction,
    *,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    local: bool = False,
    result_handler: ResultHandler | None = None,
    reraise: bool = True,
) -> Any:
    """Run an ``IsolatedFunction`` locally or on its host with friendly errors.

    When ``local=True``, defers to ``isolated_function.run_local`` (which
    invokes the function in the current Python process). ``result_handler``
    has no effect in that mode.

    Otherwise the call is dispatched through ``host.run`` and:

    * ``FalMissingDependencyError`` is translated into ``FalServerlessError``
      with a hint listing modules accessed by the function but missing from
      the environment definition.
    * ``UserFunctionException`` is unwrapped to its underlying cause when
      ``reraise=True`` (default), so callers see the original exception.
    """
    if kwargs is None:
        kwargs = {}

    if local:
        return isolated_function.run_local(*args, **kwargs)

    entrypoint = isolated_function.entrypoint
    func = None if entrypoint else isolated_function.func
    options = isolated_function.options
    try:
        return isolated_function.host.run(
            func,
            options,
            args=args,
            kwargs=kwargs,
            application_name=isolated_function.app_name,
            application_auth_mode=isolated_function.app_auth,
            result_handler=result_handler,
            entrypoint=entrypoint,
        )
    except FalMissingDependencyError as e:
        # Pickle deserialization can't fail in entrypoint mode, so func is set.
        assert func is not None
        pairs = list(find_missing_dependencies(func, options.environment))
        if not pairs:
            raise e
        lines = [
            f"\t- {used_modules!r} "
            f"(accessed through {', '.join(map(repr, references))})"
            for used_modules, references in pairs
        ]
        raise FalServerlessError(
            f"Couldn't deserialize your function on the remote server. \n\n"
            f"[Hint] {func.__name__!r} function uses the following modules "
            "which weren't present in the environment definition:\n" + "\n".join(lines)
        ) from None
    except Exception as exc:
        cause = exc.__cause__
        if reraise and isinstance(exc, UserFunctionException) and cause:
            # re-raise original exception without our wrappers
            raise cause
        raise
