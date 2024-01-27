from __future__ import annotations

import inspect
from functools import wraps
from typing import (
    Callable,
    TypeVar,
)

from typing_extensions import ParamSpec

ArgsT = ParamSpec("ArgsT")
ReturnT = TypeVar("ReturnT", covariant=True)


def cached(func: Callable[ArgsT, ReturnT]) -> Callable[ArgsT, ReturnT]:
    """Cache the result of the given function in-memory."""
    import hashlib

    try:
        source_code = inspect.getsource(func).encode("utf-8")
    except OSError:
        # TODO: explain the reason for this (e.g. we don't know how to
        # check if you sent us the same function twice).
        print(f"[warning] Function {func.__name__} can not be cached...")
        return func

    cache_key = hashlib.sha256(source_code).hexdigest()

    @wraps(func)
    def wrapper(
        *args: ArgsT.args,
        **kwargs: ArgsT.kwargs,
    ) -> ReturnT:
        from functools import lru_cache

        # HACK: Using the isolate module as a global cache.
        import isolate

        if not hasattr(isolate, "__cached_functions__"):
            isolate.__cached_functions__ = {}

        if cache_key not in isolate.__cached_functions__:
            isolate.__cached_functions__[cache_key] = lru_cache(maxsize=None)(func)

        return isolate.__cached_functions__[cache_key](*args, **kwargs)

    return wrapper
