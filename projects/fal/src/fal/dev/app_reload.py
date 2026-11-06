"""Decide how an edit reaches a running `fal dev app` process.

Three outcomes, from cheapest to most expensive:

* reload: the endpoint code changed. The class is swapped in place and the
  instance state that ``setup()`` built (weights on the GPU) is kept.
* restart: ``setup()`` or something it uses changed, so the weights must be
  loaded again. The app process is restarted on the same runner.
* rebuild: the environment changed (requirements, image, machine type), which
  needs a new runner.

``setup()`` dependencies are found by walking the names its code references:
local functions, classes and modules are followed, and constants are compared
by value. Module globals that this code assigns hold runtime state, so the
child carries their values across a reload instead of comparing them. Code
reached only through objects stored on ``self`` is not seen, and caches kept in
other module globals reset on reload; `fal dev app restart` covers both.
"""

from __future__ import annotations

import dis
import enum
import inspect
import textwrap
import types
from pathlib import Path
from typing import Any

# Host settings that are baked into the runner's environment.
ENVIRONMENT_KEYS = ("kind", "image", "python_version", "machine_type", "num_gpus")
CONSTANT_TYPES = (str, bytes, int, float, complex, bool, type(None), tuple, frozenset)


def _source(obj: Any) -> str | None:
    try:
        return textwrap.dedent(inspect.getsource(obj))
    except (OSError, TypeError):
        return None


def _is_local(obj: Any, root: Path) -> bool:
    try:
        path = inspect.getsourcefile(obj) or ""
    except TypeError:
        return False
    return bool(path) and Path(path).resolve().is_relative_to(root)


def _code_names(code: types.CodeType) -> set[str]:
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _code_names(const)
    return names


def _store_globals(code: types.CodeType) -> set[str]:
    names = {
        ins.argval for ins in dis.get_instructions(code) if ins.opname == "STORE_GLOBAL"
    }
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _store_globals(const)
    return names


def _comparable(value: Any) -> str | None:
    """A stable repr for a constant, or None for values without one."""
    if isinstance(value, CONSTANT_TYPES):
        return repr(value)
    if callable(value) and not isinstance(value, enum.Enum):
        return None
    # Containers, paths, enums, dataclass/pydantic config objects: anything
    # whose repr shows its value rather than its address.
    try:
        text = repr(value)
    except Exception:
        return None
    return None if " at 0x" in text else text


def _analyze(cls: type, root: Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Walk setup() and the local code it references.

    Returns the fingerprint (source and constant values by name) and the module
    globals that code assigns, which hold runtime state rather than constants.
    """
    stack: list[Any] = [
        klass.__dict__["setup"]
        for klass in reversed(cls.__mro__)
        if "setup" in klass.__dict__ and _is_local(klass.__dict__["setup"], root)
    ]
    fingerprint: dict[str, str] = {}
    writes: dict[str, set[str]] = {}
    constants: dict[str, tuple[str, str]] = {}
    seen: set[int] = set()
    while stack:
        item = stack.pop()
        fn = inspect.unwrap(getattr(item, "__func__", item))
        code = getattr(fn, "__code__", None)
        if code is None or id(code) in seen:
            continue
        seen.add(id(code))
        fingerprint[f"{fn.__module__}.{fn.__qualname__}"] = _source(fn) or repr(
            code.co_code
        )
        stored = _store_globals(code)
        if stored:
            writes.setdefault(fn.__module__, set()).update(stored)

        names = _code_names(code)
        scopes: list[tuple[str, Any]] = [(fn.__module__, fn.__globals__), ("self", cls)]
        index = 0
        while index < len(scopes):
            scope_name, scope = scopes[index]
            index += 1
            for name in names:
                value = (
                    scope.get(name)
                    if isinstance(scope, dict)
                    else getattr(scope, name, None)
                )
                if value is None:
                    continue
                key = f"{scope_name}.{name}"
                if isinstance(value, types.ModuleType):
                    known = {s for s, _ in scopes}
                    if value.__name__ not in known and _is_local(value, root):
                        scopes.append((value.__name__, vars(value)))
                elif isinstance(value, (types.FunctionType, types.MethodType)):
                    if _is_local(value, root):
                        stack.append(value)
                elif isinstance(value, type):
                    if value is not cls and _is_local(value, root):
                        fingerprint[key] = _source(value) or value.__qualname__
                elif (text := _comparable(value)) is not None:
                    constants[key] = (scope_name, text)
    for key, (scope_name, text) in constants.items():
        if key.rsplit(".", 1)[-1] not in writes.get(scope_name, ()):
            fingerprint[key] = text
    return fingerprint, writes


def setup_fingerprint(cls: type, root: str | Path) -> dict[str, str]:
    """Source of ``setup()`` and of the local code and constants it references."""
    return _analyze(cls, Path(root).resolve())[0]


def setup_global_writes(cls: type, root: str | Path) -> dict[str, set[str]]:
    """Module globals that ``setup()`` (or code it calls) assigns, by module."""
    return _analyze(cls, Path(root).resolve())[1]


def environment_fingerprint(cls: type) -> dict[str, str]:
    """The settings that would need a new runner to change."""
    host = getattr(cls, "host_kwargs", {}) or {}
    result = {"requirements": repr(getattr(cls, "requirements", None))}
    for key in ENVIRONMENT_KEYS:
        value = getattr(cls, key, None)
        if value is None:
            value = host.get(key)
        if value is not None and hasattr(value, "dockerfile_str"):
            value = value.dockerfile_str
        result[key] = repr(value)
    return result


def changed_keys(old: dict[str, str], new: dict[str, str]) -> list[str]:
    return sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))


def decide(
    old_setup: dict[str, str],
    new_setup: dict[str, str],
    old_env: dict[str, str],
    new_env: dict[str, str],
) -> tuple[str, str]:
    """``(action, reason)`` where action is reload, restart or rebuild."""
    env = changed_keys(old_env, new_env)
    if env:
        return "rebuild", f"{', '.join(env)} changed; the runner must be rebuilt"
    setup = changed_keys(old_setup, new_setup)
    if setup:
        names = ", ".join(k.rsplit(".", 1)[-1] for k in setup)
        return "restart", f"setup() or what it uses changed ({names})"
    return "reload", "endpoint code changed"
