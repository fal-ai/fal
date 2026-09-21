from __future__ import annotations

import functools
import importlib
import pkgutil
import subprocess
import sys
from typing import ForwardRef

import cloudpickle
import pytest

import fal
from fal._serialization import patch_pickle


def test_forward_ref_serialization_drops_patch_version_cache():
    patch_pickle()

    ref = ForwardRef("App")
    if hasattr(ref, "__resolved_str_cache__"):
        ref.__resolved_str_cache__ = "App"

    payload = cloudpickle.dumps(ref)

    assert b"__resolved_str_cache__" not in payload
    restored = cloudpickle.loads(payload)
    assert restored.__forward_arg__ == ref.__forward_arg__


def test_no_foreign_lru_cache_wrappers_in_pickled_by_value_modules():
    """Pin the class of mistake: a lru_cache wrapper that cannot ship by value.

    ``include_module("fal")`` registers the whole package pickle-by-value, and
    ``_patch_lru_cache`` reduces a wrapper to ``(create_lru_cache, (__wrapped__,
    ...))``. That is fine while ``__wrapped__`` also ships by value. When it does
    not, cloudpickle cannot reach it by reference and pickles it by value along
    with its module's globals, including privates that only exist on some CPython
    patch releases.

    ``from urllib.parse import urlsplit`` did exactly that: ``urlsplit`` is
    cache-wrapped on 3.11 and later, so every deploy died on the runner with
    ``Can't get attribute '_check_bracketed_netloc'``. Import the module and
    qualify the call instead, since modules pickle by reference.
    """
    patch_pickle()
    by_value_roots = getattr(
        cloudpickle.cloudpickle, "_PICKLE_BY_VALUE_MODULES", {"fal", "tblib"}
    )

    for module in list(pkgutil.walk_packages(fal.__path__, "fal.")):
        try:
            importlib.import_module(module.name)
        except Exception:
            continue  # optional extra, not something this guard cares about

    foreign = []
    for name, module in list(sys.modules.items()):
        if not name.startswith("fal") or module is None:
            continue
        for attribute, value in vars(module).items():
            if not isinstance(value, functools._lru_cache_wrapper):
                continue
            wrapped = getattr(value, "__wrapped__", None)
            root, _, _ = (getattr(wrapped, "__module__", "") or "").partition(".")
            if root not in by_value_roots:
                foreign.append(f"{name}.{attribute}")

    assert foreign == []


# Run in a subprocess, never in-process. In-process the assertion would be
# vacuous: cloudpickle's dynamic_subimport builds a throwaway module and never
# writes sys.modules, and the dynamic class tracker short-circuits to the live
# class, so nothing is imported and no blocked module could bite. The blocker
# also evicts modules and installs a raising meta_path finder, which would
# poison every later test in the process.
_LOAD_WITH_BLOCKED_IMPORT = """
import importlib.abc, pickle, sys

blocked = set(sys.argv[1].split(","))


class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise ImportError("blocked for test: " + fullname)
        return None


for name in [n for n in list(sys.modules) if n.split(".")[0] in blocked]:
    del sys.modules[name]
sys.meta_path.insert(0, Blocker())

with open(sys.argv[2], "rb") as handle:
    pickle.loads(handle.read())
print("LOADED")
"""

# Every distribution a plain @fal.function environment does not install, so none
# of them may be needed to *load* the toolkit. starlette and anyio are the ones
# the toolkit can actually leak today, since `from fastapi import Request` binds
# starlette.requests.Request and blocking fastapi would never fire on it. Only
# pydantic is exempt: the toolkit pickle genuinely imports it at load time, and
# the runner base image provides it.
_SERVE_ONLY_MODULES = [
    "anyio",
    "fastapi",
    "httpx",
    "packaging",
    "starlette",
    "starlette_exporter",
    "structlog",
    "tblib",
    "tomli",
    "tomli_w",
    "uvicorn",
]


# Dumped in a child too, not with a plain cloudpickle.dumps here. A by-value
# module pickle carries the module's __dict__, so a FastAPI app left in
# fal.ref's current-app slot by an earlier test in the same worker would put
# fastapi into this pickle and make the result depend on test order. Only a
# serving process sets an app, and it installs SERVE_REQUIREMENTS, so this is a
# property of the test rather than of the product.
_DUMP_TOOLKIT_EXPORT = """
import sys

import cloudpickle

from fal._serialization import patch_pickle

patch_pickle()

from fal.toolkit import File, Image

with open(sys.argv[2], "wb") as handle:
    handle.write(cloudpickle.dumps({"File": File, "Image": Image}[sys.argv[1]]))
"""


def _dump_in_a_clean_process(exported: str, tmp_path) -> bytes:
    target = tmp_path / f"{exported}.pkl"
    result = subprocess.run(
        [sys.executable, "-c", _DUMP_TOOLKIT_EXPORT, exported, str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return target.read_bytes()


def _load_with_blocked_imports(payload: bytes, blocked: list[str], tmp_path):
    target = tmp_path / "payload.pkl"
    target.write_bytes(payload)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            _LOAD_WITH_BLOCKED_IMPORT,
            ",".join(blocked),
            str(target),
        ],
        capture_output=True,
        text=True,
        check=False,  # the failure mode is the assertion
    )


@pytest.mark.parametrize("exported", ["File", "Image"])
def test_toolkit_deserializes_without_serve_only_dependencies(exported, tmp_path):
    """A plain @fal.function environment installs no SERVE_REQUIREMENTS, so the
    toolkit's pickle must not need any of them in order to *load*.

    cloudpickle records a module found in a captured function's globals as a
    subimport that runs at deserialization. A module-level ``import httpx`` in
    _upload_policy therefore made httpx a hard requirement of every isolated
    environment, and any function returning a File or an Image died on the runner
    with "FalSerializationError: Could not find module 'httpx'".

    The whole serve-only set is blocked rather than one name, so a future
    module-level import cannot simply move the failure to the next module.
    """
    payload = _dump_in_a_clean_process(exported, tmp_path)

    result = _load_with_blocked_imports(payload, _SERVE_ONLY_MODULES, tmp_path)

    assert result.returncode == 0, result.stderr
    assert "LOADED" in result.stdout


# Re-introduces the exact defect: a function reachable from File.from_bytes that
# reads httpx as a *global*. Injecting a module attribute is not enough, since
# capture is per function, over the names a code object actually reads.
_DUMP_WITH_THE_REGRESSION = """
import sys

import cloudpickle
import httpx

from fal._serialization import patch_pickle

patch_pickle()

import fal.toolkit.file._upload_policy as up

up.httpx = httpx
exec("def _new_client():\\n    return httpx.Client()", up.__dict__)
up._new_client = up.__dict__["_new_client"]

from fal.toolkit import File

with open(sys.argv[1], "wb") as handle:
    handle.write(cloudpickle.dumps(File))
"""


def test_the_guard_catches_the_regression_it_exists_for(tmp_path):
    """Negative control for the test above: a clean dump passing proves nothing
    unless a dirty one fails. Dumping and loading in one process would not do,
    because it short-circuits to the live class and imports nothing.
    """
    target = tmp_path / "regressed.pkl"
    dump = subprocess.run(
        [sys.executable, "-c", _DUMP_WITH_THE_REGRESSION, str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert dump.returncode == 0, dump.stderr

    result = _load_with_blocked_imports(
        target.read_bytes(), _SERVE_ONLY_MODULES, tmp_path
    )

    assert result.returncode != 0
    assert "blocked for test: httpx" in result.stderr


def test_the_blocked_import_helper_actually_blocks(tmp_path):
    """Positive control. Without it the test above passes for free the day the
    blocker stops working.

    An httpx object is used rather than anything from fal: its class pickles by
    reference, so loading it must import httpx. That is exactly the load-time
    requirement the test above asserts the toolkit does not have.
    """
    import httpx  # noqa: PLC0415

    result = _load_with_blocked_imports(
        cloudpickle.dumps(httpx.Timeout(1.0)), ["httpx"], tmp_path
    )

    assert result.returncode != 0
    assert "blocked for test: httpx" in result.stderr
