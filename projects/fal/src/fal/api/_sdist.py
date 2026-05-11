"""Build, upload, and rewrite local-path requirements for fal apps.

When a ``[tool.fal.apps.*]`` block in ``pyproject.toml`` lists ``.`` or
``.[extras]`` in its ``requirements``, the user means "install this project
itself on the worker." We can't ship the source tree as a pip requirement,
so we build an sdist locally, upload it to the fal CDN via
``fal.toolkit.File``, and rewrite the requirement to ``<package> @ <url>``
(or ``<package>[extras] @ <url>``) so the worker can pip-install it.

This module is the pure helper. ``FalServerlessHost`` calls
``materialize_local_paths`` on the way to dispatch.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

from fal.project import _load_toml

# ``on_progress(event, payload)`` callback signature used by
# ``materialize_local_paths`` to surface build/upload phases to a CLI front-end.
# Events emitted (with the keys their payload carries):
#
#   "build_started"  -> {"package_name": str, "project_root": Path}
#   "build_finished" -> {"sdist_path": Path, "sdist_size": int}
#   "upload_started" -> {"sdist_path": Path, "sdist_size": int}
#   "upload_finished"-> {"url": str, "cached": bool, "sdist_size": int | None}
#
# Only the upload_finished event fires on a cache hit; in that case
# ``cached`` is True and ``sdist_size`` is None (no upload happened).
# Callers are free to ignore unknown events. Keeping the contract loose so
# we can add phases later without breaking integrations.
ProgressCallback = Callable[[str, dict], None]

# Match ``.`` or ``.[extras]`` exactly. Other path-like forms (``./subdir``,
# ``-e .``, ``file://...``) are intentionally left alone.
_LOCAL_PATH_RE = re.compile(r"^\.(\[[^\]]+\])?$")

# In-process cache to avoid re-building/re-uploading the sdist on every
# dispatch within a single ``fal run``/``fal deploy`` invocation. Keyed by
# resolved project root path. The user's source tree isn't expected to change
# mid-invocation; a fresh process re-runs the build either way, so this cache
# tradeoff matches user expectation ("I ran fal once, it packaged once").
#
# Note: hashing the sdist content would be more conservative but doesn't help
# in practice — ``python -m build`` is non-deterministic (file mtimes inside
# the tarball drift between back-to-back builds), so a content-hash cache
# misses on every call.
_SDIST_URL_CACHE: dict[str, tuple[str, str]] = {}
# Global lock serializing the build+upload sequence across the whole
# process. A finer per-root lock would let independent projects build
# in parallel, but the realistic concurrent shape today is at most one
# project per ``fal run``/``fal deploy`` invocation, so a single lock
# keeps the implementation simple. The trade-off: two threads packaging
# *different* roots in the same process will serialize.
_SDIST_URL_CACHE_LOCK = threading.Lock()

Requirements = Union[List[str], List[List[str]]]


def has_local_path(requirements: Any) -> bool:
    """True if ``requirements`` contains any ``.``/``.[extras]`` entry."""
    if not isinstance(requirements, list):
        return False
    for item in requirements:
        if isinstance(item, list):
            if has_local_path(item):
                return True
        elif isinstance(item, str) and _LOCAL_PATH_RE.match(item.strip()):
            return True
    return False


def materialize_local_paths(
    requirements: Requirements,
    project_root: Union[str, Path],
    on_progress: Optional[ProgressCallback] = None,
) -> Requirements:
    """Rewrite ``.``/``.[extras]`` entries to ``<package> @ <url>``.

    Builds an sdist of ``project_root`` once per project root per process and
    uploads it to the fal CDN via ``fal.toolkit.File.from_path``; subsequent
    calls within the same process reuse the cached URL. No-op when no
    local-path entry is present.

    Preserves the input shape (flat list vs layered list-of-lists).

    ``on_progress`` is invoked at each phase boundary; see
    :data:`ProgressCallback` for the event contract. Defaults to a no-op so
    library callers don't have to care about presentation.
    """
    if not has_local_path(requirements):
        return requirements
    project_root_path = Path(project_root)
    package_name, url = _resolve_sdist_url(project_root_path, on_progress)
    return _walk_and_rewrite(requirements, package_name, url)


def _walk_and_rewrite(
    requirements: Requirements, package_name: str, url: str
) -> Requirements:
    out: list = []
    for item in requirements:
        if isinstance(item, list):
            out.append(_walk_and_rewrite(item, package_name, url))
        elif isinstance(item, str):
            out.append(_rewrite_one(item, package_name, url))
        else:
            out.append(item)
    return out  # type: ignore[return-value]


def _rewrite_one(req: str, package_name: str, url: str) -> str:
    match = _LOCAL_PATH_RE.match(req.strip())
    if match is None:
        return req
    extras = match.group(1) or ""
    return f"{package_name}{extras} @ {url}"


def _resolve_sdist_url(
    project_root: Path,
    on_progress: Optional[ProgressCallback] = None,
) -> Tuple[str, str]:
    def _emit(event: str, **payload: Any) -> None:
        if on_progress is not None:
            on_progress(event, payload)

    cache_key = str(project_root.resolve())

    # Hold the lock for the entire check-build-upload-populate sequence so
    # a concurrent dispatch waits and re-uses the populated entry instead
    # of running its own build + upload in parallel. See the comment on
    # ``_SDIST_URL_CACHE_LOCK`` for the cross-root serialization caveat.
    with _SDIST_URL_CACHE_LOCK:
        cached = _SDIST_URL_CACHE.get(cache_key)
        if cached is not None:
            package_name, url = cached
            _emit("upload_finished", url=url, cached=True, sdist_size=None)
            return package_name, url

        package_name = _read_package_name(project_root)
        _emit("build_started", package_name=package_name, project_root=project_root)
        sdist = _build_sdist(project_root)
        try:
            sdist_size = sdist.stat().st_size
            _emit("build_finished", sdist_path=sdist, sdist_size=sdist_size)

            _emit("upload_started", sdist_path=sdist, sdist_size=sdist_size)
            url = _upload_sdist(sdist)
            _SDIST_URL_CACHE[cache_key] = (package_name, url)
            _emit("upload_finished", url=url, sdist_size=sdist_size, cached=False)
        finally:
            shutil.rmtree(sdist.parent, ignore_errors=True)
        return package_name, url


def _read_package_name(project_root: Path) -> str:
    pyproject = project_root / "pyproject.toml"
    if not pyproject.is_file():
        raise RuntimeError(
            f"Cannot resolve local-path requirement: "
            f"no pyproject.toml at {pyproject}"
        )
    data = _load_toml(pyproject)
    name = data.get("project", {}).get("name")
    if not isinstance(name, str) or not name:
        raise RuntimeError(
            f"Cannot resolve local-path requirement: "
            f"[project].name is missing or empty in {pyproject}"
        )
    return name


def _build_sdist(project_root: Path) -> Path:
    """Run ``python -m build --sdist`` for ``project_root`` and return the
    resulting tarball path.

    Output streams live to stdout/stderr so the user sees what's happening
    between the section rules drawn by the caller. We don't capture it —
    the failure message tells them to look at the live output above.

    The temp ``outdir`` is owned by the caller on success (it cleans up
    after consuming the sdist). On any failure — expected (build error,
    missing artefact) or unexpected (``KeyboardInterrupt``, OS errors) —
    we delete it here before re-raising so it never leaks.
    """
    outdir = Path(tempfile.mkdtemp(prefix="fal-sdist-"))
    try:
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--sdist",
                    "--outdir",
                    str(outdir),
                    str(project_root),
                ],
                check=False,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Could not invoke `{sys.executable} -m build`: {e}"
            ) from e

        if result.returncode != 0:
            raise RuntimeError(
                f"sdist build for {project_root} failed (exit "
                f"{result.returncode}). See output above."
            )

        # The outdir is a fresh ``mkdtemp`` and ``python -m build --sdist``
        # writes exactly one artefact into it. Refuse to guess if that
        # invariant is ever violated.
        candidates = list(outdir.glob("*.tar.gz"))
        if len(candidates) != 1:
            raise RuntimeError(
                f"sdist build for {project_root} produced "
                f"{len(candidates)} .tar.gz artefact(s) in {outdir}, "
                "expected exactly one."
            )
    except BaseException:
        shutil.rmtree(outdir, ignore_errors=True)
        raise
    return candidates[0]


def _upload_sdist(sdist_path: Path) -> str:
    from fal.toolkit import File  # noqa: PLC0415

    file = File.from_path(sdist_path)
    return file.url
