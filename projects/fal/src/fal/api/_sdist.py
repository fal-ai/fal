"""Build, upload, and rewrite local-path requirements for fal apps.

When a ``[tool.fal.apps.*]`` block in ``pyproject.toml`` lists ``.``,
``.[extras]``, or another local project path in its ``requirements``, the
user means "install this project on the worker." We can't ship the source
tree as a pip requirement, so we build an sdist locally, upload it to the fal
CDN via ``fal.toolkit.File``, and rewrite the requirement to
``<package> @ <url>`` (or ``<package>[extras] @ <url>``) so the worker can
pip-install it.

This module is the pure helper. ``FalServerlessHost`` calls
``materialize_local_paths`` on the way to dispatch.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
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
#   "upload_finished"-> {"url": str, "sdist_size": int}
#
# Callers are free to ignore unknown events. Keeping the contract loose so
# we can add phases later without breaking integrations.
ProgressCallback = Callable[[str, dict], None]

_EXTRAS_SUFFIX_RE = re.compile(r"(?P<extras>\[[^\]]+\])$")

EXPIRATION_DURATION_SECONDS = 60 * 60

Requirements = Union[List[str], List[List[str]]]


@dataclass(frozen=True)
class _LocalPathRequirement:
    path: Path
    extras: str = ""


def has_local_path(requirements: Any, project_root: Union[str, Path]) -> bool:
    """True if ``requirements`` contains a local path entry.

    Requirement strings are resolved relative to ``project_root``; if the path
    part points at an existing directory, it is treated as local.
    """
    if not isinstance(requirements, list):
        return False
    base_path = Path(project_root)
    for item in requirements:
        if isinstance(item, list):
            if has_local_path(item, base_path):
                return True
        elif isinstance(item, str) and _parse_local_path_requirement(item, base_path):
            return True
    return False


def materialize_local_paths(
    requirements: Requirements,
    project_root: Union[str, Path],
    on_progress: Optional[ProgressCallback] = None,
) -> Requirements:
    """Rewrite local path entries to ``<package> @ <url>``.

    Builds an sdist for each referenced local project and uploads it to the fal
    CDN via ``fal.toolkit.File.from_path``. No-op when no local-path entry is
    present.

    Preserves the input shape (flat list vs layered list-of-lists).

    ``on_progress`` is invoked at each phase boundary; see
    :data:`ProgressCallback` for the event contract. Defaults to a no-op so
    library callers don't have to care about presentation.
    """
    project_root_path = Path(project_root)
    resolved_sdists: dict[Path, tuple[str, str]] = {}
    return _walk_and_rewrite(
        requirements, project_root_path, resolved_sdists, on_progress
    )


def _walk_and_rewrite(
    requirements: Requirements,
    project_root: Path,
    resolved_sdists: dict[Path, tuple[str, str]],
    on_progress: Optional[ProgressCallback],
) -> Requirements:
    out: list = []
    for item in requirements:
        if isinstance(item, list):
            out.append(
                _walk_and_rewrite(item, project_root, resolved_sdists, on_progress)
            )
        elif isinstance(item, str):
            out.append(_rewrite_one(item, project_root, resolved_sdists, on_progress))
        else:
            out.append(item)
    return out  # type: ignore[return-value]


def _rewrite_one(
    req: str,
    project_root: Path,
    resolved_sdists: dict[Path, tuple[str, str]],
    on_progress: Optional[ProgressCallback],
) -> str:
    local_path_req = _parse_local_path_requirement(req, project_root)
    if local_path_req is None:
        return req

    local_project_root = local_path_req.path
    package_name_url = resolved_sdists.get(local_project_root)
    if package_name_url is None:
        package_name_url = _resolve_sdist_url(local_project_root, on_progress)
        resolved_sdists[local_project_root] = package_name_url

    package_name, url = package_name_url
    return f"{package_name}{local_path_req.extras} @ {url}"


def _parse_local_path_requirement(
    req: str, project_root: Path
) -> Optional[_LocalPathRequirement]:
    req = req.strip()
    if not req:
        return None

    path_part = req
    extras = ""
    extras_match = _EXTRAS_SUFFIX_RE.search(req)
    if extras_match is not None:
        candidate_path = req[: extras_match.start()].strip()
        if not candidate_path:
            return None
        path_part = candidate_path
        extras = extras_match.group("extras")

    path = _resolve_path(path_part, project_root)
    if path.is_dir():
        return _LocalPathRequirement(path, extras)
    return None


def _resolve_path(path: str, project_root: Path) -> Path:
    result = Path(path).expanduser()
    if not result.is_absolute():
        result = project_root / result
    return result.resolve()


def _resolve_sdist_url(
    project_root: Path,
    on_progress: Optional[ProgressCallback] = None,
) -> Tuple[str, str]:
    def _emit(event: str, **payload: Any) -> None:
        if on_progress is not None:
            on_progress(event, payload)

    package_name = _read_package_name(project_root)
    _emit("build_started", package_name=package_name, project_root=project_root)
    sdist = _build_sdist(project_root)
    try:
        sdist_size = sdist.stat().st_size
        _emit("build_finished", sdist_path=sdist, sdist_size=sdist_size)

        _emit("upload_started", sdist_path=sdist, sdist_size=sdist_size)
        url = _upload_sdist(sdist)
        _emit("upload_finished", url=url, sdist_size=sdist_size)
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

    save_kwargs = {
        "object_lifecycle_preference": {
            "expiration_duration_seconds": EXPIRATION_DURATION_SECONDS
        }
    }
    file = File.from_path(
        sdist_path,
        save_kwargs=save_kwargs,
        fallback_save_kwargs=save_kwargs,
    )
    return file.url
