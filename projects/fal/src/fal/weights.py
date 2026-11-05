from __future__ import annotations

import posixpath
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from fal.api import Host

# Same HF_HOME that LLMApp sets on runners (fal.toolkit.llm._cache.HF_HOME_DIR),
# so a later boot finds these files in its hub cache instead of downloading
# them. Kept as a literal: importing it loads all of LLMApp, and the CLI imports
# this module while building its parser. A unit test pins the two together.
DEFAULT_HF_HOME = "/data/.fal/hf"
DEFAULT_HF_CACHE_DIR = DEFAULT_HF_HOME + "/hub"
DEFAULT_WORKERS = 8
DEFAULT_MACHINE_TYPE = "M"
# Generous so a multi-hundred-GB repo finishes; a cut-off pull resumes on rerun.
REQUEST_TIMEOUT_S = 6 * 3600

_REQUIREMENTS = ["huggingface_hub[hf_xet]>=1.0"]


def validate_dest(dest: Optional[str]) -> Optional[str]:
    """Normalize ``dest``; it must be an absolute path inside ``/data``."""
    if dest is None:
        return None
    normalized = posixpath.normpath(dest)
    if not normalized.startswith("/data/"):
        raise ValueError(
            f"--dest must be a directory inside /data (not /data itself), "
            f"got {dest!r}. "
            "The CPU job can only persist files under /data."
        )
    return normalized


def _advance_marker(repo_id: str, path: str, advance: Any) -> Optional[bool]:
    """Move LLMApp's unpinned marker to ``path`` if it is the default branch head.

    Returns whether the marker moved, or None when it was not attempted.
    """
    import os  # noqa: PLC0415

    from huggingface_hub import HfApi  # noqa: PLC0415

    commit = os.path.basename(path)
    info = HfApi().model_info(repo_id)
    # Only the default branch's head may become what unpinned apps serve.
    if info.sha != commit:
        return None
    date = info.last_modified.isoformat() if info.last_modified else None
    return advance(repo_id, path, commit, date, hf_home=DEFAULT_HF_HOME)


def _download(
    repo_id: str,
    revision: Optional[str],
    dest: Optional[str],
    workers: int,
    advance_marker: Any = None,
) -> dict[str, Any]:
    import os  # noqa: PLC0415
    import time  # noqa: PLC0415

    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    kwargs: dict[str, Any] = {"revision": revision, "max_workers": workers}
    if dest is None:
        kwargs["cache_dir"] = DEFAULT_HF_CACHE_DIR
    else:
        kwargs["local_dir"] = dest

    start = time.monotonic()
    try:
        path = snapshot_download(repo_id, **kwargs)
    except Exception as exc:
        if type(exc).__name__ in ("GatedRepoError", "RepositoryNotFoundError"):
            raise RuntimeError(
                f"Cannot download {repo_id!r}: it is private, gated or does not "
                "exist. Only public repositories are supported."
            ) from exc
        raise
    elapsed = time.monotonic() - start

    # Snapshot entries are symlinks into shared blobs; count each file once.
    total = 0
    seen = set()
    for root, _, files in os.walk(path):
        for name in files:
            try:
                st = os.stat(os.path.join(root, name))
            except FileNotFoundError:
                continue
            if (st.st_dev, st.st_ino) not in seen:
                seen.add((st.st_dev, st.st_ino))
                total += st.st_size
    result: dict[str, Any] = {"path": path, "bytes": total, "seconds": elapsed}
    if dest is None and advance_marker is not None:
        # The weights are on /data either way; a marker failure only means
        # unpinned apps keep their current snapshot.
        try:
            result["marker_advanced"] = _advance_marker(repo_id, path, advance_marker)
        except Exception as exc:
            result["marker_error"] = repr(exc)
    return result


def pull_weights(
    repo_id: str,
    *,
    revision: Optional[str] = None,
    dest: Optional[str] = None,
    workers: int = DEFAULT_WORKERS,
    machine_type: str = DEFAULT_MACHINE_TYPE,
    host: Optional[Host] = None,
) -> dict[str, Any]:
    """Download public Hugging Face weights into ``/data`` on a CPU machine.

    Runs an ephemeral fal function so the next GPU runner finds the weights on
    ``/data`` instead of downloading them at boot. Without ``dest`` the files
    land in the Hugging Face hub cache LLMApp uses (``HF_HOME=/data/.fal/hf``
    makes ``from_pretrained(repo_id)`` hit it); with ``dest`` they are written
    as a plain directory. ``host`` selects the target (for example a team);
    the default is the logged-in account. When LLMApp is available, a pull of
    the default branch's head also advances LLMApp's unpinned snapshot marker,
    so new runners of apps without ``model_revision`` serve the pulled commit.
    """
    import fal  # noqa: PLC0415

    try:
        # Resolved here and shipped by value: the CPU job has no fal install.
        from fal.toolkit.llm._cache import (  # noqa: PLC0415
            advance_snapshot_marker,
        )
    except ImportError:
        advance_snapshot_marker = None

    if workers < 1:
        raise ValueError("workers must be at least 1")
    dest = validate_dest(dest)
    remote = fal.function(
        requirements=_REQUIREMENTS,
        machine_type=machine_type,
        keep_alive=0,
        request_timeout=REQUEST_TIMEOUT_S,
        host=host,
    )(_download)
    return remote(repo_id, revision, dest, workers, advance_snapshot_marker)
