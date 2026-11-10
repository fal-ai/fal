"""Persist the CUDA driver's PTX JIT cache on the persistent volume.

Blackwell (sm_120) wheels often ship PTX only, so the driver compiles kernels
on every boot and loses them with the container. Keeping the cache on /data
turns that into a one-time cost per GPU model and driver version.

Kept free of fal imports so it is cheap to load before any CUDA init.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path, PurePath
from typing import MutableMapping

# No fal.toolkit imports: this loads on the runner before user code, and
# importing anything under fal.toolkit pulls the whole toolkit's dependencies.
FAL_PERSISTENT_DIR = PurePath("/data")

# The CUDA driver JIT cache defaults to 256 MB; vLLM's kernels overflow it.
CUDA_CACHE_MAXSIZE = 4 << 30
_DRIVER_VERSION_FILE = "/proc/driver/nvidia/version"
_GPU_INFO_GLOB = "/proc/driver/nvidia/gpus/*/information"
_DRIVER_RE = re.compile(r"Kernel Module\s+(?:for \S+\s+)?([\d.]+)")
_MODEL_RE = re.compile(r"^Model:\s*(.+)$", re.MULTILINE)
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")

CUDA_CACHE_ROOT = Path(FAL_PERSISTENT_DIR) / ".fal" / "cuda_cache"


def _slug(value: str) -> str:
    return _SAFE.sub("-", value).strip("-") or "default"


def nvidia_driver_version() -> str:
    """The loaded NVIDIA kernel driver version, or "" without a GPU."""
    try:
        with open(_DRIVER_VERSION_FILE) as f:
            m = _DRIVER_RE.search(f.read())
    except OSError:
        return ""
    return m.group(1) if m else ""


def nvidia_gpu_model() -> str:
    """The first GPU's model name (the arch key), or "" when unreadable."""
    for path in sorted(glob.glob(_GPU_INFO_GLOB)):
        try:
            with open(path) as f:
                m = _MODEL_RE.search(f.read())
        except OSError:
            continue
        if m:
            return m.group(1).strip()
    return ""


def cuda_cache_dir(
    *,
    root: str | os.PathLike[str] | None = None,
    driver: str | None = None,
    model: str | None = None,
) -> Path | None:
    """Cache dir for this GPU model and driver, or None on a CPU host."""
    driver = nvidia_driver_version() if driver is None else driver
    if not driver:
        return None
    model = nvidia_gpu_model() if model is None else model
    base = Path(root) if root is not None else CUDA_CACHE_ROOT
    return base / _slug(model or "gpu") / _slug(driver)


def apply_cuda_cache_env(
    *,
    root: str | os.PathLike[str] | None = None,
    driver: str | None = None,
    model: str | None = None,
    env: MutableMapping[str, str] | None = None,
) -> str | None:
    """Point the CUDA JIT cache at the persistent volume; return the path.

    Honors a value the user already set, and does nothing on CPU hosts or when
    the persistent volume is not mounted. Must run before CUDA initializes,
    since the driver reads these variables once.
    """
    target = os.environ if env is None else env
    path = cuda_cache_dir(root=root, driver=driver, model=model)
    if path is None:
        return None
    if root is None and not os.path.isdir(FAL_PERSISTENT_DIR):
        return None
    if target.get("CUDA_CACHE_DISABLE") == "1":
        return None
    if "CUDA_CACHE_PATH" in target:
        # The user owns the location; only fill in the size if they did not.
        target.setdefault("CUDA_CACHE_MAXSIZE", str(CUDA_CACHE_MAXSIZE))
        return target["CUDA_CACHE_PATH"]
    chosen = str(path)
    try:
        os.makedirs(chosen, exist_ok=True)
        if not os.access(chosen, os.W_OK | os.X_OK):
            raise PermissionError(f"{chosen} is not writable")
    except OSError as exc:
        # Leave the env alone: the driver keeps its default cache location
        # rather than pointing at a directory it cannot write.
        print(f"Warning: CUDA JIT cache dir {chosen} is unusable: {exc!r}")
        return None
    target["CUDA_CACHE_PATH"] = chosen
    target.setdefault("CUDA_CACHE_MAXSIZE", str(CUDA_CACHE_MAXSIZE))
    return chosen
