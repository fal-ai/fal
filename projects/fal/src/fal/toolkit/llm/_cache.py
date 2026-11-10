from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from fal._cuda_cache import CUDA_CACHE_MAXSIZE, nvidia_driver_version
from fal.toolkit.utils.download_utils import FAL_PERSISTENT_DIR

ENGINE_CACHE_ROOT = Path(FAL_PERSISTENT_DIR) / ".fal" / "engine_cache"
HF_HOME_DIR = Path(FAL_PERSISTENT_DIR) / ".fal" / "hf"

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
# engine_cache_env values that are settings, not directories.
NON_PATH_KEYS = frozenset({"CUDA_CACHE_MAXSIZE"})


def _slug(value: str) -> str:
    return _SAFE.sub("-", value).strip("-") or "default"


def namespace_dir(
    namespace: str = "default", *, root: str | os.PathLike[str] | None = None
) -> Path:
    """Directory holding one namespace's compile caches and boot markers."""
    return (Path(root) if root is not None else ENGINE_CACHE_ROOT) / _slug(namespace)


def snapshot_marker_path(
    model: str,
    revision: str | None = None,
    *,
    hf_home: str | os.PathLike[str] | None = None,
) -> Path:
    """Where LLMApp records a completed ``snapshot_download`` of ``model``.

    Kept beside, not inside, HF's snapshot folders: HF's own cache layout
    cannot tell a finished download from an interrupted one. It sits next to
    the effective hub cache (``HF_HUB_CACHE`` if set) so marker and files
    share a volume and a lifetime.
    """
    if hf_home is not None:
        hub = Path(hf_home) / "hub"
    elif os.environ.get("HF_HUB_CACHE"):
        hub = Path(os.environ["HF_HUB_CACHE"])
    else:
        hub = Path(os.environ.get("HF_HOME") or HF_HOME_DIR) / "hub"
    name = _slug(model) + (f"@{_slug(revision)}" if revision else "")
    # Slugs collapse "/" and "-", so a hash keeps distinct ids apart.
    digest = hashlib.sha1(f"{model}@{revision or ''}".encode()).hexdigest()[:8]
    return hub.parent / "llm_app_snapshots" / f"{name}-{digest}.json"


def read_marker(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_marker(path: Path, data: Mapping[str, Any]) -> None:
    """Atomic write: readers never see a partial marker.

    The temp name is unique per call, since containers sharing a volume can
    have the same pid.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(data))
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# Weight file formats a model repo may ship; training_args.bin is a pickled
# config, not weights.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".ckpt")
_NOT_WEIGHTS = frozenset({"training_args.bin"})


def snapshot_intact(path: str) -> bool:
    """Config and weights resolve (HF snapshots are symlinks).

    At least one weight file must survive, and every shard named by a
    ``*.index.json`` weight map must resolve: a snapshot whose entries
    vanished after the completion marker was written is not intact.
    """
    if not os.path.isfile(os.path.join(path, "config.json")):
        return False
    weights = 0
    indexes = []
    for root, _, files in os.walk(path):
        for name in files:
            full = os.path.join(root, name)
            if name.endswith(_WEIGHT_SUFFIXES) and name not in _NOT_WEIGHTS:
                if not os.path.exists(full):
                    return False
                weights += 1
            elif name.endswith(".index.json"):
                indexes.append((root, full))
    if not weights:
        return False
    for root, index in indexes:
        try:
            with open(index) as f:
                shards = set(json.load(f)["weight_map"].values())
        except (OSError, ValueError, KeyError, AttributeError, TypeError):
            return False
        if not all(
            isinstance(s, str) and os.path.exists(os.path.join(root, s)) for s in shards
        ):
            return False
    return True


def _parse_date(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def advance_snapshot_marker(
    model: str,
    path: str,
    commit: str,
    commit_date: str | None,
    *,
    hf_home: str | os.PathLike[str] | None = None,
) -> bool:
    """Point the unpinned marker of ``model`` at ``path`` if it is newer.

    Snapshot markers (``snapshot_marker_path``) are JSON objects::

        {"model": <repo id>, "revision": <requested revision or null>,
         "path": <absolute per-commit snapshot dir>,
         "commit": <40-hex sha, the snapshot dir name>,
         "commit_date": <ISO-8601 UTC commit time, or null if unknown>}

    The unpinned marker is ``snapshot_marker_path(model, None)``: what an app
    without ``model_revision`` boots from. It is replaced when missing,
    unreadable or without a commit, or when ``commit_date`` is known and the
    recorded one is unknown or older; the same commit is a no-op. Newest means
    commit date, never file mtime. Only a default-branch commit belongs here.
    Returns whether the marker was written.
    """
    marker = snapshot_marker_path(model, None, hf_home=hf_home)
    marker.parent.mkdir(parents=True, exist_ok=True)
    # Runners and the weights-pull CLI share the volume: serialize the
    # read-compare-write so an older commit cannot land last.
    with open(f"{marker}.lock", "a") as lock:
        _lock(lock)
        return _advance_locked(marker, model, path, commit, commit_date)


def _lock(f: Any) -> None:
    try:
        import fcntl  # noqa: PLC0415  (POSIX only; best effort elsewhere)
    except ImportError:
        return
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)


def _advance_locked(
    marker: Path, model: str, path: str, commit: str, commit_date: str | None
) -> bool:
    current = read_marker(marker)
    if current is not None and current.get("commit") == commit:
        return False
    current_path = (current or {}).get("path")
    usable = isinstance(current_path, str) and snapshot_intact(current_path)
    # A marker whose snapshot was pruned is replaced whatever the dates say.
    if usable and isinstance(current.get("commit"), str):  # type: ignore[union-attr]
        new = _parse_date(commit_date)
        old = _parse_date(current.get("commit_date"))
        if new is None or (old is not None and new <= old):
            return False
    write_marker(
        marker,
        {
            "model": model,
            "revision": None,
            "path": path,
            "commit": commit,
            "commit_date": commit_date,
        },
    )
    return True


def engine_cache_env(
    namespace: str = "default",
    *,
    root: str | os.PathLike[str] | None = None,
    hf_home: str | os.PathLike[str] | None = None,
    driver: str | None = None,
) -> dict[str, str]:
    """Env vars that point engine compile caches at the persistent volume.

    Compiled artifacts (torch Inductor and C++ extensions, Triton, DeepGEMM,
    FlashInfer, TileLang, CuTe DSL, vLLM, SGLang) land under
    ``<root>/<namespace>/``, so a new runner reuses them instead of recompiling.
    ``namespace`` should change whenever cached artifacts could be incompatible
    (engine version, GPU type). HF caches are shared across namespaces, since
    downloaded weights are not engine specific.

    The CUDA driver's PTX JIT cache is also kept, keyed by driver version
    (detected when ``driver`` is None), since a driver upgrade invalidates it.
    ``fal.App`` already sets ``CUDA_CACHE_PATH`` before ``setup()`` (see
    ``persist_cuda_cache``), and ``apply_engine_cache_env`` never overrides a
    set variable, so inside an App that default wins over this namespaced path.

    Args:
        namespace: Subdirectory name for the cache set. Change it when cached
            artifacts could be incompatible.
        root: Cache root. Defaults to the persistent ``/data`` engine cache
            directory.
        hf_home: Hugging Face cache directory shared across namespaces.
        driver: CUDA driver version used to key the PTX JIT cache. Detected
            when ``None``.

    Returns:
        Mapping from environment variable name to value.
    """
    base = Path(root) if root is not None else ENGINE_CACHE_ROOT
    base = base / _slug(namespace)
    hf = Path(hf_home) if hf_home is not None else HF_HOME_DIR
    driver = nvidia_driver_version() if driver is None else driver
    env = {
        "TORCHINDUCTOR_CACHE_DIR": str(base / "inductor"),
        "TRITON_CACHE_DIR": str(base / "triton"),
        "DG_JIT_CACHE_DIR": str(base / "deep_gemm"),
        "SGLANG_DG_CACHE_DIR": str(base / "deep_gemm"),
        "FLASHINFER_WORKSPACE_BASE": str(base / "flashinfer"),
        "TILELANG_CACHE_DIR": str(base / "tilelang"),
        "TORCH_EXTENSIONS_DIR": str(base / "torch_extensions"),
        "CUTE_DSL_CACHE_DIR": str(base / "cute_dsl"),
        "VLLM_CACHE_ROOT": str(base / "vllm"),
        "SGLANG_CACHE_DIR": str(base / "sglang"),
        "HF_HOME": str(hf),
        "HF_HUB_CACHE": str(hf / "hub"),
    }
    if driver:
        env["CUDA_CACHE_PATH"] = str(base / "cuda" / _slug(driver))
        env["CUDA_CACHE_MAXSIZE"] = str(CUDA_CACHE_MAXSIZE)
    return env


def apply_engine_cache_env(
    namespace: str = "default",
    *,
    root: str | os.PathLike[str] | None = None,
    hf_home: str | os.PathLike[str] | None = None,
    driver: str | None = None,
    env: MutableMapping[str, str] | None = None,
) -> Mapping[str, str]:
    """Create the cache dirs and set the env vars that are not already set.

    Values the user set explicitly win.

    Args:
        namespace: Subdirectory name for the cache set.
        root: Cache root. Defaults to the persistent ``/data`` engine cache
            directory.
        hf_home: Hugging Face cache directory shared across namespaces.
        driver: CUDA driver version used to key the PTX JIT cache.
        env: Mapping to update. Defaults to ``os.environ``.

    Returns:
        The effective environment values, including ones the user had set.
    """
    target = os.environ if env is None else env
    effective: dict[str, str] = {}
    values = engine_cache_env(namespace, root=root, hf_home=hf_home, driver=driver)
    for key, value in values.items():
        effective[key] = target.setdefault(key, value)
        if key not in NON_PATH_KEYS:
            os.makedirs(effective[key], exist_ok=True)
    return effective
