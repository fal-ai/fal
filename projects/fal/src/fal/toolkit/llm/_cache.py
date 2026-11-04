from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Mapping, MutableMapping

from fal.toolkit.utils.download_utils import FAL_PERSISTENT_DIR

ENGINE_CACHE_ROOT = Path(FAL_PERSISTENT_DIR) / ".fal" / "engine_cache"
HF_HOME_DIR = Path(FAL_PERSISTENT_DIR) / ".fal" / "hf"

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(value: str) -> str:
    return _SAFE.sub("-", value).strip("-") or "default"


def engine_cache_env(
    namespace: str = "default",
    *,
    root: str | os.PathLike[str] | None = None,
    hf_home: str | os.PathLike[str] | None = None,
) -> dict[str, str]:
    """Env vars that point engine compile caches at the persistent volume.

    Compiled artifacts (torch Inductor, Triton, DeepGEMM, FlashInfer, TileLang,
    vLLM, SGLang) land under ``<root>/<namespace>/``, so a new runner reuses
    them instead of recompiling. ``namespace`` should change whenever cached
    artifacts could be incompatible (engine version, GPU type). HF caches are
    shared across namespaces, since downloaded weights are not engine specific.
    """
    base = Path(root) if root is not None else ENGINE_CACHE_ROOT
    base = base / _slug(namespace)
    hf = Path(hf_home) if hf_home is not None else HF_HOME_DIR
    return {
        "TORCHINDUCTOR_CACHE_DIR": str(base / "inductor"),
        "TRITON_CACHE_DIR": str(base / "triton"),
        "DG_JIT_CACHE_DIR": str(base / "deep_gemm"),
        "SGLANG_DG_CACHE_DIR": str(base / "deep_gemm"),
        "FLASHINFER_WORKSPACE_BASE": str(base / "flashinfer"),
        "TILELANG_CACHE_DIR": str(base / "tilelang"),
        "VLLM_CACHE_ROOT": str(base / "vllm"),
        "SGLANG_CACHE_DIR": str(base / "sglang"),
        "HF_HOME": str(hf),
        "HF_HUB_CACHE": str(hf / "hub"),
    }


def apply_engine_cache_env(
    namespace: str = "default",
    *,
    root: str | os.PathLike[str] | None = None,
    hf_home: str | os.PathLike[str] | None = None,
    env: MutableMapping[str, str] | None = None,
) -> Mapping[str, str]:
    """Create the cache dirs and set the env vars that are not already set.

    Values the user set explicitly win. Returns the effective values.
    """
    target = os.environ if env is None else env
    effective: dict[str, str] = {}
    for key, value in engine_cache_env(namespace, root=root, hf_home=hf_home).items():
        effective[key] = target.setdefault(key, value)
        os.makedirs(effective[key], exist_ok=True)
    return effective
