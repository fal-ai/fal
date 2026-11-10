from __future__ import annotations

import pytest

from fal._cuda_cache import nvidia_driver_version
from fal.toolkit.llm import apply_engine_cache_env, engine_cache_env


def test_engine_cache_env_namespaced(tmp_path):
    env = engine_cache_env(
        "vllm RTX/6000", root=tmp_path, hf_home=tmp_path / "hf", driver=""
    )
    assert env["TORCHINDUCTOR_CACHE_DIR"] == str(
        tmp_path / "vllm-RTX-6000" / "inductor"
    )
    assert env["HF_HUB_CACHE"] == str(tmp_path / "hf" / "hub")
    for key in (
        "TRITON_CACHE_DIR",
        "DG_JIT_CACHE_DIR",
        "FLASHINFER_WORKSPACE_BASE",
        "TILELANG_CACHE_DIR",
        "VLLM_CACHE_ROOT",
        "SGLANG_CACHE_DIR",
    ):
        assert key in env
    # No driver (CPU host): the CUDA JIT cache keeps its default location.
    assert "CUDA_CACHE_PATH" not in env


def test_cuda_jit_cache_keyed_by_driver(tmp_path):
    env = engine_cache_env("ns", root=tmp_path, driver="580.95.05")
    assert env["CUDA_CACHE_PATH"] == str(tmp_path / "ns" / "cuda" / "580.95.05")
    assert int(env["CUDA_CACHE_MAXSIZE"]) > 256 << 20
    target: dict[str, str] = {}
    apply_engine_cache_env(
        "ns", root=tmp_path, hf_home=tmp_path / "hf", driver="580.95.05", env=target
    )
    assert (tmp_path / "ns" / "cuda" / "580.95.05").is_dir()
    assert not any(p.name == target["CUDA_CACHE_MAXSIZE"] for p in tmp_path.rglob("*"))


def test_apply_respects_existing_and_creates_dirs(tmp_path):
    target = {"TRITON_CACHE_DIR": str(tmp_path / "mine")}
    effective = apply_engine_cache_env(
        "ns", root=tmp_path / "root", hf_home=tmp_path / "hf", driver="", env=target
    )
    assert effective["TRITON_CACHE_DIR"] == str(tmp_path / "mine")
    assert (tmp_path / "mine").is_dir()
    assert (tmp_path / "root" / "ns" / "inductor").is_dir()
    assert target["VLLM_CACHE_ROOT"] == str(tmp_path / "root" / "ns" / "vllm")


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  610.57.04  "
            "Release Build",
            "610.57.04",
        ),
        (
            "NVRM version: NVIDIA UNIX x86_64 Kernel Module  550.54.15  Tue Mar",
            "550.54.15",
        ),
        ("garbage", ""),
        (None, ""),
    ],
)
def test_nvidia_driver_version(tmp_path, monkeypatch, text, expected):
    path = tmp_path / "version"
    if text is not None:
        path.write_text(text)
    monkeypatch.setattr("fal._cuda_cache._DRIVER_VERSION_FILE", str(path))
    assert nvidia_driver_version() == expected
