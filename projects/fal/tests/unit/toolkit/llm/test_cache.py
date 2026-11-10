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
        "TORCH_EXTENSIONS_DIR",
        "CUTE_DSL_CACHE_DIR",
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


def _snap(tmp_path, weights=(), index=None):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    for name in weights:
        (snap / name).write_bytes(b"x")
    if index is not None:
        import json

        (snap / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {f"w{i}": s for i, s in enumerate(index)}})
        )
    return str(snap)


def test_snapshot_without_weights_is_not_intact(tmp_path):
    from fal.toolkit.llm._cache import snapshot_intact

    assert not snapshot_intact(_snap(tmp_path))


def test_snapshot_with_weights_is_intact(tmp_path):
    from fal.toolkit.llm._cache import snapshot_intact

    assert snapshot_intact(_snap(tmp_path, ["model.safetensors"]))


@pytest.mark.parametrize(
    "name",
    ["model.bin", "model.pt", "model.pth", "model-Q4_K_M.gguf", "model.ckpt"],
)
def test_snapshot_with_other_weight_formats_is_intact(tmp_path, name):
    from fal.toolkit.llm._cache import snapshot_intact

    assert snapshot_intact(_snap(tmp_path, [name]))


def test_snapshot_with_only_training_args_is_not_intact(tmp_path):
    from fal.toolkit.llm._cache import snapshot_intact

    assert not snapshot_intact(_snap(tmp_path, ["training_args.bin"]))


def test_snapshot_missing_indexed_shard_is_not_intact(tmp_path):
    from fal.toolkit.llm._cache import snapshot_intact

    path = _snap(
        tmp_path, ["m-1.safetensors"], index=["m-1.safetensors", "m-2.safetensors"]
    )
    assert not snapshot_intact(path)


def test_snapshot_all_indexed_shards_present_is_intact(tmp_path):
    from fal.toolkit.llm._cache import snapshot_intact

    shards = ["m-1.safetensors", "m-2.safetensors"]
    assert snapshot_intact(_snap(tmp_path, shards, index=shards))
