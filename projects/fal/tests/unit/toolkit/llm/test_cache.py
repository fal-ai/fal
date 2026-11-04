from __future__ import annotations

from fal.toolkit.llm import apply_engine_cache_env, engine_cache_env


def test_engine_cache_env_namespaced(tmp_path):
    env = engine_cache_env("vllm RTX/6000", root=tmp_path, hf_home=tmp_path / "hf")
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


def test_apply_respects_existing_and_creates_dirs(tmp_path):
    target = {"TRITON_CACHE_DIR": str(tmp_path / "mine")}
    effective = apply_engine_cache_env(
        "ns", root=tmp_path / "root", hf_home=tmp_path / "hf", env=target
    )
    assert effective["TRITON_CACHE_DIR"] == str(tmp_path / "mine")
    assert (tmp_path / "mine").is_dir()
    assert (tmp_path / "root" / "ns" / "inductor").is_dir()
    assert target["VLLM_CACHE_ROOT"] == str(tmp_path / "root" / "ns" / "vllm")
