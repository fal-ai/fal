from __future__ import annotations

import fastapi
import pytest

import fal
from fal import _cuda_cache as cuda_cache
from fal._cuda_cache import apply_cuda_cache_env, cuda_cache_dir
from fal.toolkit.utils.download_utils import FAL_PERSISTENT_DIR


@pytest.fixture
def isolate_agent_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("IS_ISOLATE_AGENT", "1")


def test_dir_keyed_by_model_and_driver(tmp_path):
    path = cuda_cache_dir(root=tmp_path, driver="580.95.05", model="NVIDIA RTX/6000")
    assert path == tmp_path / "NVIDIA-RTX-6000" / "580.95.05"


def test_no_driver_is_cpu_host(tmp_path):
    assert cuda_cache_dir(root=tmp_path, driver="") is None
    env: dict[str, str] = {}
    assert apply_cuda_cache_env(root=tmp_path, driver="", env=env) is None
    assert env == {}


def test_sets_env_and_creates_dir(tmp_path):
    env: dict[str, str] = {}
    got = apply_cuda_cache_env(root=tmp_path, driver="580.1", model="m", env=env)
    assert got == env["CUDA_CACHE_PATH"] == str(tmp_path / "m" / "580.1")
    assert env["CUDA_CACHE_MAXSIZE"] == str(4 << 30)
    assert (tmp_path / "m" / "580.1").is_dir()


def test_user_values_win(tmp_path):
    env = {"CUDA_CACHE_PATH": str(tmp_path / "mine"), "CUDA_CACHE_MAXSIZE": "7"}
    apply_cuda_cache_env(root=tmp_path, driver="580.1", model="m", env=env)
    assert env == {"CUDA_CACHE_PATH": str(tmp_path / "mine"), "CUDA_CACHE_MAXSIZE": "7"}


def test_missing_data_volume_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(cuda_cache, "FAL_PERSISTENT_DIR", str(tmp_path / "absent"))
    env: dict[str, str] = {}
    assert apply_cuda_cache_env(driver="580.1", model="m", env=env) is None
    assert env == {}


def test_gpu_model_from_proc(tmp_path, monkeypatch):
    info = tmp_path / "0000:01:00.0" / "information"
    info.parent.mkdir()
    info.write_text("Model: \t\t NVIDIA RTX PRO 6000 Blackwell\nIRQ: 1\n")
    monkeypatch.setattr(
        cuda_cache, "_GPU_INFO_GLOB", str(tmp_path / "*" / "information")
    )
    assert cuda_cache.nvidia_gpu_model() == "NVIDIA RTX PRO 6000 Blackwell"
    monkeypatch.setattr(cuda_cache, "_GPU_INFO_GLOB", str(tmp_path / "none" / "*"))
    assert cuda_cache.nvidia_gpu_model() == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(("opt_out", "expected"), [(False, True), (True, False)])
async def test_app_lifespan_applies_before_setup(
    isolate_agent_env, monkeypatch, opt_out, expected
):
    seen: list[str] = []
    monkeypatch.setattr(fal.app, "apply_cuda_cache_env", lambda: seen.append("applied"))

    class A(fal.App):
        persist_cuda_cache = not opt_out

        def setup(self):
            seen.append("setup")

    async with A().lifespan(fastapi.FastAPI()):
        pass
    assert seen == (["applied", "setup"] if expected else ["setup"])


def test_unwritable_dir_leaves_env_untouched(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    env: dict[str, str] = {}
    assert (
        apply_cuda_cache_env(root=blocker, driver="580.1", model="m", env=env) is None
    )
    assert env == {}


def test_persistent_dir_matches_toolkit():
    assert cuda_cache.FAL_PERSISTENT_DIR == FAL_PERSISTENT_DIR


@pytest.mark.asyncio
async def test_app_lifespan_survives_cache_failure(isolate_agent_env, monkeypatch):
    def boom():
        raise RuntimeError("nope")

    monkeypatch.setattr(fal.app, "apply_cuda_cache_env", boom)
    seen: list[str] = []

    class A(fal.App):
        def setup(self):
            seen.append("setup")

    async with A().lifespan(fastapi.FastAPI()):
        pass
    assert seen == ["setup"]


def test_cache_disabled_by_user_is_noop(tmp_path):
    env = {"CUDA_CACHE_DISABLE": "1"}
    assert (
        apply_cuda_cache_env(root=tmp_path, driver="580.1", model="m", env=env) is None
    )
    assert env == {"CUDA_CACHE_DISABLE": "1"}
    assert not (tmp_path / "m").exists()


def test_existing_unwritable_dir_leaves_env_untouched(tmp_path, monkeypatch):
    (tmp_path / "m" / "580.1").mkdir(parents=True)
    monkeypatch.setattr(cuda_cache.os, "access", lambda *_: False)
    env: dict[str, str] = {}
    assert (
        apply_cuda_cache_env(root=tmp_path, driver="580.1", model="m", env=env) is None
    )
    assert env == {}
