"""The LLMApp examples import, resolve their config and build an engine command."""

from __future__ import annotations

import json
import py_compile
import runpy
from pathlib import Path

import pytest

from fal.toolkit.llm._presets import Machine

EXAMPLES = Path(__file__).parents[4] / "examples"

RTX = Machine("RTX", "NVIDIA RTX PRO 6000 Blackwell Server Edition", 97887, 1)

# (file, class, app name, model)
APPS = [
    ("llm_app.py", "QwenVLLM", "qwen-vllm", "Qwen/Qwen2.5-7B-Instruct"),
    ("llm_agent.py", "QwenAgent", "qwen-agent", "Qwen/Qwen3-8B"),
    ("llm_replicas.py", "QwenReplicas", "qwen-replicas", "Qwen/Qwen2.5-7B-Instruct"),
    ("llm_sglang.py", "QwenSGLang", "qwen-sglang", "Qwen/Qwen2.5-7B-Instruct"),
]


@pytest.fixture(autouse=True)
def isolate_agent_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("IS_ISOLATE_AGENT", "1")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


def load(filename: str, cls: str):
    return runpy.run_path(str(EXAMPLES / filename))[cls]


@pytest.mark.parametrize("filename,cls,name,model", APPS)
def test_app_name_and_model(filename, cls, name, model):
    app = load(filename, cls)
    assert app.app_name == name
    assert app.model == model


@pytest.mark.parametrize("filename,cls,name,model", APPS)
def test_engine_command_serves_the_model(
    filename, cls, name, model, tmp_path, monkeypatch
):
    app_cls = load(filename, cls)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen2ForCausalLM"],
                "hidden_size": 3584,
                "num_hidden_layers": 28,
                "intermediate_size": 18944,
                "vocab_size": 152064,
                "num_attention_heads": 28,
                "num_key_value_heads": 4,
                "max_position_embeddings": 32768,
            }
        )
    )
    # One engine sees this many GPUs: num_gpus split across the replicas.
    machine = Machine(
        RTX.gpu_type,
        RTX.gpu_name,
        RTX.gpu_memory_mib,
        (app_cls.num_gpus or 1) // app_cls.engine_replicas,
    )
    monkeypatch.setattr(app_cls, "_machine", lambda self: machine)
    cmd = app_cls().engine_command(str(tmp_path))
    assert str(tmp_path) in cmd
    assert cmd[cmd.index("--served-model-name") + 1] == app_cls.served_model_name
    if app_cls.engine == "vllm":
        assert cmd[:2] == ["vllm", "serve"]
    else:
        assert cmd[1:3] == ["-m", "sglang.launch_server"]
        assert cmd[cmd.index("--model-path") + 1] == str(tmp_path)


def test_llm_app_leaves_the_sequence_cap_to_the_presets(tmp_path, monkeypatch):
    app_cls = load("llm_app.py", "QwenVLLM")
    assert "--max-num-seqs" not in app_cls.engine_args
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["Qwen2ForCausalLM"], "hidden_size": 3584})
    )
    monkeypatch.setattr(app_cls, "_machine", lambda self: RTX)
    cmd = app_cls().engine_command(str(tmp_path))
    assert cmd[cmd.index("--max-num-seqs") + 1] == str(app_cls.max_multiplexing)


def test_agent_enables_tool_calling_and_deadline(tmp_path, monkeypatch):
    app_cls = load("llm_agent.py", "QwenAgent")
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3ForCausalLM"], "hidden_size": 4096})
    )
    monkeypatch.setattr(app_cls, "_machine", lambda self: RTX)
    cmd = app_cls().engine_command(str(tmp_path))
    assert app_cls.engine_profile == "agent"
    assert app_cls.request_deadline_s
    assert "--enable-auto-tool-choice" in cmd
    assert cmd[cmd.index("--tool-call-parser") + 1] == "hermes"


def test_replicas_split_the_gpus():
    app_cls = load("llm_replicas.py", "QwenReplicas")
    assert app_cls.num_gpus == 2
    assert app_cls.engine_replicas == 2


def test_client_script_compiles(tmp_path):
    py_compile.compile(
        str(EXAMPLES / "llm_client.py"),
        cfile=str(tmp_path / "llm_client.pyc"),
        doraise=True,
    )
