from __future__ import annotations

import warnings

import pytest

from fal.toolkit.llm import LLMApp, engine_preset_args, merge_engine_args

PRO6000 = "RTX"  # what get_gpu_type() returns for "NVIDIA RTX PRO 6000 ..."


@pytest.fixture(autouse=True)
def isolate_agent_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("IS_ISOLATE_AGENT", "1")


def test_single_gpu_has_no_tp_or_all_reduce_flag():
    assert engine_preset_args("vllm", PRO6000, 1) == []
    assert engine_preset_args("sglang", "H100", 1) == []


def test_pcie_gpu_multi_gpu_gets_tp_and_all_reduce():
    assert engine_preset_args("vllm", PRO6000, 4) == [
        "--tensor-parallel-size",
        "4",
        "--disable-custom-all-reduce",
    ]
    assert engine_preset_args("sglang", "L40S", 2) == [
        "--tp-size",
        "2",
        "--disable-custom-all-reduce",
    ]


def test_nvlink_gpu_gets_tp_only():
    assert engine_preset_args("vllm", "H100", 8) == ["--tensor-parallel-size", "8"]
    # A100 must not be mistaken for A10.
    assert engine_preset_args("vllm", "A100", 2) == ["--tensor-parallel-size", "2"]
    assert engine_preset_args("vllm", "A10G", 2)[-1] == "--disable-custom-all-reduce"


def test_max_num_seqs_flag_per_engine():
    assert engine_preset_args("vllm", "H100", 1, 64) == ["--max-num-seqs", "64"]
    assert engine_preset_args("sglang", "H100", 1, 64) == [
        "--max-running-requests",
        "64",
    ]


def test_unknown_engine_raises():
    with pytest.raises(ValueError):
        engine_preset_args("trt", "H100", 1)


@pytest.mark.parametrize(
    "user",
    [
        ["--max-num-seqs", "64"],
        ["--max-num-seqs=64"],
        ["--max_num_seqs", "64"],
    ],
)
def test_user_valued_flag_wins(user):
    preset = engine_preset_args("vllm", "H100", 2, 32)
    assert merge_engine_args(preset, user) == ["--tensor-parallel-size", "2", *user]


def test_user_tp_alias_wins():
    preset = engine_preset_args("sglang", "H100", 4)
    assert merge_engine_args(preset, ["--tp", "4", "--x"]) == ["--tp", "4", "--x"]


@pytest.mark.parametrize(
    "user", [["--pipeline-parallel-size", "2"], ["--dp-size=2"], ["-pp", "2"]]
)
def test_user_pipeline_or_data_parallel_drops_tp_preset(user):
    preset = engine_preset_args("vllm", PRO6000, 4)
    assert merge_engine_args(preset, user) == ["--disable-custom-all-reduce", *user]


def test_user_boolean_flag_wins_without_eating_next_preset_flag():
    preset = engine_preset_args("vllm", PRO6000, 4, 16)
    merged = merge_engine_args(preset, ["--disable-custom-all-reduce"])
    assert merged == [
        "--tensor-parallel-size",
        "4",
        "--max-num-seqs",
        "16",
        "--disable-custom-all-reduce",
    ]
    merged = merge_engine_args(preset, ["--no-disable-custom-all-reduce"])
    assert "--disable-custom-all-reduce" not in merged[:-1]
    assert merged[:4] == ["--tensor-parallel-size", "4", "--max-num-seqs", "16"]


def test_user_value_not_mistaken_for_flag():
    preset = ["--tensor-parallel-size", "2"]
    assert merge_engine_args(preset, ["--max-model-len", "-1"]) == [
        *preset,
        "--max-model-len",
        "-1",
    ]


def _command(app_cls) -> list[str]:
    return app_cls().engine_command("/m")


def _make(monkeypatch, **attrs):
    class Sub(LLMApp):
        model = "org/m"
        engine_port = 1234

    for k, v in attrs.items():
        setattr(Sub, k, v)
    monkeypatch.setattr(Sub, "_gpu_type", staticmethod(lambda: PRO6000))
    return Sub


def test_engine_command_merges_once_and_user_last(monkeypatch):
    sub = _make(
        monkeypatch,
        num_gpus=4,
        max_multiplexing=32,
        engine_args=["--max-num-seqs", "32", "--dtype", "bf16"],
    )
    cmd = _command(sub)
    assert cmd.count("--tensor-parallel-size") == 1
    assert cmd.count("--disable-custom-all-reduce") == 1
    assert cmd.count("--max-num-seqs") == 1
    assert cmd[-4:] == ["--max-num-seqs", "32", "--dtype", "bf16"]
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "4"


def test_engine_command_sglang_uses_runtime_gpu_and_defaults(monkeypatch):
    sub = _make(monkeypatch, engine="sglang", num_gpus=None, max_multiplexing=8)
    cmd = _command(sub)
    assert "--tp-size" not in cmd
    assert cmd[-2:] == ["--max-running-requests", "8"]


def test_presets_off_leaves_engine_args_untouched(monkeypatch):
    sub = _make(
        monkeypatch,
        num_gpus=4,
        max_multiplexing=32,
        engine_presets=False,
        engine_args=["--dtype", "bf16"],
    )
    cmd = _command(sub)
    assert cmd[-2:] == ["--dtype", "bf16"]
    assert "--tensor-parallel-size" not in cmd
    assert "--max-num-seqs" not in cmd


def test_warns_on_max_seqs_mismatch():
    with pytest.warns(UserWarning, match="max_multiplexing is 16"):

        class A(LLMApp):
            model = "m"
            max_multiplexing = 16
            engine_args = ["--max-num-seqs=32"]


def test_warns_on_tp_mismatch():
    with pytest.warns(UserWarning, match="num_gpus is 4"):

        class A(LLMApp):
            model = "m"
            num_gpus = 4
            engine_args = ["--tensor-parallel-size", "2"]


def test_no_warning_when_args_agree():
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        class A(LLMApp):
            model = "m"
            num_gpus = 2
            max_multiplexing = 16
            engine_args = ["--tp-size=2", "--max-num-seqs", "16"]

        class C(LLMApp):
            model = "m"
            num_gpus = 4
            engine_args = [
                "--tensor-parallel-size",
                "2",
                "--pipeline-parallel-size",
                "2",
            ]

        class B(LLMApp):
            model = "m"


def test_subclass_of_configured_app_does_not_rewarn():
    with pytest.warns(UserWarning):

        class A(LLMApp):
            model = "m"
            num_gpus = 4
            engine_args = ["--tensor-parallel-size", "2"]

    with warnings.catch_warnings():
        warnings.simplefilter("error")

        class B(A):
            model = "other"


def test_every_preset_flag_is_recognized_as_user_override():
    # merge_engine_args matches preset tokens through the same alias table.
    for engine in ("vllm", "sglang"):
        preset = engine_preset_args(engine, "RTX", 2, 8)
        assert merge_engine_args(preset, preset) == preset
