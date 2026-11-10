from __future__ import annotations

import json
import sys
import types
import warnings

import pytest

from fal.toolkit.llm import (
    LLMApp,
    _app,
    _cache,
    _presets,
    engine_preset_args,
    merge_engine_args,
)
from fal.toolkit.llm._presets import (
    RECIPES,
    Machine,
    ModelFacts,
    _estimate_active_params,
    classify_gpu,
    engine_arg_warnings,
    engine_preset_env,
    model_facts,
    short_gpu_type,
)

PRO6000 = "RTX"  # what get_gpu_type() returns for "NVIDIA RTX PRO 6000 ..."
PRO6000_NAME = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
H100_NAME = "NVIDIA H100 80GB HBM3"


@pytest.fixture(autouse=True)
def isolate_agent_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("IS_ISOLATE_AGENT", "1")


def _without_details(args):
    """Drop the always-on usage flag so tests can compare the rest."""
    return [a for a in args if a != _presets._PROMPT_DETAILS_FLAG]


def _base(engine, gpu_type, num_gpus, seqs=None):
    """Preset args from the generic path (no GPU name, no model facts)."""
    return _without_details(
        engine_preset_args(
            engine,
            Machine(gpu_type, None, None, num_gpus),
            ModelFacts(),
            max_multiplexing=seqs,
        )
    )


def test_vllm_reports_prompt_token_details_by_default():
    args = engine_preset_args(
        "vllm", Machine("H100", None, None, 1), ModelFacts(), max_multiplexing=8
    )
    assert "--enable-prompt-tokens-details" in args
    sglang = engine_preset_args(
        "sglang", Machine("H100", None, None, 1), ModelFacts(), max_multiplexing=8
    )
    assert "--enable-prompt-tokens-details" not in sglang
    # A user can turn it off, or restate it, without a duplicate.
    for user in (
        ["--no-enable-prompt-tokens-details"],
        ["--enable-prompt-tokens-details"],
    ):
        merged = merge_engine_args(args, user)
        assert merged.count("--enable-prompt-tokens-details") <= 1
        assert merged[-len(user) :] == user
    assert "--enable-prompt-tokens-details" not in merge_engine_args(
        args, ["--no-enable-prompt-tokens-details"]
    )


def _flags(args):
    """Flag to value (None for boolean switches), so tests ignore ordering."""
    out = {}
    i = 0
    while i < len(args):
        if args[i] in _presets._BOOLEAN_FLAGS:
            out[args[i]] = None
            i += 1
        else:
            out[args[i]] = args[i + 1]
            i += 2
    return out


def test_single_gpu_has_no_tp_or_all_reduce_flag():
    assert _base("vllm", PRO6000, 1) == []
    assert _base("sglang", "H100", 1) == []


def test_pcie_gpu_multi_gpu_gets_tp_and_all_reduce():
    assert _base("vllm", PRO6000, 4) == [
        "--tensor-parallel-size",
        "4",
        "--disable-custom-all-reduce",
    ]
    assert _base("sglang", "L40S", 2) == [
        "--tp-size",
        "2",
        "--disable-custom-all-reduce",
    ]


def test_nvlink_gpu_gets_tp_only():
    assert _base("vllm", "H100", 8) == ["--tensor-parallel-size", "8"]
    # A100 must not be mistaken for A10.
    assert _base("vllm", "A100", 2) == ["--tensor-parallel-size", "2"]
    assert _base("vllm", "A10G", 2)[-1] == "--disable-custom-all-reduce"


def test_max_num_seqs_flag_per_engine():
    assert _base("vllm", "H100", 1, 64) == ["--max-num-seqs", "64"]
    assert _base("sglang", "H100", 1, 64) == ["--max-running-requests", "64"]


def test_unknown_engine_raises():
    with pytest.raises(ValueError):
        _base("trt", "H100", 1)


@pytest.mark.parametrize(
    "user",
    [
        ["--max-num-seqs", "64"],
        ["--max-num-seqs=64"],
        ["--max_num_seqs", "64"],
    ],
)
def test_user_valued_flag_wins(user):
    preset = _base("vllm", "H100", 2, 32)
    assert merge_engine_args(preset, user) == ["--tensor-parallel-size", "2", *user]


def test_user_tp_alias_wins():
    preset = _base("sglang", "H100", 4)
    assert merge_engine_args(preset, ["--tp", "4", "--x"]) == ["--tp", "4", "--x"]


@pytest.mark.parametrize(
    "user", [["--pipeline-parallel-size", "2"], ["--dp-size=2"], ["-pp", "2"]]
)
def test_user_pipeline_or_data_parallel_drops_tp_preset(user):
    preset = _base("vllm", PRO6000, 4)
    assert merge_engine_args(preset, user) == ["--disable-custom-all-reduce", *user]


def test_user_boolean_flag_wins_without_eating_next_preset_flag():
    preset = _base("vllm", PRO6000, 4, 16)
    merged = merge_engine_args(preset, ["--disable-custom-all-reduce"])
    assert merged == [
        "--tensor-parallel-size",
        "4",
        "--max-num-seqs",
        "32",
        "--disable-custom-all-reduce",
    ]
    merged = merge_engine_args(preset, ["--no-disable-custom-all-reduce"])
    assert "--disable-custom-all-reduce" not in merged[:-1]
    assert merged[:4] == ["--tensor-parallel-size", "4", "--max-num-seqs", "32"]


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
    monkeypatch.setattr(Sub, "_gpu_name", staticmethod(lambda: None))
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
            engine_args = ["--max-num-seqs=8"]


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

        # The defaulted max_multiplexing is not a user value to compare against.
        class D(LLMApp):
            model = "m"
            engine_args = ["--max-num-seqs", "64"]


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


def test_keep_alive_default_and_override():
    class Default(LLMApp):
        model = "m"

    class Short(LLMApp):
        model = "m"
        keep_alive = 120

    assert Default.keep_alive == 600
    assert Default.host_kwargs["keep_alive"] == 600
    assert Short.host_kwargs["keep_alive"] == 120


def test_max_multiplexing_is_not_defaulted():
    # Left to the user and the platform: a class default would count as
    # explicit and block a pyproject or CLI override.
    class One(LLMApp):
        model = "m"

    assert One.max_multiplexing is None
    assert "max_multiplexing" not in One.__dict__


def test_max_multiplexing_user_value_is_kept():
    class A(LLMApp):
        model = "m"
        num_gpus = 4
        max_multiplexing = 32

    class B(A):
        model = "other"

    assert A.max_multiplexing == B.max_multiplexing == 32
    assert A.host_kwargs["max_multiplexing"] == 32


QWEN3_8B = {
    "architectures": ["Qwen3ForCausalLM"],
    "hidden_size": 4096,
    "num_hidden_layers": 36,
    "intermediate_size": 12288,
    "vocab_size": 151936,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
}
LLAMA_8B = {**QWEN3_8B, "architectures": ["LlamaForCausalLM"]}
GPT_OSS_120B = {
    "architectures": ["GptOssForCausalLM"],
    "hidden_size": 2880,
    "num_hidden_layers": 36,
    "num_local_experts": 128,
    "num_experts_per_tok": 4,
    "intermediate_size": 2880,
    "vocab_size": 201088,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 64,
}
DENSE_70B = {
    "architectures": ["Qwen3ForCausalLM"],
    "hidden_size": 8192,
    "num_hidden_layers": 80,
    "intermediate_size": 28672,
    "vocab_size": 128256,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
}
# Stand-in for a large MoE; only the id and the expert count matter here.
DSV4_FLASH = {"architectures": ["DeepseekV4ForCausalLM"], "n_routed_experts": 256}
DSV4_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
FP8_QUANT = {"quant_method": "fp8"}
EAGLE3_QWEN3_8B = {
    "method": "eagle3",
    "model": "RedHatAI/Qwen3-8B-speculator.eagle3",
    "num_speculative_tokens": 3,
}


def _run(engine, config, machine, **kwargs):
    model = model_facts(
        kwargs.pop("model_id", None), config, kwargs.pop("active_params_b", None)
    )
    return _without_details(
        engine_preset_args(
            engine,
            machine,
            model,
            max_multiplexing=kwargs.pop("max_multiplexing", None),
            **kwargs,
        )
    )


def _rtx(engine="vllm", config=None, **kwargs):
    machine = Machine(PRO6000, PRO6000_NAME, 97887, kwargs.pop("num_gpus", 1))
    return _run(engine, config, machine, **kwargs)


def _h100(engine="vllm", config=None, **kwargs):
    machine = Machine("H100", H100_NAME, 81559, kwargs.pop("num_gpus", 1))
    return _run(engine, config, machine, **kwargs)


def test_every_preset_flag_is_recognized_as_user_override():
    # merge_engine_args matches preset tokens through the same alias table, so
    # every flag any recipe can emit must be a known group.
    for engine in ("vllm", "sglang"):
        preset = _base(engine, "RTX", 2, 8)
        assert merge_engine_args(preset, preset) == preset
    for recipe in RECIPES:
        for flag in (*recipe.args, *_presets._MAX_SEQS_FLAG.values()):
            assert _presets._flag_group(flag) is not None, flag
    # Every recipe at once.
    for args in (
        _rtx("vllm", QWEN3_8B, model_id="Qwen/Qwen3-8B", fp8_weights=True),
        _rtx("vllm", DSV4_FLASH, model_id=DSV4_ID, num_gpus=4),
        _h100("vllm", GPT_OSS_120B),
    ):
        assert merge_engine_args(args, args) == args


def test_every_recipe_names_gpu_count_gpu_engine_and_numbers():
    for recipe in RECIPES:
        assert "vLLM" in recipe.source or "SGLang" in recipe.source
        assert any(tag in recipe.source for tag in ("1x ", "2x ", "4x ", "8x "))
        assert any(gpu in recipe.source for gpu in ("RTX PRO 6000", "B200", "B300"))
        assert any(ch.isdigit() for ch in recipe.source)


@pytest.mark.parametrize(
    "name,memory,expected",
    [
        ("NVIDIA RTX PRO 6000 Blackwell Server Edition", 97887, "RTX_PRO_6000"),
        ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", 97887, "RTX_PRO_6000"),
        ("NVIDIA RTX PRO 6000 Blackwell Server Edition", None, "RTX_PRO_6000"),
        ("NVIDIA GeForce RTX 5090", 32607, "NVIDIA_GEFORCE_RTX_5090"),
        ("NVIDIA H100 80GB HBM3", 81559, "H100"),
        ("NVIDIA H100 PCIe", 81559, "H100"),
        ("NVIDIA H200", 143771, "NVIDIA_H200"),
        ("NVIDIA B200", 183359, "B200"),
        ("NVIDIA B300 SXM6 AC", 288000, "B300"),
        ("NVIDIA RTX 6000 Ada Generation", 49140, "NVIDIA_RTX_6000_ADA_GENERATION"),
        ("", None, "UNKNOWN"),
    ],
)
def test_classify_gpu(name, memory, expected):
    assert classify_gpu(name, memory) == expected


def test_short_gpu_type_matches_get_gpu_type():
    assert short_gpu_type(PRO6000_NAME) == "RTX"
    assert short_gpu_type("NVIDIA H100 80GB HBM3") == "H100"
    assert short_gpu_type("") == "UNKNOWN"


def test_estimate_active_params_dense_and_moe():
    is_moe, active = _estimate_active_params(QWEN3_8B)
    assert is_moe is False
    assert 7 < active < 9
    is_moe, active = _estimate_active_params(GPT_OSS_120B)
    assert is_moe is True
    assert 4 < active < 7
    assert _estimate_active_params(DENSE_70B)[1] > 40


def test_estimate_active_params_missing_fields():
    assert _estimate_active_params({}) == (None, None)
    assert _estimate_active_params({"hidden_size": 4096}) == (False, None)
    partial = {k: v for k, v in GPT_OSS_120B.items() if k != "num_experts_per_tok"}
    assert _estimate_active_params(partial) == (True, None)
    nested = {"text_config": QWEN3_8B}
    assert _estimate_active_params(nested) == _estimate_active_params(QWEN3_8B)


def test_model_facts_resolves_once():
    facts = model_facts("Qwen/Qwen3-8B", QWEN3_8B)
    assert facts.is_moe is False
    assert 7 < facts.active_params_b < 9
    assert facts.architectures == {"Qwen3ForCausalLM"}
    assert not facts.prequantized
    assert model_facts("m", QWEN3_8B, active_params_b=70).active_params_b == 70
    assert model_facts("m", None) == ModelFacts("m")


@pytest.mark.parametrize(
    "quant,expected",
    [
        ({"quant_method": "fp8"}, True),
        ({"quant_method": "awq"}, True),
        ({"quant_method": "gptq"}, True),
        ({"quant_method": "mxfp4"}, True),
        ({}, True),
        (None, False),
    ],
)
@pytest.mark.parametrize("nested", [False, True])
def test_any_quantization_config_counts_as_prequantized(quant, expected, nested):
    if nested:
        config = {**QWEN3_8B, "text_config": {**QWEN3_8B}}
        if quant is not None:
            config["text_config"]["quantization_config"] = quant
        config = {k: v for k, v in config.items() if k != "hidden_size"}
    else:
        config = {**QWEN3_8B}
        if quant is not None:
            config["quantization_config"] = quant
    assert model_facts("m", config).prequantized is expected


def test_prequantized_checkpoint_skips_quantization_flag():
    config = {**QWEN3_8B, "quantization_config": FP8_QUANT}
    args = _rtx("vllm", config, model_id="Qwen/Qwen3-8B-FP8", fp8_weights=True)
    assert "--quantization" not in args
    awq = {**QWEN3_8B, "quantization_config": {"quant_method": "awq"}}
    assert "--quantization" not in _rtx("vllm", awq, fp8_weights=True)
    args = _rtx("vllm", QWEN3_8B, model_id="Qwen/Qwen3-8B", fp8_weights=True)
    assert _flags(args)["--quantization"] == "fp8"


# Recipe 1: 1x RTX PRO 6000 or H100, vLLM, cap 256.


@pytest.mark.parametrize("run", [_rtx, _h100])
def test_single_gpu_seqs_recipe_on_rtx_and_h100(run):
    assert _flags(run("vllm", LLAMA_8B, max_multiplexing=32)) == {
        "--max-num-seqs": "256"
    }
    assert _flags(run("vllm", QWEN3_8B)) == {"--max-num-seqs": "256"}


def test_qwen3_8b_on_h100_gets_only_the_cap():
    # FP8 KV was noise on H100 and EAGLE3 was not measured there.
    flags = _flags(_h100("vllm", QWEN3_8B, model_id="Qwen/Qwen3-8B"))
    assert flags == {"--max-num-seqs": "256"}


def test_seqs_recipe_is_vllm_only():
    assert _rtx("sglang", QWEN3_8B) == []
    assert _flags(_rtx("sglang", QWEN3_8B, max_multiplexing=32)) == {
        "--max-running-requests": "32"
    }


def test_max_seqs_emitted_once_and_never_lowered():
    for engine in ("vllm", "sglang"):
        args = _rtx(engine, QWEN3_8B, max_multiplexing=32)
        assert args.count(_presets._MAX_SEQS_FLAG[engine]) == 1
    assert _flags(_rtx("vllm", LLAMA_8B, max_multiplexing=512)) == {
        "--max-num-seqs": "512"
    }
    assert _rtx("vllm", QWEN3_8B, max_multiplexing=512).count("--max-num-seqs") == 1


def test_rtx_large_dense_and_unknown_model_unchanged():
    assert _rtx("vllm", DENSE_70B, max_multiplexing=16) == ["--max-num-seqs", "16"]
    assert _rtx("vllm", None, max_multiplexing=16) == ["--max-num-seqs", "16"]
    assert _rtx("vllm", None) == []


def test_active_params_override_beats_estimate():
    assert _flags(_rtx("vllm", DENSE_70B, active_params_b=8)) == {
        "--max-num-seqs": "256"
    }
    assert _rtx("vllm", QWEN3_8B, active_params_b=70) == []


def test_other_gpus_unchanged():
    model = model_facts("Qwen/Qwen3-8B", QWEN3_8B)
    a100 = Machine("A100", "NVIDIA A100-SXM4-80GB", 81920, 1)
    assert _without_details(
        engine_preset_args("vllm", a100, model, max_multiplexing=32)
    ) == ["--max-num-seqs", "32"]
    geforce = Machine("RTX", "NVIDIA GeForce RTX 5090", 32607, 1)
    assert _without_details(engine_preset_args("vllm", geforce, model)) == []


# Recipes are measured at a given GPU count: 1x recipes never apply at TP > 1.


def test_single_gpu_recipes_absent_on_multi_gpu():
    tp = {"--tensor-parallel-size": "4", "--disable-custom-all-reduce": None}
    qwen = _rtx("vllm", QWEN3_8B, model_id="Qwen/Qwen3-8B", num_gpus=4)
    assert _flags(qwen) == tp
    gpt_oss_dense_view = {
        k: v for k, v in GPT_OSS_120B.items() if k != "num_local_experts"
    }
    assert _flags(_rtx("vllm", gpt_oss_dense_view, num_gpus=4)) == tp
    # gpt-oss is MoE: only the multi-GPU MoE cap applies, not its 1x FP8 KV.
    assert _flags(_rtx("vllm", GPT_OSS_120B, num_gpus=4)) == {
        **tp,
        "--max-num-seqs": "128",
    }
    h100 = _h100("vllm", GPT_OSS_120B, num_gpus=4)
    assert _flags(h100) == {"--tensor-parallel-size": "4"}


def test_multi_gpu_moe_on_rtx_gets_cap_128():
    flags = _flags(_rtx("vllm", DSV4_FLASH, num_gpus=4))
    assert flags["--max-num-seqs"] == "128"
    assert "--kv-cache-dtype" not in flags
    # Never lowered below a larger router concurrency.
    flags = _flags(_rtx("vllm", DSV4_FLASH, num_gpus=4, max_multiplexing=512))
    assert flags["--max-num-seqs"] == "512"
    # Dense multi-GPU is not covered.
    assert "--max-num-seqs" not in _flags(_rtx("vllm", DENSE_70B, num_gpus=4))


# Recipe 3: Qwen3-8B on 1x RTX PRO 6000.


@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-8B", "Qwen/Qwen3-8B-FP8"])
@pytest.mark.parametrize("profile", ["throughput", "latency"])
def test_qwen3_8b_rtx_recipe_without_fp8_gate(model_id, profile):
    flags = _flags(
        _rtx("vllm", QWEN3_8B, model_id=model_id, profile=profile, max_multiplexing=32)
    )
    assert flags.pop("--speculative-config") == json.dumps(EAGLE3_QWEN3_8B)
    assert flags == {"--max-num-seqs": "256", "--kv-cache-dtype": "fp8"}


def test_qwen3_8b_recipe_needs_known_model_id_and_vllm():
    other = _flags(_rtx("vllm", QWEN3_8B, model_id="org/other"))
    assert other == {"--max-num-seqs": "256"}
    assert _flags(_rtx("vllm", QWEN3_8B)) == {"--max-num-seqs": "256"}
    # Without a config the model-id recipe still applies.
    flags = _flags(_rtx("vllm", None, model_id="Qwen/Qwen3-8B"))
    assert flags["--max-num-seqs"] == "256" and "--speculative-config" in flags
    assert _rtx("sglang", QWEN3_8B, model_id="Qwen/Qwen3-8B") == []


def test_fp8_weights_stays_opt_in():
    base = _flags(_rtx("vllm", QWEN3_8B, model_id="Qwen/Qwen3-8B"))
    assert "--quantization" not in base
    on = _flags(_rtx("vllm", QWEN3_8B, model_id="Qwen/Qwen3-8B", fp8_weights=True))
    assert on == {**base, "--quantization": "fp8"}
    sglang = _flags(_rtx("sglang", QWEN3_8B, fp8_weights=True))
    assert sglang == {"--quantization": "fp8"}


# Checkpoint substitution and the max_throughput profile.

RTX_MACHINE = Machine(PRO6000, PRO6000_NAME, 97887, 1)
FP8_CKPT = "Qwen/Qwen3-8B-FP8"
NVFP4_CKPT = "nvidia/Qwen3-8B-NVFP4"


@pytest.mark.parametrize("profile", ["throughput", "latency"])
def test_checkpoint_default_is_fp8_on_rtx(profile):
    assert (
        _presets.checkpoint_for("vllm", RTX_MACHINE, "Qwen/Qwen3-8B", profile=profile)
        == FP8_CKPT
    )
    # True also lands on the measured checkpoint, not online quantization.
    assert (
        _presets.checkpoint_for(
            "vllm", RTX_MACHINE, "qwen/qwen3-8b", profile=profile, fp8_weights=True
        )
        == FP8_CKPT
    )


def test_checkpoint_fp8_opt_out_and_no_match_cases():
    def pick(model_id="Qwen/Qwen3-8B", machine=RTX_MACHINE, engine="vllm", **kw):
        return _presets.checkpoint_for(engine, machine, model_id, **kw)

    assert pick(fp8_weights=False) is None
    assert pick(machine=Machine("H100", H100_NAME, 81559, 1)) is None
    assert pick(machine=Machine(PRO6000, PRO6000_NAME, 97887, 4)) is None
    assert pick(machine=Machine(PRO6000, None, None, 1)) is None
    assert pick(engine="sglang") is None
    assert pick(model_id="org/other") is None
    assert pick(model_id="openai/gpt-oss-120b") is None


def test_default_run_keeps_bf16_recipe_flags_and_adds_no_online_quant():
    # The served FP8 checkpoint carries quantization_config, so even
    # fp8_weights=True adds no --quantization.
    config = {**QWEN3_8B, "quantization_config": FP8_QUANT}
    args = _rtx("vllm", config, model_id="Qwen/Qwen3-8B", fp8_weights=True)
    flags = _flags(args)
    assert "--quantization" not in flags
    assert flags["--kv-cache-dtype"] == "fp8" and flags["--max-num-seqs"] == "256"
    assert flags["--speculative-config"] == json.dumps(EAGLE3_QWEN3_8B)


@pytest.mark.parametrize(
    "config,model_id,profile",
    [
        ({**QWEN3_8B, "quantization_config": FP8_QUANT}, "Qwen/Qwen3-8B", "throughput"),
        ({**QWEN3_8B, "quantization_config": FP8_QUANT}, "Qwen/Qwen3-8B", "latency"),
        (GPT_OSS_120B, "openai/gpt-oss-120b", "throughput"),
    ],
)
def test_fp8_and_gpt_oss_keep_cap_256_and_default_memory(config, model_id, profile):
    # Cap 512 and memory 0.95 measured no gain with FP8 weights (4,111 vs
    # 4,206 tok/s at 256 streams) and cost single stream (168 vs 208 tok/s);
    # they are neutral on gpt-oss. Only the NVFP4 profile keeps them.
    flags = _flags(_rtx("vllm", config, model_id=model_id, profile=profile))
    assert flags["--max-num-seqs"] == "256"
    assert "--gpu-memory-utilization" not in flags


def test_max_throughput_on_rtx_qwen3_8b():
    assert (
        _presets.checkpoint_for(
            "vllm", RTX_MACHINE, "Qwen/Qwen3-8B", profile="max_throughput"
        )
        == NVFP4_CKPT
    )
    # Opting out of FP8 weights does not opt out of an explicit max_throughput.
    assert (
        _presets.checkpoint_for(
            "vllm",
            RTX_MACHINE,
            "Qwen/Qwen3-8B",
            profile="max_throughput",
            fp8_weights=False,
        )
        == NVFP4_CKPT
    )
    config = {**QWEN3_8B, "quantization_config": {"quant_method": "modelopt"}}
    flags = _flags(
        _rtx("vllm", config, model_id="Qwen/Qwen3-8B", profile="max_throughput")
    )
    assert flags == {
        "--max-num-seqs": "512",
        "--gpu-memory-utilization": "0.95",
        "--kv-cache-dtype": "fp8",
        "--speculative-config": json.dumps(EAGLE3_QWEN3_8B),
    }


def test_max_throughput_elsewhere_equals_throughput():
    for run, config, model_id in (
        (_h100, QWEN3_8B, "Qwen/Qwen3-8B"),
        (_rtx, LLAMA_8B, "org/other"),
        (_rtx, GPT_OSS_120B, "openai/gpt-oss-120b"),
        (_rtx, QWEN3_8B, "qwen/qwen3-8b-other"),
    ):
        assert run("vllm", config, model_id=model_id, profile="max_throughput") == run(
            "vllm", config, model_id=model_id, profile="throughput"
        )
    machine = Machine(PRO6000, PRO6000_NAME, 97887, 4)
    assert (
        _presets.checkpoint_for(
            "vllm", machine, "Qwen/Qwen3-8B", profile="max_throughput"
        )
        is None
    )
    assert _rtx("sglang", QWEN3_8B, profile="max_throughput") == []


def test_max_throughput_user_args_win():
    preset = _rtx("vllm", QWEN3_8B, model_id="Qwen/Qwen3-8B", profile="max_throughput")
    merged = merge_engine_args(
        preset, ["--gpu-memory-utilization=0.9", "--max-num-seqs", "64"]
    )
    assert "--gpu-memory-utilization" not in merged[:-3]
    assert merged[-3:] == ["--gpu-memory-utilization=0.9", "--max-num-seqs", "64"]
    assert merged.count("--max-num-seqs") == 1


def _checkpoint_app(monkeypatch, **attrs):
    sub = _make(monkeypatch, model="Qwen/Qwen3-8B", **attrs)
    monkeypatch.setattr(sub, "_gpu_name", staticmethod(lambda: (PRO6000_NAME, 97887)))
    app = sub()
    app._boot = {}
    return app


@pytest.mark.parametrize(
    "attrs,expected",
    [
        ({}, FP8_CKPT),
        ({"fp8_weights": True}, FP8_CKPT),
        ({"fp8_weights": False}, "Qwen/Qwen3-8B"),
        ({"engine_profile": "max_throughput"}, NVFP4_CKPT),
        ({"engine_presets": False}, "Qwen/Qwen3-8B"),
        ({"engine": "sglang"}, "Qwen/Qwen3-8B"),
        ({"num_gpus": 4}, "Qwen/Qwen3-8B"),
    ],
)
def test_app_model_source_and_served_name(monkeypatch, attrs, expected):
    app = _checkpoint_app(monkeypatch, **attrs)
    assert app._model_source() == expected
    # Clients and the engine's served name keep the user's model.
    assert app._served_name() == "Qwen/Qwen3-8B"
    cmd = app.engine_command("/m")
    assert cmd[cmd.index("--served-model-name") + 1] == "Qwen/Qwen3-8B"
    assert ("model_source" in app._boot) == (expected != "Qwen/Qwen3-8B")


def test_app_resolve_downloads_the_substitute(monkeypatch, capsys, tmp_path):
    seen = []

    def download(repo, revision=None):
        # The recipe checkpoint is never pinned to the app's model_revision.
        assert revision is None
        seen.append(repo)
        return "/p"

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    app = _checkpoint_app(monkeypatch)
    assert app._resolve_model() == "/p"
    assert seen == [FP8_CKPT]
    # Local-first reuse and the upstream check follow the checkpoint repo.
    assert app._model_repo == FP8_CKPT
    assert app._boot["served_repo"] == FP8_CKPT
    marker = _cache.snapshot_marker_path(FP8_CKPT)
    assert _cache.read_marker(marker) == {
        "model": FP8_CKPT,
        "revision": None,
        "path": "/p",
    }
    assert app._boot["model_source"] == FP8_CKPT
    assert FP8_CKPT in capsys.readouterr().out
    off = _checkpoint_app(monkeypatch, fp8_weights=False)
    assert off._resolve_model() == "/p"
    assert seen[-1] == "Qwen/Qwen3-8B"
    assert "model_source" not in off._boot


def test_app_falls_back_to_own_model_when_substitute_fails(
    monkeypatch, capsys, tmp_path
):
    seen = []
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))

    def download(repo, revision=None):
        seen.append(repo)
        if repo == FP8_CKPT:
            raise OSError("offline")
        return "/own"

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    app = _checkpoint_app(monkeypatch)
    assert app._resolve_model() == "/own"
    assert seen == [FP8_CKPT, "Qwen/Qwen3-8B"]
    assert "model_source" not in app._boot
    assert app._model_repo == "Qwen/Qwen3-8B"
    assert app._boot["served_repo"] == "Qwen/Qwen3-8B"
    assert "unavailable" in capsys.readouterr().out


@pytest.mark.parametrize(
    "profile,checkpoint", [("throughput", FP8_CKPT), ("max_throughput", NVFP4_CKPT)]
)
def test_fallback_drops_the_checkpoint_recipe_flags(
    monkeypatch, capsys, tmp_path, profile, checkpoint
):
    own = tmp_path / "own"
    own.mkdir()
    (own / "config.json").write_text(json.dumps(QWEN3_8B))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))

    def download(repo, revision=None):
        if repo == checkpoint:
            raise OSError("offline")
        return str(own)

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    app = _checkpoint_app(monkeypatch, engine_profile=profile)
    path = app._resolve_model()
    cmd = app.engine_command(path)
    # Only the generic single-GPU cap is left; 0.95 memory, cap 512, FP8 KV
    # and EAGLE3 were tuned for the checkpoint.
    for flag in (
        "--gpu-memory-utilization",
        "--kv-cache-dtype",
        "--speculative-config",
        "--quantization",
    ):
        assert flag not in cmd
    assert cmd[cmd.index("--max-num-seqs") + 1] == "256"


def test_skip_checkpoint_drops_only_that_recipe_case_insensitively():
    kept = _rtx("vllm", QWEN3_8B, model_id="Qwen/Qwen3-8B", profile="max_throughput")
    assert "--gpu-memory-utilization" in kept
    for skip in (NVFP4_CKPT, NVFP4_CKPT.upper()):
        args = _rtx(
            "vllm",
            QWEN3_8B,
            model_id="Qwen/Qwen3-8B",
            profile="max_throughput",
            skip_checkpoint=skip,
        )
        assert args == ["--max-num-seqs", "256"]
    # Another checkpoint's name leaves the matching recipe alone.
    other = _rtx(
        "vllm",
        QWEN3_8B,
        model_id="Qwen/Qwen3-8B",
        profile="max_throughput",
        skip_checkpoint=FP8_CKPT,
    )
    assert other == kept


def test_swap_check_is_case_insensitive(monkeypatch):
    def app_for(model):
        sub = _make(monkeypatch, model=model)
        monkeypatch.setattr(
            sub, "_gpu_name", staticmethod(lambda: (PRO6000_NAME, 97887))
        )
        app = sub()
        app._boot = {}
        return app

    # Already the recipe's checkpoint, in another case: served as given.
    same = app_for("qwen/qwen3-8b-fp8")
    assert same._model_source() == "qwen/qwen3-8b-fp8"
    assert "model_source" not in same._boot
    # Recipe matching is case-insensitive too, so this one is swapped.
    other = app_for("QWEN/Qwen3-8B")
    assert other._model_source() == FP8_CKPT
    # A pin on the model applies however the repo is spelled.
    seen = []
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda repo, revision=None: seen.append(revision) or "/p"
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(type(same), "persistent_caches", False)
    monkeypatch.setattr(type(same), "model_revision", "a" * 40)
    same._resolve_repo(FP8_CKPT)
    assert seen == ["a" * 40]


def test_pinned_revision_is_never_swapped_for_a_checkpoint(
    monkeypatch, capsys, tmp_path
):
    seen = []
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda repo, revision=None: (
        seen.append((repo, revision)) or "/p"
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    pin = "a" * 40
    app = _checkpoint_app(monkeypatch, model_revision=pin)
    assert app._resolve_model() == "/p"
    assert seen == [("Qwen/Qwen3-8B", pin)]
    assert "model_source" not in app._boot
    assert "pinned" in capsys.readouterr().out
    # The app's pin names the app's own model; another repo resolves unpinned.
    seen.clear()
    app._resolve_repo(FP8_CKPT)
    assert seen == [(FP8_CKPT, None)]


def _snap(root, commit):
    path = root / commit
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"x")
    return str(path)


def test_local_model_dir_is_never_swapped(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Qwen" / "Qwen3-8B").mkdir(parents=True)
    assert _checkpoint_app(monkeypatch)._model_source() == "Qwen/Qwen3-8B"


def test_checkpoint_markers_are_separate_from_the_model(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    bf16, fp8 = _snap(tmp_path, "b" * 40), _snap(tmp_path, "f" * 40)
    # A completed bf16 download is on the volume.
    _cache.write_marker(
        _cache.snapshot_marker_path("Qwen/Qwen3-8B"),
        {"model": "Qwen/Qwen3-8B", "revision": None, "path": bf16},
    )
    seen = []
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda repo, revision=None: seen.append(repo) or fp8
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    # The checkpoint is downloaded, not the bf16 snapshot adopted.
    assert _checkpoint_app(monkeypatch)._resolve_model() == fp8
    assert seen == [FP8_CKPT]
    # The next boot reuses the checkpoint's own marker without a download.
    again = _checkpoint_app(monkeypatch)
    assert again._resolve_model() == fp8
    assert seen == [FP8_CKPT]
    assert again._boot["resolve_source"] == "local"
    assert again._served_commit == "f" * 40


def test_fallback_does_not_keep_the_checkpoint_commit(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    fp8 = _snap(tmp_path, "f" * 40)
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda repo, revision=None: (
        fp8 if repo == FP8_CKPT else "/own"
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    def advance(repo, *args, **kwargs):
        # Fails after the checkpoint download set its commit.
        raise RuntimeError("marker bug")

    monkeypatch.setattr(_app, "advance_snapshot_marker", advance)
    app = _checkpoint_app(monkeypatch)
    assert app._resolve_model() == "/own"
    # /own is not a per-commit dir, so no commit is known for what is served.
    assert app._served_commit is None
    assert app._boot["served_repo"] == "Qwen/Qwen3-8B"


def test_max_throughput_profile_is_valid_and_says_when_unmeasured(capsys):
    class Max(LLMApp):
        model = "m"
        engine_profile = "max_throughput"

    assert Max.engine_profile == "max_throughput"
    _h100(config=QWEN3_8B, model_id="Qwen/Qwen3-8B", profile="max_throughput")
    assert "max_throughput: no measured recipe" in capsys.readouterr().out
    _rtx(config=QWEN3_8B, model_id="Qwen/Qwen3-8B", profile="max_throughput")
    assert "no measured recipe" not in capsys.readouterr().out


def test_app_substituted_checkpoint_gets_no_online_quant(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({**QWEN3_8B, "quantization_config": FP8_QUANT})
    )
    app = _checkpoint_app(monkeypatch, fp8_weights=True)
    cmd = app.engine_command(str(tmp_path))
    assert "--quantization" not in cmd
    assert cmd[cmd.index("--kv-cache-dtype") + 1] == "fp8"


def test_app_one_nvidia_smi_call_with_checkpoint_resolution(monkeypatch):
    calls = {"smi": 0}

    def fake_detect():
        calls["smi"] += 1
        return PRO6000_NAME, 97887

    class Sub(LLMApp):
        model = "Qwen/Qwen3-8B"

    monkeypatch.setattr(_app, "detect_gpu", fake_detect)
    app = Sub()
    app._boot = {}
    assert app._model_source() == FP8_CKPT
    app.engine_command("/m")
    assert calls["smi"] == 1


def test_app_user_engine_args_win_with_checkpoint(monkeypatch):
    app = _checkpoint_app(monkeypatch, engine_args=["--kv-cache-dtype", "auto"])
    assert app._model_source() == FP8_CKPT
    cmd = app.engine_command("/m")
    assert cmd.count("--kv-cache-dtype") == 1
    assert cmd[-2:] == ["--kv-cache-dtype", "auto"]


def test_user_flags_win_over_qwen3_recipe():
    preset = _rtx("vllm", QWEN3_8B, model_id="Qwen/Qwen3-8B")
    merged = merge_engine_args(
        preset, ["--max-num-seqs", "64", "--kv-cache-dtype=auto"]
    )
    assert _flags(merged[:-3]).keys() == {"--speculative-config"}
    assert merged[-3:] == ["--max-num-seqs", "64", "--kv-cache-dtype=auto"]
    merged = merge_engine_args(preset, ["--speculative-config", "{}"])
    assert merged.count("--speculative-config") == 1
    assert merged[-1] == "{}"


def test_user_quantization_wins():
    args = _rtx("vllm", QWEN3_8B, model_id="Qwen/Qwen3-8B", fp8_weights=True)
    merged = merge_engine_args(args, ["--quantization", "awq"])
    assert merged[-2:] == ["--quantization", "awq"]
    assert merged.count("--quantization") == 1
    merged = merge_engine_args(args, ["-q=awq"])
    assert "--quantization" not in merged
    assert merged[-1] == "-q=awq"


@pytest.mark.parametrize(
    "user",
    [
        ["--speculative-config", "{}"],
        ["--speculative-config={}"],
        ["--speculative-model", "x"],
        ["--num-speculative-tokens=2"],
    ],
)
def test_any_user_speculative_flag_suppresses_the_preset_one(user):
    preset = _rtx("vllm", QWEN3_8B, model_id="Qwen/Qwen3-8B")
    merged = merge_engine_args(preset, user)
    assert "--speculative-config" not in merged[: -len(user)]
    assert merged[-len(user) :] == user


# Recipe 4 and the EAGLE3 denylist: gpt-oss-120b.


@pytest.mark.parametrize("run", [_rtx, _h100])
def test_gpt_oss_gets_cap_and_fp8_kv_on_rtx_and_h100(run):
    for profile in ("throughput", "latency"):
        flags = _flags(run("vllm", GPT_OSS_120B, profile=profile, max_multiplexing=32))
        assert flags == {"--max-num-seqs": "256", "--kv-cache-dtype": "fp8"}
    # Matched by architecture; a never-lowered larger cap is kept.
    flags = _flags(run("vllm", GPT_OSS_120B, max_multiplexing=512))
    assert flags == {"--max-num-seqs": "512", "--kv-cache-dtype": "fp8"}


def test_gpt_oss_recipe_is_vllm_only_and_not_for_other_models():
    assert _rtx("sglang", GPT_OSS_120B) == []
    assert "--kv-cache-dtype" not in _flags(_rtx("vllm", QWEN3_8B))
    assert "--kv-cache-dtype" not in _flags(_rtx("vllm", LLAMA_8B))


def test_denylisted_architecture_never_gets_speculative_decoding(monkeypatch):
    spec_recipe = _presets.Recipe(
        source="test",
        gpus=(_presets.RTX_PRO_6000,),
        engines=("vllm",),
        args={"--speculative-config": "{}", "--x": "1"},
    )
    monkeypatch.setattr(_presets, "RECIPES", (*RECIPES, spec_recipe))
    flags = _flags(_rtx("vllm", GPT_OSS_120B))
    assert "--speculative-config" not in flags
    assert flags["--x"] == "1"
    assert "--speculative-config" in _flags(_rtx("vllm", QWEN3_8B))


def test_denylist_warns_once_when_user_sets_speculative_config(capsys):
    _rtx("vllm", GPT_OSS_120B, user_args=["--speculative-config", "{}"])
    out = capsys.readouterr().out
    assert out.count("speculative decoding") == 1
    assert "GptOssForCausalLM" in out
    _rtx("vllm", GPT_OSS_120B, user_args=["--speculative-model=x"])
    assert "speculative decoding" in capsys.readouterr().out
    # No warning without a user flag, or for other architectures.
    _rtx("vllm", GPT_OSS_120B)
    _rtx("vllm", QWEN3_8B, user_args=["--speculative-config", "{}"])
    assert capsys.readouterr().out == ""


def test_engine_command_warns_for_denylisted_user_spec(monkeypatch, tmp_path, capsys):
    (tmp_path / "config.json").write_text(json.dumps(GPT_OSS_120B))
    sub = _make(monkeypatch, engine_args=["--speculative-config", "{}"])
    monkeypatch.setattr(sub, "_gpu_name", staticmethod(lambda: (PRO6000_NAME, 97887)))
    cmd = sub().engine_command(str(tmp_path))
    assert cmd.count("--speculative-config") == 1
    assert cmd[cmd.index("--speculative-config") + 1] == "{}"
    assert capsys.readouterr().out.count("speculative decoding") == 1


# Recipe 6: DeepSeek V4 Flash on 2x and 4x RTX PRO 6000.


def test_deepseek_v4_flash_recipe_on_four_rtx(capsys):
    args = _rtx("vllm", DSV4_FLASH, model_id=DSV4_ID, num_gpus=4)
    assert _flags(args) == {
        "--tensor-parallel-size": "4",
        "--disable-custom-all-reduce": None,
        "--max-num-seqs": "128",
        "--trust-remote-code": None,
        "--safetensors-load-strategy": "prefetch",
        "--kv-cache-dtype": "fp8_ds_mla",
    }
    assert "2 GPUs (num_gpus = 2)" in capsys.readouterr().out
    # --trust-remote-code is a bare switch: no value token follows it.
    i = args.index("--trust-remote-code")
    assert args[i + 1].startswith("--")
    # Matched by lowercase id, with or without a config.
    assert "--trust-remote-code" in _rtx(
        "vllm", None, model_id=DSV4_ID.lower(), num_gpus=4
    )


def test_deepseek_v4_flash_default_layout_on_two_rtx(capsys):
    args = _rtx("vllm", DSV4_FLASH, model_id=DSV4_ID, num_gpus=2)
    assert _flags(args) == {
        "--tensor-parallel-size": "2",
        "--disable-custom-all-reduce": None,
        "--max-num-seqs": "128",
        "--trust-remote-code": None,
        "--safetensors-load-strategy": "prefetch",
        "--kv-cache-dtype": "fp8_ds_mla",
        "--gpu-memory-utilization": "0.95",
    }
    assert "num_gpus = 2" not in capsys.readouterr().out
    # Other multi-GPU counts get the base entry only.
    assert "--kv-cache-dtype" not in _rtx(
        "vllm", DSV4_FLASH, model_id=DSV4_ID, num_gpus=8
    )


@pytest.mark.parametrize(
    "user",
    [
        ["--data-parallel-size", "4", "--enable-expert-parallel"],
        ["--pipeline-parallel-size=4"],
        ["--tensor-parallel-size", "2", "--pipeline-parallel-size", "2"],
    ],
)
def test_user_layout_skips_the_tp_fp8_kv_recipes(user, capsys):
    preset = _rtx("vllm", DSV4_FLASH, model_id=DSV4_ID, num_gpus=4, user_args=user)
    assert "recipes measured with the tensor parallel" in capsys.readouterr().out
    merged = merge_engine_args(preset, user)
    flags = _flags(merged[: -len(user)])
    # FP8 KV was measured with the tensor parallel preset only.
    assert "--kv-cache-dtype" not in flags
    assert "--tensor-parallel-size" not in flags
    assert flags["--max-num-seqs"] == "128"
    assert "--trust-remote-code" in flags
    two = _rtx(
        "vllm", DSV4_FLASH, model_id=DSV4_ID, num_gpus=2, user_args=["--dp", "2"]
    )
    assert "--gpu-memory-utilization" not in two


def test_deepseek_v4_flash_recipe_needs_the_model_gpu_count_and_vllm():
    assert "--trust-remote-code" not in _rtx("vllm", DSV4_FLASH, num_gpus=4)
    assert "--trust-remote-code" not in _rtx("vllm", DSV4_FLASH, model_id=DSV4_ID)
    assert "--trust-remote-code" not in _rtx(
        "sglang", DSV4_FLASH, model_id=DSV4_ID, num_gpus=4
    )
    assert "--trust-remote-code" not in _h100(
        "vllm", DSV4_FLASH, model_id=DSV4_ID, num_gpus=4
    )


@pytest.mark.parametrize(
    "user,absent",
    [
        (["--trust-remote-code"], ["--trust-remote-code"]),
        (["--no-trust-remote-code"], ["--trust-remote-code"]),
        (["--safetensors-load-strategy=eager"], ["--safetensors-load-strategy"]),
        (["--safetensors_load_strategy", "eager"], ["--safetensors-load-strategy"]),
    ],
)
def test_user_wins_over_deepseek_recipe_flags(user, absent):
    preset = _rtx("vllm", DSV4_FLASH, model_id=DSV4_ID, num_gpus=4)
    merged = merge_engine_args(preset, user)
    assert not set(absent) & set(merged[: -len(user)])
    assert merged[-len(user) :] == user
    # The other recipe flag and the cap are untouched.
    kept = _flags(merged[: -len(user)])
    assert kept["--max-num-seqs"] == "128"
    assert kept["--tensor-parallel-size"] == "4"


# Profiles and review fixes.


def test_latency_profile_unknown_model_and_sglang_add_nothing(capsys):
    throughput = _rtx("vllm", LLAMA_8B)
    assert _rtx("vllm", LLAMA_8B, profile="latency", model_id="org/other") == throughput
    assert "no measured recipe" in capsys.readouterr().out
    assert _rtx("sglang", QWEN3_8B, profile="latency", model_id="Qwen/Qwen3-8B") == []
    assert "no measured recipe" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "user", [["--speculative-config", "{}"], ["--speculative-model", "x"]]
)
def test_latency_print_skipped_when_user_set_speculative_config(capsys, user):
    _rtx("vllm", LLAMA_8B, profile="latency", model_id="org/other", user_args=user)
    assert "no measured recipe" not in capsys.readouterr().out


def test_new_recipe_is_a_table_edit(monkeypatch):
    recipe = _presets.Recipe(
        source="test",
        gpus=(_presets.RTX_PRO_6000,),
        engines=("vllm",),
        args={"--x": "1"},
        architectures=frozenset({"LlamaForCausalLM"}),
    )
    monkeypatch.setattr(_presets, "RECIPES", (*RECIPES, recipe))
    assert _flags(_rtx("vllm", LLAMA_8B))["--x"] == "1"
    assert "--x" not in _flags(_rtx("vllm", QWEN3_8B))


def test_invalid_profile_raises():
    with pytest.raises(ValueError, match="engine_profile"):

        class Bad(LLMApp):
            model = "m"
            engine_profile = "fast"

    with pytest.raises(ValueError):
        engine_preset_args(
            "vllm", Machine("H100", None, None, 1), ModelFacts(), profile="fast"
        )


def test_engine_command_reads_config_json(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(GPT_OSS_120B))
    sub = _make(monkeypatch, max_multiplexing=32)
    monkeypatch.setattr(sub, "_gpu_name", staticmethod(lambda: (PRO6000_NAME, 97887)))
    cmd = sub().engine_command(str(tmp_path))
    assert cmd[cmd.index("--max-num-seqs") + 1] == "256"
    assert cmd[cmd.index("--kv-cache-dtype") + 1] == "fp8"
    # No config.json: base behavior.
    cmd = sub().engine_command(str(tmp_path / "missing"))
    assert cmd[cmd.index("--max-num-seqs") + 1] == "32"
    assert "--kv-cache-dtype" not in cmd


@pytest.mark.parametrize("fp8_weights", [False, True])
def test_engine_command_qwen3_8b_recipe_end_to_end(monkeypatch, tmp_path, fp8_weights):
    (tmp_path / "config.json").write_text(json.dumps(QWEN3_8B))
    sub = _make(
        monkeypatch,
        model="Qwen/Qwen3-8B",
        max_multiplexing=256,
        fp8_weights=fp8_weights,
        engine_args=["--max-model-len", "8192"],
    )
    monkeypatch.setattr(sub, "_gpu_name", staticmethod(lambda: (PRO6000_NAME, 97887)))
    cmd = sub().engine_command(str(tmp_path))
    assert cmd[cmd.index("--max-num-seqs") + 1] == "256"
    assert cmd[cmd.index("--kv-cache-dtype") + 1] == "fp8"
    assert '"eagle3"' in cmd[cmd.index("--speculative-config") + 1]
    assert ("--quantization" in cmd) is fp8_weights
    assert cmd[-2:] == ["--max-model-len", "8192"]


def test_one_gpu_detection_per_boot(monkeypatch):
    calls = {"smi": 0, "type": 0}

    def fake_detect():
        calls["smi"] += 1
        return PRO6000_NAME, 97887

    def fake_type():
        calls["type"] += 1
        return "RTX"

    class Sub(LLMApp):
        model = "org/m"
        num_gpus = 2

    monkeypatch.setattr(_app, "detect_gpu", fake_detect)
    monkeypatch.setattr(Sub, "_gpu_type", staticmethod(fake_type))
    cmd = Sub().engine_command("/m")
    assert calls == {"smi": 1, "type": 0}
    # The PCIe-only decision came from the detected name.
    assert "--disable-custom-all-reduce" in cmd


def test_detect_failure_falls_back_to_gpu_type(monkeypatch):
    class Sub(LLMApp):
        model = "org/m"
        num_gpus = 2

    monkeypatch.setattr(_app, "detect_gpu", lambda: None)
    monkeypatch.setattr(Sub, "_gpu_type", staticmethod(lambda: "RTX"))
    assert "--disable-custom-all-reduce" in Sub().engine_command("/m")


def test_detect_gpu_returns_none_without_nvidia_smi(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError

    monkeypatch.setattr(_presets.subprocess, "run", boom)
    assert _presets.detect_gpu() is None


GPT_OSS_WINDOW = {**GPT_OSS_120B, "max_position_embeddings": 131072}


def _flag_value(args, flag):
    return args[args.index(flag) + 1]


def test_agent_profile_vllm_flags_for_gpt_oss():
    args = _rtx(config=GPT_OSS_WINDOW, model_id="openai/gpt-oss-120b", profile="agent")
    assert "--enable-prefix-caching" in args
    assert "--enable-auto-tool-choice" in args
    assert _flag_value(args, "--tool-call-parser") == "openai"
    assert _flag_value(args, "--max-model-len") == "131072"
    # Measured gpt-oss recipe still applies.
    assert _flag_value(args, "--kv-cache-dtype") == "fp8"


@pytest.mark.parametrize(
    "arch,parser",
    [
        ("Qwen3ForCausalLM", "hermes"),
        ("LlamaForCausalLM", "llama3_json"),
    ],
)
def test_agent_profile_tool_parser_per_family(arch, parser):
    config = {**QWEN3_8B, "architectures": [arch], "max_position_embeddings": 32768}
    mid = "meta-llama/Llama-3.1-8B-Instruct" if "Llama" in arch else "Qwen/Qwen3-8B"
    args = _h100(config=config, model_id=mid, profile="agent")
    assert _flag_value(args, "--tool-call-parser") == parser
    # Window never exceeds what the model supports.
    assert _flag_value(args, "--max-model-len") == "32768"


def test_agent_profile_caps_window_at_128k():
    config = {**GPT_OSS_120B, "max_position_embeddings": 1_000_000}
    args = _rtx(config=config, model_id="x/y", profile="agent")
    assert _flag_value(args, "--max-model-len") == "131072"


def test_agent_profile_unknown_family_and_window_say_so(capsys):
    config = {**QWEN3_8B, "architectures": ["FooForCausalLM"]}
    args = _rtx(config=config, model_id="x/y", profile="agent")
    out = capsys.readouterr().out
    assert "no known tool-call parser" in out and "window unknown" in out
    assert "--tool-call-parser" not in args and "--max-model-len" not in args
    assert "--enable-prefix-caching" in args


def test_agent_profile_is_vllm_only_and_other_profiles_untouched():
    assert "--enable-prefix-caching" not in _rtx(
        engine="sglang", config=GPT_OSS_WINDOW, profile="agent"
    )
    plain = _rtx(config=GPT_OSS_WINDOW, model_id="x/y")
    assert "--tool-call-parser" not in plain
    assert "--max-model-len" not in plain


def test_agent_profile_user_flags_win():
    preset = _rtx(config=GPT_OSS_WINDOW, model_id="x/y", profile="agent")
    merged = merge_engine_args(
        preset,
        ["--max-model-len", "65536", "--tool-call-parser", "hermes"]
        + ["--no-enable-prefix-caching"],
    )
    assert merged.count("--max-model-len") == 1
    assert _flag_value(merged, "--max-model-len") == "65536"
    assert _flag_value(merged, "--tool-call-parser") == "hermes"
    assert "--enable-prefix-caching" not in merged


def test_agent_app_defaults_offload_and_hints_and_overrides(monkeypatch):
    monkeypatch.setattr(_app, "_container_ram_bytes", lambda: 100 << 30)

    class Agent(LLMApp):
        model = "org/m"
        engine_profile = "agent"

    monkeypatch.setattr(Agent, "_gpu_type", staticmethod(lambda: PRO6000))
    monkeypatch.setattr(Agent, "_gpu_name", staticmethod(lambda: None))
    cmd = Agent().engine_command("/m")
    cfg = json.loads(cmd[cmd.index("--kv-transfer-config") + 1])
    assert cfg["kv_connector_extra_config"]["cpu_bytes_to_use"] == 70 << 30
    assert Agent()._prefix_hints_on() is True

    class Tuned(Agent):
        kv_cpu_offload = 0.25
        prefix_hints = False

    cmd = Tuned().engine_command("/m")
    cfg = json.loads(cmd[cmd.index("--kv-transfer-config") + 1])
    assert cfg["kv_connector_extra_config"]["cpu_bytes_to_use"] == 25 << 30
    assert Tuned()._prefix_hints_on() is False

    class Off(Agent):
        kv_cpu_offload = 0

    assert "--kv-transfer-config" not in Off().engine_command("/m")

    class NoPresets(Agent):
        engine_presets = False

    assert "--kv-transfer-config" not in NoPresets().engine_command("/m")
    with pytest.raises(NotImplementedError):
        Tuned().provide_hints()
    assert Agent().provide_hints() == []


def test_non_agent_app_keeps_offload_and_hints_off(monkeypatch):
    monkeypatch.setattr(_app, "_container_ram_bytes", lambda: 100 << 30)

    class Plain(LLMApp):
        model = "m"
        engine_presets = False

    assert "--kv-transfer-config" not in Plain().engine_command("/m")
    assert Plain()._prefix_hints_on() is False


@pytest.mark.parametrize(
    "arch,model_id,parser",
    [
        ("Qwen3MoeForCausalLM", "Qwen/Qwen3-Coder-30B-A3B-Instruct", "qwen3_coder"),
        ("Qwen3MoeForCausalLM", "Qwen/Qwen3-30B-A3B", "hermes"),
        ("LlamaForCausalLM", "meta-llama/Llama-2-7b-chat-hf", None),
        ("LlamaForCausalLM", "meta-llama/Meta-Llama-3-8B-Instruct", None),
        ("Qwen3NextForCausalLM", "Qwen/Qwen3-Coder-Next", "qwen3_coder"),
        ("LlamaForCausalLM", "NousResearch/Hermes-3-Llama-3.1-8B", None),
        ("Qwen2ForCausalLM", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", None),
        ("Qwen2ForCausalLM", "Qwen/Qwen2.5-7B-Instruct", "hermes"),
        ("Qwen3_5ForConditionalGeneration", "Qwen/Qwen3.8-27B-FP8", "qwen3_coder"),
        ("Qwen3_5ForConditionalGeneration", "org/merged-27b", None),
    ],
)
def test_agent_tool_parser_same_architecture_exceptions(arch, model_id, parser):
    facts = ModelFacts(model_id, {"architectures": [arch]})
    assert _presets.tool_call_parser(facts) == parser


def test_agent_profile_skips_qwen3_fp8_eagle_recipe():
    args = _rtx(config=QWEN3_8B, model_id="Qwen/Qwen3-8B", profile="agent")
    assert "--speculative-config" not in args


def test_agent_offload_without_cgroup_limit_sizes_from_fal_ram_not_host(
    monkeypatch, capsys
):
    # Inside a fal RTX PRO 6000 runner the cgroup files read empty and
    # /proc/meminfo shows the 1.5 TiB host; the tier must follow the 60 GiB
    # container allotment instead.
    monkeypatch.setattr(_app, "_container_ram_bytes", lambda: 0)

    class Agent(LLMApp):
        model = "org/m"
        engine_profile = "agent"

    monkeypatch.setattr(Agent, "_gpu_type", staticmethod(lambda: PRO6000))
    monkeypatch.setattr(
        Agent,
        "_gpu_name",
        staticmethod(lambda: ("NVIDIA RTX PRO 6000 Blackwell Server Edition", 97887)),
    )
    cmd = Agent().engine_command("/m")
    cfg = json.loads(cmd[cmd.index("--kv-transfer-config") + 1])
    assert cfg["kv_connector_extra_config"]["cpu_bytes_to_use"] == int(
        60 * (1 << 30) * 0.7
    )
    assert "no container memory limit is readable" in capsys.readouterr().out


# Qwen3.8 (Qwen3_5ForConditionalGeneration): multimodal config, text model in
# text_config, served FP8.
QWEN38_27B = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "text_config": {
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "intermediate_size": 17408,
        "vocab_size": 248320,
        "max_position_embeddings": 262144,
    },
    "quantization_config": FP8_QUANT,
}
QWEN38_ID = "Qwen/Qwen3.8-27B-FP8"


def test_agent_profile_qwen38_flags():
    args = _rtx(config=QWEN38_27B, model_id=QWEN38_ID, profile="agent")
    assert _flag_value(args, "--tool-call-parser") == "qwen3_coder"
    assert "--enable-auto-tool-choice" in args
    assert _flag_value(args, "--reasoning-parser") == "qwen3"
    assert "--language-model-only" in args
    kwargs = json.loads(_flag_value(args, "--default-chat-template-kwargs"))
    assert kwargs == {"reasoning_effort": "low"}
    # Window comes from text_config, capped at 128k.
    assert _flag_value(args, "--max-model-len") == "131072"
    assert _flag_value(args, "--gpu-memory-utilization") == "0.92"
    # Already FP8: no online quantization on top.
    assert "--quantization" not in _rtx(
        config=QWEN38_27B, model_id=QWEN38_ID, profile="agent", fp8_weights=True
    )
    assert merge_engine_args(args, args) == args


def test_qwen38_flags_are_agent_and_vllm_only():
    for profile in ("throughput", "latency", "max_throughput"):
        args = _rtx(config=QWEN38_27B, model_id=QWEN38_ID, profile=profile)
        for flag in (
            "--reasoning-parser",
            "--language-model-only",
            "--default-chat-template-kwargs",
            "--gpu-memory-utilization",
        ):
            assert flag not in args, (profile, flag)
    sglang = _rtx("sglang", QWEN38_27B, model_id=QWEN38_ID, profile="agent")
    assert "--reasoning-parser" not in sglang


def test_qwen38_user_flags_win():
    preset = _rtx(config=QWEN38_27B, model_id=QWEN38_ID, profile="agent")
    user = [
        "--default-chat-template-kwargs",
        '{"reasoning_effort": "high"}',
        "--no-language-model-only",
        "--reasoning-parser=deepseek_r1",
    ]
    merged = merge_engine_args(preset, user)
    assert merged.count("--default-chat-template-kwargs") == 1
    assert _flag_value(merged, "--default-chat-template-kwargs") == user[1]
    assert "--language-model-only" not in merged
    assert "--reasoning-parser" not in merged
    assert merged[-len(user) :] == user


@pytest.mark.parametrize("model_id", ["Qwen/Qwen3.8-27B", "qwen/qwen3.8-27b-fp8"])
def test_qwen38_agent_checkpoint_is_fp8_on_one_rtx(model_id):
    def pick(machine=RTX_MACHINE, profile="agent", **kw):
        return _presets.checkpoint_for("vllm", machine, model_id, profile=profile, **kw)

    assert pick() == QWEN38_ID
    assert pick(fp8_weights=False) is None
    assert pick(profile="throughput") is None
    assert pick(machine=Machine("H100", H100_NAME, 81559, 1)) is None
    assert pick(machine=Machine(PRO6000, PRO6000_NAME, 97887, 2)) is None


@pytest.mark.parametrize(
    "model_id", ["org/merged-27b", "someone/Qwen3.8-27B-abliterated"]
)
def test_qwen38_finetune_gets_no_arch_flags(model_id):
    args = _rtx(config=QWEN38_27B, model_id=model_id, profile="agent")
    for flag in (
        "--reasoning-parser",
        "--language-model-only",
        "--default-chat-template-kwargs",
    ):
        assert flag not in args, flag


def test_qwen38_checkpoint_fallback_and_bf16_paths():
    bf16 = {k: v for k, v in QWEN38_27B.items() if k != "quantization_config"}
    # Checkpoint unavailable: the recipe goes, the architecture flags stay.
    args = _rtx(
        config=bf16,
        model_id="Qwen/Qwen3.8-27B",
        profile="agent",
        skip_checkpoint=QWEN38_ID,
    )
    assert "--gpu-memory-utilization" not in args
    assert _flag_value(args, "--reasoning-parser") == "qwen3"
    # bf16 weights with fp8_weights: online FP8 plus the same flags.
    args = _rtx(
        config=bf16, model_id="Qwen/Qwen3.8-27B", profile="agent", fp8_weights=True
    )
    assert _flag_value(args, "--quantization") == "fp8"
    assert _flag_value(args, "--gpu-memory-utilization") == "0.92"
    assert "--language-model-only" in args


GLM53_ID = "zai-org/GLM-5.3"
GLM53_FLASH_ID = "zai-org/GLM-5.3-Flash"
DSV41_ID = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
GLM_CONFIG = {"architectures": ["GlmMoeDsaForCausalLM"], "n_routed_experts": 256}
DSV41_CONFIG = {"architectures": ["DeepseekV4ForCausalLM"], "n_routed_experts": 256}
SPEC_FLAGS = ("--speculative-config", "--speculative-algorithm")


def _blackwell(name, engine, config, model_id, num_gpus, **kwargs):
    machine = Machine(name.split()[1], name, 183359, num_gpus)
    return _run(engine, config, machine, model_id=model_id, **kwargs)


def _b300(engine, config, model_id, num_gpus=8, **kwargs):
    return _blackwell("NVIDIA B300", engine, config, model_id, num_gpus, **kwargs)


def _b200(engine, config, model_id, num_gpus=8, **kwargs):
    return _blackwell("NVIDIA B200", engine, config, model_id, num_gpus, **kwargs)


def _has_spec(args):
    return any(a in args for a in SPEC_FLAGS)


def test_seqs_are_twice_the_concurrency_on_several_gpus():
    assert _base("vllm", "H100", 8, 8) == ["--tensor-parallel-size", "8"] + [
        "--max-num-seqs",
        "16",
    ]
    assert _base("sglang", "H100", 2, 32)[-2:] == ["--max-running-requests", "64"]


def test_seqs_stay_at_the_concurrency_on_one_gpu():
    assert _base("vllm", "H100", 1, 64) == ["--max-num-seqs", "64"]


def test_recipe_min_seqs_own_the_cap_instead_of_doubling():
    args = _b300("sglang", DSV41_CONFIG, DSV41_ID, num_gpus=2, max_multiplexing=8)
    assert _flags(args)["--max-running-requests"] == "64"
    args = _rtx("vllm", DSV4_FLASH, model_id=DSV4_ID, num_gpus=4, max_multiplexing=100)
    assert _flags(args)["--max-num-seqs"] == "128"


def test_no_seqs_warning_at_twice_the_concurrency_but_below_it_warns():
    assert not engine_arg_warnings(["--max-num-seqs", "32"], 8, 16)
    assert not engine_arg_warnings(["--max-num-seqs", "256"], 8, 16)
    (message,) = engine_arg_warnings(["--max-num-seqs", "8"], 8, 16)
    assert "queue requests" in message


def test_glm53_b300_keeps_mtp_up_to_the_limit(capsys):
    args = _b300("vllm", GLM_CONFIG, GLM53_ID, max_multiplexing=8)
    flags = _flags(args)
    assert json.loads(flags["--speculative-config"]) == {
        "method": "mtp",
        "num_speculative_tokens": 3,
    }
    assert flags["--tensor-parallel-size"] == "8"
    assert flags["--tool-call-parser"] == "glm47"
    assert flags["--safetensors-load-strategy"] == "prefetch"
    assert "--no-enable-flashinfer-autotune" in args
    assert "vllm-project/vllm#64580" in capsys.readouterr().out


def test_glm53_b300_drops_mtp_above_the_limit(capsys):
    args = _b300("vllm", GLM_CONFIG, GLM53_ID, max_multiplexing=16)
    assert not _has_spec(args)
    assert "--tool-call-parser" in args
    assert "dropping speculative decoding" in capsys.readouterr().out


def test_spec_max_conc_unknown_concurrency_keeps_spec(capsys):
    args = _b300("vllm", GLM_CONFIG, GLM53_ID)
    assert _has_spec(args)
    assert "max_multiplexing is not set" in capsys.readouterr().out


def test_spec_max_conc_class_value_overrides_the_recipe(capsys):
    kept = _b300("vllm", GLM_CONFIG, GLM53_ID, max_multiplexing=16, spec_max_conc=32)
    assert _has_spec(kept)
    off = _b300("vllm", GLM_CONFIG, GLM53_ID, max_multiplexing=1, spec_max_conc=0)
    assert not _has_spec(off)


def test_spec_max_conc_zero_disables_even_without_max_multiplexing():
    args = _b300("vllm", GLM_CONFIG, GLM53_ID, spec_max_conc=0)
    assert not _has_spec(args)


def test_spec_max_conc_zero_keeps_a_user_speculative_config():
    user = ["--speculative-config", "{}"]
    preset = _b300("vllm", GLM_CONFIG, GLM53_ID, spec_max_conc=0, user_args=user)
    merged = merge_engine_args(preset, user)
    assert merged.count("--speculative-config") == 1
    assert merged[-2:] == user


def test_user_can_re_enable_flashinfer_autotune():
    preset = _b300("vllm", GLM_CONFIG, GLM53_ID)
    assert "--no-enable-flashinfer-autotune" in preset
    merged = merge_engine_args(preset, ["--enable-flashinfer-autotune"])
    assert "--no-enable-flashinfer-autotune" not in merged


def test_class_spec_max_conc_zero_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        class A(LLMApp):
            model = "m"
            max_multiplexing = 16
            spec_max_conc = 0


def test_gb200_is_not_b200():
    assert classify_gpu("NVIDIA GB200", 190000) == "NVIDIA_GB200"
    assert classify_gpu("NVIDIA GB300", 280000) == "NVIDIA_GB300"


def test_dsv41_recipe_is_skipped_above_its_measured_concurrency(capsys):
    args = _b300("sglang", DSV41_CONFIG, DSV41_ID, 2, max_multiplexing=48)
    assert "--dsv4-attn-backend" not in args
    assert not _has_spec(args)
    assert "skipped" in capsys.readouterr().out
    env = engine_preset_env(
        "sglang",
        Machine("B300", "NVIDIA B300", 288000, 2),
        model_facts(DSV41_ID, DSV41_CONFIG),
        max_multiplexing=48,
    )
    assert env == {}


def test_flash_mtp_is_gated_at_32():
    kept = _b200("vllm", GLM_CONFIG, GLM53_FLASH_ID, max_multiplexing=32)
    assert _has_spec(kept)
    assert not _has_spec(_b200("vllm", GLM_CONFIG, GLM53_FLASH_ID, max_multiplexing=33))


def test_user_speculative_config_wins_but_warns_over_the_limit(capsys):
    user = ["--speculative-config", '{"method":"mtp","num_speculative_tokens":1}']
    preset = _b300("vllm", GLM_CONFIG, GLM53_ID, max_multiplexing=16, user_args=user)
    assert "your --speculative-config is passed as given" in capsys.readouterr().out
    merged = merge_engine_args(preset, user)
    assert merged.count("--speculative-config") == 1
    assert merged[-2:] == user


def test_spec_max_conc_without_a_gate_leaves_single_gpu_recipes_alone():
    args = _rtx(
        "vllm",
        QWEN3_8B,
        model_id="Qwen/Qwen3-8B",
        fp8_weights=True,
        max_multiplexing=256,
    )
    assert json.loads(_flags(args)["--speculative-config"]) == EAGLE3_QWEN3_8B


def test_spec_denylist_still_wins_over_a_recipe_gate():
    args = _h100("vllm", GPT_OSS_120B, max_multiplexing=8, spec_max_conc=100)
    assert not _has_spec(args)


def test_class_level_spec_max_conc_warns_at_class_creation():
    with pytest.warns(UserWarning, match="spec_max_conc 8"):

        class A(LLMApp):
            model = "m"
            max_multiplexing = 16
            spec_max_conc = 8

    with pytest.warns(UserWarning, match="your --speculative-config is kept"):

        class B(LLMApp):
            model = "m"
            max_multiplexing = 16
            spec_max_conc = 8
            engine_args = ["--speculative-config", "{}"]

    with warnings.catch_warnings():
        warnings.simplefilter("error")

        class C(LLMApp):
            model = "m"
            max_multiplexing = 8
            spec_max_conc = 8


def test_glm53_b200_has_no_mtp_and_wrong_shapes_do_not_match():
    args = _b200("vllm", GLM_CONFIG, GLM53_ID, max_multiplexing=4)
    assert not _has_spec(args)
    assert _flags(args)["--tool-call-parser"] == "glm47"
    # Four GPUs, another model, or SGLang: no recipe.
    assert "--tool-call-parser" not in _b300("vllm", GLM_CONFIG, GLM53_ID, 4)
    assert "--tool-call-parser" not in _b300("vllm", GLM_CONFIG, "other/model")
    assert "--tool-call-parser" not in _b300("sglang", GLM_CONFIG, GLM53_ID)


def test_glm53_flash_b200_vllm_mtp3():
    args = _b200("vllm", GLM_CONFIG, GLM53_FLASH_ID, max_multiplexing=16)
    flags = _flags(args)
    assert flags["--tokenizer-mode"] == "glm5_next"
    assert flags["--reasoning-parser"] == "glm47"
    assert json.loads(flags["--speculative-config"])["method"] == "mtp"
    assert not _has_spec(
        _b200("vllm", GLM_CONFIG, GLM53_FLASH_ID, max_multiplexing=16, spec_max_conc=0)
    )


def test_dsv41_flash_b300_sglang_tp2_dspark():
    args = _b300("sglang", DSV41_CONFIG, DSV41_ID, num_gpus=2, max_multiplexing=32)
    flags = _flags(args)
    assert flags["--tp-size"] == "2"
    assert flags["--speculative-algorithm"] == "DSPARK"
    assert flags["--speculative-dspark-block-size"] == "5"
    assert flags["--dsv4-attn-backend"] == "trtllm"
    assert "--enable-decoder-swa-bounded-replay" in args
    # A user algorithm replaces every preset speculative flag.
    user = ["--speculative-algorithm", "EAGLE"]
    merged = merge_engine_args(args, user)
    assert "--speculative-dspark-block-size" not in merged
    # Four GPUs (DEP4) is not expressible: no recipe.
    assert "--dsv4-attn-backend" not in _b300("sglang", DSV41_CONFIG, DSV41_ID, 4)


def test_recipe_env_for_matching_machines_only():
    def env(engine, name, model_id, num_gpus):
        machine = Machine(name.split()[1], name, 183359, num_gpus)
        return engine_preset_env(
            engine, machine, model_facts(model_id, GLM_CONFIG), user_args=()
        )

    assert (
        env("vllm", "NVIDIA B300", GLM53_ID, 8)[
            "VLLM_SPARSE_INDEXER_MAX_DECODE_LOGITS_MB"
        ]
        == "0"
    )
    assert env("vllm", "NVIDIA B200", GLM53_ID, 8) == {}
    sglang = env("sglang", "NVIDIA B300", DSV41_ID, 2)
    assert sglang["SGLANG_DEFAULT_THINKING"] == "1"
    assert sglang["NCCL_NVLS_ENABLE"] == "0"


def test_agent_profile_does_not_duplicate_the_recipe_tool_parser(capsys):
    args = _b300("vllm", GLM_CONFIG, GLM53_ID, max_multiplexing=8, profile="agent")
    assert args.count("--tool-call-parser") == 1
    assert args.count("--enable-auto-tool-choice") == 1
    assert "no known tool-call parser" not in capsys.readouterr().out
