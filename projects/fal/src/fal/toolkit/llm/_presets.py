import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

# Preset flag per engine. Every flag here takes a value except the boolean
# switches (see _BOOLEAN_FLAGS), which a recipe lists with an empty value.
_TP_FLAG = {"vllm": "--tensor-parallel-size", "sglang": "--tp-size"}
_MAX_SEQS_FLAG = {"vllm": "--max-num-seqs", "sglang": "--max-running-requests"}
_KV_DTYPE_FLAG = "--kv-cache-dtype"
_SPEC_FLAG = "--speculative-config"
_QUANT_FLAG = "--quantization"
_ALL_REDUCE_FLAG = "--disable-custom-all-reduce"
_TRUST_REMOTE_CODE_FLAG = "--trust-remote-code"
_LOAD_STRATEGY_FLAG = "--safetensors-load-strategy"
_GPU_MEM_FLAG = "--gpu-memory-utilization"
_PROMPT_DETAILS_FLAG = "--enable-prompt-tokens-details"
_PREFIX_CACHING_FLAG = "--enable-prefix-caching"
_AUTO_TOOL_FLAG = "--enable-auto-tool-choice"
_TOOL_PARSER_FLAG = "--tool-call-parser"
_MAX_MODEL_LEN_FLAG = "--max-model-len"
_REASONING_PARSER_FLAG = "--reasoning-parser"
_LANGUAGE_ONLY_FLAG = "--language-model-only"
_TEMPLATE_KWARGS_FLAG = "--default-chat-template-kwargs"
_BOOLEAN_FLAGS = frozenset(
    {
        _ALL_REDUCE_FLAG,
        _TRUST_REMOTE_CODE_FLAG,
        _PROMPT_DETAILS_FLAG,
        _PREFIX_CACHING_FLAG,
        _AUTO_TOOL_FLAG,
        _LANGUAGE_ONLY_FLAG,
    }
)

# Spellings that mean the same thing to an engine, keyed by a canonical group.
_FLAG_GROUPS: Dict[str, str] = {
    "--tensor-parallel-size": "tp",
    "--tp-size": "tp",
    "--tp": "tp",
    "-tp": "tp",
    "--max-num-seqs": "max_seqs",
    "--max-running-requests": "max_seqs",
    _KV_DTYPE_FLAG: "kv_dtype",
    _SPEC_FLAG: "spec",
    "--speculative-model": "spec",
    "--num-speculative-tokens": "spec",
    _QUANT_FLAG: "quant",
    "-q": "quant",
    _ALL_REDUCE_FLAG: "all_reduce",
    _TRUST_REMOTE_CODE_FLAG: "trust_remote_code",
    _PROMPT_DETAILS_FLAG: "prompt_details",
    _LOAD_STRATEGY_FLAG: "load_strategy",
    _GPU_MEM_FLAG: "gpu_mem",
    _PREFIX_CACHING_FLAG: "prefix_caching",
    _AUTO_TOOL_FLAG: "auto_tool",
    _TOOL_PARSER_FLAG: "tool_parser",
    _MAX_MODEL_LEN_FLAG: "max_model_len",
    _REASONING_PARSER_FLAG: "reasoning_parser",
    _LANGUAGE_ONLY_FLAG: "language_only",
    _TEMPLATE_KWARGS_FLAG: "template_kwargs",
    "--pipeline-parallel-size": "pp",
    "--pp-size": "pp",
    "-pp": "pp",
    "--data-parallel-size": "dp",
    "--dp-size": "dp",
    "--dp": "dp",
    "-dp": "dp",
}
# A user split across pipeline or data parallel ranks owns the GPU layout, so
# the tensor parallel preset (all GPUs) is dropped.
_LAYOUT_GROUPS = frozenset({"tp", "pp", "dp"})

# GPUs without NVLink between cards (PCIe only), matched against the name
# `fal.toolkit.compilation.get_gpu_type()` returns. That function keeps only
# the first word after "NVIDIA ", so RTX PRO 6000, RTX 4090 and RTX 5090 all
# arrive as "RTX" and GeForce boards as "GeForce". A100 must not match A10,
# hence exact names for the short ones.
_NO_NVLINK_PREFIXES = ("RTX", "GEFORCE", "L40")
_NO_NVLINK_EXACT = frozenset({"A10", "A10G", "L4"})

RTX_PRO_6000 = "RTX_PRO_6000"
H100 = "H100"
# Architectures whose EAGLE3 drafts crash vLLM 0.32 at startup.
_SPEC_DENYLIST = frozenset({"GptOssForCausalLM"})
# max_throughput trades quality for speed and is opt-in; models without a
# max_throughput recipe get the throughput one.
# agent: long multi-turn tool-calling sessions (coding agents). Adds prefix
# caching, the tool-call parser and a 128k window here; LLMApp also turns on
# the CPU KV tier (0.7) and prefix_hints unless the app sets them.
PROFILES = ("throughput", "latency", "max_throughput", "agent")
# The Qwen3-8B FP8 + EAGLE3 recipes are not measured with long context, prefix
# caching and the CPU KV tier, so the agent profile does not take them.
_SAFE_PROFILES = ("throughput", "latency")
_ENGINES = tuple(_TP_FLAG)

AGENT_PROFILE = "agent"
AGENT_MAX_MODEL_LEN = 131072
AGENT_KV_CPU_OFFLOAD = 0.7
# vLLM --tool-call-parser per model family, matched on architecture. Families
# not listed get no parser (a note is printed) rather than a wrong one.
# Qwen3.8 dense; other versions on this architecture are untested.
_QWEN3_5 = "Qwen3_5ForConditionalGeneration"
_QWEN_ARCHS = frozenset(
    {
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3NextForCausalLM",
        _QWEN3_5,
    }
)
_TOOL_PARSERS: Tuple[Tuple[FrozenSet[str], str], ...] = (
    (frozenset({"GptOssForCausalLM"}), "openai"),
    # Qwen3.5 and later chat templates emit <tool_call><function=...> XML.
    (frozenset({_QWEN3_5}), "qwen3_coder"),
    (
        frozenset(
            {
                "Qwen2ForCausalLM",
                "Qwen3ForCausalLM",
                "Qwen3MoeForCausalLM",
                "Qwen3NextForCausalLM",
            }
        ),
        "hermes",
    ),
)
# More agent profile flags per architecture, for the publisher's own models
# only (Hub id prefix), since a fine-tune may change the chat template or
# rely on vision. Qwen3.8 thinks at xhigh effort unless asked otherwise; low
# effort halved the thinking tokens on the longest agent task at the same
# pass rate. omp never sends reasoning_effort, so the server default is low
# (a request can still raise it). The checkpoint is multimodal: the agent
# profile serves text only.
_AGENT_ARCH_ARGS: Tuple[Tuple[FrozenSet[str], str, Mapping[str, str]], ...] = (
    (
        frozenset({_QWEN3_5}),
        "qwen/",
        {
            _REASONING_PARSER_FLAG: "qwen3",
            _LANGUAGE_ONLY_FLAG: "",
            _TEMPLATE_KWARGS_FLAG: json.dumps({"reasoning_effort": "low"}),
        },
    ),
)


def _normalize_gpu(gpu_type: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", gpu_type.upper())


def _is_pcie_only(gpu_type: str, gpu_name: Optional[str] = None) -> bool:
    # The full nvidia-smi name is more precise than get_gpu_type()'s first word.
    name = _normalize_gpu(gpu_name or gpu_type)
    name = name.removeprefix("NVIDIA")
    return name in _NO_NVLINK_EXACT or name.startswith(_NO_NVLINK_PREFIXES)


def classify_gpu(name: str, memory_mib: Optional[int] = None) -> str:
    """Stable key for a GPU from its nvidia-smi name and memory (MiB).

    RTX PRO 6000 (Blackwell, SM120, ~96 GB) gets ``"RTX_PRO_6000"`` and every
    H100 variant gets ``"H100"``. Anything else returns its normalized name, so
    GeForce RTX 5090 (32 GB) and other RTX cards never collide with them.
    """
    upper = name.upper()
    if "RTX PRO 6000" in upper and (memory_mib is None or memory_mib >= 80 * 1024):
        return RTX_PRO_6000
    if "H100" in upper:
        return H100
    return re.sub(r"[^A-Z0-9]+", "_", upper).strip("_") or "UNKNOWN"


def short_gpu_type(name: str) -> str:
    """What ``get_gpu_type()`` returns for an nvidia-smi name (first word)."""
    match = re.search(r"NVIDIA [a-zA-Z0-9]*", name)
    return match.group(0)[7:] if match else "UNKNOWN"


def detect_gpu() -> Optional[Tuple[str, Optional[int]]]:
    """Name and memory (MiB) of the first GPU from nvidia-smi, None on failure."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        ).stdout
        line = out.strip().splitlines()[0]
        name, _, memory = line.rpartition(",")
        try:
            memory_mib: Optional[int] = int(float(memory))
        except ValueError:
            memory_mib = None
        return (name.strip(), memory_mib) if name.strip() else None
    except Exception:
        return None


def load_model_config(model_path: str) -> Optional[Dict[str, Any]]:
    """``config.json`` of a local model directory, None if absent or unreadable."""
    try:
        with open(os.path.join(model_path, "config.json")) as f:
            config = json.load(f)
        return config if isinstance(config, dict) else None
    except Exception:
        return None


@dataclass(frozen=True)
class Machine:
    """The hardware an engine will run on."""

    gpu_type: str  # get_gpu_type() spelling, e.g. "H100" or "RTX"
    gpu_name: Optional[str]  # full nvidia-smi name
    gpu_memory_mib: Optional[int]
    num_gpus: int

    @property
    def gpu_key(self) -> Optional[str]:
        if self.gpu_name is None:
            return None
        return classify_gpu(self.gpu_name, self.gpu_memory_mib)


@dataclass(frozen=True)
class ModelFacts:
    """What the presets know about the model; unknown values are ``None``."""

    model_id: Optional[str] = None
    config: Optional[Mapping[str, Any]] = None
    is_moe: Optional[bool] = None
    active_params_b: Optional[float] = None
    prequantized: bool = False

    @property
    def architectures(self) -> FrozenSet[str]:
        archs = (self.config or {}).get("architectures")
        return frozenset(archs) if isinstance(archs, list) else frozenset()


def _num(config: Mapping[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = config.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _text_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = config.get("text_config")
    if not isinstance(config.get("hidden_size"), (int, float)) and isinstance(
        nested, Mapping
    ):
        return nested
    return config


def _estimate_active_params(
    config: Mapping[str, Any],
) -> Tuple[Optional[bool], Optional[float]]:
    """``(is_moe, active_params_b)`` from a Hugging Face ``config.json``.

    Active parameters per token, in billions: attention projections, the
    dense MLP or the routed experts per token (plus shared experts), and input
    plus untied output embeddings (an upper bound for tied models). Layer norms
    and bias terms are ignored. ``None`` means the config lacks what is needed.
    """
    config = _text_config(config)
    experts = _num(config, "num_local_experts", "num_experts", "n_routed_experts")
    if experts is not None:
        is_moe: Optional[bool] = experts > 0
    else:
        is_moe = False if _num(config, "hidden_size") else None

    hidden = _num(config, "hidden_size")
    layers = _num(config, "num_hidden_layers")
    vocab = _num(config, "vocab_size")
    heads = _num(config, "num_attention_heads")
    if not (hidden and layers and vocab and heads):
        return is_moe, None
    kv_heads = _num(config, "num_key_value_heads") or heads
    head_dim = _num(config, "head_dim") or hidden / heads
    attn = hidden * head_dim * (2 * heads + 2 * kv_heads)
    if is_moe:
        ffn = _num(config, "moe_intermediate_size", "intermediate_size")
        per_token = _num(config, "num_experts_per_tok", "experts_per_token")
        if ffn is None or per_token is None:
            return is_moe, None
        shared = _num(config, "n_shared_experts", "num_shared_experts") or 0
        mlp = 3 * hidden * ffn * (per_token + shared)
    else:
        ffn = _num(config, "intermediate_size")
        if ffn is None:
            return is_moe, None
        mlp = 3 * hidden * ffn
    total = layers * (attn + mlp) + 2 * vocab * hidden
    return is_moe, total / 1e9


def _is_prequantized(config: Mapping[str, Any]) -> bool:
    # Any method (fp8, awq, gptq, mxfp4): adding --quantization fp8 on top of
    # a quantized checkpoint fails at launch.
    nested = config.get("text_config")
    for source in (config, nested if isinstance(nested, Mapping) else {}):
        if isinstance(source.get("quantization_config"), Mapping):
            return True
    return False


def model_facts(
    model_id: Optional[str],
    config: Optional[Mapping[str, Any]],
    active_params_b: Optional[float] = None,
) -> ModelFacts:
    """Facts from ``config.json``; ``active_params_b`` overrides the estimate."""
    if not config:
        return ModelFacts(model_id, None, None, active_params_b, False)
    is_moe, estimate = _estimate_active_params(config)
    active = active_params_b if active_params_b is not None else estimate
    return ModelFacts(model_id, config, is_moe, active, _is_prequantized(config))


@dataclass(frozen=True)
class Recipe:
    """A measured engine setting. Every set field must match for it to apply.

    ``args`` is added for each engine in ``engines`` (flag to value, empty value
    for a boolean switch). ``min_seqs`` raises the max concurrent sequences to
    at least that value. ``checkpoint`` is a Hub repo downloaded and served in
    place of the user's model (clients keep the user's name). A checkpoint
    recipe is picked before the download, so it matches on model id, GPU,
    engine, GPU count and profile only. ``fp8_weights`` marks the FP8 weights
    default, which the app's ``fp8_weights = False`` opts out of.
    """

    source: str  # what was measured, for the next reader
    gpus: Tuple[str, ...]  # Machine.gpu_key values
    engines: Tuple[str, ...]
    args: Mapping[str, str] = field(default_factory=dict)
    min_seqs: Optional[int] = None
    single_gpu: Optional[bool] = None  # True: 1 GPU, False: tensor parallel
    architectures: FrozenSet[str] = frozenset()  # any of; empty matches all
    model_ids: FrozenSet[str] = frozenset()  # lowercase Hub ids; empty matches all
    moe: Optional[bool] = None
    max_active_b: Optional[float] = None  # needs known active parameters
    profiles: Tuple[str, ...] = PROFILES
    checkpoint: Optional[str] = None
    fp8_weights: bool = False
    gpu_counts: Tuple[int, ...] = ()  # exact GPU counts; empty matches all
    # Only when the presets own the GPU layout (no user tp, pp or dp).
    preset_layout: bool = False
    note: Optional[str] = None  # printed at boot when the recipe applies

    def matches(
        self,
        engine: str,
        machine: Machine,
        model: ModelFacts,
        profile: str,
    ) -> bool:
        if (
            machine.gpu_key not in self.gpus
            or engine not in self.engines
            or profile not in self.profiles
            or (
                self.single_gpu is not None
                and self.single_gpu != (machine.num_gpus == 1)
            )
            or (self.gpu_counts and machine.num_gpus not in self.gpu_counts)
        ):
            return False
        if self.architectures and not self.architectures & model.architectures:
            return False
        if self.model_ids and (model.model_id or "").lower() not in self.model_ids:
            return False
        if self.moe is not None and model.is_moe is not self.moe:
            return False
        if self.max_active_b is not None and not (
            model.active_params_b is not None
            and model.active_params_b < self.max_active_b
        ):
            return False
        return True


# vLLM downloads the EAGLE3 draft at engine start into HF_HOME (on /data with
# persistent caches): the first boot fetches it, later boots reuse it. It needs
# a vLLM version with EAGLE3 speculator support.
_EAGLE3_QWEN3_8B = {
    _SPEC_FLAG: json.dumps(
        {
            "method": "eagle3",
            "model": "RedHatAI/Qwen3-8B-speculator.eagle3",
            "num_speculative_tokens": 3,
        }
    )
}

_RTX_AND_H100 = (RTX_PRO_6000, H100)
_DSV4_FLASH = frozenset({"deepseek-ai/deepseek-v4-flash-0731"})

# Measured recipes (all vLLM v0.32.0, 2k prompt / 256 output, via fal.run). To
# add one, append an entry; flags are merged like any preset, so user
# engine_args win. --max-num-batched-tokens stays at the vLLM default: 16384
# gave no gain.
RECIPES: Tuple[Recipe, ...] = (
    Recipe(
        source=(
            "vLLM 0.32.0, 1x RTX PRO 6000 and 1x H100, Qwen3-8B and gpt-oss-120b: "
            "a cap of 64 left 1.6x to 2x throughput unused, 256 uses it"
        ),
        gpus=_RTX_AND_H100,
        engines=("vllm",),
        min_seqs=256,
        single_gpu=True,
        max_active_b=40.0,
    ),
    Recipe(
        source=(
            "vLLM 0.32.0, 4x RTX PRO 6000, DeepSeek V4 Flash: a cap of 64 dips "
            "throughput at 32 streams, 128 does not (64 left 1.6x to 2x unused "
            "on every model tested)"
        ),
        gpus=(RTX_PRO_6000,),
        engines=("vllm",),
        min_seqs=128,
        single_gpu=False,
        moe=True,
    ),
    # Measured with the pre-quantized FP8 checkpoint, which is served in place
    # of Qwen/Qwen3-8B; with fp8_weights = False the same flags apply to bf16
    # weights (not remeasured).
    Recipe(
        source=(
            "vLLM 0.32.0, 1x RTX PRO 6000, Qwen3-8B-FP8: FP8 KV + cap 256 + "
            "EAGLE3 (3 tokens) peaked at 4,206 tok/s and 208 tok/s single "
            "stream; quality within run noise (MMLU-Pro -2.2 pts) vs bf16, "
            "ai-designs/2026-11-05-rtx-quant-quality.md"
        ),
        gpus=(RTX_PRO_6000,),
        engines=("vllm",),
        args={_KV_DTYPE_FLAG: "fp8", **_EAGLE3_QWEN3_8B},
        min_seqs=256,
        single_gpu=True,
        model_ids=frozenset({"qwen/qwen3-8b", "qwen/qwen3-8b-fp8"}),
        profiles=_SAFE_PROFILES,
        checkpoint="Qwen/Qwen3-8B-FP8",
        fp8_weights=True,
    ),
    # Opt-in: the NVFP4 checkpoint loses 8.2 pts on MMLU-Pro (GSM8K and IFEval
    # hold) for about 1.4x the FP8 peak.
    Recipe(
        source=(
            "vLLM 0.32.0, 1x RTX PRO 6000, Qwen3-8B NVFP4: FP8 KV + EAGLE3 + cap "
            "512 + gpu-memory-utilization 0.95 peaked at 6,067 tok/s; MMLU-Pro "
            "-8.2 pts vs bf16, outside run noise"
        ),
        gpus=(RTX_PRO_6000,),
        engines=("vllm",),
        args={
            _KV_DTYPE_FLAG: "fp8",
            _GPU_MEM_FLAG: "0.95",
            **_EAGLE3_QWEN3_8B,
        },
        min_seqs=512,
        single_gpu=True,
        model_ids=frozenset({"qwen/qwen3-8b"}),
        profiles=("max_throughput",),
        checkpoint="nvidia/Qwen3-8B-NVFP4",
    ),
    Recipe(
        source=(
            "vLLM 0.32.0, 1x RTX PRO 6000, Qwen3.8-27B-FP8, agent profile with "
            "low reasoning effort: 8 of 8 omp coding tasks in 2 runs, slowest "
            "task 206s vs 459s at the default effort (median unchanged), "
            "ai-designs/2026-11-05-omp-open-model-shootout.md"
        ),
        gpus=(RTX_PRO_6000,),
        engines=("vllm",),
        args={_GPU_MEM_FLAG: "0.92"},
        single_gpu=True,
        model_ids=frozenset({"qwen/qwen3.8-27b", "qwen/qwen3.8-27b-fp8"}),
        profiles=(AGENT_PROFILE,),
        checkpoint="Qwen/Qwen3.8-27B-FP8",
        fp8_weights=True,
    ),
    # gpt-oss on SM120 runs the Marlin W4A16 MoE path. The native b12x backend
    # crashes on expert biases, so never force a b12x backend here.
    Recipe(
        source=(
            "vLLM 0.32.0, 1x RTX PRO 6000 and 1x H100, gpt-oss-120b: cap 256 + FP8 "
            "KV cache; FP8 KV is required on H100 (bf16 KV preempts at high "
            "load) and quality-safe (no measurable loss on RTX PRO 6000). No "
            "EAGLE3: both NVIDIA drafts crash vLLM 0.32 at startup"
        ),
        gpus=_RTX_AND_H100,
        engines=("vllm",),
        args={_KV_DTYPE_FLAG: "fp8"},
        min_seqs=256,
        single_gpu=True,
        architectures=frozenset({"GptOssForCausalLM"}),
    ),
    Recipe(
        source=(
            "vLLM 0.32.0, 2x and 4x RTX PRO 6000, DeepSeek V4 Flash 0731: cap 128 "
            "+ --disable-custom-all-reduce (base) + --trust-remote-code + "
            "prefetch weight loading"
        ),
        gpus=(RTX_PRO_6000,),
        engines=("vllm",),
        args={_TRUST_REMOTE_CODE_FLAG: "", _LOAD_STRATEGY_FLAG: "prefetch"},
        min_seqs=128,
        single_gpu=False,
        model_ids=_DSV4_FLASH,
    ),
    # Layouts measured on PCIe (2k prompt / 256 out, tokens per GPU-hour at 128
    # streams): TP2 1.72M, DP4 + EP 1.24M, TP4 0.92M, PP4 0.87M. DP4 + EP has
    # the best 4-GPU peak (1,382 tok/s) but drops to 348 at 32 streams, so it
    # only suits steady high load (engine_args --data-parallel-size 4
    # --enable-expert-parallel). The FP8 KV entries assume the tensor
    # parallel preset, so a user layout skips them.
    Recipe(
        source=(
            "vLLM 0.32.0, 2x RTX PRO 6000, DeepSeek V4 Flash 0731: TP2 + FP8 KV "
            "(fp8_ds_mla) + gpu-memory-utilization 0.95 + cap 128: 955 tok/s at "
            "128 streams, 80 tok/s single stream, 1.72M tok/GPU-h vs 0.92M on TP4"
        ),
        gpus=(RTX_PRO_6000,),
        engines=("vllm",),
        args={_KV_DTYPE_FLAG: "fp8_ds_mla", _GPU_MEM_FLAG: "0.95"},
        gpu_counts=(2,),
        model_ids=_DSV4_FLASH,
        preset_layout=True,
    ),
    Recipe(
        source=(
            "vLLM 0.32.0, 4x RTX PRO 6000, DeepSeek V4 Flash 0731: TP4 + FP8 KV "
            "(fp8_ds_mla): 1,021 tok/s at 128 streams, TTFT p50 at 64 streams "
            "3.5s vs 7.0s without"
        ),
        gpus=(RTX_PRO_6000,),
        engines=("vllm",),
        args={_KV_DTYPE_FLAG: "fp8_ds_mla"},
        gpu_counts=(4,),
        model_ids=_DSV4_FLASH,
        preset_layout=True,
        note=(
            "DeepSeek V4 Flash: 2 GPUs (num_gpus = 2) measured 1.9x more tokens "
            "per GPU-hour than 4, at the same single-stream speed"
        ),
    ),
)


def checkpoint_for(
    engine: str,
    machine: Machine,
    model_id: str,
    *,
    profile: str = "throughput",
    fp8_weights: Optional[bool] = None,
) -> Optional[str]:
    """Hub repo to download and serve instead of ``model_id``, or None.

    Runs before the download, so only recipes that match on model id, GPU,
    engine, GPU count and profile can apply. ``fp8_weights=False`` skips the
    FP8 weights default.
    """
    facts = ModelFacts(model_id)
    for recipe in RECIPES:
        if (
            recipe.checkpoint
            and not (recipe.fp8_weights and fp8_weights is False)
            and recipe.matches(engine, machine, facts, profile)
        ):
            return recipe.checkpoint
    return None


def _has_flag(args: Sequence[str], group: str) -> bool:
    return any(_flag_group(token) == group for token in args)


def engine_preset_args(
    engine: str,
    machine: Machine,
    model: ModelFacts,
    *,
    profile: str = "throughput",
    fp8_weights: bool = False,
    max_multiplexing: Optional[int] = None,
    user_args: Sequence[str] = (),
    skip_checkpoint: Optional[str] = None,
) -> List[str]:
    """Engine arguments that match the machine, for ``vllm`` or ``sglang``.

    Generic base:

    - ``machine.num_gpus > 1``: tensor parallel across all GPUs
      (``--tensor-parallel-size`` for vLLM, ``--tp-size`` for SGLang), plus
      ``--disable-custom-all-reduce`` on PCIe-only GPUs (RTX PRO 6000, RTX
      4090/5090, L40/L40S, L4, A10), since the custom all-reduce kernel needs
      NVLink.
    - ``max_multiplexing``: ``--max-num-seqs`` (vLLM) or
      ``--max-running-requests`` (SGLang), matching the router's concurrency.
      A recipe may raise the engine cap above it, never lower it.
    - vLLM: ``--enable-prompt-tokens-details``, so ``usage`` carries
      ``prompt_tokens_details.cached_tokens``.
    - ``profile="agent"`` (vLLM): ``--enable-prefix-caching``, the tool-call
      parser for the model family (``--enable-auto-tool-choice``,
      ``--tool-call-parser``), ``--max-model-len`` of the model's window up
      to 128k, and per-architecture flags (Qwen3.8: reasoning parser, text
      only, low default reasoning effort).
    - ``fp8_weights``: online ``--quantization fp8`` (any GPU) unless the
      checkpoint is already quantized (any ``quantization_config``, which
      includes a checkpoint substituted by ``checkpoint_for``).

    Measured recipes (see ``RECIPES``) add more for specific GPU, engine, model
    and profile combinations. Speculative decoding is never added for an
    architecture in ``_SPEC_DENYLIST``. ``user_args`` (the app's
    ``engine_args``) drives messages about the presets and, when it sets a
    tensor, pipeline or data parallel size, skips recipes that assume the
    tensor parallel preset.
    ``skip_checkpoint`` drops the recipes tuned for that checkpoint, for when
    it could not be served and the original weights are used instead.

    Use ``merge_engine_args`` to put user arguments on top.
    """
    if engine not in _TP_FLAG:
        raise ValueError(f"unknown engine {engine!r}; expected 'vllm' or 'sglang'")
    if profile not in PROFILES:
        raise ValueError(f"unknown engine_profile {profile!r}; expected {PROFILES}")
    skipped = skip_checkpoint.lower() if skip_checkpoint else None
    user_layout = any(_has_flag(user_args, g) for g in _LAYOUT_GROUPS)
    recipes = [
        r
        for r in RECIPES
        if r.matches(engine, machine, model, profile)
        and not (skipped and r.checkpoint and r.checkpoint.lower() == skipped)
    ]
    if user_layout and any(r.preset_layout for r in recipes):
        print(
            "[llm-app] engine_args set the parallel layout, so recipes measured "
            "with the tensor parallel preset (e.g. FP8 KV cache) are skipped"
        )
        recipes = [r for r in recipes if not r.preset_layout]
    for recipe in recipes:
        if recipe.note:
            print(f"[llm-app] {recipe.note}")
    spec_denied = model.architectures & _SPEC_DENYLIST
    user_spec = _has_flag(user_args, "spec")
    if spec_denied and user_spec:
        print(
            f"[llm-app] speculative decoding on {sorted(spec_denied)} crashes "
            "vLLM 0.32 at startup; your --speculative-config is passed as given"
        )
    seqs = max_multiplexing
    for recipe in recipes:
        if recipe.min_seqs is not None:
            seqs = max(seqs or 0, recipe.min_seqs)

    args: List[str] = []
    if machine.num_gpus > 1:
        args += [_TP_FLAG[engine], str(machine.num_gpus)]
        if _is_pcie_only(machine.gpu_type, machine.gpu_name):
            args.append(_ALL_REDUCE_FLAG)
    if seqs is not None:
        args += [_MAX_SEQS_FLAG[engine], str(seqs)]
    if engine == "vllm":
        # Report prompt_tokens_details.cached_tokens in usage, so clients can
        # see prefix-cache hits. vLLM only: SGLang's equivalent is unverified.
        args.append(_PROMPT_DETAILS_FLAG)
    for recipe in recipes:
        for flag, value in recipe.args.items():
            if flag == _SPEC_FLAG and spec_denied:
                continue
            args += [flag] if flag in _BOOLEAN_FLAGS else [flag, value]
    if profile == AGENT_PROFILE and engine == "vllm":
        args += _agent_args(model)
    # Online FP8 for a bf16 checkpoint; an FP8 one already has it.
    if fp8_weights and not model.prequantized:
        args += [_QUANT_FLAG, "fp8"]
    if (
        profile == "latency"
        and engine == "vllm"
        and machine.gpu_key == RTX_PRO_6000
        and _SPEC_FLAG not in args
        and not user_spec
    ):
        print(f"[llm-app] latency profile: no measured recipe for {model.model_id!r}")
    if profile == "max_throughput" and not any(
        "throughput" not in r.profiles for r in recipes
    ):
        # Only generic recipes matched, so this is the throughput profile.
        print(
            "[llm-app] max_throughput: no measured recipe for "
            f"{model.model_id!r} on this machine, using throughput"
        )
    return args


def tool_call_parser(model: ModelFacts) -> Optional[str]:
    mid = (model.model_id or "").lower()
    # Same architecture, different tool format: Qwen3-Coder and Llama 2.
    if "coder" in mid and model.architectures & {
        "Qwen3MoeForCausalLM",
        "Qwen3NextForCausalLM",
    }:
        return "qwen3_coder"
    # llama3_json is for Llama 3.1 and later only.
    if "LlamaForCausalLM" in model.architectures:
        meta = mid.startswith("meta-llama/") and re.search(r"llama-?3\.[1-9]", mid)
        return "llama3_json" if meta else None
    # Distills and fine-tunes share the Qwen architectures but not the format.
    qwen = model.architectures & _QWEN_ARCHS
    if qwen and ("qwen" not in mid or "deepseek" in mid or "distill" in mid):
        return None
    for architectures, parser in _TOOL_PARSERS:
        if architectures & model.architectures:
            return parser
    return None


def _agent_args(model: ModelFacts) -> List[str]:
    """vLLM flags of the agent profile (see ``PROFILES``)."""
    args = [_PREFIX_CACHING_FLAG]
    parser = tool_call_parser(model)
    if parser:
        args += [_AUTO_TOOL_FLAG, _TOOL_PARSER_FLAG, parser]
    else:
        print(
            "[llm-app] agent profile: no known tool-call parser for "
            f"{model.model_id!r}; set --enable-auto-tool-choice and "
            "--tool-call-parser in engine_args"
        )
    mid = (model.model_id or "").lower()
    for architectures, prefix, extra in _AGENT_ARCH_ARGS:
        if architectures & model.architectures and mid.startswith(prefix):
            for flag, value in extra.items():
                args += [flag] if flag in _BOOLEAN_FLAGS else [flag, value]
    window = _num(_text_config(model.config or {}), "max_position_embeddings")
    if window is None:
        print(
            "[llm-app] agent profile: model window unknown, leaving "
            "--max-model-len to the engine"
        )
    else:
        args += [_MAX_MODEL_LEN_FLAG, str(int(min(window, AGENT_MAX_MODEL_LEN)))]
    return args


def _flag_group(token: str) -> Optional[str]:
    if not token.startswith("-"):
        return None
    name = token.split("=", 1)[0].replace("_", "-")
    # vLLM accepts --no-<flag> for boolean switches.
    if name.startswith("--no-"):
        name = "--" + name[5:]
    return _FLAG_GROUPS.get(name)


def _user_flag_value(args: Sequence[str], group: str) -> Optional[str]:
    """Value of the last ``--flag value`` or ``--flag=value`` in ``group``."""
    found: Optional[str] = None
    for i, token in enumerate(args):
        if _flag_group(token) != group:
            continue
        if "=" in token:
            found = token.split("=", 1)[1]
        elif i + 1 < len(args):
            found = args[i + 1]
    return found


def merge_engine_args(preset: Sequence[str], user: Sequence[str]) -> List[str]:
    """Preset arguments followed by user arguments, user winning per flag.

    A preset flag is dropped (with its value, unless it is a boolean switch)
    when the user arguments mention it as ``--flag value``, ``--flag=value``
    or bare ``--flag``. User arguments are never altered and come last.
    ``preset`` must come from ``engine_preset_args``.
    """
    user_groups = {g for g in map(_flag_group, user) if g}
    if user_groups & _LAYOUT_GROUPS:
        user_groups.add("tp")
    merged: List[str] = []
    i = 0
    while i < len(preset):
        flag = preset[i]
        width = 1 if flag in _BOOLEAN_FLAGS else 2
        if _flag_group(flag) not in user_groups:
            merged += preset[i : i + width]
        i += width
    return [*merged, *user]


def _int_or_none(value: Optional[str]) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def engine_arg_warnings(
    user: Sequence[str], num_gpus: int, max_multiplexing: Optional[int]
) -> List[str]:
    """Messages for user arguments that disagree with the app's resources."""
    messages: List[str] = []
    sizes = {g: _int_or_none(_user_flag_value(user, g)) for g in ("tp", "pp", "dp")}
    if any(v is not None for v in sizes.values()):
        ranks = 1
        for v in sizes.values():
            ranks *= v or 1
        if ranks != num_gpus:
            messages.append(
                f"engine_args use {ranks} GPUs (tensor x pipeline x data "
                f"parallel) but num_gpus is {num_gpus}"
            )
    seqs = _int_or_none(_user_flag_value(user, "max_seqs"))
    if seqs is not None and max_multiplexing is not None and seqs != max_multiplexing:
        messages.append(
            f"engine_args set max concurrent sequences {seqs} but "
            f"max_multiplexing is {max_multiplexing}; the router would "
            "starve the engine or queue requests inside it"
        )
    return messages
