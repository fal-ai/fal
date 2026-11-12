import json
import os
import re
import subprocess
from dataclasses import dataclass
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

from ._recipes import (
    _AUTO_TOOL_FLAG,
    _CACHE_REPORT_FLAG,
    _FLASHINFER_AUTOTUNE_OFF_FLAG,
    _GPU_MEM_FLAG,
    _KV_DTYPE_FLAG,
    _LOAD_STRATEGY_FLAG,
    _REASONING_PARSER_FLAG,
    _SPEC_FLAG,
    _SWA_REPLAY_FLAG,
    _TOKENIZER_MODE_FLAG,
    _TOOL_PARSER_FLAG,
    _TRUST_REMOTE_CODE_FLAG,
    _WEIGHT_PREFETCH_FLAG,
    AGENT_PROFILE,
    B200,
    B300,
    H100,
    PROFILES,
    RECIPES,
    RTX_PRO_6000,
    Recipe,
)

# Preset flag per engine. Every flag here takes a value except the boolean
# switches (see _BOOLEAN_FLAGS), which a recipe lists with an empty value.
_TP_FLAG = {"vllm": "--tensor-parallel-size", "sglang": "--tp-size"}
_MAX_SEQS_FLAG = {"vllm": "--max-num-seqs", "sglang": "--max-running-requests"}
_QUANT_FLAG = "--quantization"
_ALL_REDUCE_FLAG = "--disable-custom-all-reduce"
_PROMPT_DETAILS_FLAG = "--enable-prompt-tokens-details"
_PREFIX_CACHING_FLAG = "--enable-prefix-caching"
_MAX_MODEL_LEN_FLAG = "--max-model-len"
_LANGUAGE_ONLY_FLAG = "--language-model-only"
_TEMPLATE_KWARGS_FLAG = "--default-chat-template-kwargs"
_BOOLEAN_FLAGS = frozenset(
    {
        _FLASHINFER_AUTOTUNE_OFF_FLAG,
        _WEIGHT_PREFETCH_FLAG,
        _SWA_REPLAY_FLAG,
        _CACHE_REPORT_FLAG,
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
    "--speculative-algorithm": "spec",
    "--speculative-dspark-block-size": "spec",
    "--speculative-num-steps": "spec",
    "--speculative-eagle-topk": "spec",
    "--speculative-num-draft-tokens": "spec",
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
    _TOKENIZER_MODE_FLAG: "tokenizer_mode",
    "--enable-flashinfer-autotune": "flashinfer_autotune",
    _WEIGHT_PREFETCH_FLAG: "weight_prefetch",
    _SWA_REPLAY_FLAG: "swa_replay",
    _CACHE_REPORT_FLAG: "cache_report",
    "--chunked-prefill-size": "chunked_prefill",
    "--cuda-graph-max-bs-decode": "cuda_graph_decode",
    "--schedule-policy": "schedule_policy",
    "--dsv4-attn-backend": "dsv4_attn_backend",
    "--mem-fraction-static": "mem_fraction",
    "--prefill-decode-interval": "prefill_decode_interval",
    "--swa-prefix-tails": "swa_prefix_tails",
    "--watchdog-timeout": "watchdog_timeout",
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

# Architectures whose EAGLE3 drafts crash vLLM 0.32 at startup.
_SPEC_DENYLIST = frozenset({"GptOssForCausalLM"})
_ENGINES = tuple(_TP_FLAG)

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
# Qwen3 hybrid-thinking checkpoints (Qwen3-8B, Qwen3-32B, Qwen3-30B-A3B) think
# by default and emit <think>...</think> in the reply. Without a reasoning
# parser that text lands in message.content; with it, in reasoning_content.
# Coder and the *-Instruct-2507 variants never think and are left alone.
_QWEN3_THINKING_ARCHS = frozenset({"Qwen3ForCausalLM", "Qwen3MoeForCausalLM"})
_NON_THINKING_MARKERS = ("coder", "instruct")


def _qwen3_thinks(model: "ModelFacts") -> bool:
    mid = (model.model_id or "").lower()
    return (
        bool(model.architectures & _QWEN3_THINKING_ARCHS)
        and mid.startswith("qwen/qwen3-")
        and not any(m in mid for m in _NON_THINKING_MARKERS)
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
    # Not GB200 / GB300 (Grace Blackwell, a different node shape).
    if re.search(r"(?<![A-Z0-9])B300", upper):
        return B300
    if re.search(r"(?<![A-Z0-9])B200", upper):
        return B200
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


def _matching_recipes(
    engine: str,
    machine: Machine,
    model: ModelFacts,
    profile: str,
    user_args: Sequence[str],
    skip_checkpoint: Optional[str],
    max_multiplexing: Optional[int] = None,
    announce: bool = True,
) -> List[Recipe]:
    skipped = skip_checkpoint.lower() if skip_checkpoint else None
    user_layout = any(_has_flag(user_args, g) for g in _LAYOUT_GROUPS)
    recipes = [
        r
        for r in RECIPES
        if r.matches(engine, machine, model, profile)
        and not (skipped and r.checkpoint and r.checkpoint.lower() == skipped)
    ]
    over = [
        r
        for r in recipes
        if r.max_conc is not None
        and max_multiplexing is not None
        and max_multiplexing > r.max_conc
    ]
    if over and announce:
        print(
            "[llm-app] max_multiplexing is above the concurrency a recipe was "
            f"measured at ({over[0].max_conc}), so it is skipped"
        )
    recipes = [r for r in recipes if r not in over]
    if user_layout and any(r.preset_layout for r in recipes) and announce:
        print(
            "[llm-app] engine_args set the parallel layout, so recipes measured "
            "with the tensor parallel preset (e.g. FP8 KV cache) are skipped"
        )
    if user_layout:
        recipes = [r for r in recipes if not r.preset_layout]
    return recipes


def engine_preset_env(
    engine: str,
    machine: Machine,
    model: ModelFacts,
    *,
    profile: str = "throughput",
    user_args: Sequence[str] = (),
    skip_checkpoint: Optional[str] = None,
    max_multiplexing: Optional[int] = None,
) -> Dict[str, str]:
    """Environment of the matching recipes; the container's and ``engine_env`` win."""
    env: Dict[str, str] = {}
    for recipe in _matching_recipes(
        engine,
        machine,
        model,
        profile,
        user_args,
        skip_checkpoint,
        max_multiplexing,
        announce=False,
    ):
        env.update(recipe.env)
    return env


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
    spec_max_conc: Optional[int] = None,
) -> List[str]:
    """Engine arguments that match the machine, for ``vllm`` or ``sglang``.

    Generic base:

    - ``machine.num_gpus > 1``: tensor parallel across all GPUs
      (``--tensor-parallel-size`` for vLLM, ``--tp-size`` for SGLang), plus
      ``--disable-custom-all-reduce`` on PCIe-only GPUs (RTX PRO 6000, RTX
      4090/5090, L40/L40S, L4, A10), since the custom all-reduce kernel needs
      NVLink.
    - ``max_multiplexing``: ``--max-num-seqs`` (vLLM) or
      ``--max-running-requests`` (SGLang). Multi-GPU: twice the router's
      concurrency, so bursts and retries are admitted (llm-serving sets 2x
      everywhere). Not doubled when a matching recipe sets its own
      ``min_seqs`` (that measured cap applies, raised to the router's
      concurrency) or when engine_args set a pipeline or data parallel layout
      (the cap is per rank). One GPU: exactly the router's concurrency,
      because the engine's memory profiling and CUDA graphs scale with the cap
      and one card has no KV to spare. A recipe may raise the engine cap above
      it, never lower it.
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
    architecture in ``_SPEC_DENYLIST``. A recipe's speculative decoding is
    kept only while ``max_multiplexing`` is at or below its ``spec_max_conc``
    (``spec_max_conc`` here overrides the recipe's; ``0`` never keeps it); a
    user ``--speculative-config`` always wins but is warned about when over the
    limit. ``user_args`` (the app's
    ``engine_args``) drives messages about the presets and, when it sets a
    tensor, pipeline or data parallel size, skips recipes that assume the
    tensor parallel preset.
    ``skip_checkpoint`` drops the recipes tuned for that checkpoint, for when
    it could not be served and the original weights are used instead.

    Use ``merge_engine_args`` to put user arguments on top.

    Args:
        engine: ``"vllm"`` or ``"sglang"``.
        machine: The GPU type, name, memory and count one engine will see.
        model: Facts about the model (architectures, size, window).
        profile: ``"throughput"``, ``"latency"``, ``"max_throughput"`` or
            ``"agent"``.
        fp8_weights: Apply online FP8 quantization.
        max_multiplexing: The app's expected concurrency, mapped to the
            engine's sequence cap.
        user_args: The app's ``engine_args``.
        skip_checkpoint: Recipe checkpoint that could not be served.
        spec_max_conc: Overrides the recipes' speculative decoding limit.

    Returns:
        Engine arguments to pass to the launch command.

    Raises:
        ValueError: If ``engine`` or ``profile`` is unknown.
    """
    if engine not in _TP_FLAG:
        raise ValueError(f"unknown engine {engine!r}; expected 'vllm' or 'sglang'")
    if profile not in PROFILES:
        raise ValueError(f"unknown engine_profile {profile!r}; expected {PROFILES}")
    recipes = _matching_recipes(
        engine, machine, model, profile, user_args, skip_checkpoint, max_multiplexing
    )
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
    spec_off: List[Recipe] = []
    for recipe in recipes:
        limit = spec_max_conc if spec_max_conc is not None else recipe.spec_max_conc
        if limit is None or not any(_flag_group(f) == "spec" for f in recipe.args):
            continue
        if limit == 0:
            if not user_spec:
                spec_off.append(recipe)
        elif max_multiplexing is None:
            print(
                f"[llm-app] speculative decoding is only measured up to "
                f"{limit} concurrent requests; max_multiplexing is not set, so "
                "it is kept"
            )
        elif max_multiplexing > limit:
            if user_spec:
                print(
                    f"[llm-app] max_multiplexing {max_multiplexing} is above "
                    f"spec_max_conc {limit}, where speculative decoding "
                    "usually hurts; your --speculative-config is passed as given"
                )
            else:
                spec_off.append(recipe)
                print(
                    f"[llm-app] dropping speculative decoding: max_multiplexing "
                    f"{max_multiplexing} is above spec_max_conc {limit} "
                    "(it loses once KV is tight)"
                )
    seqs = max_multiplexing
    # A recipe with its own cap was measured at that cap; doubling past it is
    # untested (it also scales profiling memory and CUDA graphs).
    if (
        seqs is not None
        and machine.num_gpus > 1
        and not any(_has_flag(user_args, g) for g in ("pp", "dp"))
        and all(r.min_seqs is None for r in recipes)
    ):
        seqs *= 2
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
            if _flag_group(flag) == "spec" and (spec_denied or recipe in spec_off):
                continue
            args += [flag] if flag in _BOOLEAN_FLAGS else [flag, value]
    if profile == AGENT_PROFILE and engine == "vllm":
        args += _agent_args(model, any(_TOOL_PARSER_FLAG in r.args for r in recipes))
    # Online FP8 for a bf16 checkpoint; an FP8 one already has it.
    if fp8_weights and not model.prequantized:
        args += [_QUANT_FLAG, "fp8"]
    if (
        profile == "latency"
        and engine == "vllm"
        and machine.gpu_key == RTX_PRO_6000
        and not any(_flag_group(a) == "spec" for a in args)
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


def _agent_args(model: ModelFacts, recipe_tools: bool = False) -> List[str]:
    """vLLM flags of the agent profile (see ``PROFILES``).

    ``recipe_tools``: a recipe already sets the tool-call parser.
    """
    args = [_PREFIX_CACHING_FLAG]
    parser = tool_call_parser(model)
    if recipe_tools:
        pass
    elif parser:
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
    if _qwen3_thinks(model):
        args += [_REASONING_PARSER_FLAG, "qwen3"]
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

    Args:
        preset: Arguments from ``engine_preset_args``.
        user: The app's own ``engine_args``.

    Returns:
        The merged argument list.
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
    user: Sequence[str],
    num_gpus: int,
    max_multiplexing: Optional[int],
    spec_max_conc: Optional[int] = None,
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
    # Above the router's concurrency is harmless headroom (the presets use 2x
    # on several GPUs); below it the router queues requests inside the engine.
    if seqs is not None and max_multiplexing is not None and seqs < max_multiplexing:
        messages.append(
            f"engine_args set max concurrent sequences {seqs} but "
            f"max_multiplexing is {max_multiplexing}; the router would "
            "queue requests inside the engine"
        )
    # 0 is the documented opt-out, so it is not worth a warning.
    if (
        spec_max_conc
        and max_multiplexing is not None
        and max_multiplexing > spec_max_conc
    ):
        if _has_flag(user, "spec"):
            messages.append(
                f"max_multiplexing {max_multiplexing} is above spec_max_conc "
                f"{spec_max_conc}, where speculative decoding usually hurts; "
                "your --speculative-config is kept"
            )
        else:
            messages.append(
                f"max_multiplexing {max_multiplexing} is above spec_max_conc "
                f"{spec_max_conc}: if a preset recipe applies, its speculative "
                "decoding is dropped"
            )
    return messages
