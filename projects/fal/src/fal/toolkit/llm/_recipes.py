"""Measured engine tuning catalog.

Each ``Recipe`` records settings measured for one model family on one GPU
layout. Selection, normalization and argument precedence live in
``_presets``; this module holds the data and the predicate that matches a
recipe to a model and machine, so tuning changes are reviewed apart from the
algorithms.
"""

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, FrozenSet, Mapping, Optional, Tuple

if TYPE_CHECKING:
    from ._presets import Machine, ModelFacts

# Preset flags a recipe lists. Every flag takes a value except the boolean
# switches, which a recipe lists with an empty value.
_KV_DTYPE_FLAG = "--kv-cache-dtype"
_SPEC_FLAG = "--speculative-config"
_TRUST_REMOTE_CODE_FLAG = "--trust-remote-code"
_LOAD_STRATEGY_FLAG = "--safetensors-load-strategy"
_GPU_MEM_FLAG = "--gpu-memory-utilization"
_AUTO_TOOL_FLAG = "--enable-auto-tool-choice"
_TOOL_PARSER_FLAG = "--tool-call-parser"
_REASONING_PARSER_FLAG = "--reasoning-parser"
_TOKENIZER_MODE_FLAG = "--tokenizer-mode"
_FLASHINFER_AUTOTUNE_OFF_FLAG = "--no-enable-flashinfer-autotune"
_WEIGHT_PREFETCH_FLAG = "--weight-loader-prefetch-checkpoints"
_SWA_REPLAY_FLAG = "--enable-decoder-swa-bounded-replay"
_CACHE_REPORT_FLAG = "--enable-cache-report"

RTX_PRO_6000 = "RTX_PRO_6000"
H100 = "H100"
B200 = "B200"
B300 = "B300"
# max_throughput trades quality for speed and is opt-in; models without a
# max_throughput recipe get the throughput one.
# agent: long multi-turn tool-calling sessions (coding agents). Adds prefix
# caching, the tool-call parser and a 128k window here; LLMApp also turns on
# the CPU KV tier (0.7) and prefix_hints unless the app sets them.
PROFILES = ("throughput", "latency", "max_throughput", "agent")
# The Qwen3-8B FP8 + EAGLE3 recipes are not measured with long context, prefix
# caching and the CPU KV tier, so the agent profile does not take them.
_SAFE_PROFILES = ("throughput", "latency")
AGENT_PROFILE = "agent"


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

    ``spec_max_conc`` keeps the recipe's speculative decoding flags only while
    the app's ``max_multiplexing`` (its expected concurrency) is at or below
    it: speculation wins at low to mid concurrency and loses once KV is tight.
    ``None`` leaves it always on. ``max_conc`` skips the whole recipe above
    that concurrency (the settings were measured up to it). ``env`` is set on
    the engine process (the container's environment and ``engine_env`` win).
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
    spec_max_conc: Optional[int] = None
    max_conc: Optional[int] = None
    env: Mapping[str, str] = field(default_factory=dict)

    def matches(
        self,
        engine: str,
        machine: "Machine",
        model: "ModelFacts",
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

_MTP3 = {_SPEC_FLAG: json.dumps({"method": "mtp", "num_speculative_tokens": 3})}
# GLM-5.3 on vLLM: the model's own tool and reasoning parsers, sequential
# weight prefetch (141 shards read through JuiceFS mmap are ~40x slower), and
# FlashInfer autotune off (its warmup faults in the TRT-LLM MoE routing kernel
# on this nightly).
_GLM53_VLLM_ARGS = {
    _AUTO_TOOL_FLAG: "",
    _TOOL_PARSER_FLAG: "glm47",
    _REASONING_PARSER_FLAG: "glm47",
    _LOAD_STRATEGY_FLAG: "prefetch",
    _FLASHINFER_AUTOTUNE_OFF_FLAG: "",
}
_GLM53_NOTE = (
    "GLM-5.3 needs a vLLM nightly with vllm-project/vllm#64580 (dd4e4014 or "
    "newer); v0.33.0 and earlier change 0.4-0.75% of the DSA indexer's top-2048 "
    "selections. Use a container image that has it"
)

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
    # GLM-5.3 and DSv4.1-Flash below come from fal-ai-community/llm-serving
    # (commit 8fddeef, AgentX 1.0.12 official preset). Not remeasured here.
    Recipe(
        source=(
            "vLLM nightly dd4e4014, 8x B300, GLM-5.3 (llm-serving "
            "glm-5.3/vllm/b300/launch.sh): TP8 + MTP3 only to C8; at C16 MTP's "
            "KV cost (2.75M vs 3.46M tokens) took p90 from 37.8 to 22.4 "
            "tok/s/user and TTFT p90 from 3.3 to 14 s"
        ),
        gpus=(B300,),
        engines=("vllm",),
        args={**_GLM53_VLLM_ARGS, **_MTP3},
        gpu_counts=(8,),
        model_ids=frozenset({"zai-org/glm-5.3"}),
        preset_layout=True,
        spec_max_conc=8,
        # Independent of MTP: the chunked SM100 decode-indexer path faults on
        # B300 at startup with a 1M max_model_len.
        env={"VLLM_SPARSE_INDEXER_MAX_DECODE_LOGITS_MB": "0"},
        note=_GLM53_NOTE,
    ),
    Recipe(
        source=(
            "vLLM nightly dd4e4014, 8x B200, GLM-5.3 (llm-serving "
            "glm-5.3/vllm/b200/launch.sh): TP8 with MTP off, since with it one "
            "1M-token request needs 47 GiB of KV and 8x B200 have 35 GiB left"
        ),
        gpus=(B200,),
        engines=("vllm",),
        args=_GLM53_VLLM_ARGS,
        gpu_counts=(8,),
        model_ids=frozenset({"zai-org/glm-5.3"}),
        preset_layout=True,
        note=_GLM53_NOTE,
    ),
    # llm-serving keeps MTP off here until its quality is checked against its
    # golden eval; set spec_max_conc = 0 on the app to drop it.
    Recipe(
        source=(
            "vLLM nightly 40941f2c, 8x B200, GLM-5.3-Flash (llm-serving "
            "glm-5.3-flash/README.md): TP8 + MTP3 at C32 gave 168.8 vs 144.9 "
            "output tok/s/GPU and 120.4 vs 74.6 p90 tok/s/user; C16 p90 180.2 "
            "vs 105.9. Quality not yet checked upstream"
        ),
        gpus=(B200,),
        engines=("vllm",),
        args={
            _TOKENIZER_MODE_FLAG: "glm5_next",
            _AUTO_TOOL_FLAG: "",
            _TOOL_PARSER_FLAG: "glm47",
            _REASONING_PARSER_FLAG: "glm47",
            **_MTP3,
        },
        gpu_counts=(8,),
        model_ids=frozenset({"zai-org/glm-5.3-flash"}),
        preset_layout=True,
        spec_max_conc=32,
        note=(
            "GLM-5.3-Flash MTP3 was measured on a vLLM nightly (40941f2c), not "
            "v0.32; upstream has not validated its quality yet. Set "
            "spec_max_conc = 0 to serve without it"
        ),
    ),
    # The c32 variant of the B300 recipe (conc <= 32 upstream; above 32 it uses
    # mem 0.8, prefill-decode-interval 64, swa-prefix-tails 3072 and no prefill
    # CUDA graph, not expressed here).
    Recipe(
        source=(
            "SGLang nightly-dev-cu13-20261102, 2x B300, DeepSeek-V4.1-Flash "
            "(llm-serving deepseek-v4.1-flash/sglang/b300/launch.sh): TP2 + "
            "DSpark block 5 at every concurrency, cap 64; C32 p90 226 tok/s/user"
            " (acceptance forced to 3.51, not a quality validation)"
        ),
        gpus=(B300,),
        engines=("sglang",),
        args={
            "--trust-remote-code": "",
            "--reasoning-parser": "auto",
            "--tool-call-parser": "auto",
            _CACHE_REPORT_FLAG: "",
            _WEIGHT_PREFETCH_FLAG: "",
            "--watchdog-timeout": "3600",
            "--chunked-prefill-size": "16384",
            "--cuda-graph-max-bs-decode": "64",
            "--schedule-policy": "shortest-prefill-first",
            "--dsv4-attn-backend": "trtllm",
            "--mem-fraction-static": "0.85",
            "--prefill-decode-interval": "96",
            "--swa-prefix-tails": "2048",
            _SWA_REPLAY_FLAG: "",
            "--speculative-algorithm": "DSPARK",
            "--speculative-dspark-block-size": "5",
        },
        min_seqs=64,
        gpu_counts=(2,),
        max_conc=32,
        model_ids=frozenset({"deepseek-ai/deepseek-v4.1-flash-1017"}),
        preset_layout=True,
        env={
            "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": "1",
            "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT": "per_rank",
            "SGLANG_DEFAULT_THINKING": "1",
            "SGLANG_DSV41_REASONING_EFFORT": "high",
            "SGLANG_TIMEOUT_KEEP_ALIVE": "900",
            "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
            "NCCL_NVLS_ENABLE": "0",
            "NCCL_MNNVL_ENABLE": "0",
        },
        note=(
            "DeepSeek-V4.1-Flash on B300 was measured on the SGLang nightly "
            "lmsysorg/sglang nightly-dev-cu13-20261102, with the thinking-on "
            "env defaults (SGLANG_DEFAULT_THINKING=1, reasoning effort high)"
        ),
    ),
)
