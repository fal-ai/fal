"""Qwen3.8-27B-FP8 on 1x RTX PRO 6000, the app under tuning for the RTX loop.

Edit ENGINE_ARGS (or the class body) and run `fal dev app restart`, which syncs
first and then restarts the app process with the new flags, keeping the box, the
weights and the compile caches on /data. Never re-run `start` unless requirements
or the image change.

Use `restart`, not `sync`. `sync` decides reload-vs-restart by walking setup() and
the names it references, but only when setup() is defined in the synced tree
(`analyze_setup` checks `"setup" in klass.__dict__`). This class inherits setup()
from LLMApp, so nothing is fingerprinted, every edit comes back as "reload:
endpoint code changed", the class is swapped in place, and the vLLM engine keeps
running on the old flags. The next bench would then score the previous
configuration under the new label.

  cd projects/fal && uv run --python 3.12 fal dev app start \
    tests/bench/llm_tuning/llm_app_qwen38_tune.py::Qwen38Tune \
    --name qwen38tune --keep-alive 600

`start` fixes the sync root at the folder holding this file, and `sync`/`restart`
take no path, so they act on whatever root that first `start` chose. Point `start`
at the copy you intend to edit for the rest of the loop.
"""

from fal.container import ContainerImage
from fal.toolkit.llm import LLMApp

DOCKERFILE = """
FROM vllm/vllm-openai:v0.32.0
"""
_IMG = ContainerImage.from_dockerfile_str(DOCKERFILE)

# The knobs under test. Keep one variable per run so a delta has one cause.
# Spelled out rather than taken from an engine_profile: a profile hides what
# changed between two runs, which is the one thing this loop needs to see.
ENGINE_ARGS = [
    "--gpu-memory-utilization", "0.92",
    "--max-model-len", "32768",
    "--enable-prefix-caching",
    # Without this, usage carries no prompt_tokens_details, so every row records
    # cached_tokens as null and no prefix-cache claim can be checked.
    "--enable-prompt-tokens-details",
    "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
    "--reasoning-parser", "qwen3",
    "--language-model-only",
    # Overlap the scheduler's CPU work with GPU execution. Decode tok/s at 32
    # streams sits near the floor, and per-step overhead is what caps it.
    "--async-scheduling",
    # Provisional. vLLM picks FLASH_ATTN (FlashAttention 2) on SM120 out of
    # FLASH_ATTN, FLASHINFER, TRITON_ATTN and FLEX_ATTENTION. At a 12k agent
    # prompt and 8 streams, the only operating point measured that holds the
    # latency floor, FlashInfer is about 2% ahead. The 8% gap at 32 streams is
    # real throughput but neither backend holds the floor there. One sample
    # each and run order is not controlled for, so this is not recipe material
    # yet. The flag is the only way in: the VLLM_ATTENTION_BACKEND env var is
    # gone in 0.32 and only warns. See ai-designs/2026-11-06-rtx-tuning-loop.md.
    "--attention-backend", "FLASHINFER",
]


class Qwen38Tune(LLMApp, kind="container", image=_IMG):
    machine_type = "GPU-RTXPRO6000"
    num_gpus = 1
    startup_timeout = 1800
    keep_alive = 600
    min_concurrency = 0
    max_concurrency = 1
    max_multiplexing = 64

    engine = "vllm"
    model = "Qwen/Qwen3.8-27B-FP8"
    served_model_name = "qwen3.8-27b"
    engine_args = ENGINE_ARGS
