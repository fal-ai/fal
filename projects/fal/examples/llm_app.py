"""OpenAI-compatible Qwen2.5-7B on vLLM with fal.toolkit.llm.LLMApp.

Deploy:  fal deploy examples/llm_app.py::QwenVLLM
Call:    POST https://fal.run/<user>/qwen-vllm/v1/chat/completions
"""

from __future__ import annotations

from fal.container import ContainerImage
from fal.toolkit.llm import LLMApp

DOCKERFILE = """
FROM vllm/vllm-openai:v0.32.0
"""


class QwenVLLM(
    LLMApp,
    kind="container",
    image=ContainerImage.from_dockerfile_str(DOCKERFILE),
    name="qwen-vllm",
):
    machine_type = "GPU-RTXPRO6000"
    num_gpus = 1
    startup_timeout = 1800
    max_multiplexing = 32

    engine = "vllm"
    model = "Qwen/Qwen2.5-7B-Instruct"
    served_model_name = "qwen2.5-7b-instruct"
    engine_args = [
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "32",
        "--gpu-memory-utilization",
        "0.90",
    ]
