"""OpenAI-compatible Qwen2.5-7B on SGLang with fal.toolkit.llm.LLMApp.

Deploy:  fal deploy examples/llm_sglang.py::QwenSGLang
Call:    POST https://fal.run/<user>/qwen-sglang/v1/chat/completions

Set ``engine = "sglang"`` and use an image that ships SGLang. LLMApp starts
``python -m sglang.launch_server`` and keeps the same OpenAI routes, health
check and presets as with vLLM. The image tag below is the SGLang nightly
the presets' recipes were measured on; pin the tag you have tested.
"""

from __future__ import annotations

from fal.container import ContainerImage
from fal.toolkit.llm import LLMApp

DOCKERFILE = """
FROM lmsysorg/sglang:nightly-dev-cu13-20261102
"""


class QwenSGLang(
    LLMApp,
    kind="container",
    image=ContainerImage.from_dockerfile_str(DOCKERFILE),
    name="qwen-sglang",
):
    machine_type = "GPU-RTXPRO6000"
    num_gpus = 1
    startup_timeout = 1800
    max_multiplexing = 64

    engine = "sglang"
    model = "Qwen/Qwen2.5-7B-Instruct"
    served_model_name = "qwen2.5-7b-instruct"
    engine_args = [
        "--context-length",
        "8192",
    ]
