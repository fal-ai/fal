"""Two vLLM engines on one 2-GPU runner with engine_replicas.

Deploy:  fal deploy examples/llm_replicas.py::QwenReplicas
Call:    POST https://fal.run/<user>/qwen-replicas/v1/chat/completions

A 7B model fits on one GPU, so tensor parallelism across both GPUs mostly
adds communication cost. With ``engine_replicas = 2`` the runner starts one
engine per GPU (each on its own port, starting at ``engine_port``) and routes
every request to one of them: the engine that served the same session id
before, else the one with the fewest requests in flight. Affinity is not
strict: a session whose engine is far busier than the other moves (and loses
its cached prefix), so a hot id cannot pile every request onto one GPU.
``num_gpus`` must divide evenly by ``engine_replicas``. The health check fails
when either engine is down.

Run this once on real GPUs and compare it to a single-engine app before you
rely on it for throughput.
"""

from __future__ import annotations

from fal.container import ContainerImage
from fal.toolkit.llm import LLMApp

DOCKERFILE = """
FROM vllm/vllm-openai:v0.32.0
"""


class QwenReplicas(
    LLMApp,
    kind="container",
    image=ContainerImage.from_dockerfile_str(DOCKERFILE),
    name="qwen-replicas",
):
    machine_type = "GPU-RTXPRO6000"
    num_gpus = 2
    startup_timeout = 1800
    max_multiplexing = 128

    engine = "vllm"
    model = "Qwen/Qwen2.5-7B-Instruct"
    served_model_name = "qwen2.5-7b-instruct"
    engine_replicas = 2
    engine_args = [
        "--max-model-len",
        "8192",
    ]
