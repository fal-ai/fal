"""Tool-calling agent backend on vLLM with the "agent" engine profile.

Deploy:  fal deploy examples/llm_agent.py::QwenAgent
Call:    POST https://fal.run/<user>/qwen-agent/v1/chat/completions

Long multi-turn tool-calling sessions reuse a large prompt prefix on every
turn. ``engine_profile = "agent"`` turns on prefix caching, the model
family's tool-call parser (``hermes`` for Qwen3), a longer context window,
the CPU KV tier and ``prefix_hints``. With ``prefix_hints`` the client sends a
stable session id on every turn, so the gateway prefers the runner that
already holds that conversation's KV cache.
``request_deadline_s`` aborts any request that runs longer than 10 minutes.

Client:

    import os
    import uuid

    from openai import OpenAI

    client = OpenAI(
        base_url="https://fal.run/<user>/qwen-agent/v1",
        api_key=os.environ["FAL_KEY"],
    )
    session = uuid.uuid4().hex  # opaque and random, never a user identifier
    response = client.chat.completions.create(
        model="qwen3-8b",
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        extra_headers={"X-Fal-Runner-Hint": session},
        extra_body={"prompt_cache_key": session},
    )
    print(response.choices[0].message.tool_calls)

Send the same ``session`` value on every later turn of the conversation.
"""

from __future__ import annotations

from fal.container import ContainerImage
from fal.toolkit.llm import LLMApp

DOCKERFILE = """
FROM vllm/vllm-openai:v0.32.0
"""


class QwenAgent(
    LLMApp,
    kind="container",
    image=ContainerImage.from_dockerfile_str(DOCKERFILE),
    name="qwen-agent",
):
    machine_type = "GPU-RTXPRO6000"
    num_gpus = 1
    startup_timeout = 1800
    max_multiplexing = 64

    engine = "vllm"
    model = "Qwen/Qwen3-8B"
    served_model_name = "qwen3-8b"
    engine_profile = "agent"
    request_deadline_s = 600.0
