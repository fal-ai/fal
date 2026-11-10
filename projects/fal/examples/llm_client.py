"""Call an LLMApp endpoint with the OpenAI Python SDK.

Run:  FAL_USER=<user> FAL_APP=qwen-vllm FAL_KEY=<key> python examples/llm_client.py

This is a plain script, not a fal app. Install the client with
``pip install openai``. ``FAL_USER`` is your fal account name and ``FAL_APP``
is the ``name=`` of the deployed app (for example ``qwen-vllm`` from
``llm_app.py``). Set ``FAL_MODEL`` to the app's ``served_model_name`` when
it is not ``qwen2.5-7b-instruct``. ``X-Fal-Store-IO: 0`` asks fal not to store
request payloads.
"""

from __future__ import annotations

import os
import uuid

from openai import OpenAI

MODEL = os.environ.get("FAL_MODEL", "qwen2.5-7b-instruct")


def make_client() -> OpenAI:
    user = os.environ["FAL_USER"]
    app = os.environ["FAL_APP"]
    return OpenAI(
        base_url=f"https://fal.run/{user}/{app}/v1",
        api_key=os.environ["FAL_KEY"],
        default_headers={"X-Fal-Store-IO": "0"},
    )


def main() -> None:
    client = make_client()
    messages = [{"role": "user", "content": "Name three uses of a hash map."}]

    # Non-streaming call. finish_reason is "stop" (natural end) or "length"
    # (max_tokens reached).
    response = client.chat.completions.create(model=MODEL, messages=messages)
    print(response.choices[0].message.content)
    print("finish_reason:", response.choices[0].finish_reason)

    # Streaming call that stops early. Closing the stream drops the connection,
    # which aborts the request in the engine and frees its KV cache. Streams
    # that LLMApp ends itself report finish_reason "deadline",
    # "idle_timeout" or "shutdown".
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Count from 1 to 500."}],
        stream=True,
    )
    received = 0
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            received += 1
            if received >= 20:
                break
    stream.close()
    print("stopped after", received, "chunks")

    # A session id keeps a multi-turn conversation on the runner that holds
    # its KV cache (needs prefix_hints, which engine_profile="agent" turns on).
    session = uuid.uuid4().hex
    for question in ("Pick a color.", "Why that one?"):
        messages.append({"role": "user", "content": question})
        reply = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            extra_headers={"X-Fal-Runner-Hint": session},
            extra_body={"prompt_cache_key": session},
        )
        answer = reply.choices[0].message.content
        messages.append({"role": "assistant", "content": answer})
        print(answer, "| finish_reason:", reply.choices[0].finish_reason)


if __name__ == "__main__":
    main()
