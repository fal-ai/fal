"""Runs per request, and is reloaded on every save (along with post.py), with the weights kept on the GPU."""

import torch
from post import finish


def infer(
    state, prompt="a red fox in fresh snow, golden hour", steps=4, guidance_scale=1.0, width=1024, height=1024, seed=0
):
    pipe = state["pipe"]
    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images[0]
    return finish(image)
