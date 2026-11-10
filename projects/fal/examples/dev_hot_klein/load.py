"""Runs once per server process: loads FLUX.2 [klein] 4B onto the GPU.

Saving this file restarts the hot server, which loads the weights again.
"""

import torch
from diffusers import Flux2KleinPipeline

# Apache 2.0, not gated: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
MODEL = "black-forest-labs/FLUX.2-klein-4B"


def load():
    pipe = Flux2KleinPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    return {"pipe": pipe}
