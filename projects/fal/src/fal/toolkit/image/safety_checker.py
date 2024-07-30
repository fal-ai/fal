from typing import Any

import fal

from . import filter_by
from .nsfw_filter.model import get_model


@fal.cached
def load_safety_checker():
    import torch
    from diffusers.pipelines.stable_diffusion.safety_checker import (
        StableDiffusionSafetyChecker,
    )
    from transformers import AutoFeatureExtractor

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "CompVis/stable-diffusion-safety-checker",
        torch_dtype="float16",
    )
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        "CompVis/stable-diffusion-safety-checker",
        torch_dtype=torch.float16,
    ).to("cuda")

    return feature_extractor, safety_checker


def run_safety_checker(
    pil_images: list[object],
) -> list[bool]:
    import numpy as np
    import torch

    feature_extractor, safety_checker = load_safety_checker()

    safety_checker_input = feature_extractor(pil_images, return_tensors="pt").to("cuda")

    np_image = [np.array(val) for val in pil_images]

    _, has_nsfw_concept = safety_checker(
        images=np_image,
        clip_input=safety_checker_input.pixel_values.to(torch.float16),
    )

    return has_nsfw_concept


def run_safety_checker_v2(pil_images: list, nsfw_threshold: float = 0.5) -> list[bool]:
    import torch

    model, processor = get_model()

    has_nsfw_concept = []

    with torch.no_grad():
        for pil_image in pil_images:
            inputs = processor(
                images=pil_image.convert("RGB"),
                return_tensors="pt",
            )
            outputs = model(**inputs)
            logits = (
                outputs.logits.squeeze()
            )  # Remove batch dimension to simplify indexing

            # Apply softmax to convert logits to probabilities
            probabilities = torch.softmax(logits, dim=0)

            nsfw_class_index = model.config.label2id.get(
                "nsfw", None
            )  # Replace "NSFW" with the exact class name if different

            # Validate that NSFW class index is found
            if nsfw_class_index is not None:
                nsfw_probability = probabilities[int(nsfw_class_index)].item()
                print("NSFW probability:", nsfw_probability)
                has_nsfw_concept.append(nsfw_probability > nsfw_threshold)
            else:
                raise ValueError("NSFW class not found in model output.")

    return has_nsfw_concept


def postprocess_images(
    pil_images: list[object],
    enable_safety_checker: bool = True,
    safety_checker_version: int = 2,
) -> dict[str, Any]:
    outputs: dict[str, list[Any]] = {
        "images": pil_images,
    }

    if enable_safety_checker:
        safety_checker_fn = (
            run_safety_checker_v2 if safety_checker_version == 2 else run_safety_checker
        )
        outputs["has_nsfw_concepts"] = safety_checker_fn(pil_images)  # type: ignore
    else:
        outputs["has_nsfw_concepts"] = [False] * len(pil_images)

    outputs["images"] = filter_by(
        outputs["has_nsfw_concepts"],
        outputs["images"],
    )

    return outputs
