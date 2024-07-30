from pydantic import BaseModel, Field

import fal
from fal.toolkit.image import read_image_from_url

from .env import get_requirements
from .model import get_model


class NSFWImageDetectionInput(BaseModel):
    image_url: str = Field(
        description="Input image url.",
        examples=[
            "https://storage.googleapis.com/falserverless/model_tests/remove_background/elephant.jpg",
        ],
    )


class NSFWImageDetectionOutput(BaseModel):
    nsfw_probability: float = Field(
        description="The probability of the image being NSFW.",
    )


def check_nsfw_content(pil_image: object):
    import torch

    model, processor = get_model()

    with torch.no_grad():
        inputs = processor(images=pil_image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits.squeeze()  # Remove batch dimension to simplify indexing

    # Apply softmax to convert logits to probabilities
    probabilities = torch.softmax(logits, dim=0)

    nsfw_class_index = model.config.label2id.get(
        "nsfw", None
    )  # Replace "NSFW" with the exact class name if different

    # Validate that NSFW class index is found
    if nsfw_class_index is not None:
        nsfw_probability = probabilities[int(nsfw_class_index)].item()
        return nsfw_probability
    else:
        raise ValueError("NSFW class not found in model output.")


def run_nsfw_estimation(
    input: NSFWImageDetectionInput,
) -> NSFWImageDetectionOutput:
    img = read_image_from_url(input.image_url)
    nsfw_probability = check_nsfw_content(img)

    return NSFWImageDetectionOutput(nsfw_probability=nsfw_probability)


@fal.function(
    requirements=get_requirements(),
    machine_type="GPU-A6000",
    serve=True,
)
def run_nsfw_estimation_on_fal(
    input: NSFWImageDetectionInput,
) -> NSFWImageDetectionOutput:
    return run_nsfw_estimation(input)


if __name__ == "__main__":
    local = run_nsfw_estimation_on_fal.on(serve=False)
    result = local(
        NSFWImageDetectionInput(
            image_url="https://storage.googleapis.com/falserverless/model_tests/remove_background/elephant.jpg",
        )
    )
    print(result)
