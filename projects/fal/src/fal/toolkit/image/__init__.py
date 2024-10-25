from __future__ import annotations

import urllib.request
from functools import lru_cache
from typing import TYPE_CHECKING

from .image import *  # noqa: F403

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


def filter_by(
    has_nsfw_concepts: list[bool],
    images: list[PILImage],
) -> list[PILImage]:
    from PIL import Image as PILImage

    return [
        (
            PILImage.new("RGB", (image.width, image.height), (0, 0, 0))
            if has_nsfw
            else image
        )
        for image, has_nsfw in zip(images, has_nsfw_concepts)
    ]


def preprocess_image(image_pil, convert_to_rgb=True, fix_orientation=True):
    from PIL import ImageOps, ImageSequence

    # For MPO (multi picture object) format images, we only need the first image
    images = []
    for image in ImageSequence.Iterator(image_pil):
        img = image

        if convert_to_rgb:
            img = img.convert("RGB")

        if fix_orientation:
            img = ImageOps.exif_transpose(img)

        images.append(img)

        break

    return images[0]


@lru_cache(maxsize=64)
def read_image_from_url(
    url: str, convert_to_rgb: bool = True, fix_orientation: bool = True
):
    from fastapi import HTTPException
    from PIL import Image

    TEMP_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:21.0) "
            "Gecko/20100101 Firefox/21.0"
        ),
    }

    try:
        request = urllib.request.Request(url, headers=TEMP_HEADERS)
        response = urllib.request.urlopen(request)
        image_pil = Image.open(response)
    except Exception:
        import traceback

        traceback.print_exc()
        raise HTTPException(422, f"Could not load image from url: {url}")

    return preprocess_image(image_pil, convert_to_rgb, fix_orientation)
