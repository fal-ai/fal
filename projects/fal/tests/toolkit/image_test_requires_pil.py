from __future__ import annotations

import base64
import io

from fal.toolkit.image.image import Image
from PIL import Image as PILImage


# taken from chatgpt
def images_are_equal(img1: PILImage.Image, img2: PILImage.Image) -> bool:
    if img1.size != img2.size:
        return False

    pixels1 = list(img1.getdata())
    pixels2 = list(img2.getdata())

    for p1, p2 in zip(pixels1, pixels2):
        if p1 != p2:
            return False

    return True


def test_image_matches():
    # 1x1 white png image
    base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx5QAAAABJRU5ErkJggg=="
    image_bytes = base64.b64decode(base64_image)
    pil_image = PILImage.open(io.BytesIO(image_bytes))
    image_file: Image = Image.from_pil(pil_image, format="png", repository="in_memory")
    output_pil_image = PILImage.open(io.BytesIO(image_file.as_bytes()))
    assert images_are_equal(output_pil_image, pil_image)
