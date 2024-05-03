from __future__ import annotations

from base64 import b64encode
from io import BytesIO
from typing import Literal, overload

import pytest
from fal.toolkit import Image
from PIL import Image as PILImage
from pydantic import BaseModel, Field
from pydantic import __version__ as pydantic_version


@overload
def get_image(as_bytes: Literal[False] = False) -> PILImage.Image: ...


@overload
def get_image(as_bytes: Literal[True]) -> bytes: ...


def get_image(as_bytes: bool = False):
    from PIL import Image

    pil_image = Image.new("RGB", (1, 1), (255, 255, 255))
    if not as_bytes:
        return pil_image

    return pil_image_to_bytes(pil_image)


def pil_image_to_bytes(image: PILImage.Image) -> bytes:
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    return image_bytes.getvalue()


def fal_image_downloaded(image: Image):
    return image.file_size is not None


def fal_image_url_matches(image: Image, url: str):
    return image.url == url


def fal_image_content_matches(image: Image, content: bytes):
    image1 = PILImage.open(BytesIO(image.as_bytes()))
    image2 = PILImage.open(BytesIO(content))
    return images_are_equal(image1, image2)


def image_to_data_uri(image: PILImage.Image) -> str:
    image_bytes = pil_image_to_bytes(image)
    b64_encoded = b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64_encoded}"


def images_are_equal(img1: PILImage.Image, img2: PILImage.Image) -> bool:
    pixels1 = list(img1.getdata())
    pixels2 = list(img2.getdata())
    return pixels1 == pixels2


def assert_fal_images_equals(fal_image_1: Image, fal_image_2: Image):
    assert fal_image_content_matches(fal_image_1, fal_image_2.as_bytes())

    assert (
        fal_image_1.file_size == fal_image_2.file_size
    ), "Image file size should match"
    assert (
        fal_image_1.content_type == fal_image_2.content_type
    ), "Content type should match"
    assert fal_image_1.url == fal_image_2.url, "URL should match"
    assert fal_image_1.width == fal_image_2.width, "Width should match"
    assert fal_image_1.height == fal_image_2.height, "Height should match"


def test_fal_image_from_pil(isolated_client):
    @isolated_client(requirements=["pillow", f"pydantic=={pydantic_version}"])
    def fal_image_from_bytes_remote():
        pil_image = get_image()
        return Image.from_pil(pil_image, repository="in_memory")

    fal_image = fal_image_from_bytes_remote()
    assert fal_image_content_matches(fal_image, get_image(as_bytes=True))


def test_fal_image_from_bytes(isolated_client):
    @isolated_client(requirements=["pillow", f"pydantic=={pydantic_version}"])
    def fal_image_from_bytes_remote():
        image_bytes = get_image(as_bytes=True)
        return Image.from_bytes(image_bytes, repository="in_memory", format="png")

    fal_image = fal_image_from_bytes_remote()
    assert fal_image_content_matches(fal_image, get_image(as_bytes=True))


@pytest.mark.parametrize(
    "image_url",
    [
        "https://commons.wikimedia.org/static/images/project-logos/commonswiki-1.5x.png",
        image_to_data_uri(get_image()),
    ],
)
def test_fal_image_input(isolated_client, image_url):
    class TestInput(BaseModel):
        image: Image = Field()

    @isolated_client(requirements=["pillow", f"pydantic=={pydantic_version}"])
    def init_image_on_fal(input: TestInput) -> Image:
        return TestInput(image=input.image).image

    test_input = TestInput(image=image_url)
    image = init_image_on_fal(test_input)

    # Image is not downloaded until it is needed
    assert not fal_image_downloaded(image)
    assert fal_image_url_matches(image, image_url)

    # Expect value error if we try to access the file content for input file
    with pytest.raises(ValueError):
        image.as_bytes()


def test_fal_image_input_to_pil(isolated_client):
    class TestInput(BaseModel):
        image: Image = Field()

    @isolated_client(requirements=["pillow", f"pydantic=={pydantic_version}"])
    def init_image_on_fal(input: TestInput) -> bytes:
        input_image = TestInput(image=input.image).image
        pil_image = input_image.to_pil()
        return pil_image_to_bytes(pil_image)

    test_input = TestInput(image=Image.from_pil(get_image()))
    image_bytes = init_image_on_fal(test_input)

    assert image_bytes == get_image(as_bytes=True)
