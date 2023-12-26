from __future__ import annotations

from base64 import b64encode
from io import BytesIO

import pytest
from fal.toolkit import Image, mainify
from PIL import Image as PILImage
from pydantic import BaseModel, Field


@mainify
def get_image(as_pil: bool = True):
    pil_image = PILImage.new("RGB", (1, 1), (255, 255, 255))
    if as_pil:
        return pil_image

    return pil_image_to_bytes(pil_image)


@mainify
def pil_image_to_bytes(image: PILImage.Image) -> bytes:
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    return image_bytes.getvalue()


def fal_image_downloaded(image: Image):
    return image.file_size != None


def fal_image_url_matches(image: Image, url: str):
    return image.url == url


def fal_image_content_matches(image: Image, content: bytes):
    image1 = PILImage.open(BytesIO(image.as_bytes()))
    image2 = PILImage.open(BytesIO(content))
    return images_are_equal(image1, image2)


@mainify
def image_to_data_uri(image: PILImage.Image) -> str:
    image_bytes = pil_image_to_bytes(image)
    b64_encoded = b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64_encoded}"


def images_are_equal(img1: PILImage.Image, img2: PILImage.Image) -> bool:
    pixels1 = list(img1.getdata())
    pixels2 = list(img2.getdata())
    return pixels1 == pixels2


def assert_fal_images_equal(fal_image_1: Image, fal_image_2: Image):
    assert (
        fal_image_1.file_size == fal_image_2.file_size
    ), "Image file size should match"
    assert (
        fal_image_1.content_type == fal_image_2.content_type
    ), "Content type should match"
    assert fal_image_1.url == fal_image_2.url, "URL should match"
    assert fal_image_1.width == fal_image_2.width, "Width should match"
    assert fal_image_1.height == fal_image_2.height, "Height should match"


def test_image_matches():
    pil_image = get_image()

    image_file = Image.from_pil(pil_image, repository="in_memory")
    output_pil_image = PILImage.open(BytesIO(image_file.as_bytes()))

    assert images_are_equal(output_pil_image, pil_image)


def test_fal_image_from_pil(isolated_client):
    def fal_image_from_pil():
        pil_image = get_image()
        return Image.from_pil(pil_image, repository="in_memory")

    @isolated_client(requirements=["pillow", "pydantic==1.10.12"])
    def fal_image_from_bytes_remote():
        return fal_image_from_pil()

    local_image = fal_image_from_pil()
    remote_image = fal_image_from_bytes_remote()

    assert fal_image_content_matches(remote_image, get_image(as_pil=False))

    assert_fal_images_equal(local_image, remote_image)


def test_fal_image_from_bytes(isolated_client):
    image_bytes = get_image(as_pil=False)

    def fal_image_from_bytes():
        return Image.from_bytes(image_bytes, repository="in_memory")

    @isolated_client(requirements=["pillow", "pydantic==1.10.12"])
    def fal_image_from_bytes_remote():
        return fal_image_from_bytes()

    local_image = fal_image_from_bytes()
    remote_image = fal_image_from_bytes_remote()

    assert fal_image_content_matches(remote_image, image_bytes)
    assert_fal_images_equal(local_image, remote_image)


@pytest.mark.parametrize(
    "image_url",
    [
        "https://storage.googleapis.com/falserverless/model_tests/remove_background/elephant.jpg",
        image_to_data_uri(get_image()),
    ],
)
def test_fal_image_input(isolated_client, image_url):
    class TestInput(BaseModel):
        image: Image = Field()

    def test_input():
        return TestInput(image=image_url).image

    @isolated_client(requirements=["pillow", "pydantic==1.10.12"])
    def test_input_remote():
        return test_input()

    local_input_image = test_input()
    remote_input_image = test_input_remote()

    # Image is not downloaded until it is needed
    assert not fal_image_downloaded(local_input_image)
    assert not fal_image_downloaded(remote_input_image)

    assert fal_image_url_matches(local_input_image, image_url)

    # Image will be downloaded when trying to access its content
    assert_fal_images_equal(local_input_image, remote_input_image)
    assert fal_image_downloaded(local_input_image)
