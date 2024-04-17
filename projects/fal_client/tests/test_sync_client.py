import io
import pytest
import fal_client
import httpx
from PIL import Image


@pytest.fixture
def client() -> fal_client.SyncClient:
    return fal_client.SyncClient()


def test_fal_client(client: fal_client.SyncClient):
    output = client.run(
        "fal-ai/fast-sdxl",
        arguments={
            "prompt": "a cat",
        },
    )
    assert len(output["images"]) == 1

    handle = client.submit(
        "fal-ai/fast-sdxl/image-to-image",
        arguments={
            "image_url": output["images"][0]["url"],
            "prompt": "an orange cat",
            "seed": 42,
        },
    )

    result = handle.get()
    assert result["seed"] == 42

    status = handle.status(with_logs=False)
    assert isinstance(status, fal_client.Completed)
    assert status.logs is None

    status_w_logs = handle.status(with_logs=True)
    assert isinstance(status_w_logs, fal_client.Completed)
    assert status_w_logs.logs is not None


def test_fal_client_streaming(client: fal_client.SyncClient):
    events = []
    for event in client.stream(
        "fal-ai/llavav15-13b",
        arguments={
            "image_url": "https://llava-vl.github.io/static/images/monalisa.jpg",
            "prompt": "Do you know who drew this painting?",
        },
    ):
        events.append(event)
        print(event)

    assert len(events) >= 2
    assert not events[-1]["partial"]


def test_fal_client_upload(
    client: fal_client.SyncClient,
    tmp_path,
):
    url = client.upload(b"Hello, world!", content_type="text/plain")
    response = httpx.get(url)
    response.raise_for_status()

    assert response.text == "Hello, world!"

    fake_file = tmp_path / "fake.txt"
    fake_file.write_text("from fake.txt")

    url = client.upload_file(fake_file)
    response = httpx.get(url)
    response.raise_for_status()

    assert response.text == "from fake.txt"

    image = Image.new("RGB", (100, 100))

    url = client.upload_image(image)
    response = httpx.get(url)
    response.raise_for_status()

    response_image = Image.open(io.BytesIO(response.content))
    assert response_image.size == (100, 100)
    assert response_image.mode == "RGB"
    assert response_image.getpixel((0, 0)) == (0, 0, 0)


def test_fal_client_encode(client: fal_client.SyncClient, tmp_path):
    image = Image.new("RGB", (1024, 1024))

    url = fal_client.encode_image(image)
    response = client.run(
        "fal-ai/fast-sdxl/image-to-image",
        arguments={"image_url": url, "prompt": "a cat"},
    )

    assert len(response["images"]) == 1
    assert response["images"][0]["width"] == 1024
    assert response["images"][0]["height"] == 1024
