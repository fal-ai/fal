import pytest
import fal_client


@pytest.fixture
async def client() -> fal_client.AsyncClient:
    return fal_client.AsyncClient()


async def test_fal_client(client: fal_client.AsyncClient):
    output = await client.run(
        "fal-ai/fast-sdxl",
        data={
            "prompt": "a cat",
        },
    )
    assert len(output["images"]) == 1

    handle = await client.submit(
        "fal-ai/fast-sdxl/image-to-image",
        data={
            "image_url": output["images"][0]["url"],
            "prompt": "an orange cat",
            "seed": 42,
        },
    )

    result = await handle.get()
    assert result["seed"] == 42

    status = await handle.status(with_logs=False)
    assert isinstance(status, fal_client.Completed)
    assert status.logs is None

    status_w_logs = await handle.status(with_logs=True)
    assert isinstance(status_w_logs, fal_client.Completed)
    assert status_w_logs.logs is not None


async def test_fal_client_streaming(client: fal_client.AsyncClient):
    events = []
    async for event in client.stream(
        "fal-ai/llavav15-13b",
        data={
            "image_url": "https://llava-vl.github.io/static/images/monalisa.jpg",
            "prompt": "Do you know who drew this painting?",
        },
    ):
        events.append(event)
        print(event)

    assert len(events) >= 2
    assert not events[-1]["partial"]
