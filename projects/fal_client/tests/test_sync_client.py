import pytest
import fal_client


@pytest.fixture
def client() -> fal_client.SyncClient:
    return fal_client.SyncClient()


def test_fal_client(client: fal_client.SyncClient):
    output = client.run(
        "fal-ai/fast-sdxl",
        data={
            "prompt": "a cat",
        },
    )
    assert len(output["images"]) == 1

    handle = client.submit(
        "fal-ai/fast-sdxl/image-to-image",
        data={
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
        data={
            "image_url": "https://llava-vl.github.io/static/images/monalisa.jpg",
            "prompt": "Do you know who drew this painting?",
        },
    ):
        events.append(event)
        print(event)

    assert len(events) >= 2
    assert not events[-1]["partial"]
