import io
import pytest
import fal_client
import httpx
from PIL import Image

from fal_client.client import _maybe_retry_request


@pytest.fixture
def client() -> fal_client.SyncClient:
    client = fal_client.SyncClient()
    try:
        client._get_key()
    except fal_client.auth.MissingCredentialsError:
        pytest.skip("Missing credentials")
    return client


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

    assert client.result("fal-ai/fast-sdxl/image-to-image", handle.request_id) == result

    status = handle.status(with_logs=False)
    assert isinstance(status, fal_client.Completed)
    assert status.logs is None

    status_w_logs = handle.status(with_logs=True)
    assert isinstance(status_w_logs, fal_client.Completed)
    assert status_w_logs.logs is not None

    new_handle = client.get_handle("fal-ai/fast-sdxl/image-to-image", handle.request_id)
    assert new_handle == handle

    assert client.status("fal-ai/fast-sdxl/image-to-image", handle.request_id) == status

    output = client.subscribe(
        "fal-ai/fast-sdxl",
        arguments={
            "prompt": "a cat",
        },
        hint="lora:a",
    )
    assert len(output["images"]) == 1

    output = client.run(
        "fal-ai/fast-sdxl",
        arguments={
            "prompt": "a cat",
        },
        hint="lora:a",
    )
    assert len(output["images"]) == 1


def test_fal_client_streaming(client: fal_client.SyncClient):
    events = []
    for event in client.stream(
        "fal-ai/any-llm",
        arguments={
            "model": "google/gemini-flash-1.5",
            "prompt": "Tell me a joke",
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


@pytest.mark.parametrize(
    "exc, retried",
    [
        # not retryable
        (Exception("test"), False),
        (httpx.RequestError(message="test"), False),
        (
            httpx.HTTPStatusError(
                message="test",
                request=httpx.Request("GET", "https://example.com"),
                response=httpx.Response(status_code=400),
            ),
            False,
        ),
        # retryable
        (httpx.TimeoutException(message="test"), True),
        (httpx.ConnectTimeout(message="test"), True),
        (httpx.ReadTimeout(message="test"), True),
        (httpx.WriteTimeout(message="test"), True),
        (httpx.PoolTimeout(message="test"), True),
        (httpx.NetworkError(message="test"), True),
        (httpx.ConnectError(message="test"), True),
        (httpx.ReadError(message="test"), True),
        (httpx.WriteError(message="test"), True),
        (httpx.CloseError(message="test"), True),
        (httpx.ProtocolError(message="test"), True),
        (httpx.LocalProtocolError(message="test"), True),
        (httpx.RemoteProtocolError(message="test"), True),
        (httpx.ProxyError(message="test"), True),
        (httpx.UnsupportedProtocol(message="test"), True),
        (
            httpx.HTTPStatusError(
                message="test",
                request=httpx.Request("GET", "https://example.com"),
                response=httpx.Response(status_code=408),
            ),
            True,
        ),
        (
            httpx.HTTPStatusError(
                message="test",
                request=httpx.Request("GET", "https://example.com"),
                response=httpx.Response(status_code=409),
            ),
            True,
        ),
        (
            httpx.HTTPStatusError(
                message="test",
                request=httpx.Request("GET", "https://example.com"),
                response=httpx.Response(status_code=429),
            ),
            True,
        ),
    ],
)
def test_retry(mocker, exc, retried, monkeypatch):
    monkeypatch.setattr(fal_client.client, "MAX_ATTEMPTS", 4)
    monkeypatch.setattr(fal_client.client, "BASE_DELAY", 0.1)
    monkeypatch.setattr(fal_client.client, "MAX_DELAY", 0.1)

    httpx_client = mocker.Mock()
    httpx_client.request = mocker.Mock(side_effect=exc)

    with pytest.raises(Exception):
        _maybe_retry_request(httpx_client, "GET", "https://example.com")

    if retried:
        assert httpx_client.request.call_count == 4
    else:
        assert httpx_client.request.call_count == 1
