import io
import httpx
import pytest
import fal_client
from fal_client.client import _async_maybe_retry_request

from PIL import Image


@pytest.fixture
async def client() -> fal_client.AsyncClient:
    client = fal_client.AsyncClient()
    try:
        client._get_key()
    except fal_client.auth.MissingCredentialsError:
        pytest.skip("No credentials found")
    return client


async def test_fal_client(client: fal_client.AsyncClient):
    output = await client.run(
        "fal-ai/fast-sdxl",
        arguments={
            "prompt": "a cat",
        },
    )
    assert len(output["images"]) == 1

    handle = await client.submit(
        "fal-ai/fast-sdxl/image-to-image",
        arguments={
            "image_url": output["images"][0]["url"],
            "prompt": "an orange cat",
            "seed": 42,
        },
    )

    result = await handle.get()
    assert result["seed"] == 42

    assert (
        await client.result("fal-ai/fast-sdxl/image-to-image", handle.request_id)
        == result
    )

    status = await handle.status(with_logs=False)
    assert isinstance(status, fal_client.Completed)
    assert status.logs is None

    new_handle = client.get_handle("fal-ai/fast-sdxl/image-to-image", handle.request_id)
    assert new_handle == handle

    status_w_logs = await handle.status(with_logs=True)
    assert isinstance(status_w_logs, fal_client.Completed)
    assert status_w_logs.logs is not None

    assert (
        await client.status(
            "fal-ai/fast-sdxl/image-to-image",
            handle.request_id,
        )
        == status
    )

    output = await client.subscribe(
        "fal-ai/fast-sdxl",
        arguments={
            "prompt": "a cat",
        },
        hint="lora:a",
    )
    assert len(output["images"]) == 1

    output = await client.run(
        "fal-ai/fast-sdxl",
        arguments={
            "prompt": "a cat",
        },
        hint="lora:a",
    )
    assert len(output["images"]) == 1


async def test_fal_client_streaming(client: fal_client.AsyncClient):
    events = []
    async for event in client.stream(
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


async def test_fal_client_upload(
    client: fal_client.AsyncClient,
    tmp_path,
):
    async with httpx.AsyncClient() as httpx_client:
        url = await client.upload(b"Hello, world!", content_type="text/plain")
        response = await httpx_client.get(url)
        response.raise_for_status()

        assert response.text == "Hello, world!"

        fake_file = tmp_path / "fake.txt"
        fake_file.write_text("from fake.txt")

        url = await client.upload_file(fake_file)
        response = await httpx_client.get(url)
        response.raise_for_status()

        assert response.text == "from fake.txt"

        image = Image.new("RGB", (100, 100))

        url = await client.upload_image(image)
        response = await httpx_client.get(url)
        response.raise_for_status()

        response_image = Image.open(io.BytesIO(response.content))
        assert response_image.size == (100, 100)
        assert response_image.mode == "RGB"
        assert response_image.getpixel((0, 0)) == (0, 0, 0)


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
        (
            httpx.HTTPStatusError(
                message="test",
                request=httpx.Request("GET", "https://example.com"),
                response=httpx.Response(status_code=502, text="nginx error"),
            ),
            True,
        ),
        (
            httpx.HTTPStatusError(
                message="test",
                request=httpx.Request("GET", "https://example.com"),
                response=httpx.Response(status_code=504, text="nginx error"),
            ),
            True,
        ),
    ],
)
async def test_retry(mocker, exc, retried, monkeypatch):
    monkeypatch.setattr(fal_client.client, "MAX_ATTEMPTS", 4)
    monkeypatch.setattr(fal_client.client, "BASE_DELAY", 0.1)
    monkeypatch.setattr(fal_client.client, "MAX_DELAY", 0.1)

    httpx_client = mocker.Mock()
    httpx_client.request = mocker.Mock(side_effect=exc)

    with pytest.raises(Exception):
        await _async_maybe_retry_request(httpx_client, "GET", "https://example.com")

    if retried:
        assert httpx_client.request.call_count == 4
    else:
        assert httpx_client.request.call_count == 1
