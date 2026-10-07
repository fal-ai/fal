import inspect

import httpx
import pytest

from fal_client.client import AsyncClient, SyncClient


async def resolve(value):
    return await value if inspect.isawaitable(value) else value


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_oauth_provider_refreshes_retained_queue_handles(
    monkeypatch, asynchronous
):
    monkeypatch.setenv("FAL_KEY", "ambient-key")
    monkeypatch.setattr("fal_client.client._get_retry_delay", lambda *args: 0)
    queue_url = "https://queue.fal.run/fal-ai/test/requests/request-id"
    requests = []
    tokens = iter(f"token-{n}" for n in range(1, 10))

    def provider():
        return next(tokens)

    async def async_provider():
        return provider()

    def respond(request):
        requests.append(request)
        if len(requests) == 2:
            return httpx.Response(429)
        if request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "request_id": "request-id",
                    "response_url": queue_url,
                    "status_url": queue_url + "/status",
                    "cancel_url": queue_url + "/cancel",
                },
            )
        if request.url.path.endswith("/status"):
            return httpx.Response(
                200, json={"status": "COMPLETED", "logs": [], "metrics": {}}
            )
        return httpx.Response(200, json={"output": "ok"})

    transport = (
        "AsyncBackupDomainTransport" if asynchronous else "BackupDomainTransport"
    )
    monkeypatch.setattr(
        f"fal_client.client.{transport}", lambda: httpx.MockTransport(respond)
    )
    cls = AsyncClient if asynchronous else SyncClient
    client = cls(access_token=async_provider if asynchronous else provider)
    http = await resolve(client._client)
    try:
        handle = await resolve(client.submit("fal-ai/test", {}))
        await resolve(handle.status())
        assert await resolve(handle.get()) == {"output": "ok"}
        await resolve(handle.cancel())
    finally:
        await resolve(http.aclose() if asynchronous else http.close())
    assert [r.headers["Authorization"] for r in requests] == [
        f"Bearer token-{n}" for n in range(1, 7)
    ]
    assert requests[1].url.params["logs"] == "false"
    assert requests[-1].method == "PUT" and requests[-1].url.path.endswith("/cancel")


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_oauth_bearer_stays_on_trusted_api_origins(monkeypatch, asynchronous):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200)

    transport = (
        "AsyncBackupDomainTransport" if asynchronous else "BackupDomainTransport"
    )
    monkeypatch.setattr(
        f"fal_client.client.{transport}", lambda: httpx.MockTransport(respond)
    )
    client = (AsyncClient if asynchronous else SyncClient)(access_token="token")
    http = await resolve(client._client)
    try:
        for url in (
            "https://rest.fal.ai/storage/upload/initiate",
            "https://v3.fal.media/files/file?signature=object-capability",
            "https://queue.fal.run.attacker.test/path",
            "http://fal.run/path",
            "https://fal.run:8443/path",
        ):
            await resolve(http.get(url, headers={"Authorization": "Bearer override"}))
    finally:
        await resolve(http.aclose() if asynchronous else http.close())
    assert requests[0].headers["Authorization"] == "Bearer token"
    assert all("Authorization" not in r.headers for r in requests[1:])
    assert "token" not in repr(client)


def test_oauth_and_key_cannot_be_combined():
    for cls in (SyncClient, AsyncClient):
        with pytest.raises(ValueError, match="either key or access_token"):
            cls(key="key", access_token="token")
