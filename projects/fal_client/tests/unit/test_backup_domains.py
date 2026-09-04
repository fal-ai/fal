import inspect
import json
from contextlib import asynccontextmanager
from functools import partial

import httpx
import pytest

from fal_client.client import (
    AsyncClient,
    AsyncBackupDomainTransport,
    BackupDomainTransport,
    MAX_ATTEMPTS,
    Completed,
    FalClientHTTPError,
    SyncClient,
    _fallback_url,
)


@pytest.fixture
def make_client(monkeypatch):
    @asynccontextmanager
    async def create(asynchronous, handler, **kwargs):
        transport_type = (
            AsyncBackupDomainTransport if asynchronous else BackupDomainTransport
        )
        monkeypatch.setattr(
            f"fal_client.client.{transport_type.__name__}",
            partial(transport_type, transport=httpx.MockTransport(handler)),
        )

        cls = AsyncClient if asynchronous else SyncClient
        client = cls(key="test-key", **kwargs)
        http = await resolve(client._client)
        try:
            yield client
        finally:
            await resolve(http.aclose() if asynchronous else http.close())

    return create


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://fal.run/a/b?x=a%2Fb", "https://falrun.com/a/b?x=a%2Fb"),
        ("https://queue.fal.run:8443/a", "https://queue.falrun.com:8443/a"),
        ("https://run.fal.dev/a", None),
        ("https://falrun.com/a", None),
        ("https://fal.run.example.com/a", None),
        ("https://rest.fal.ai/tokens/", None),
        ("https://v3.fal.media/files/upload", None),
    ],
)
def test_fallback_domains(url, expected):
    assert _fallback_url(httpx.URL(url)) == expected


async def resolve(value):
    return await value if inspect.isawaitable(value) else value


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_run_and_queue_fallback(make_client, asynchronous):
    requests = []
    queue_url = "https://queue.fal.run/fal-ai/test/requests/request-id"

    def respond(request):
        requests.append(request)
        if request.url.host in ("fal.run", "queue.fal.run"):
            raise httpx.ConnectError("connection failed", request=request)
        if request.url.path.endswith("/status"):
            return httpx.Response(
                200, json={"status": "COMPLETED", "logs": [], "metrics": {}}
            )
        if request.url.host == "queue.falrun.com" and request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "request_id": "request-id",
                    "response_url": queue_url,
                    "status_url": queue_url + "/status",
                    "cancel_url": queue_url + "/cancel",
                },
            )
        return httpx.Response(200, json={"output": "ok"})

    async with make_client(asynchronous, respond) as client:
        assert await resolve(
            client.run(
                "fal-ai/test",
                {"prompt": "test"},
                headers={"x-test": "yes"},
                start_timeout=30,
            )
        ) == {"output": "ok"}
        handle = await resolve(
            client.submit(
                "fal-ai/test",
                {"prompt": "test"},
                webhook_url="https://example.com/hook?a=b",
                start_timeout=30,
            )
        )
        assert isinstance(await resolve(handle.status(with_logs=True)), Completed)
        assert await resolve(handle.get()) == {"output": "ok"}
        await resolve(handle.cancel())
        restored = await resolve(client.get_handle("fal-ai/test", "request-id"))
        assert isinstance(await resolve(restored.status()), Completed)

    # run, submit, status, get, cancel, get_handle + its status, each retried once
    # against the backup domain.
    assert len(requests) == 7 * 2
    for primary, backup in zip(requests[::2], requests[1::2]):
        assert backup.url == _fallback_url(primary.url)
        assert backup.method == primary.method
        assert backup.content == primary.content
        assert backup.headers["authorization"] == "Key test-key"
        assert backup.headers["host"] == backup.url.host
        assert backup.extensions["timeout"] == primary.extensions["timeout"]
        assert {k: v for k, v in backup.headers.items() if k != "host"} == {
            k: v for k, v in primary.headers.items() if k != "host"
        }
    assert requests[1].headers["x-test"] == "yes"
    assert json.loads(requests[1].content) == {"prompt": "test"}
    assert requests[3].url.params["fal_webhook"] == "https://example.com/hook?a=b"
    assert requests[5].url.params["logs"] == "true"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "host,outcome",
    [
        ("fal.run", 200),
        ("fal.run", 500),
        ("fal.run", httpx.ReadError("read failed")),
        ("fal.run", httpx.WriteError("write failed")),
        ("custom.example.com", httpx.ConnectError("connect failed")),
    ],
)
async def test_http_does_not_fallback(make_client, asynchronous, host, outcome):
    requests = []

    def respond(request):
        requests.append(request)
        if isinstance(outcome, Exception):
            raise outcome
        return httpx.Response(outcome)

    async with make_client(asynchronous, respond) as client:
        http = await resolve(client._client)
        if isinstance(outcome, Exception):
            with pytest.raises(type(outcome)) as exc:
                await resolve(http.post(f"https://{host}/model", json={}))
            assert exc.value is outcome
        else:
            response = await resolve(http.post(f"https://{host}/model", json={}))
            assert response.status_code == outcome
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_stream_fallback(make_client, asynchronous):
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.host == "fal.run":
            raise httpx.ConnectError("DNS failed", request=request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content='data: {"output":"ok"}\n\n',
        )

    async with make_client(asynchronous, respond) as client:
        stream = client.stream("fal-ai/test", {"prompt": "test"})
        events = [event async for event in stream] if asynchronous else list(stream)
        assert events == [{"output": "ok"}]
    assert [r.url.host for r in requests] == ["fal.run", "falrun.com"]
    assert requests[1].content == requests[0].content
    assert requests[1].headers["authorization"] == "Key test-key"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_stream_failure_after_open_does_not_fallback(make_client, asynchronous):
    requests = []
    closed = []

    class FailingStream(httpx.SyncByteStream, httpx.AsyncByteStream):
        def __iter__(self):
            yield b'data: {"output":"first"}\n\n'
            raise httpx.ReadError("stream interrupted")

        async def __aiter__(self):
            for chunk in self:
                yield chunk

        def close(self):
            closed.append(True)

        async def aclose(self):
            self.close()

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=FailingStream()
        )

    async with make_client(asynchronous, respond) as client:
        with pytest.raises(httpx.ReadError, match="stream interrupted"):
            stream = client.stream("fal-ai/test", {})
            if asynchronous:
                async for _ in stream:
                    pass
            else:
                for _ in stream:
                    pass
    assert len(requests) == 1
    assert closed == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "operation,timeout,default_timeout,expected_read,expected_connect",
    [
        ("run", None, 120, None, 5),
        ("run", 20, 120, 20, 5),
        ("run", 2, 120, 2, 2),
        ("stream", None, 120, None, 5),
        ("stream", 20, 120, 20, 5),
        ("stream", 2, 120, 2, 2),
        ("submit", None, 120, 120, 5),
        ("submit", None, 2, 2, 2),
        ("subscribe", None, 120, 120, 5),
        ("status", None, 120, 120, 5),
    ],
)
async def test_public_connect_timeouts_and_start_timeout(
    make_client,
    asynchronous,
    operation,
    timeout,
    default_timeout,
    expected_read,
    expected_connect,
):
    requests = []
    queue_url = "https://queue.fal.run/fal-ai/test/requests/test-id"

    def respond(request):
        requests.append(request)
        if request.url.host in ("fal.run", "queue.fal.run"):
            raise httpx.ConnectTimeout("handshake hung", request=request)
        if request.url.path.endswith("/status"):
            return httpx.Response(
                200, json={"status": "COMPLETED", "logs": [], "metrics": {}}
            )
        if request.url.path.endswith("/stream"):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content='data: {"ok":true}\n\n',
            )
        if request.url.host == "queue.falrun.com" and request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "request_id": "test-id",
                    "response_url": queue_url,
                    "status_url": queue_url + "/status",
                    "cancel_url": queue_url + "/cancel",
                },
            )
        return httpx.Response(200, json={"ok": True})

    async with make_client(
        asynchronous, respond, default_timeout=default_timeout
    ) as client:
        if operation == "stream":
            stream = client.stream("fal-ai/test", {}, timeout=timeout)
            result = [event async for event in stream] if asynchronous else list(stream)
            assert result == [{"ok": True}]
        elif operation == "status":
            await resolve(getattr(client, operation)("fal-ai/test", "test-id"))
        elif operation == "run":
            assert await resolve(
                client.run("fal-ai/test", {}, timeout=timeout, start_timeout=30)
            ) == {"ok": True}
        else:
            await resolve(getattr(client, operation)("fal-ai/test", {}, start_timeout=30))
    for request in requests:
        assert request.extensions["timeout"] == {
            "connect": expected_connect,
            "read": expected_read,
            "write": expected_read,
            "pool": expected_read,
        }
    hosts = (
        ["fal.run", "falrun.com"]
        if operation in ("run", "stream")
        else ["queue.fal.run", "queue.falrun.com"]
    )
    assert [request.url.host for request in requests] == hosts * (
        3 if operation == "subscribe" else 1
    )
    if operation not in ("stream", "status"):
        assert requests[0].headers["X-Fal-Request-Timeout"] == "30.0"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "host,timeout,expected_connect",
    [
        ("fal.run", None, 5),
        ("falrun.com", None, 5),
        ("queue.falrun.com", None, 5),
        ("custom.example.com", None, None),
        ("custom.example.com", httpx.USE_CLIENT_DEFAULT, 120),
        ("rest.fal.ai", httpx.USE_CLIENT_DEFAULT, 120),
        ("v3.fal.media", httpx.USE_CLIENT_DEFAULT, 120),
    ],
)
async def test_connect_timeout_is_capped_only_on_mapped_domains(
    make_client, asynchronous, host, timeout, expected_connect
):
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.host == "fal.run":
            raise httpx.ConnectTimeout("primary unavailable", request=request)
        return httpx.Response(200, json={})

    async with make_client(asynchronous, respond) as client:
        http = await resolve(client._client)
        await resolve(http.get(f"https://{host}/model", timeout=timeout))
    assert requests
    for request in requests:
        assert request.extensions["timeout"] == {
            "connect": expected_connect,
            "read": None if timeout is None else 120,
            "write": None if timeout is None else 120,
            "pool": None if timeout is None else 120,
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_backup_log_omits_paths_and_credentials(
    make_client, caplog, asynchronous
):
    def respond(request):
        if request.url.host == "fal.run":
            raise httpx.ConnectTimeout("failed with secret-token", request=request)
        return httpx.Response(200, json={})

    async with make_client(asynchronous, respond) as client:
        http = await resolve(client._client)
        response = await resolve(
            http.get("https://fal.run/private-path?fal_jwt_token=secret-token")
        )

    assert response.url.host == "fal.run"
    records = [
        record for record in caplog.records if record.name == "fal_client.client"
    ]
    assert len(records) == 1
    assert records[0].levelname == "WARNING"
    assert records[0].message == (
        "Connection to fal.run failed (ConnectTimeout); trying backup domain falrun.com"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "asynchronous,follow_redirects", [(False, None), (True, None), (True, True)]
)
async def test_public_run_preserves_redirect_policy(
    make_client, asynchronous, follow_redirects
):
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.path == "/fal-ai/model":
            return httpx.Response(
                307, headers={"location": "/fal-ai/redirected?input=a%2Fb"}
            )
        if request.url.host == "fal.run":
            raise httpx.ConnectError("redirected host unavailable", request=request)
        return httpx.Response(200, json={"ok": True})

    async with make_client(asynchronous, respond) as client:
        if follow_redirects is not None:
            http = await resolve(client._client)
            http.follow_redirects = follow_redirects
        if asynchronous and follow_redirects is None:
            with pytest.raises(FalClientHTTPError) as exc:
                await client.run("fal-ai/model", {"prompt": "test"})
            assert exc.value.response.status_code == 307
            assert len(requests) == 1
            return
        assert await resolve(client.run("fal-ai/model", {"prompt": "test"})) == {
            "ok": True
        }
    assert [(r.url.host, r.url.path) for r in requests] == [
        ("fal.run", "/fal-ai/model"),
        ("fal.run", "/fal-ai/redirected"),
        ("falrun.com", "/fal-ai/redirected"),
    ]
    assert all(r.method == "POST" for r in requests)
    assert len({r.content for r in requests}) == 1
    assert requests[1].url.query == requests[2].url.query == b"input=a%2Fb"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "start_timeout,primary_type,backup_type",
    [
        # A backup that also fails to connect re-raises the primary error, so the
        # primary's type decides whether the request is retried.
        (None, httpx.ConnectError, httpx.ConnectError),
        (None, httpx.ConnectTimeout, httpx.ConnectTimeout),
        # A backup that connects and then fails keeps its own error and type.
        (None, httpx.ConnectError, httpx.ReadTimeout),
        # A caller-supplied start_timeout stops retries for timeout errors only.
        (30, httpx.ConnectTimeout, httpx.ConnectTimeout),
        (30, httpx.ConnectError, httpx.ConnectError),
        (30, httpx.ConnectError, httpx.ReadTimeout),
    ],
)
async def test_failure_type_preserves_retry_policy(
    make_client,
    monkeypatch,
    asynchronous,
    start_timeout,
    primary_type,
    backup_type,
):
    requests = []
    errors = []
    monkeypatch.setattr("fal_client.client._get_retry_delay", lambda *args: 0)

    def respond(request):
        requests.append(request)
        error = (
            primary_type("primary failure", request=request)
            if request.url.host == "fal.run"
            else backup_type("backup failure", request=request)
        )
        errors.append(error)
        raise error

    expected_type = backup_type if backup_type is httpx.ReadTimeout else primary_type
    async with make_client(asynchronous, respond) as client:
        with pytest.raises(expected_type) as exc:
            await resolve(client.run("fal-ai/test", {}, start_timeout=start_timeout))
        assert type(exc.value) is expected_type
        primary_error, backup_error = errors[-2:]
        if backup_type is httpx.ReadTimeout:
            assert exc.value is backup_error
            assert exc.value.__context__ is primary_error
        else:
            assert exc.value is primary_error
            assert exc.value.__context__ is backup_error
        assert exc.value.__cause__ is None
    attempts = (
        1
        if start_timeout is not None
        and issubclass(expected_type, httpx.TimeoutException)
        else MAX_ATTEMPTS
    )
    assert [r.url.host for r in requests] == ["fal.run", "falrun.com"] * attempts
    assert exc.value.request.url.host == "fal.run"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_backup_transport_closes_injected_transport(asynchronous):
    closed = []

    class Transport(httpx.MockTransport):
        def close(self):
            closed.append(True)

        async def aclose(self):
            closed.append(True)

    transport = Transport(lambda request: httpx.Response(200))
    if asynchronous:
        async with httpx.AsyncClient(
            transport=AsyncBackupDomainTransport(transport=transport)
        ) as client:
            await client.get("https://fal.run/test")
    else:
        with httpx.Client(
            transport=BackupDomainTransport(transport=transport)
        ) as client:
            client.get("https://fal.run/test")
    assert closed == [True]
