import pytest
import httpx
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from fal_client.client import (
    Queued,
    InProgress,
    Completed,
    _BaseRequestHandle,
    FalClientHTTPError,
    SyncClient,
    AsyncClient,
    SyncRequestHandle,
    AsyncRequestHandle,
)


@pytest.mark.parametrize(
    "data, result, raised",
    [
        (
            {"status": "IN_QUEUE", "queue_position": 123},
            Queued(position=123),
            False,
        ),
        (
            {"status": "IN_PROGRESS", "logs": [{"msg": "foo"}, {"msg": "bar"}]},
            InProgress(logs=[{"msg": "foo"}, {"msg": "bar"}]),
            False,
        ),
        (
            {"status": "COMPLETED", "logs": [{"msg": "foo"}, {"msg": "bar"}]},
            Completed(logs=[{"msg": "foo"}, {"msg": "bar"}], metrics={}),
            False,
        ),
        (
            {
                "status": "COMPLETED",
                "logs": [{"msg": "foo"}, {"msg": "bar"}],
                "metrics": {"m1": "v1", "m2": "v2"},
            },
            Completed(
                logs=[{"msg": "foo"}, {"msg": "bar"}], metrics={"m1": "v1", "m2": "v2"}
            ),
            False,
        ),
        (
            {"status": "FOO"},
            ValueError,
            True,
        ),
    ],
)
def test_parse_status(data, result, raised):
    handle = _BaseRequestHandle("foo", "bar", "baz", "qux")

    if raised:
        with pytest.raises(result):
            handle._parse_status(data)
    else:
        assert handle._parse_status(data) == result


def test_sync_client_run_with_headers():
    """Test that custom headers are passed through in run()"""
    with patch("fal_client.client._maybe_retry_request") as mock_request:
        mock_response = Mock()
        mock_response.json.return_value = {"result": "success"}
        mock_request.return_value = mock_response

        client = SyncClient(key="test-key")
        custom_headers = {"X-Custom-Header": "test-value", "X-Trace-Id": "123"}

        result = client.run(
            "test-app",
            {"input": "data"},
            headers=custom_headers,
        )

        assert result == {"result": "success"}
        # Verify headers were passed to the request
        call_kwargs = mock_request.call_args[1]
        assert "headers" in call_kwargs
        assert call_kwargs["headers"]["X-Custom-Header"] == "test-value"
        assert call_kwargs["headers"]["X-Trace-Id"] == "123"


def test_sync_client_run_with_headers_and_hint():
    """Test that custom headers are merged with hint header"""
    with patch("fal_client.client._maybe_retry_request") as mock_request:
        mock_response = Mock()
        mock_response.json.return_value = {"result": "success"}
        mock_request.return_value = mock_response

        client = SyncClient(key="test-key")
        custom_headers = {"X-Custom-Header": "test-value"}

        client.run(
            "test-app",
            {"input": "data"},
            hint="lora:a",
            headers=custom_headers,
        )

        # Verify both hint and custom headers are present
        call_kwargs = mock_request.call_args[1]
        assert "headers" in call_kwargs
        assert call_kwargs["headers"]["X-Fal-Runner-Hint"] == "lora:a"
        assert call_kwargs["headers"]["X-Custom-Header"] == "test-value"


def test_sync_client_submit_with_headers():
    """Test that custom headers are passed through in submit()"""
    with patch("fal_client.client._maybe_retry_request") as mock_request:
        mock_response = Mock()
        mock_response.json.return_value = {
            "request_id": "req-123",
            "response_url": "http://response",
            "status_url": "http://status",
            "cancel_url": "http://cancel",
        }
        mock_request.return_value = mock_response

        client = SyncClient(key="test-key")
        custom_headers = {"X-Request-Id": "abc-123"}

        handle = client.submit(
            "test-app",
            {"input": "data"},
            headers=custom_headers,
        )

        assert handle.request_id == "req-123"
        # Verify headers were passed to the request
        call_kwargs = mock_request.call_args[1]
        assert "headers" in call_kwargs
        assert call_kwargs["headers"]["X-Request-Id"] == "abc-123"


def test_sync_client_submit_with_headers_and_priority():
    """Test that custom headers are merged with priority header"""
    with patch("fal_client.client._maybe_retry_request") as mock_request:
        mock_response = Mock()
        mock_response.json.return_value = {
            "request_id": "req-123",
            "response_url": "http://response",
            "status_url": "http://status",
            "cancel_url": "http://cancel",
        }
        mock_request.return_value = mock_response

        client = SyncClient(key="test-key")
        custom_headers = {"X-Custom": "value"}

        client.submit(
            "test-app",
            {"input": "data"},
            priority="low",
            headers=custom_headers,
        )

        # Verify both priority and custom headers are present
        call_kwargs = mock_request.call_args[1]
        assert "headers" in call_kwargs
        assert call_kwargs["headers"]["X-Fal-Queue-Priority"] == "low"
        assert call_kwargs["headers"]["X-Custom"] == "value"


def test_sync_client_subscribe_with_headers():
    """Test that custom headers are passed through in subscribe()"""
    with patch("fal_client.client._maybe_retry_request") as mock_request:
        # Mock submit response
        submit_response = Mock()
        submit_response.json.return_value = {
            "request_id": "req-123",
            "response_url": "http://response",
            "status_url": "http://status",
            "cancel_url": "http://cancel",
        }

        # Mock status response
        status_response = Mock()
        status_response.json.return_value = {
            "status": "COMPLETED",
            "logs": [],
        }

        # Mock result response
        result_response = Mock()
        result_response.json.return_value = {"result": "done"}

        mock_request.side_effect = [submit_response, status_response, result_response]

        client = SyncClient(key="test-key")
        custom_headers = {"X-Trace-Id": "trace-123"}

        result = client.subscribe(
            "test-app",
            {"input": "data"},
            headers=custom_headers,
        )

        assert result == {"result": "done"}
        # Verify headers were passed to the submit request (first call)
        first_call_kwargs = mock_request.call_args_list[0][1]
        assert "headers" in first_call_kwargs
        assert first_call_kwargs["headers"]["X-Trace-Id"] == "trace-123"


@pytest.mark.asyncio
async def test_async_client_run_with_headers():
    """Test that custom headers are passed through in async run()"""
    with patch(
        "fal_client.client._async_maybe_retry_request", new_callable=AsyncMock
    ) as mock_request:
        mock_response = Mock()
        mock_response.json.return_value = {"result": "success"}
        mock_request.return_value = mock_response

        client = AsyncClient(key="test-key")
        custom_headers = {"X-Custom-Header": "test-value", "X-Trace-Id": "123"}

        result = await client.run(
            "test-app",
            {"input": "data"},
            headers=custom_headers,
        )

        assert result == {"result": "success"}
        # Verify headers were passed to the request
        call_kwargs = mock_request.call_args[1]
        assert "headers" in call_kwargs
        assert call_kwargs["headers"]["X-Custom-Header"] == "test-value"
        assert call_kwargs["headers"]["X-Trace-Id"] == "123"


@pytest.mark.asyncio
async def test_async_client_submit_with_headers():
    """Test that custom headers are passed through in async submit()"""
    with patch(
        "fal_client.client._async_maybe_retry_request", new_callable=AsyncMock
    ) as mock_request:
        mock_response = Mock()
        mock_response.json.return_value = {
            "request_id": "req-456",
            "response_url": "http://response",
            "status_url": "http://status",
            "cancel_url": "http://cancel",
        }
        mock_request.return_value = mock_response

        client = AsyncClient(key="test-key")
        custom_headers = {"X-Request-Id": "xyz-789"}

        handle = await client.submit(
            "test-app",
            {"input": "data"},
            headers=custom_headers,
        )

        assert handle.request_id == "req-456"
        # Verify headers were passed to the request
        call_kwargs = mock_request.call_args[1]
        assert "headers" in call_kwargs
        assert call_kwargs["headers"]["X-Request-Id"] == "xyz-789"


@pytest.mark.asyncio
async def test_async_client_subscribe_with_headers():
    """Test that custom headers are passed through in async subscribe()"""
    with patch(
        "fal_client.client._async_maybe_retry_request", new_callable=AsyncMock
    ) as mock_request:
        # Mock submit response
        submit_response = Mock()
        submit_response.json.return_value = {
            "request_id": "req-789",
            "response_url": "http://response",
            "status_url": "http://status",
            "cancel_url": "http://cancel",
        }

        # Mock status response
        status_response = Mock()
        status_response.json.return_value = {
            "status": "COMPLETED",
            "logs": [],
        }

        # Mock result response
        result_response = Mock()
        result_response.json.return_value = {"result": "async_done"}

        mock_request.side_effect = [submit_response, status_response, result_response]

        client = AsyncClient(key="test-key")
        custom_headers = {"X-Correlation-Id": "corr-456"}

        result = await client.subscribe(
            "test-app",
            {"input": "data"},
            headers=custom_headers,
        )

        assert result == {"result": "async_done"}
        # Verify headers were passed to the submit request (first call)
        first_call_kwargs = mock_request.call_args_list[0][1]
        assert "headers" in first_call_kwargs
        assert first_call_kwargs["headers"]["X-Correlation-Id"] == "corr-456"


def test_sync_handle_retries(monkeypatch):
    import fal_client.client as client_mod

    monkeypatch.setattr(client_mod, "MAX_ATTEMPTS", 3)
    monkeypatch.setattr(client_mod, "BASE_DELAY", 0.0)
    monkeypatch.setattr(client_mod, "MAX_DELAY", 0.0)

    client = httpx.Client()
    handle = SyncRequestHandle(
        request_id="r-sync",
        response_url="http://resp",
        status_url="http://status",
        cancel_url="http://cancel",
        client=client,
    )

    # Mock iter_events to skip waiting
    def _iter_events(self, with_logs: bool = False, interval: float = 0.1):
        return iter([Completed(logs=[], metrics={})])

    monkeypatch.setattr(SyncRequestHandle, "iter_events", _iter_events, raising=True)

    # Prepare sequence:
    # 1) status: 408 -> 429 -> 200(COMPLETED)
    # 2) get:    408 -> 429 -> 200({"ok": true})
    # 3) cancel: 408 -> 429 -> 200()
    req_status = httpx.Request("GET", "http://status")
    status_408 = httpx.Response(408, request=req_status)
    status_429 = httpx.Response(429, request=req_status)
    status_ok = httpx.Response(
        200,
        content=b'{"status": "COMPLETED", "logs": [], "metrics": {}}',
        headers={"Content-Type": "application/json"},
        request=req_status,
    )

    req_get = httpx.Request("GET", "http://resp")
    get_408 = httpx.Response(408, request=req_get)
    get_429 = httpx.Response(429, request=req_get)
    get_ok = httpx.Response(
        200,
        content=b'{"ok": true}',
        headers={"Content-Type": "application/json"},
        request=req_get,
    )

    req_put = httpx.Request("PUT", "http://cancel")
    cancel_408 = httpx.Response(408, request=req_put)
    cancel_429 = httpx.Response(429, request=req_put)
    cancel_ok = httpx.Response(200, request=req_put)

    client.request = Mock(
        side_effect=[
            status_408,
            status_429,
            status_ok,
            get_408,
            get_429,
            get_ok,
            cancel_408,
            cancel_429,
            cancel_ok,
        ]
    )

    # Status
    status = handle.status(with_logs=False)
    assert isinstance(status, Completed)

    # Get
    result = handle.get()
    assert result == {"ok": True}

    # Cancel
    handle.cancel()

    # 3 per operation
    assert client.request.call_count == 9


@pytest.mark.asyncio
async def test_async_handle_retries(monkeypatch):
    import fal_client.client as client_mod

    monkeypatch.setattr(client_mod, "MAX_ATTEMPTS", 3)
    monkeypatch.setattr(client_mod, "BASE_DELAY", 0.0)
    monkeypatch.setattr(client_mod, "MAX_DELAY", 0.0)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    client = httpx.AsyncClient()
    handle = AsyncRequestHandle(
        request_id="r-async",
        response_url="http://resp",
        status_url="http://status",
        cancel_url="http://cancel",
        client=client,
    )

    # Mock iter_events to skip waiting
    async def _iter_events(self, with_logs: bool = False, interval: float = 0.1):
        yield Completed(logs=[], metrics={})

    monkeypatch.setattr(AsyncRequestHandle, "iter_events", _iter_events, raising=True)

    # Prepare sequence:
    # 1) status: 408 -> 429 -> 200(COMPLETED)
    # 2) get:    408 -> 429 -> 200({"ok": true})
    # 3) cancel: 408 -> 429 -> 200()
    req_status = httpx.Request("GET", "http://status")
    status_408 = httpx.Response(408, request=req_status)
    status_429 = httpx.Response(429, request=req_status)
    status_ok = httpx.Response(
        200,
        content=b'{"status": "COMPLETED", "logs": [], "metrics": {}}',
        headers={"Content-Type": "application/json"},
        request=req_status,
    )

    req_get = httpx.Request("GET", "http://resp")
    get_408 = httpx.Response(408, request=req_get)
    get_429 = httpx.Response(429, request=req_get)
    get_ok = httpx.Response(
        200,
        content=b'{"ok": true}',
        headers={"Content-Type": "application/json"},
        request=req_get,
    )

    req_put = httpx.Request("PUT", "http://cancel")
    cancel_408 = httpx.Response(408, request=req_put)
    cancel_429 = httpx.Response(429, request=req_put)
    cancel_ok = httpx.Response(200, request=req_put)

    client.request = AsyncMock(
        side_effect=[
            status_408,
            status_429,
            status_ok,
            get_408,
            get_429,
            get_ok,
            cancel_408,
            cancel_429,
            cancel_ok,
        ]
    )

    # Status
    status = await handle.status(with_logs=False)
    assert isinstance(status, Completed)

    # Get
    result = await handle.get()
    assert result == {"ok": True}

    # Cancel
    await handle.cancel()

    # 3 per operation
    assert client.request.await_count == 9


@pytest.mark.parametrize("status_code", [500, 503])
def test_sync_get_does_not_retry_on_500_503(monkeypatch, status_code):
    import fal_client.client as client_mod

    # No retries; still set to ensure if retry attempted it would be fast
    monkeypatch.setattr(client_mod, "MAX_ATTEMPTS", 3)
    monkeypatch.setattr(client_mod, "BASE_DELAY", 0.0)
    monkeypatch.setattr(client_mod, "MAX_DELAY", 0.0)

    client = httpx.Client()
    handle = SyncRequestHandle(
        request_id="r-sync-no-retry",
        response_url="http://resp",
        status_url="http://status",
        cancel_url="http://cancel",
        client=client,
    )

    # Mock iter_events to skip waiting
    def _iter_events(self, with_logs: bool = False, interval: float = 0.1):
        return iter([Completed(logs=[], metrics={})])

    monkeypatch.setattr(SyncRequestHandle, "iter_events", _iter_events, raising=True)

    req_get = httpx.Request("GET", "http://resp")
    get_err = httpx.Response(status_code, request=req_get)
    client.request = Mock(side_effect=[get_err])

    with pytest.raises(FalClientHTTPError):
        handle.get()

    assert client.request.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [500, 503])
async def test_async_get_does_not_retry_on_500_503(monkeypatch, status_code):
    import fal_client.client as client_mod

    # No retries; still set to ensure if retry attempted it would be fast
    monkeypatch.setattr(client_mod, "MAX_ATTEMPTS", 3)
    monkeypatch.setattr(client_mod, "BASE_DELAY", 0.0)
    monkeypatch.setattr(client_mod, "MAX_DELAY", 0.0)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    client = httpx.AsyncClient()
    handle = AsyncRequestHandle(
        request_id="r-async-no-retry",
        response_url="http://resp",
        status_url="http://status",
        cancel_url="http://cancel",
        client=client,
    )

    # Mock iter_events to skip waiting
    async def _iter_events(self, with_logs: bool = False, interval: float = 0.1):
        yield Completed(logs=[], metrics={})

    monkeypatch.setattr(AsyncRequestHandle, "iter_events", _iter_events, raising=True)

    req_get = httpx.Request("GET", "http://resp")
    get_err = httpx.Response(status_code, request=req_get)
    client.request = AsyncMock(side_effect=[get_err])

    with pytest.raises(FalClientHTTPError):
        await handle.get()

    assert client.request.await_count == 1


def test_sync_handle_retries_ingress(monkeypatch):
    import fal_client.client as client_mod

    monkeypatch.setattr(client_mod, "MAX_ATTEMPTS", 3)
    monkeypatch.setattr(client_mod, "BASE_DELAY", 0.0)
    monkeypatch.setattr(client_mod, "MAX_DELAY", 0.0)

    client = httpx.Client()
    handle = SyncRequestHandle(
        request_id="r-sync-ingress",
        response_url="http://resp",
        status_url="http://status",
        cancel_url="http://cancel",
        client=client,
    )

    # Mock iter_events to skip waiting
    def _iter_events(self, with_logs: bool = False, interval: float = 0.1):
        return iter([Completed(logs=[], metrics={})])

    monkeypatch.setattr(SyncRequestHandle, "iter_events", _iter_events, raising=True)

    # Prepare sequence with ingress errors (nginx body, no x-fal-request-id)
    req_status = httpx.Request("GET", "http://status")
    status_ingress = httpx.Response(
        503,
        content=b"<html>nginx error</html>",
        headers={"Content-Type": "text/html"},
        request=req_status,
    )
    status_ok = httpx.Response(
        200,
        content=b'{"status": "COMPLETED", "logs": [], "metrics": {}}',
        headers={"Content-Type": "application/json"},
        request=req_status,
    )

    req_get = httpx.Request("GET", "http://resp")
    get_ingress = httpx.Response(
        503,
        content=b"gateway timeout via nginx",
        headers={"Content-Type": "text/plain"},
        request=req_get,
    )
    get_ok = httpx.Response(
        200,
        content=b'{"ok": true}',
        headers={"Content-Type": "application/json"},
        request=req_get,
    )

    req_put = httpx.Request("PUT", "http://cancel")
    cancel_ingress = httpx.Response(
        503,
        content=b"bad gateway - nginx",
        headers={"Content-Type": "text/plain"},
        request=req_put,
    )
    cancel_ok = httpx.Response(200, request=req_put)

    client.request = Mock(
        side_effect=[
            status_ingress,
            status_ok,
            get_ingress,
            get_ok,
            cancel_ingress,
            cancel_ok,
        ]
    )

    # Status
    status = handle.status(with_logs=False)
    assert isinstance(status, Completed)

    # Get
    result = handle.get()
    assert result == {"ok": True}

    # Cancel
    handle.cancel()

    # 2 per operation
    assert client.request.call_count == 6


@pytest.mark.asyncio
async def test_async_handle_retries_ingress(monkeypatch):
    import fal_client.client as client_mod

    monkeypatch.setattr(client_mod, "MAX_ATTEMPTS", 3)
    monkeypatch.setattr(client_mod, "BASE_DELAY", 0.0)
    monkeypatch.setattr(client_mod, "MAX_DELAY", 0.0)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    client = httpx.AsyncClient()
    handle = AsyncRequestHandle(
        request_id="r-async-ingress",
        response_url="http://resp",
        status_url="http://status",
        cancel_url="http://cancel",
        client=client,
    )

    # Mock iter_events to skip waiting
    async def _iter_events(self, with_logs: bool = False, interval: float = 0.1):
        yield Completed(logs=[], metrics={})

    monkeypatch.setattr(AsyncRequestHandle, "iter_events", _iter_events, raising=True)

    # Prepare sequence with ingress errors (nginx body, no x-fal-request-id)
    req_status = httpx.Request("GET", "http://status")
    status_ingress = httpx.Response(
        503,
        content=b"<html>nginx error</html>",
        headers={"Content-Type": "text/html"},
        request=req_status,
    )
    status_ok = httpx.Response(
        200,
        content=b'{"status": "COMPLETED", "logs": [], "metrics": {}}',
        headers={"Content-Type": "application/json"},
        request=req_status,
    )

    req_get = httpx.Request("GET", "http://resp")
    get_ingress = httpx.Response(
        503,
        content=b"gateway timeout via nginx",
        headers={"Content-Type": "text/plain"},
        request=req_get,
    )
    get_ok = httpx.Response(
        200,
        content=b'{"ok": true}',
        headers={"Content-Type": "application/json"},
        request=req_get,
    )

    req_put = httpx.Request("PUT", "http://cancel")
    cancel_ingress = httpx.Response(
        503,
        content=b"bad gateway - nginx",
        headers={"Content-Type": "text/plain"},
        request=req_put,
    )
    cancel_ok = httpx.Response(200, request=req_put)

    client.request = AsyncMock(
        side_effect=[
            status_ingress,
            status_ok,
            get_ingress,
            get_ok,
            cancel_ingress,
            cancel_ok,
        ]
    )

    # Status
    status = await handle.status(with_logs=False)
    assert isinstance(status, Completed)

    # Get
    result = await handle.get()
    assert result == {"ok": True}

    # Cancel
    await handle.cancel()

    # 2 per operation
    assert client.request.await_count == 6
