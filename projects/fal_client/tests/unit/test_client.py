import asyncio
import time
import json
from contextlib import asynccontextmanager, contextmanager
from typing import Dict, Optional
from unittest.mock import AsyncMock, Mock, patch

import httpx
import msgpack
import pytest

from fal_client.client import (
    AsyncClient,
    AsyncRequestHandle,
    CDN_URL,
    Completed,
    FAL_CDN_FALLBACK_URL,
    FalClientHTTPError,
    FalClientTimeoutError,
    InProgress,
    Queued,
    RealtimeConnection,
    RealtimeError,
    REST_URL,
    SyncClient,
    SyncRequestHandle,
    USER_AGENT,
    _BaseRequestHandle,
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


def test_sync_upload_falls_back_to_cdn():
    with patch("fal_client.client._maybe_retry_request") as mock_request, patch(
        "fal_client.client.SyncClient._get_cdn_client"
    ) as mock_cdn_client:
        fallback_response = Mock()
        fallback_response.json.return_value = {"access_url": "https://fallback/file"}
        mock_request.side_effect = [Exception("boom"), fallback_response]
        mock_cdn_client.return_value = Mock()

        client = SyncClient(key="test-key")
        url = client.upload(b"hello", content_type="text/plain")

    assert url == "https://fallback/file"
    assert mock_request.call_args_list[0][0][2] == f"{CDN_URL}/files/upload"
    assert (
        mock_request.call_args_list[1][0][2] == f"{FAL_CDN_FALLBACK_URL}/files/upload"
    )


def test_sync_upload_falls_back_to_storage():
    with patch("fal_client.client._maybe_retry_request") as mock_request, patch(
        "fal_client.client.SyncClient._get_cdn_client"
    ) as mock_cdn_client:
        init_response = Mock()
        init_response.json.return_value = {
            "upload_url": "https://upload.example.com/put",
            "file_url": "https://file.example.com/file",
        }
        mock_request.side_effect = [
            Exception("boom"),
            Exception("boom"),
            init_response,
            Mock(),
        ]
        mock_cdn_client.return_value = Mock()

        client = SyncClient(key="test-key")
        url = client.upload(b"hello", content_type="text/plain")

    assert url == "https://file.example.com/file"
    assert mock_request.call_args_list[0][0][2] == f"{CDN_URL}/files/upload"
    assert (
        mock_request.call_args_list[1][0][2] == f"{FAL_CDN_FALLBACK_URL}/files/upload"
    )
    assert (
        mock_request.call_args_list[2][0][2]
        == f"{REST_URL}/storage/upload/initiate?storage_type=gcs"
    )
    assert mock_request.call_args_list[3][0][2] == "https://upload.example.com/put"


def test_sync_upload_respects_repository_order():
    with patch("fal_client.client._maybe_retry_request") as mock_request:
        cdn_response = Mock()
        cdn_response.json.return_value = {"access_url": "https://cdn-only/file"}
        mock_request.return_value = cdn_response

        client = SyncClient(key="test-key")
        url = client.upload(
            b"hello",
            content_type="text/plain",
            repository="cdn",
            fallback_repository=[],
        )

    assert url == "https://cdn-only/file"
    assert (
        mock_request.call_args_list[0][0][2] == f"{FAL_CDN_FALLBACK_URL}/files/upload"
    )


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
async def test_async_upload_falls_back_to_cdn():
    with patch(
        "fal_client.client._async_maybe_retry_request", new_callable=AsyncMock
    ) as mock_request, patch(
        "fal_client.client.AsyncClient._get_cdn_client", new_callable=AsyncMock
    ) as mock_cdn_client:
        fallback_response = httpx.Response(
            status_code=200, json={"access_url": "https://fallback/file"}
        )
        mock_request.side_effect = [Exception("boom"), fallback_response]
        mock_cdn_client.return_value = Mock()

        client = AsyncClient(key="test-key")
        url = await client.upload(b"hello", content_type="text/plain")

    assert url == "https://fallback/file"
    assert mock_request.call_args_list[0][0][2] == f"{CDN_URL}/files/upload"
    assert (
        mock_request.call_args_list[1][0][2] == f"{FAL_CDN_FALLBACK_URL}/files/upload"
    )


@pytest.mark.asyncio
async def test_async_upload_falls_back_to_storage():
    with patch(
        "fal_client.client._async_maybe_retry_request", new_callable=AsyncMock
    ) as mock_request, patch(
        "fal_client.client.AsyncClient._get_cdn_client", new_callable=AsyncMock
    ) as mock_cdn_client:
        init_response = httpx.Response(
            status_code=200,
            json={
                "upload_url": "https://upload.example.com/put",
                "file_url": "https://file.example.com/file",
            },
        )
        mock_request.side_effect = [
            Exception("boom"),
            Exception("boom"),
            init_response,
            Mock(),
        ]
        mock_cdn_client.return_value = Mock()

        client = AsyncClient(key="test-key")
        url = await client.upload(b"hello", content_type="text/plain")

    assert url == "https://file.example.com/file"
    assert mock_request.call_args_list[0][0][2] == f"{CDN_URL}/files/upload"
    assert (
        mock_request.call_args_list[1][0][2] == f"{FAL_CDN_FALLBACK_URL}/files/upload"
    )
    assert (
        mock_request.call_args_list[2][0][2]
        == f"{REST_URL}/storage/upload/initiate?storage_type=gcs"
    )
    assert mock_request.call_args_list[3][0][2] == "https://upload.example.com/put"


@pytest.mark.asyncio
async def test_async_upload_respects_repository_order():
    with patch(
        "fal_client.client._async_maybe_retry_request", new_callable=AsyncMock
    ) as mock_request:
        cdn_response = httpx.Response(
            status_code=200, json={"access_url": "https://cdn-only/file"}
        )
        mock_request.return_value = cdn_response

        client = AsyncClient(key="test-key")
        url = await client.upload(
            b"hello",
            content_type="text/plain",
            repository="cdn",
            fallback_repository=[],
        )

    assert url == "https://cdn-only/file"
    assert (
        mock_request.call_args_list[0][0][2] == f"{FAL_CDN_FALLBACK_URL}/files/upload"
    )


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


def test_realtime_connection_decodes_messages():
    fake_ws = Mock()
    payload = {"foo": "bar"}
    fake_ws.recv.side_effect = [
        json.dumps({"type": "x-fal-message", "action": "ping"}),
        msgpack.packb(payload, use_bin_type=True),
    ]
    connection = RealtimeConnection(fake_ws)

    assert connection.recv() == payload


def test_realtime_connection_raises_on_error():
    fake_ws = Mock()
    fake_ws.recv.return_value = json.dumps(
        {"type": "x-fal-error", "error": "BAD_REQUEST", "reason": "nope"}
    )
    connection = RealtimeConnection(fake_ws)

    with pytest.raises(RealtimeError):
        connection.recv()


def test_sync_client_realtime_builds_url(mocker):
    client = SyncClient(key="test-key")
    token_response = Mock()
    token_response.json.return_value = "jwt-token"
    mock_request = mocker.patch(
        "fal_client.client._maybe_retry_request", return_value=token_response
    )

    fake_ws = Mock()
    fake_ws.recv.side_effect = [msgpack.packb({"ok": True}, use_bin_type=True)]

    @contextmanager
    def fake_connect(url: str, headers: Optional[Dict[str, str]] = None):
        assert url.startswith("wss://")
        assert "fal_jwt_token=jwt-token" in url
        assert "max_buffering=10" in url
        assert headers is None
        yield fake_ws

    mocker.patch("fal_client.client._connect_sync_ws", fake_connect)

    with client.realtime("1234-test", max_buffering=10) as connection:
        result = connection.recv()

    assert result == {"ok": True}
    assert mock_request.call_args[1]["json"]["allowed_apps"] == ["test"]


@pytest.mark.asyncio
async def test_async_client_realtime_builds_url(mocker):
    client = AsyncClient(key="test-key")
    token_response = Mock()
    token_response.json.return_value = "jwt-token"
    mock_request = mocker.patch(
        "fal_client.client._async_maybe_retry_request",
        new_callable=AsyncMock,
        return_value=token_response,
    )

    fake_ws = AsyncMock()
    fake_ws.recv = AsyncMock(
        side_effect=[msgpack.packb({"ok": True}, use_bin_type=True)]
    )

    @asynccontextmanager
    async def fake_connect(url: str, headers: Optional[Dict[str, str]] = None):
        assert url.startswith("wss://")
        assert "fal_jwt_token=jwt-token" in url
        assert "max_buffering=5" in url
        assert headers is None
        yield fake_ws

    mocker.patch("fal_client.client._connect_async_ws", fake_connect)

    async with client.realtime("1234-test", max_buffering=5) as connection:
        result = await connection.recv()

    assert result == {"ok": True}
    assert mock_request.await_args.kwargs["json"]["allowed_apps"] == ["test"]


def test_sync_client_ws_connect_custom_path(mocker):
    client = SyncClient(key="test-key")
    token_response = Mock()
    token_response.json.return_value = "jwt-token"
    mock_request = mocker.patch(
        "fal_client.client._maybe_retry_request", return_value=token_response
    )

    fake_ws = Mock()

    @contextmanager
    def fake_connect(url: str, headers: Optional[Dict[str, str]] = None):
        assert url.startswith("wss://")
        assert "/custom" in url
        assert "fal_jwt_token=jwt-token" in url
        assert headers is None
        yield fake_ws

    mocker.patch("fal_client.client._connect_sync_ws", fake_connect)

    with client.ws_connect("1234-test", path="custom") as ws:
        assert ws is fake_ws

    assert mock_request.call_args[1]["json"]["allowed_apps"] == ["test"]


@pytest.mark.asyncio
async def test_async_client_ws_connect_custom_path(mocker):
    client = AsyncClient(key="test-key")
    token_response = Mock()
    token_response.json.return_value = "jwt-token"
    mock_request = mocker.patch(
        "fal_client.client._async_maybe_retry_request",
        new_callable=AsyncMock,
        return_value=token_response,
    )

    fake_ws = AsyncMock()

    @asynccontextmanager
    async def fake_connect(url: str, headers: Optional[Dict[str, str]] = None):
        assert url.startswith("wss://")
        assert "/chat" in url
        assert "fal_jwt_token=jwt-token" in url
        assert headers is None
        yield fake_ws

    mocker.patch("fal_client.client._connect_async_ws", fake_connect)

    async with client.ws_connect("1234-test", path="chat") as ws:
        assert ws is fake_ws

    assert mock_request.await_args.kwargs["json"]["allowed_apps"] == ["test"]


def test_sync_client_realtime_uses_headers_without_jwt(mocker):
    client = SyncClient(key="test-key")
    mock_request = mocker.patch("fal_client.client._maybe_retry_request")

    fake_ws = Mock()
    fake_ws.recv.side_effect = [msgpack.packb({"ok": True}, use_bin_type=True)]

    @contextmanager
    def fake_connect(url: str, headers: Optional[Dict[str, str]] = None):
        assert url.startswith("wss://")
        assert "fal_jwt_token=" not in url
        assert headers == {
            "Authorization": "Key test-key",
            "User-Agent": USER_AGENT,
        }
        yield fake_ws

    mocker.patch("fal_client.client._connect_sync_ws", fake_connect)

    with client.realtime("1234-test", use_jwt=False) as connection:
        result = connection.recv()

    assert result == {"ok": True}
    mock_request.assert_not_called()


@pytest.mark.asyncio
async def test_async_client_realtime_uses_headers_without_jwt(mocker):
    client = AsyncClient(key="test-key")
    mock_request = mocker.patch(
        "fal_client.client._async_maybe_retry_request", new_callable=AsyncMock
    )

    fake_ws = AsyncMock()
    fake_ws.recv = AsyncMock(
        side_effect=[msgpack.packb({"ok": True}, use_bin_type=True)]
    )

    @asynccontextmanager
    async def fake_connect(url: str, headers: Optional[Dict[str, str]] = None):
        assert url.startswith("wss://")
        assert "fal_jwt_token=" not in url
        assert headers == {
            "Authorization": "Key test-key",
            "User-Agent": USER_AGENT,
        }
        yield fake_ws

    mocker.patch("fal_client.client._connect_async_ws", fake_connect)

    async with client.realtime("1234-test", use_jwt=False) as connection:
        result = await connection.recv()

    assert result == {"ok": True}
    mock_request.assert_not_called()


def test_sync_client_ws_connect_uses_headers_without_jwt(mocker):
    client = SyncClient(key="test-key")
    mock_request = mocker.patch("fal_client.client._maybe_retry_request")

    fake_ws = Mock()

    @contextmanager
    def fake_connect(url: str, headers: Optional[Dict[str, str]] = None):
        assert url.startswith("wss://")
        assert "/custom" in url
        assert "fal_jwt_token=" not in url
        assert headers == {
            "Authorization": "Key test-key",
            "User-Agent": USER_AGENT,
        }
        yield fake_ws

    mocker.patch("fal_client.client._connect_sync_ws", fake_connect)

    with client.ws_connect("1234-test", path="custom", use_jwt=False) as ws:
        assert ws is fake_ws

    mock_request.assert_not_called()


@pytest.mark.asyncio
async def test_async_client_ws_connect_uses_headers_without_jwt(mocker):
    client = AsyncClient(key="test-key")
    mock_request = mocker.patch(
        "fal_client.client._async_maybe_retry_request", new_callable=AsyncMock
    )

    fake_ws = AsyncMock()

    @asynccontextmanager
    async def fake_connect(url: str, headers: Optional[Dict[str, str]] = None):
        assert url.startswith("wss://")
        assert "/chat" in url
        assert "fal_jwt_token=" not in url
        assert headers == {
            "Authorization": "Key test-key",
            "User-Agent": USER_AGENT,
        }
        yield fake_ws

    mocker.patch("fal_client.client._connect_async_ws", fake_connect)

    async with client.ws_connect("1234-test", path="chat", use_jwt=False) as ws:
        assert ws is fake_ws

    mock_request.assert_not_called()


# Tests for start_timeout parameter


def test_sync_client_run_with_start_timeout():
    """Test that start_timeout adds X-Fal-Request-Timeout header in run()."""
    with patch("fal_client.client._maybe_retry_request") as mock_request:
        mock_response = Mock()
        mock_response.json.return_value = {"result": "success"}
        mock_request.return_value = mock_response

        client = SyncClient(key="test-key")
        client.run("test-app", {"input": "data"}, start_timeout=30)

        call_kwargs = mock_request.call_args[1]
        assert "headers" in call_kwargs
        assert call_kwargs["headers"]["X-Fal-Request-Timeout"] == "30.0"


def test_sync_client_run_with_start_timeout_float():
    """Test that start_timeout handles float values correctly."""
    with patch("fal_client.client._maybe_retry_request") as mock_request:
        mock_response = Mock()
        mock_response.json.return_value = {"result": "success"}
        mock_request.return_value = mock_response

        client = SyncClient(key="test-key")
        client.run("test-app", {"input": "data"}, start_timeout=45.5)

        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["headers"]["X-Fal-Request-Timeout"] == "45.5"


def test_sync_client_submit_with_start_timeout():
    """Test that start_timeout adds X-Fal-Request-Timeout header in submit()."""
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
        client.submit("test-app", {"input": "data"}, start_timeout=60)

        call_kwargs = mock_request.call_args[1]
        assert "headers" in call_kwargs
        assert call_kwargs["headers"]["X-Fal-Request-Timeout"] == "60.0"


def test_sync_client_subscribe_with_start_timeout():
    """Test that start_timeout is passed through to submit() in subscribe()."""
    with patch("fal_client.client._maybe_retry_request") as mock_request:
        submit_response = Mock()
        submit_response.json.return_value = {
            "request_id": "req-123",
            "response_url": "http://response",
            "status_url": "http://status",
            "cancel_url": "http://cancel",
        }

        status_response = Mock()
        status_response.json.return_value = {"status": "COMPLETED", "logs": []}

        result_response = Mock()
        result_response.json.return_value = {"result": "done"}

        mock_request.side_effect = [submit_response, status_response, result_response]

        client = SyncClient(key="test-key")
        client.subscribe("test-app", {"input": "data"}, start_timeout=90)

        # Check the first call (submit) has the header
        first_call_kwargs = mock_request.call_args_list[0][1]
        assert "headers" in first_call_kwargs
        assert first_call_kwargs["headers"]["X-Fal-Request-Timeout"] == "90.0"


@pytest.mark.asyncio
async def test_async_client_run_with_start_timeout():
    """Test that start_timeout adds X-Fal-Request-Timeout header in async run()."""
    with patch(
        "fal_client.client._async_maybe_retry_request", new_callable=AsyncMock
    ) as mock_request:
        mock_response = Mock()
        mock_response.json.return_value = {"result": "success"}
        mock_request.return_value = mock_response

        client = AsyncClient(key="test-key")
        await client.run("test-app", {"input": "data"}, start_timeout=30)

        call_kwargs = mock_request.call_args[1]
        assert "headers" in call_kwargs
        assert call_kwargs["headers"]["X-Fal-Request-Timeout"] == "30.0"


@pytest.mark.asyncio
async def test_async_client_submit_with_start_timeout():
    """Test that start_timeout adds X-Fal-Request-Timeout header in async submit()."""
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
        await client.submit("test-app", {"input": "data"}, start_timeout=60)

        call_kwargs = mock_request.call_args[1]
        assert "headers" in call_kwargs
        assert call_kwargs["headers"]["X-Fal-Request-Timeout"] == "60.0"


@pytest.mark.asyncio
async def test_async_client_subscribe_with_start_timeout():
    """Test that start_timeout is passed through to submit() in async subscribe()."""
    with patch(
        "fal_client.client._async_maybe_retry_request", new_callable=AsyncMock
    ) as mock_request:
        submit_response = Mock()
        submit_response.json.return_value = {
            "request_id": "req-789",
            "response_url": "http://response",
            "status_url": "http://status",
            "cancel_url": "http://cancel",
        }

        status_response = Mock()
        status_response.json.return_value = {"status": "COMPLETED", "logs": []}

        result_response = Mock()
        result_response.json.return_value = {"result": "async_done"}

        mock_request.side_effect = [submit_response, status_response, result_response]

        client = AsyncClient(key="test-key")
        await client.subscribe("test-app", {"input": "data"}, start_timeout=90)

        # Check the first call (submit) has the header
        first_call_kwargs = mock_request.call_args_list[0][1]
        assert "headers" in first_call_kwargs
        assert first_call_kwargs["headers"]["X-Fal-Request-Timeout"] == "90.0"


def test_sync_client_run_without_start_timeout_no_header():
    """Test that no timeout header is added when start_timeout is not specified."""
    with patch("fal_client.client._maybe_retry_request") as mock_request:
        mock_response = Mock()
        mock_response.json.return_value = {"result": "success"}
        mock_request.return_value = mock_response

        client = SyncClient(key="test-key")
        client.run("test-app", {"input": "data"})

        call_kwargs = mock_request.call_args[1]
        assert "X-Fal-Request-Timeout" not in call_kwargs.get("headers", {})


def test_sync_client_run_with_start_timeout_and_hint():
    """Test that start_timeout works alongside other headers like hint."""
    with patch("fal_client.client._maybe_retry_request") as mock_request:
        mock_response = Mock()
        mock_response.json.return_value = {"result": "success"}
        mock_request.return_value = mock_response

        client = SyncClient(key="test-key")
        client.run("test-app", {"input": "data"}, start_timeout=30, hint="lora:a")

        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["headers"]["X-Fal-Request-Timeout"] == "30.0"
        assert call_kwargs["headers"]["X-Fal-Runner-Hint"] == "lora:a"


def test_sync_subscribe_timeout_while_waiting_submit(monkeypatch):
    """Timeout during submit should raise FalClientTimeoutError with request_id=None."""

    def slow_submit(*args, **kwargs):
        time.sleep(3)

    with patch("fal_client.client._maybe_retry_request", side_effect=slow_submit):
        client = SyncClient(key="test-key")
        with pytest.raises(FalClientTimeoutError) as exc:
            client.subscribe("app", {}, client_timeout=1.5)
        assert exc.value.request_id is None


def test_sync_subscribe_timeout_while_waiting_handle(monkeypatch):
    """Timeout during handle.get() should cancel and raise with request_id."""
    call_count = 0

    def mock_request(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        resp = Mock()
        if call_count == 1:
            resp.json.return_value = {
                "request_id": "req-123",
                "response_url": "http://response",
                "status_url": "http://status",
                "cancel_url": "http://cancel",
            }
        else:
            time.sleep(3)
            resp.json.return_value = {"status": "IN_QUEUE", "queue_position": 1}
        return resp

    with patch("fal_client.client._maybe_retry_request", side_effect=mock_request):
        with patch("fal_client.client._maybe_cancel_request") as mock_cancel:
            client = SyncClient(key="test-key")
            with pytest.raises(FalClientTimeoutError) as exc:
                client.subscribe("app", {}, client_timeout=1.5)
            assert exc.value.request_id == "req-123"
            # Wait for background cancel to be executed
            time.sleep(0.2)
            mock_cancel.assert_called_once()
            handle = mock_cancel.call_args.args[0]
            assert handle.request_id == "req-123"


@pytest.mark.asyncio
async def test_async_subscribe_timeout_while_waiting_submit():
    """Timeout during submit should raise FalClientTimeoutError with request_id=None."""

    async def slow_submit(*args, **kwargs):
        await asyncio.sleep(3)

    with patch(
        "fal_client.client._async_maybe_retry_request",
        new_callable=AsyncMock,
        side_effect=slow_submit,
    ):
        client = AsyncClient(key="test-key")
        with pytest.raises(FalClientTimeoutError) as exc:
            await client.subscribe("app", {}, client_timeout=1.5)
        assert exc.value.request_id is None


@pytest.mark.asyncio
async def test_async_subscribe_timeout_while_waiting_handle():
    """Timeout during handle.get() should cancel and raise with request_id."""
    call_count = 0

    async def mock_request(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        resp = Mock()
        if call_count == 1:
            resp.json.return_value = {
                "request_id": "req-456",
                "response_url": "http://r",
                "status_url": "http://s",
                "cancel_url": "http://c",
            }
        else:
            await asyncio.sleep(5)
            resp.json.return_value = {"status": "IN_QUEUE", "queue_position": 1}
        return resp

    with patch(
        "fal_client.client._async_maybe_retry_request",
        new_callable=AsyncMock,
        side_effect=mock_request,
    ):
        with patch(
            "fal_client.client._async_maybe_cancel_request",
            new_callable=AsyncMock,
        ) as mock_cancel:
            client = AsyncClient(key="test-key")
            with pytest.raises(FalClientTimeoutError) as exc:
                await client.subscribe("app", {}, client_timeout=1.5)
            assert exc.value.request_id == "req-456"
            # Wait for background cancel task
            await asyncio.sleep(0.2)
            mock_cancel.assert_called_once()
            handle = mock_cancel.call_args.args[0]
            assert handle.request_id == "req-456"


@pytest.mark.asyncio
async def test_async_subscribe_with_async_callbacks():
    """Test that async callbacks are awaited in AsyncClient.subscribe"""
    with patch(
        "fal_client.client._async_maybe_retry_request", new_callable=AsyncMock
    ) as mock_request:
        submit_response = Mock()
        submit_response.json.return_value = {
            "request_id": "req-abc",
            "response_url": "http://response",
            "status_url": "http://status",
            "cancel_url": "http://cancel",
        }

        status_response = Mock()
        status_response.json.return_value = {
            "status": "COMPLETED",
            "logs": [],
        }

        result_response = Mock()
        result_response.json.return_value = {"result": "done"}

        # status is checked twice: once in on_queue_update loop, once in handle.get()
        status_response_2 = Mock()
        status_response_2.json.return_value = {
            "status": "COMPLETED",
            "logs": [],
        }

        mock_request.side_effect = [
            submit_response,
            status_response,
            status_response_2,
            result_response,
        ]

        enqueue_called = False
        queue_updates = []

        async def async_on_enqueue(request_id: str):
            nonlocal enqueue_called
            enqueue_called = True

        async def async_on_queue_update(status):
            queue_updates.append(status)

        client = AsyncClient(key="test-key")
        result = await client.subscribe(
            "test-app",
            {"input": "data"},
            on_enqueue=async_on_enqueue,
            on_queue_update=async_on_queue_update,
        )

        assert result == {"result": "done"}
        assert enqueue_called
        assert len(queue_updates) == 1


@pytest.mark.asyncio
async def test_async_subscribe_with_sync_callbacks():
    """Test that sync callbacks works correctly in AsyncClient.subscribe"""
    with patch(
        "fal_client.client._async_maybe_retry_request", new_callable=AsyncMock
    ) as mock_request:
        submit_response = Mock()
        submit_response.json.return_value = {
            "request_id": "req-def",
            "response_url": "http://response",
            "status_url": "http://status",
            "cancel_url": "http://cancel",
        }

        status_response = Mock()
        status_response.json.return_value = {
            "status": "COMPLETED",
            "logs": [],
        }

        status_response_2 = Mock()
        status_response_2.json.return_value = {
            "status": "COMPLETED",
            "logs": [],
        }

        result_response = Mock()
        result_response.json.return_value = {"result": "done"}

        mock_request.side_effect = [
            submit_response,
            status_response,
            status_response_2,
            result_response,
        ]

        enqueue_called = False
        queue_updates = []

        def sync_on_enqueue(request_id: str):
            nonlocal enqueue_called
            enqueue_called = True

        def sync_on_queue_update(status):
            queue_updates.append(status)

        client = AsyncClient(key="test-key")
        result = await client.subscribe(
            "test-app",
            {"input": "data"},
            on_enqueue=sync_on_enqueue,
            on_queue_update=sync_on_queue_update,
        )

        assert result == {"result": "done"}
        assert enqueue_called
        assert len(queue_updates) == 1
