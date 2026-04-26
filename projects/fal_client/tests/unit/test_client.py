from __future__ import annotations

import asyncio
import time
import json
from contextlib import asynccontextmanager, contextmanager
from typing import Dict, Optional
from unittest.mock import AsyncMock, Mock, call, patch

import httpx
import msgpack
import pytest

from fal_client.client import (
    AsyncClient,
    AsyncMultipartUpload,
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
    StorageACL,
    StorageACLRule,
    StorageSettings,
    SyncClient,
    SyncRequestHandle,
    USER_AGENT,
    _BaseRequestHandle,
    _raise_for_status,
)


def test_clients_remain_hashable():
    assert isinstance(hash(AsyncClient(key="test-key")), int)
    assert isinstance(hash(SyncClient(key="test-key")), int)


@pytest.mark.asyncio
async def test_async_client_get_handle_uses_async_client_cache():
    client = AsyncClient(key="test-key")

    handle = await client.get_handle("fal-ai/fast-sdxl", "request-id")

    assert isinstance(handle, AsyncRequestHandle)
    assert await client._client is handle.client


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
            {
                "status": "COMPLETED",
                "logs": [],
                "metrics": {},
                "error": "Runner disconnected",
                "error_type": "runner_disconnected",
            },
            Completed(
                logs=[],
                metrics={},
                error="Runner disconnected",
                error_type="runner_disconnected",
            ),
            False,
        ),
        (
            {
                "status": "COMPLETED",
                "logs": None,
                "metrics": {"inference_time": 1.5},
                "error": "Request timed out",
                "error_type": "request_timeout",
            },
            Completed(
                logs=None,
                metrics={"inference_time": 1.5},
                error="Request timed out",
                error_type="request_timeout",
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
    ) as mock_cdn_context:
        fallback_response = Mock()
        fallback_response.json.return_value = {"access_url": "https://fallback/file"}
        mock_request.side_effect = [Exception("boom"), fallback_response]
        mock_cdn_context.return_value = Mock()
        settings = StorageSettings(expires_in=3600)

        client = SyncClient(key="test-key")
        url = client.upload(
            b"hello",
            content_type="text/plain",
            lifecycle=settings,
        )

    assert url == "https://fallback/file"
    assert mock_request.call_args_list[0][0][2] == f"{CDN_URL}/files/upload"
    assert (
        mock_request.call_args_list[1][0][2] == f"{FAL_CDN_FALLBACK_URL}/files/upload"
    )
    expected = json.dumps({"expiration_duration_seconds": 3600})
    assert (
        mock_request.call_args_list[0][1]["headers"]["X-Fal-Object-Lifecycle"]
        == expected
    )
    assert (
        mock_request.call_args_list[1][1]["headers"][
            "X-Fal-Object-Lifecycle-Preference"
        ]
        == expected
    )


def test_sync_upload_falls_back_to_storage():
    with patch("fal_client.client._maybe_retry_request") as mock_request, patch(
        "fal_client.client.SyncClient._get_cdn_client"
    ) as mock_cdn_context:
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
        mock_cdn_context.return_value = Mock()
        settings = StorageSettings(expires_in=3600)

        client = SyncClient(key="test-key")
        url = client.upload(
            b"hello",
            content_type="text/plain",
            lifecycle=settings,
        )

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
    expected = json.dumps({"expiration_duration_seconds": 3600})
    assert (
        mock_request.call_args_list[2][1]["headers"][
            "X-Fal-Object-Lifecycle-Preference"
        ]
        == expected
    )


def test_sync_upload_passes_lifecycle_to_multipart(monkeypatch):
    monkeypatch.setattr("fal_client.client.MULTIPART_THRESHOLD", 1)
    settings = StorageSettings(expires_in=3600)

    with patch(
        "fal_client.client.MultipartUpload.save", return_value="https://file"
    ) as mock_save, patch("fal_client.client.SyncClient._get_cdn_client") as mock_cdn:
        mock_cdn.return_value = Mock()
        client = SyncClient(key="test-key")
        url = client.upload(
            b"hello",
            content_type="text/plain",
            lifecycle=settings,
        )

    assert url == "https://file"
    assert mock_save.call_args.kwargs["object_lifecycle_preference"] == {
        "expiration_duration_seconds": 3600,
    }


def test_sync_upload_file_passes_lifecycle_to_multipart(tmp_path, monkeypatch):
    monkeypatch.setattr("fal_client.client.MULTIPART_THRESHOLD", 1)
    file_path = tmp_path / "upload.txt"
    file_path.write_bytes(b"hello")
    settings = StorageSettings(expires_in=3600)

    with patch(
        "fal_client.client.MultipartUpload.save_file", return_value="https://file"
    ) as mock_save, patch("fal_client.client.SyncClient._get_cdn_client") as mock_cdn:
        mock_cdn.return_value = Mock()
        client = SyncClient(key="test-key")
        url = client.upload_file(file_path, lifecycle=settings)

    assert url == "https://file"
    assert mock_save.call_args.kwargs["object_lifecycle_preference"] == {
        "expiration_duration_seconds": 3600,
    }


def test_sync_upload_lifecycle_expires_in_is_normalized():
    with patch("fal_client.client._maybe_retry_request") as mock_request:
        response = Mock()
        response.json.return_value = {"access_url": "https://cdn-only/file"}
        mock_request.return_value = response

        client = SyncClient(key="test-key")
        url = client.upload(
            b"hello",
            content_type="text/plain",
            repository="cdn",
            fallback_repository=[],
            lifecycle=StorageSettings(expires_in="1h"),
        )

    assert url == "https://cdn-only/file"
    lifecycle_header = mock_request.call_args[1]["headers"]["X-Fal-Object-Lifecycle"]
    assert json.loads(lifecycle_header) == {
        "expiration_duration_seconds": 3600,
    }


def test_sync_upload_lifecycle_is_sent_to_fal_v3():
    with patch("fal_client.client._maybe_retry_request") as mock_request, patch(
        "fal_client.client.SyncClient._get_cdn_client"
    ) as mock_cdn_context:
        response = Mock()
        response.json.return_value = {"access_url": "https://v3-only/file"}
        mock_request.return_value = response
        mock_cdn_context.return_value = Mock()

        client = SyncClient(key="test-key")
        url = client.upload(
            b"hello",
            content_type="text/plain",
            repository="fal_v3",
            fallback_repository=[],
            lifecycle=StorageSettings(expires_in="1h"),
        )

    assert url == "https://v3-only/file"
    lifecycle_header = mock_request.call_args[1]["headers"]["X-Fal-Object-Lifecycle"]
    assert json.loads(lifecycle_header) == {
        "expiration_duration_seconds": 3600,
    }


def test_sync_upload_lifecycle_immediate_is_normalized():
    with patch("fal_client.client._maybe_retry_request") as mock_request:
        response = Mock()
        response.json.return_value = {"access_url": "https://cdn-only/file"}
        mock_request.return_value = response

        client = SyncClient(key="test-key")
        url = client.upload(
            b"hello",
            content_type="text/plain",
            repository="cdn",
            fallback_repository=[],
            lifecycle=StorageSettings(expires_in="immediate"),
        )

    assert url == "https://cdn-only/file"
    lifecycle_header = mock_request.call_args[1]["headers"]["X-Fal-Object-Lifecycle"]
    assert json.loads(lifecycle_header) == {
        "expiration_duration_seconds": 60,
    }


def test_sync_upload_lifecycle_never_is_normalized():
    with patch("fal_client.client._maybe_retry_request") as mock_request:
        response = Mock()
        response.json.return_value = {"access_url": "https://cdn-only/file"}
        mock_request.return_value = response

        client = SyncClient(key="test-key")
        url = client.upload(
            b"hello",
            content_type="text/plain",
            repository="cdn",
            fallback_repository=[],
            lifecycle=StorageSettings(expires_in="never"),
        )

    assert url == "https://cdn-only/file"
    assert "X-Fal-Object-Lifecycle" not in mock_request.call_args[1]["headers"]


def test_sync_upload_lifecycle_integer_must_be_positive():
    client = SyncClient(key="test-key")
    with pytest.raises(
        ValueError,
        match="Integer lifecycle expires_in value must be greater than 0 seconds",
    ):
        client.upload(
            b"hello",
            content_type="text/plain",
            repository="cdn",
            fallback_repository=[],
            lifecycle=StorageSettings(expires_in=0),
        )


def test_sync_upload_lifecycle_negative_integer_is_rejected():
    client = SyncClient(key="test-key")
    with pytest.raises(
        ValueError,
        match="Integer lifecycle expires_in value must be greater than 0 seconds",
    ):
        client.upload(
            b"hello",
            content_type="text/plain",
            repository="cdn",
            fallback_repository=[],
            lifecycle=StorageSettings(expires_in=-1),
        )


def test_sync_upload_lifecycle_boolean_is_rejected():
    client = SyncClient(key="test-key")
    with pytest.raises(
        ValueError,
        match="Boolean values are not valid lifecycle expires_in values",
    ):
        client.upload(
            b"hello",
            content_type="text/plain",
            repository="cdn",
            fallback_repository=[],
            lifecycle=StorageSettings(expires_in=True),
        )


def test_sync_upload_lifecycle_invalid_string_is_rejected():
    client = SyncClient(key="test-key")
    with pytest.raises(
        ValueError,
        match="Unsupported lifecycle expires_in value",
    ):
        client.upload(
            b"hello",
            content_type="text/plain",
            repository="cdn",
            fallback_repository=[],
            lifecycle=StorageSettings(expires_in="2h"),  # type: ignore[arg-type]
        )


def test_storage_settings_validates_expires_in_on_init():
    with pytest.raises(
        ValueError,
        match="Integer lifecycle expires_in value must be greater than 0 seconds",
    ):
        StorageSettings(expires_in=-1)


def test_sync_upload_lifecycle_includes_acl():
    with patch("fal_client.client._maybe_retry_request") as mock_request:
        response = Mock()
        response.json.return_value = {"access_url": "https://cdn-only/file"}
        mock_request.return_value = response

        client = SyncClient(key="test-key")
        url = client.upload(
            b"hello",
            content_type="text/plain",
            repository="cdn",
            fallback_repository=[],
            lifecycle=StorageSettings(
                expires_in="immediate",
                initial_acl=StorageACL(
                    default="forbid",
                    rules=[StorageACLRule(user="usr_123", decision="allow")],
                ),
            ),
        )

    assert url == "https://cdn-only/file"
    lifecycle_header = mock_request.call_args[1]["headers"]["X-Fal-Object-Lifecycle"]
    assert json.loads(lifecycle_header) == {
        "expiration_duration_seconds": 60,
        "initial_acl": {
            "default": "forbid",
            "rules": [{"user": "usr_123", "decision": "allow"}],
        },
    }


def test_sync_upload_lifecycle_skips_empty_acl_rules():
    with patch("fal_client.client._maybe_retry_request") as mock_request:
        response = Mock()
        response.json.return_value = {"access_url": "https://cdn-only/file"}
        mock_request.return_value = response

        client = SyncClient(key="test-key")
        url = client.upload(
            b"hello",
            content_type="text/plain",
            repository="cdn",
            fallback_repository=[],
            lifecycle=StorageSettings(
                initial_acl=StorageACL(
                    default="forbid",
                    rules=[],
                ),
            ),
        )

    assert url == "https://cdn-only/file"
    lifecycle_header = mock_request.call_args[1]["headers"]["X-Fal-Object-Lifecycle"]
    assert json.loads(lifecycle_header) == {
        "initial_acl": {
            "default": "forbid",
        },
    }


def test_sync_upload_lifecycle_omits_empty_acl_object():
    with patch("fal_client.client._maybe_retry_request") as mock_request:
        response = Mock()
        response.json.return_value = {"access_url": "https://cdn-only/file"}
        mock_request.return_value = response

        client = SyncClient(key="test-key")
        url = client.upload(
            b"hello",
            content_type="text/plain",
            repository="cdn",
            fallback_repository=[],
            lifecycle=StorageSettings(
                initial_acl=StorageACL(),
            ),
        )

    assert url == "https://cdn-only/file"
    request_headers = mock_request.call_args[1]["headers"]
    assert "X-Fal-Object-Lifecycle" not in request_headers
    assert "X-Fal-Object-Lifecycle-Preference" not in request_headers


def test_sync_upload_image_passes_lifecycle():
    image = Mock()

    def save(buffer, format):
        buffer.write(b"image-bytes")

    image.save.side_effect = save
    settings = StorageSettings(expires_in="1h")

    with patch.object(
        SyncClient, "upload", return_value="https://image"
    ) as mock_upload:
        client = SyncClient(key="test-key")
        url = client.upload_image(image, lifecycle=settings)

    assert url == "https://image"
    assert mock_upload.call_args.kwargs["lifecycle"] == settings


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
async def test_async_client_resolves_auth_when_no_key():
    auth = Mock(header_value="Key resolved-auth", scheme="Key", token="resolved-auth")

    with patch(
        "fal_client.client.fetch_auth_credentials_async",
        new_callable=AsyncMock,
        return_value=auth,
    ) as mock_fetch:
        client = AsyncClient()
        resolved = await client._auth

    assert resolved is auth
    mock_fetch.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_async_client_run_reuses_async_auth_cache():
    auth = Mock(header_value="Key resolved-auth", scheme="Key", token="resolved-auth")

    with patch(
        "fal_client.client.fetch_auth_credentials_async",
        new_callable=AsyncMock,
        return_value=auth,
    ) as mock_fetch, patch(
        "fal_client.client._async_maybe_retry_request", new_callable=AsyncMock
    ) as mock_request:
        mock_response = Mock()
        mock_response.json.return_value = {"result": "success"}
        mock_request.return_value = mock_response

        client = AsyncClient()
        first_result = await client.run("test-app", {"input": "first"})
        second_result = await client.run("test-app", {"input": "second"})

    assert first_result == {"result": "success"}
    assert second_result == {"result": "success"}
    mock_fetch.assert_awaited_once_with()
    assert mock_request.await_count == 2


class FakeAsyncTokenManager:
    def __await__(self):
        async def _return_self():
            return self

        return _return_self().__await__()

    async def get_token(self):
        return Mock(token_type="Bearer", token="cdn-token")


@pytest.mark.asyncio
async def test_async_upload_falls_back_to_cdn():
    @asynccontextmanager
    async def fake_cdn_client():
        yield Mock()

    with patch(
        "fal_client.client._async_maybe_retry_request", new_callable=AsyncMock
    ) as mock_request, patch(
        "fal_client.client.AsyncClient._cdn_client"
    ) as mock_cdn_context:
        fallback_response = httpx.Response(
            status_code=200, json={"access_url": "https://fallback/file"}
        )
        mock_request.side_effect = [Exception("boom"), fallback_response]
        mock_cdn_context.side_effect = lambda: fake_cdn_client()
        settings = StorageSettings(expires_in=3600)

        client = AsyncClient(key="test-key")
        client.__dict__["_token_manager"] = FakeAsyncTokenManager()
        url = await client.upload(
            b"hello",
            content_type="text/plain",
            lifecycle=settings,
        )

    assert url == "https://fallback/file"
    assert mock_request.call_args_list[0][0][2] == f"{CDN_URL}/files/upload"
    assert (
        mock_request.call_args_list[1][0][2] == f"{FAL_CDN_FALLBACK_URL}/files/upload"
    )
    expected = json.dumps({"expiration_duration_seconds": 3600})
    assert (
        mock_request.call_args_list[0][1]["headers"]["X-Fal-Object-Lifecycle"]
        == expected
    )
    assert (
        mock_request.call_args_list[1][1]["headers"][
            "X-Fal-Object-Lifecycle-Preference"
        ]
        == expected
    )


@pytest.mark.asyncio
async def test_async_upload_falls_back_to_storage():
    @asynccontextmanager
    async def fake_cdn_client():
        yield Mock()

    with patch(
        "fal_client.client._async_maybe_retry_request", new_callable=AsyncMock
    ) as mock_request, patch(
        "fal_client.client.AsyncClient._cdn_client"
    ) as mock_cdn_context:
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
        mock_cdn_context.side_effect = lambda: fake_cdn_client()
        settings = StorageSettings(expires_in=3600)

        client = AsyncClient(key="test-key")
        client.__dict__["_token_manager"] = FakeAsyncTokenManager()
        url = await client.upload(
            b"hello",
            content_type="text/plain",
            lifecycle=settings,
        )

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
    expected = json.dumps({"expiration_duration_seconds": 3600})
    assert (
        mock_request.call_args_list[2][1]["headers"][
            "X-Fal-Object-Lifecycle-Preference"
        ]
        == expected
    )


@pytest.mark.asyncio
async def test_async_upload_passes_lifecycle_to_multipart(monkeypatch):
    monkeypatch.setattr("fal_client.client.MULTIPART_THRESHOLD", 1)
    settings = StorageSettings(expires_in=3600)
    cdn_client = AsyncMock()
    cdn_client.__aenter__.return_value = Mock()
    cdn_client.__aexit__.return_value = None

    with patch(
        "fal_client.client.AsyncMultipartUpload.save",
        new_callable=AsyncMock,
        return_value="https://file",
    ) as mock_save, patch("fal_client.client.AsyncClient._cdn_client") as mock_cdn:
        mock_cdn.return_value = cdn_client
        client = AsyncClient(key="test-key")
        url = await client.upload(
            b"hello",
            content_type="text/plain",
            lifecycle=settings,
        )

    assert url == "https://file"
    assert mock_save.call_args.kwargs["object_lifecycle_preference"] == {
        "expiration_duration_seconds": 3600,
    }


@pytest.mark.asyncio
async def test_async_upload_file_passes_lifecycle_to_multipart(tmp_path, monkeypatch):
    @asynccontextmanager
    async def fake_cdn_client():
        yield Mock()

    monkeypatch.setattr("fal_client.client.MULTIPART_THRESHOLD", 1)
    file_path = tmp_path / "upload.txt"
    file_path.write_bytes(b"hello")
    settings = StorageSettings(expires_in=3600)

    with patch(
        "fal_client.client.AsyncMultipartUpload.save_file",
        new_callable=AsyncMock,
        return_value="https://file",
    ) as mock_save, patch("fal_client.client.AsyncClient._cdn_client") as mock_cdn:
        mock_cdn.return_value = fake_cdn_client()
        client = AsyncClient(key="test-key")
        url = await client.upload_file(file_path, lifecycle=settings)

    assert url == "https://file"
    assert mock_save.call_args.kwargs["object_lifecycle_preference"] == {
        "expiration_duration_seconds": 3600,
    }


@pytest.mark.asyncio
async def test_async_upload_file_uses_aiofiles_for_non_multipart_path(
    tmp_path, monkeypatch
):
    class FakeAsyncFile:
        def __init__(self, data: bytes) -> None:
            self.read = AsyncMock(return_value=data)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    file_path = tmp_path / "upload.txt"
    file_path.write_bytes(b"hello")
    fake_file = FakeAsyncFile(b"hello")
    monkeypatch.setattr("fal_client.client.MULTIPART_THRESHOLD", 1024)

    with patch("fal_client.client.aiofiles.open", return_value=fake_file) as mock_open:
        client = AsyncClient(key="test-key")
        with patch(
            "fal_client.client.AsyncClient.upload",
            new_callable=AsyncMock,
            return_value="https://file",
        ) as mock_upload:
            url = await client.upload_file(file_path)

    assert url == "https://file"
    mock_open.assert_called_once_with(file_path, "rb")
    fake_file.read.assert_awaited_once_with()
    assert mock_upload.await_count == 1
    assert mock_upload.await_args.args == (b"hello", "text/plain")
    assert mock_upload.await_args.kwargs == {
        "file_name": "upload.txt",
        "repository": None,
        "fallback_repository": None,
        "lifecycle": None,
    }


@pytest.mark.asyncio
async def test_async_multipart_save_file_uses_aiofiles(tmp_path):
    class FakeAsyncFile:
        def __init__(self, data_by_offset: dict[int, bytes]) -> None:
            self.offset: int | None = None
            self.seek = AsyncMock(side_effect=self._seek)
            self.read = AsyncMock(side_effect=lambda size: data_by_offset[self.offset])

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def _seek(self, offset: int) -> None:
            self.offset = offset

    file_path = tmp_path / "upload.txt"
    file_path.write_bytes(b"abcd")
    first_file = FakeAsyncFile({0: b"ab"})
    second_file = FakeAsyncFile({2: b"cd"})

    open_calls = {"count": 0}

    def open_side_effect(*args, **kwargs):
        open_calls["count"] += 1
        return first_file if open_calls["count"] == 1 else second_file

    with patch(
        "fal_client.client.aiofiles.open",
        side_effect=open_side_effect,
    ) as mock_open, patch.object(
        AsyncMultipartUpload,
        "create",
        new_callable=AsyncMock,
    ) as mock_create, patch.object(
        AsyncMultipartUpload,
        "upload_part",
        new_callable=AsyncMock,
    ) as mock_upload_part, patch.object(
        AsyncMultipartUpload,
        "complete",
        new_callable=AsyncMock,
        return_value="https://file",
    ) as mock_complete:
        url = await AsyncMultipartUpload.save_file(
            client=Mock(),
            token_manager=Mock(),
            file_path=file_path,
            chunk_size=2,
        )

    assert url == "https://file"
    mock_create.assert_awaited_once_with(object_lifecycle_preference=None)
    assert mock_open.call_count == 2
    first_file.seek.assert_awaited_once_with(0)
    first_file.read.assert_awaited_once_with(2)
    second_file.seek.assert_awaited_once_with(2)
    second_file.read.assert_awaited_once_with(2)
    mock_upload_part.assert_has_awaits([call(1, b"ab"), call(2, b"cd")], any_order=True)
    mock_complete.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_async_multipart_save_respects_max_concurrency():
    active_uploads = 0
    max_active_uploads = 0

    async def upload_part(self, part_number: int, data: bytes) -> None:
        nonlocal active_uploads, max_active_uploads
        active_uploads += 1
        max_active_uploads = max(max_active_uploads, active_uploads)
        await asyncio.sleep(0)
        active_uploads -= 1

    with patch.object(
        AsyncMultipartUpload,
        "create",
        new_callable=AsyncMock,
    ) as mock_create, patch.object(
        AsyncMultipartUpload,
        "upload_part",
        new=upload_part,
    ), patch.object(
        AsyncMultipartUpload,
        "complete",
        new_callable=AsyncMock,
        return_value="https://file",
    ) as mock_complete:
        url = await AsyncMultipartUpload.save(
            client=Mock(),
            token_manager=Mock(),
            file_name="upload.bin",
            data=b"abcdef",
            chunk_size=2,
            max_concurrency=2,
        )

    assert url == "https://file"
    mock_create.assert_awaited_once_with(object_lifecycle_preference=None)
    mock_complete.assert_awaited_once_with()
    assert max_active_uploads == 2


@pytest.mark.asyncio
async def test_async_upload_lifecycle_is_sent_to_fal_v3():
    @asynccontextmanager
    async def fake_cdn_client():
        yield Mock()

    with patch(
        "fal_client.client._async_maybe_retry_request", new_callable=AsyncMock
    ) as mock_request, patch(
        "fal_client.client.AsyncClient._cdn_client"
    ) as mock_cdn_context:
        response = httpx.Response(
            status_code=200, json={"access_url": "https://v3-only/file"}
        )
        mock_request.return_value = response
        mock_cdn_context.return_value = fake_cdn_client()

        client = AsyncClient(key="test-key")
        client.__dict__["_token_manager"] = FakeAsyncTokenManager()
        url = await client.upload(
            b"hello",
            content_type="text/plain",
            repository="fal_v3",
            fallback_repository=[],
            lifecycle=StorageSettings(expires_in="1h"),
        )

    assert url == "https://v3-only/file"
    lifecycle_header = mock_request.call_args[1]["headers"]["X-Fal-Object-Lifecycle"]
    assert json.loads(lifecycle_header) == {
        "expiration_duration_seconds": 3600,
    }


@pytest.mark.asyncio
async def test_async_upload_image_passes_lifecycle():
    image = Mock()

    def save(buffer, format):
        buffer.write(b"image-bytes")

    image.save.side_effect = save
    settings = StorageSettings(expires_in="1h")

    with patch.object(
        AsyncClient, "upload", new_callable=AsyncMock, return_value="https://image"
    ) as mock_upload:
        client = AsyncClient(key="test-key")
        url = await client.upload_image(image, lifecycle=settings)

    assert url == "https://image"
    assert mock_upload.call_args.kwargs["lifecycle"] == settings


@pytest.mark.asyncio
async def test_async_upload_lifecycle_includes_acl():
    with patch(
        "fal_client.client._async_maybe_retry_request", new_callable=AsyncMock
    ) as mock_request:
        response = httpx.Response(
            status_code=200, json={"access_url": "https://cdn-only/file"}
        )
        mock_request.return_value = response

        client = AsyncClient(key="test-key")
        url = await client.upload(
            b"hello",
            content_type="text/plain",
            repository="cdn",
            fallback_repository=[],
            lifecycle=StorageSettings(
                expires_in="immediate",
                initial_acl=StorageACL(
                    default="forbid",
                    rules=[StorageACLRule(user="usr_123", decision="allow")],
                ),
            ),
        )

    assert url == "https://cdn-only/file"
    lifecycle_header = mock_request.call_args[1]["headers"]["X-Fal-Object-Lifecycle"]
    assert json.loads(lifecycle_header) == {
        "expiration_duration_seconds": 60,
        "initial_acl": {
            "default": "forbid",
            "rules": [{"user": "usr_123", "decision": "allow"}],
        },
    }


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


def _make_error_response(
    status_code: int,
    body: str | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """Build a fake httpx.Response that will raise on raise_for_status."""
    resp = httpx.Response(
        status_code=status_code,
        text=body or "",
        headers=headers or {},
        request=httpx.Request("POST", "https://fal.run/test"),
    )
    return resp


class TestRaiseForStatus:
    def test_error_type_from_json_body(self):
        body = json.dumps({"detail": "not found", "error_type": "bad_request"})
        resp = _make_error_response(404, body=body)

        with pytest.raises(FalClientHTTPError) as exc_info:
            _raise_for_status(resp)

        err = exc_info.value
        assert err.status_code == 404
        assert err.message == "not found"
        assert err.error_type == "bad_request"

    def test_error_type_from_header_fallback(self):
        body = json.dumps({"detail": "Request timed out"})
        resp = _make_error_response(
            504,
            body=body,
            headers={"x-fal-error-type": "request_timeout"},
        )

        with pytest.raises(FalClientHTTPError) as exc_info:
            _raise_for_status(resp)

        err = exc_info.value
        assert err.status_code == 504
        assert err.message == "Request timed out"
        assert err.error_type == "request_timeout"

    def test_json_body_takes_precedence_over_header(self):
        body = json.dumps(
            {"detail": "Runner disconnected", "error_type": "runner_disconnected"}
        )
        resp = _make_error_response(
            503,
            body=body,
            headers={"x-fal-error-type": "internal_error"},
        )

        with pytest.raises(FalClientHTTPError) as exc_info:
            _raise_for_status(resp)

        assert exc_info.value.error_type == "runner_disconnected"

    def test_error_type_none_when_absent(self):
        body = json.dumps({"detail": "bad request"})
        resp = _make_error_response(400, body=body)

        with pytest.raises(FalClientHTTPError) as exc_info:
            _raise_for_status(resp)

        err = exc_info.value
        assert err.status_code == 400
        assert err.message == "bad request"
        assert err.error_type is None

    def test_non_json_response(self):
        resp = _make_error_response(500, body="Internal Server Error")

        with pytest.raises(FalClientHTTPError) as exc_info:
            _raise_for_status(resp)

        err = exc_info.value
        assert err.message == "Internal Server Error"
        assert err.error_type is None

    def test_success_does_not_raise(self):
        resp = httpx.Response(
            200,
            text="ok",
            request=httpx.Request("GET", "https://fal.run/test"),
        )
        _raise_for_status(resp)  # should not raise
