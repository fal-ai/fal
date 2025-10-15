import pytest
from unittest.mock import Mock, patch, AsyncMock

from fal_client.client import (
    Queued,
    InProgress,
    Completed,
    _BaseRequestHandle,
    SyncClient,
    AsyncClient,
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
