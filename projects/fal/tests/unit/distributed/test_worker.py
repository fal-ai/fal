import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from fal.distributed.utils import distributed_serialize
from fal.distributed.worker import (
    REQUEST_ID_SIZE,
    DistributedRunner,
    DistributedWorker,
)


class SimpleWorker(DistributedWorker):
    def __call__(self, **kwargs):
        return {"status": "ok"}


def test_worker_event_loop():
    """Test that a worker can be created and has a running event loop."""
    worker = SimpleWorker(rank=0, world_size=1)
    assert worker.running
    worker.shutdown()


def test_worker_task_submission():
    """Test that tasks can be submitted to a worker and return results."""
    worker = SimpleWorker(rank=0, world_size=1)
    future = worker.run_in_worker(worker.__call__)
    result = future.result()
    assert result == {"status": "ok"}
    worker.shutdown()


def test_runner_initialization():
    """Test that a distributed runner can be initialized with correct parameters."""
    runner = DistributedRunner(SimpleWorker, world_size=2)
    assert runner.world_size == 2
    assert runner.worker_cls == SimpleWorker
    assert not runner.is_alive()  # Should not be alive before start()


def test_worker_shutdown_is_safe():
    """Test that shutdown doesn't crash even if called multiple times."""
    worker = SimpleWorker(rank=0, world_size=1)
    worker.shutdown()
    worker.shutdown()  # Should not crash


def test_runner_not_alive_before_start():
    """Test that runner reports correctly before starting."""
    runner = DistributedRunner(SimpleWorker, world_size=1)
    assert not runner.is_alive()
    with pytest.raises(RuntimeError, match="not running"):
        runner.ensure_alive()


def test_worker_handles_errors_in_callable():
    """Test that worker properly handles errors raised in __call__."""

    class ErrorWorker(DistributedWorker):
        def error_method(self):
            raise ValueError("Intentional test error")

    worker = ErrorWorker(rank=0, world_size=1)
    future = worker.run_in_worker(worker.error_method)

    with pytest.raises(ValueError, match="Intentional test error"):
        future.result()

    worker.shutdown()


def test_worker_handles_async_functions():
    """Test that worker can handle async functions."""

    class AsyncWorker(DistributedWorker):
        async def async_method(self):
            await asyncio.sleep(0.01)
            return {"async": "result"}

    worker = AsyncWorker(rank=0, world_size=1)
    future = worker.run_in_worker(worker.async_method)
    result = future.result()

    assert result == {"async": "result"}
    worker.shutdown()


def test_worker_submit_when_not_running():
    """Test that submitting to a stopped worker raises error."""
    worker = SimpleWorker(rank=0, world_size=1)
    worker.shutdown()

    async def dummy():
        return {}

    with pytest.raises(RuntimeError, match="Event loop is not running"):
        worker.submit(dummy())


def test_add_streaming_result():
    """Test that streaming results are queued correctly."""
    worker = SimpleWorker(rank=0, world_size=1)
    worker.add_streaming_result({"frame": 1})
    worker.add_streaming_result({"frame": 2})

    # Check queue has items
    assert not worker.queue.empty()
    item1 = worker.queue.get_nowait()
    assert item1 is not None

    worker.shutdown()


def test_add_streaming_error():
    """Test that errors can be added to streaming queue."""
    worker = SimpleWorker(rank=0, world_size=1)
    worker.add_streaming_error(ValueError("test error"))

    assert not worker.queue.empty()
    error_item = worker.queue.get_nowait()
    assert error_item is not None

    worker.shutdown()


def test_runner_gather_errors_when_no_errors():
    """Test gather_errors returns empty list when no errors."""
    runner = DistributedRunner(SimpleWorker, world_size=1)
    errors = runner.gather_errors()
    assert errors == []


def test_runner_zmq_socket_cleanup():
    """Test that ZMQ socket can be safely closed multiple times."""
    runner = DistributedRunner(SimpleWorker, world_size=1)
    assert runner.zmq_socket is None

    runner.close_zmq_socket()  # Should not crash even if None
    assert runner.zmq_socket is None

    runner.close_zmq_socket()  # Should not crash if called again
    assert runner.zmq_socket is None


def test_runner_terminate_when_not_started():
    """Test that terminate() is safe to call even when not started."""
    runner = DistributedRunner(SimpleWorker, world_size=1)
    runner.terminate()  # Should not crash


@pytest.mark.asyncio
async def test_invoke_discards_stale_responses():
    """Test that invoke() discards responses with mismatched request IDs.

    This tests the fix for a race condition where:
    1. Request A is sent to the worker
    2. Request A is cancelled mid-flight (after send, before receive)
    3. Worker sends response for Request A
    4. Request B is sent
    5. Request B's recv loop should discard Request A's response
    """
    runner = DistributedRunner(SimpleWorker, world_size=1)

    # Create a mock socket
    mock_socket = AsyncMock()

    # Track which request_ids sent
    captured_payloads = asyncio.Queue()

    async def capture_send(msg):
        # Message format: [b"0", b"00" + request_id + payload]
        if len(msg) == 2 and msg[0] == b"0" and msg[1][:2] == b"00":
            await captured_payloads.put(msg[1][2:])

    mock_socket.send_multipart = capture_send

    # recv_multipart will return stale response first, then correct response
    call_count = 0

    # Mimic implementation of what it will do in the worker
    async def mock_echo_recv(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        captured_payload = await captured_payloads.get()
        await asyncio.sleep(0.5)
        return (b"0", captured_payload)

    mock_socket.recv_multipart = mock_echo_recv

    # Mock the runner state
    runner.zmq_socket = mock_socket
    runner.context = MagicMock()
    runner.context.processes = [MagicMock(is_alive=MagicMock(return_value=True))]

    # Call invoke with a short timeout so we don't get the response
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(runner.invoke({"test": "stale"}), timeout=0.1)

    result = await runner.invoke({"test": "correct"})

    # Verify we got the correct response, not the stale one
    assert result == {"test": "correct"}
    # Verify recv was called twice (once for stale, once for correct)
    assert call_count == 2


@pytest.mark.asyncio
async def test_stream_discards_stale_responses():
    """Test that stream() discards responses with mismatched request IDs."""
    runner = DistributedRunner(SimpleWorker, world_size=1)

    mock_socket = AsyncMock()

    stale_request_id = uuid.uuid4().bytes
    stale_chunk = stale_request_id + distributed_serialize(
        {"chunk": "stale"}, is_final=False
    )

    correct_response_data = distributed_serialize({"chunk": "correct"}, is_final=False)
    captured_request_id = None

    async def capture_send(msg):
        nonlocal captured_request_id
        if len(msg) == 2 and len(msg[1]) > 2 + REQUEST_ID_SIZE:
            captured_request_id = msg[1][2 : 2 + REQUEST_ID_SIZE]

    mock_socket.send_multipart = capture_send

    call_count = 0

    async def mock_recv(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Stale response - should be discarded
            return (b"0", stale_chunk)
        elif call_count == 2:
            # Correct chunk
            return (b"0", captured_request_id + correct_response_data)
        else:
            # DONE marker
            return (b"0", captured_request_id + b"DONE")

    mock_socket.recv_multipart = mock_recv

    runner.zmq_socket = mock_socket
    runner.context = MagicMock()
    runner.context.processes = [MagicMock(is_alive=MagicMock(return_value=True))]

    # Collect streamed results
    results = []
    async for chunk in runner.stream({"test": "payload"}):
        results.append(chunk)

    # Should only have the correct chunk, stale was discarded
    assert len(results) == 1
    assert results[0] == {"chunk": "correct"}
    # recv called 3 times: stale (discarded), correct chunk, DONE
    assert call_count == 3
