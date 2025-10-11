import pytest
import asyncio
from fal.distributed.worker import DistributedWorker, DistributedRunner

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



