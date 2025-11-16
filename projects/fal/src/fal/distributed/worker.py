from __future__ import annotations

import asyncio
import inspect
import os
import pickle
import queue
import threading
import time
import traceback
import warnings
from collections.abc import AsyncIterator, Callable, Coroutine
from concurrent.futures import Future
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from fal.distributed.utils import (
    KeepAliveTimer,
    distributed_deserialize,
    distributed_serialize,
    encode_text_event,
    launch_distributed_processes,
)

if TYPE_CHECKING:
    import torch
    import torch.multiprocessing as mp
    from zmq.sugar.socket import Socket


class DistributedWorker:
    """
    A base class for distributed workers.
    """

    queue: queue.Queue[bytes]
    loop: asyncio.AbstractEventLoop
    thread: threading.Thread

    def __init__(
        self,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.queue = queue.Queue()
        self._cancel_event = threading.Event()

        try:
            import uvloop

            self.loop = uvloop.new_event_loop()
        except ImportError:
            self.loop = asyncio.new_event_loop()

        self._start_thread()

    def _start_thread(self) -> None:
        """
        Start the thread.
        """
        self.thread = threading.Thread(target=self._run_forever, daemon=True)
        self.thread.start()

    def _run_forever(self) -> None:
        """
        Run the event loop forever.
        """
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    # Public API

    @property
    def device(self) -> torch.device:
        """
        :return: The device for the current worker.
        """
        import torch

        return torch.device(f"cuda:{self.rank}")

    @property
    def running(self) -> bool:
        """
        :return: Whether the event loop is running.
        """
        return self.thread.is_alive()

    def initialize(self, **kwargs: Any) -> None:
        """
        Initialize the worker.
        """
        import torch

        torch.cuda.set_device(self.device)
        self.rank_print(f"Initializing worker on device {self.device}")

        setup_start = time.time()
        future = self.run_in_worker(self.setup, **kwargs)
        future.result()
        setup_duration = time.time() - setup_start
        self.rank_print(f"Setup took {setup_duration:.2f} seconds")

    def add_streaming_result(
        self,
        result: Any,
        image_format: str = "jpeg",
        as_text_event: bool = False,
    ) -> None:
        """
        Add a streaming result to the queue.
        :param result: The result to add to the queue.
        """
        if as_text_event:
            serialized = encode_text_event(
                result, is_final=False, image_format=image_format
            )
        else:
            serialized = distributed_serialize(
                result, is_final=False, image_format=image_format
            )

        self.queue.put_nowait(serialized)

    def add_streaming_error(self, error: Exception) -> None:
        """
        Add an error to the queue.
        :param error: The error to add to the queue.
        """
        self.queue.put_nowait(
            distributed_serialize({"error": str(error)}, is_final=False)
        )

    def rank_print(self, message: str, debug: bool = False) -> None:
        """
        Print a message with the rank of the current worker.
        :param message: The message to print.
        :param debug: Whether to print the message as a debug message.
        """
        prefix = "[debug] " if debug else ""
        print(f"{prefix}[rank {self.rank}] {message}")

    def is_cancelled(self) -> bool:
        """
        Check if the current task has been cancelled.
        This should be called periodically in long-running __call__ methods.
        
        :return: True if cancelled, False otherwise.
        
        Example:
            class VideoWorker(DistributedWorker):
                def __call__(self, num_frames=100, **kwargs):
                    frames = []
                    for i in range(num_frames):
                        # Check for cancellation
                        if self.is_cancelled():
                            return {"status": "cancelled", "frames": frames}
                        
                        frame = self.generate_frame(i)
                        frames.append(frame)
                    return {"frames": frames}
        """
        return self._cancel_event.is_set()

    def _clear_cancel(self) -> None:
        """Clear the cancellation flag before starting new work."""
        self._cancel_event.clear()

    def submit(self, coro: Coroutine[Any, Any, Any]) -> Future[Any]:
        """
        Submit a coroutine to the event loop.
        :param coro: The coroutine to submit to the event loop.
        :return: A future that will resolve to the result of the coroutine.
        """
        if not self.running:
            raise RuntimeError("Event loop is not running.")
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def shutdown(self, timeout: Optional[Union[int, float]] = None) -> None:
        """
        Shutdown the event loop.
        :param timeout: The timeout for the shutdown.
        """
        try:
            self.run_in_worker(self.teardown).result()
        except Exception as e:
            self.rank_print(f"Error during teardown: {e}\n{traceback.format_exc()}")

        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=timeout)

    async def _run_sync_in_executor(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Run a synchronous function in the executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(func, *args, **kwargs))

    def run_in_worker(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Future[Any]:
        """
        Run a function in the worker.
        """
        if inspect.iscoroutinefunction(func):
            coro = func(*args, **kwargs)
        else:
            # Using in place of asyncio.to_thread
            # since it's not available in Python 3.8
            coro = self._run_sync_in_executor(func, *args, **kwargs)

        return self.submit(coro)

    # Overrideables

    def setup(self, **kwargs: Any) -> None:
        """
        Override this method to set up the worker.
        This method is called once per worker.
        """
        return

    def teardown(self) -> None:
        """
        Override this method to tear down the worker.
        This method is called once per worker.
        """
        return

    def __call__(self, streaming: bool = False, **kwargs: Any) -> Any:
        """
        Override this method to run the worker.
        """
        return {}


class DistributedRunner:
    """
    A class to launch and manage distributed workers.
    
    Uses separate sockets for data plane (task execution) and control plane
    (cancellation, shutdown) to ensure control messages can bypass task queues.
    """

    zmq_socket: Optional[Socket[Any]]  # Data socket (ROUTER)
    zmq_control_socket: Optional[Socket[Any]]  # Control socket (PUB)
    context: Optional[mp.ProcessContext]
    keepalive_timer: Optional[KeepAliveTimer]

    def __init__(
        self,
        worker_cls: type[DistributedWorker] = DistributedWorker,
        world_size: int = 1,
        master_addr: str = "127.0.0.1",
        master_port: int = 29500,
        worker_addr: str = "127.0.0.1",
        worker_port: int = 54923,
        worker_control_port: Optional[int] = None,  # Defaults to worker_port + 1
        timeout: int = 86400,  # 24 hours
        keepalive_payload: dict[str, Any] = {},
        keepalive_interval: Optional[Union[int, float]] = None,
        cwd: Optional[Union[str, Path]] = None,
        set_device: Optional[bool] = None,  # deprecated
    ) -> None:
        self.worker_cls = worker_cls
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.worker_addr = worker_addr
        self.worker_port = worker_port
        self.worker_control_port = worker_control_port or (worker_port + 1)
        self.timeout = timeout
        self.cwd = cwd
        self.zmq_socket = None
        self.zmq_control_socket = None
        self.context = None
        self.keepalive_payload = keepalive_payload
        self.keepalive_interval = keepalive_interval
        self.keepalive_timer = None

        if set_device is not None:
            warnings.warn("set_device is deprecated and will be removed in the future.")

    def is_alive(self) -> bool:
        """
        Check if the distributed worker processes are alive.
        :return: True if the distributed processes are alive, False otherwise.
        """
        if self.context is None:
            return False
        for process in self.context.processes:
            if not process.is_alive():
                return False
        return True

    def terminate(self, timeout: Union[int, float] = 10) -> None:
        """
        Terminates the distributed worker processes.
        This method should be called to clean up the worker processes.
        """
        if self.context is not None:
            for process in self.context.processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=timeout)

    def gather_errors(self) -> list[Exception]:
        """
        Gathers errors from the distributed worker processes.

        This method should be called to collect any errors that occurred
        during execution.

        :return: A list of exceptions raised by the worker processes.
        """
        errors = []

        if self.context is not None:
            for error_file in self.context.error_files:
                if os.path.exists(error_file):
                    with open(error_file, "rb") as f:
                        error = pickle.loads(f.read())
                        errors.append(error)

                    os.remove(error_file)

        return errors

    def ensure_alive(self) -> None:
        """
        Ensures that the distributed worker processes are alive.
        If the processes are not alive, it raises an error.
        """
        if not self.is_alive():
            raise RuntimeError(
                f"Distributed processes are not running. Errors: {self.gather_errors()}"
            )

    def get_zmq_socket(self) -> Socket[Any]:
        """
        Returns the data plane ZeroMQ socket (ROUTER pattern).
        Used for sending tasks and receiving results.
        :return: A ZeroMQ ROUTER socket.
        """
        if self.zmq_socket is not None:
            return self.zmq_socket

        import zmq
        import zmq.asyncio

        context = zmq.asyncio.Context()
        socket = context.socket(zmq.ROUTER)
        socket.bind(f"tcp://{self.worker_addr}:{self.worker_port}")
        self.zmq_socket = socket
        return socket

    def get_zmq_control_socket(self) -> Socket[Any]:
        """
        Returns the control plane ZeroMQ socket (PUB pattern).
        Used for broadcasting control messages (cancel, exit) that bypass the task queue.
        :return: A ZeroMQ PUB socket.
        """
        if self.zmq_control_socket is not None:
            return self.zmq_control_socket

        import zmq
        import zmq.asyncio

        context = zmq.asyncio.Context()
        socket = context.socket(zmq.PUB)
        socket.bind(f"tcp://{self.worker_addr}:{self.worker_control_port}")
        self.zmq_control_socket = socket
        return socket

    def close_zmq_socket(self) -> None:
        """
        Closes both data and control ZeroMQ sockets.
        """
        if self.zmq_socket is not None:
            try:
                self.zmq_socket.close()
            except Exception as e:
                print(
                    f"[debug] Error closing ZeroMQ data socket: {e}\n"
                    f"{traceback.format_exc()}"
                )
            self.zmq_socket = None

        if self.zmq_control_socket is not None:
            try:
                self.zmq_control_socket.close()
            except Exception as e:
                print(
                    f"[debug] Error closing ZeroMQ control socket: {e}\n"
                    f"{traceback.format_exc()}"
                )
            self.zmq_control_socket = None

    def run(self, **kwargs: Any) -> None:
        """
        The main function to run the distributed worker.

        This function is called by each worker process spawned by
        `torch.multiprocessing.spawn`. This method must be synchronous.
        
        Uses a dedicated background thread to monitor control socket for
        cancel/exit messages, ensuring immediate response even during task execution.
        This is the production-grade approach used by Ray and similar systems.

        :param kwargs: The arguments to pass to the worker.
        """
        import torch.distributed as dist
        import zmq

        # Set up communication
        rank = int(os.environ["RANK"])
        context = zmq.Context()
        
        # Data socket for task execution (DEALER pattern)
        data_socket = context.socket(zmq.DEALER)
        data_socket.setsockopt(zmq.IDENTITY, str(rank).encode("utf-8"))
        data_socket.connect(f"tcp://{self.worker_addr}:{self.worker_port}")
        
        # Control socket for cancel/exit messages (SUB pattern)
        control_socket = context.socket(zmq.SUB)
        control_socket.connect(f"tcp://{self.worker_addr}:{self.worker_control_port}")
        control_socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages

        # Create and setup the worker
        worker = self.worker_cls(rank, self.world_size)
        try:
            worker.initialize(**kwargs)
        except Exception as e:
            worker.rank_print(
                f"Error during initialization: {e}\n{traceback.format_exc()}"
            )
            data_socket.send(b"EXIT")
            data_socket.close()
            control_socket.close()
            return

        # Control thread management
        control_thread_running = threading.Event()
        control_thread_running.set()
        should_exit = threading.Event()
        
        def control_message_handler():
            """
            Background thread that continuously monitors control socket.
            Handles cancel/exit messages immediately, even during task execution.
            """
            worker.rank_print("Control message handler started", debug=True)
            while control_thread_running.is_set():
                try:
                    control_msg = control_socket.recv(flags=zmq.NOBLOCK)
                    
                    if control_msg == b"CANCEL":
                        worker.rank_print("⚠️ CANCEL signal received via control plane")
                        worker._cancel_event.set()
                        # Rank 0 broadcasts to other workers
                        if rank == 0:
                            try:
                                params = [b"CANCEL", False, False]
                                dist.broadcast_object_list(params, src=0)
                            except Exception as e:
                                worker.rank_print(f"Error broadcasting cancel: {e}")
                                
                    elif control_msg == b"EXIT":
                        worker.rank_print("EXIT signal received via control plane")
                        should_exit.set()
                        control_thread_running.clear()
                        # Rank 0 broadcasts to other workers
                        if rank == 0:
                            try:
                                params = [b"EXIT", False, False]
                                dist.broadcast_object_list(params, src=0)
                            except Exception as e:
                                worker.rank_print(f"Error broadcasting exit: {e}")
                        
                except zmq.Again:
                    # No message available, sleep briefly
                    time.sleep(0.01)  # 10ms polling interval
                except Exception as e:
                    worker.rank_print(
                        f"Error in control handler: {e}\n{traceback.format_exc()}"
                    )
                    
            worker.rank_print("Control message handler stopped", debug=True)
        
        # Start control message handler thread
        control_thread = threading.Thread(
            target=control_message_handler,
            daemon=True,
            name=f"control-handler-rank-{rank}"
        )
        control_thread.start()

        # Wait until all workers are ready
        data_socket.send(b"READY")
        dist.barrier()

        # Define execution methods to invoke from workers
        def execute(payload: bytes) -> Any:
            """
            Execute the worker function with the given payload synchronously.
            :param payload: The payload to send to the worker.
            :return: The result from the worker.
            """
            # Clear cancellation flag before starting new work
            worker._clear_cancel()
            
            payload_dict = distributed_deserialize(payload)
            assert isinstance(payload_dict, dict)
            payload_dict["streaming"] = False

            try:
                future = worker.run_in_worker(worker.__call__, **payload_dict)
                result = future.result()
            except Exception as e:
                error_output = {"error": str(e)}
                worker.rank_print(
                    f"Error in execution: {error_output}\n{traceback.format_exc()}"
                )
                result = error_output

            dist.barrier()
            if worker.rank != 0:
                return

            data_socket.send(distributed_serialize(result, is_final=True))

        def stream(payload: bytes, as_text_events: bool) -> None:
            """
            Stream the result from the worker function with the given payload.
            :param payload: The payload to send to the worker.
            :return: An async iterator that yields the result from the worker.
            """
            # Clear cancellation flag before starting new work
            worker._clear_cancel()
            
            payload_dict = distributed_deserialize(payload)
            assert isinstance(payload_dict, dict)
            payload_dict["streaming"] = True
            image_format = payload_dict.get("image_format", "jpeg")
            encoded_response: Optional[bytes] = None

            try:
                future = worker.run_in_worker(worker.__call__, **payload_dict)
                while not future.done():
                    try:
                        intermediate = worker.queue.get(timeout=0.1)
                        if intermediate is not None and worker.rank == 0:
                            data_socket.send(intermediate)  # already serialized
                    except queue.Empty:
                        pass
                result = future.result()
            except Exception as e:
                error_output = {"error": str(e)}
                worker.rank_print(
                    f"Error in streaming: {error_output}\n{traceback.format_exc()}"
                )
                if worker.rank == 0:
                    if as_text_events:
                        encoded_response = encode_text_event(error_output)
                    else:
                        encoded_response = distributed_serialize(error_output)
            else:
                if worker.rank == 0:
                    if as_text_events:
                        encoded_response = encode_text_event(
                            result, is_final=True, image_format=image_format
                        )
                    else:
                        encoded_response = distributed_serialize(
                            result, is_final=True, image_format=image_format
                        )

            dist.barrier()
            if worker.rank != 0:
                return

            if encoded_response is not None:
                data_socket.send(encoded_response)
            data_socket.send(b"DONE")

        # Runtime code - rank 0 master worker
        if rank == 0:
            worker.rank_print("Master worker is ready to receive tasks.")
            
            while not should_exit.is_set():
                try:
                    # Simple blocking receive with timeout (control thread handles cancel/exit)
                    data_socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout
                    serialized_data = data_socket.recv()
                    
                    streaming = serialized_data[0] == ord("1")
                    as_text_events = serialized_data[1] == ord("1")
                    serialized_data = serialized_data[2:]
                    
                    # Broadcast task to other workers
                    params = [serialized_data, streaming, as_text_events]
                    dist.broadcast_object_list(params, src=0)

                    # Legacy support for data plane exit/cancel (shouldn't be used anymore)
                    if serialized_data == b"EXIT":
                        worker.rank_print("Received exit via data plane (legacy).")
                        should_exit.set()
                    elif serialized_data == b"CANCEL":
                        worker.rank_print("Received cancel via data plane (legacy).")
                        worker._cancel_event.set()
                    elif streaming:
                        stream(serialized_data, as_text_events)
                    else:
                        execute(serialized_data)
                        
                except zmq.Again:
                    # Timeout - check should_exit and continue
                    continue
                except Exception as e:
                    if not should_exit.is_set():
                        worker.rank_print(f"Error in master worker: {e}\n{traceback.format_exc()}")
        else:
            # Non-master worker - just listen for broadcasts from rank 0
            worker.rank_print("Worker waiting for tasks.")
            
            while not should_exit.is_set():
                try:
                    # Simple blocking broadcast receive (control thread handles cancel/exit)
                    params = [None, None, None]
                    dist.broadcast_object_list(params, src=0)
                    payload, streaming, as_text_events = params  # type: ignore[assignment]
                    
                    if payload == b"EXIT":
                        worker.rank_print("Received exit via broadcast.")
                        should_exit.set()
                    elif payload == b"CANCEL":
                        worker.rank_print("Received cancel via broadcast.")
                        worker._cancel_event.set()
                    elif payload is not None:
                        if streaming:
                            stream(payload, as_text_events)  # type: ignore[arg-type]
                        else:
                            execute(payload)  # type: ignore[arg-type]
                            
                except Exception as e:
                    if not should_exit.is_set():
                        worker.rank_print(f"Error in worker: {e}\n{traceback.format_exc()}")

        # Teardown
        worker.rank_print("Worker is tearing down.")
        
        # Stop control thread
        control_thread_running.clear()
        control_thread.join(timeout=2.0)
        if control_thread.is_alive():
            worker.rank_print("Control thread did not stop cleanly", debug=True)
        
        # Shutdown worker
        try:
            worker.shutdown()
            worker.rank_print("Worker torn down successfully.")
        except Exception as e:
            worker.rank_print(f"Error during teardown: {e}\n{traceback.format_exc()}")

        # Close sockets
        data_socket.send(b"EXIT")
        data_socket.close()
        control_socket.close()

    async def start(self, timeout: int = 1800, **kwargs: Any) -> None:
        """
        Starts the distributed worker processes.
        Initializes both data and control sockets.
        :param timeout: The timeout for the distributed processes.
        """
        import zmq

        if self.is_alive():
            raise RuntimeError("Distributed processes are already running.")

        self.context = launch_distributed_processes(
            self.run,
            world_size=self.world_size,
            master_addr=self.master_addr,
            master_port=self.master_port,
            timeout=self.timeout,
            cwd=self.cwd,
            **kwargs,
        )

        try:
            ready_workers: set[int] = set()
            data_socket = self.get_zmq_socket()
            # Initialize control socket (workers will connect to it)
            control_socket = self.get_zmq_control_socket()
            # Give control socket time to be ready before workers connect
            await asyncio.sleep(0.1)
            start_time = time.perf_counter()

            while len(ready_workers) < self.world_size:
                try:
                    ident, msg = await data_socket.recv_multipart(flags=zmq.NOBLOCK)  # type: ignore[misc]

                    if msg != b"READY":
                        worker_id = ident.decode("utf-8")
                        worker_msg = msg.decode("utf-8")
                        raise RuntimeError(
                            f"Unexpected message from worker {worker_id}: {worker_msg}"
                        )

                    print(f"[debug] Worker {ident.decode('utf-8')} is ready.")
                    ready_workers.add(ident)
                except zmq.Again:
                    total_wait_time = time.perf_counter() - start_time
                    if total_wait_time > timeout:
                        raise TimeoutError(
                            f"Timeout reached after {timeout} seconds while "
                            f"waiting for workers to be ready."
                        )
                    await asyncio.sleep(0.5)
                self.ensure_alive()
        except Exception as e:
            print(f"[debug] Error during startup: {e}\n{traceback.format_exc()}")
            self.terminate(timeout=timeout)
            raise RuntimeError("Failed to start distributed processes.") from e

        print("[debug] All workers are ready and running.")
        print(f"[debug] Control socket listening on port {self.worker_control_port}")

        # Start the keepalive timer
        self.maybe_start_keepalive()

    def keepalive(self, timeout: Optional[Union[int, float]] = 60.0) -> None:
        """
        Sends the keepalive payload to the worker.
        """
        # Cancel the keepalive timer
        self.maybe_cancel_keepalive()
        loop_thread = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
            loop_thread.start()

        future = asyncio.run_coroutine_threadsafe(
            self.invoke(self.keepalive_payload), loop
        )

        try:
            future.result(timeout=timeout)
        except Exception as e:
            print(f"[debug] Error during keepalive: {e}\n{traceback.format_exc()}")
            raise RuntimeError("Failed to run keepalive.") from e
        finally:
            if loop_thread is not None:
                loop.call_soon_threadsafe(loop.stop)
                loop_thread.join(timeout=timeout)
            # Restart the keepalive timer
            self.maybe_start_keepalive()

    def maybe_start_keepalive(self) -> None:
        """
        Starts the keepalive timer if it is set.
        """
        if self.keepalive_timer is None and self.keepalive_interval is not None:
            self.keepalive_timer = KeepAliveTimer(
                self.keepalive, self.keepalive_interval, start=True
            )

    def maybe_reset_keepalive(self) -> None:
        """
        Resets the keepalive timer if it is set.
        """
        if self.keepalive_timer is not None:
            self.keepalive_timer.reset()

    def maybe_cancel_keepalive(self) -> None:
        """
        Cancels the keepalive timer if it is set.
        """
        if self.keepalive_timer is not None:
            self.keepalive_timer.cancel()
            self.keepalive_timer = None

    async def stop(self, timeout: int = 10) -> None:
        """
        Stops the distributed worker processes using the control socket.
        This bypasses any queued tasks and sends the exit signal immediately.
        :param timeout: The timeout for the distributed processes to stop.
        """
        import zmq

        if not self.is_alive():
            raise RuntimeError("Distributed processes are not running.")

        self.maybe_cancel_keepalive()
        worker_exits: set[int] = set()
        data_socket = self.get_zmq_socket()
        control_socket = self.get_zmq_control_socket()
        
        # Send EXIT via control socket for immediate delivery
        print("[debug] Sending EXIT signal via control socket")
        await control_socket.send(b"EXIT")
        
        # Small delay for PUB/SUB propagation
        await asyncio.sleep(0.1)

        wait_start = time.perf_counter()
        while len(worker_exits) < self.world_size:
            try:
                ident, msg = await data_socket.recv_multipart(flags=zmq.NOBLOCK)  # type: ignore[misc]
                if msg == b"EXIT":
                    worker_exits.add(ident)
                    if len(worker_exits) == self.world_size:
                        print("[debug] All workers have exited cleanly.")
                        await asyncio.sleep(1)  # Allow time for cleanup
                        break

            except zmq.Again:
                # No messages available, continue waiting
                await asyncio.sleep(0.1)

            if time.perf_counter() - wait_start > timeout:
                print(
                    f"[debug] Timeout reached after {timeout} seconds, "
                    f"stopping waiting."
                )
                break

            if not self.is_alive():
                print("[debug] All workers have exited prematurely.")
                break

        if self.is_alive():
            print("[debug] Some workers did not exit cleanly, terminating them.")
            self.terminate(timeout=timeout)

        self.close_zmq_socket()

    async def cancel(self) -> None:
        """
        Send cancellation signal to all workers to drop their current work.
        Uses the control socket to bypass task queues for immediate delivery.
        Workers will stop at the next point they call is_cancelled().
        Workers remain alive and ready for new tasks.
        
        Note: This is cooperative cancellation. Workers must call is_cancelled()
        in their __call__ method to respect cancellation.
        
        Example:
            runner = DistributedRunner(MyWorker, world_size=2)
            await runner.start()
            
            # Start long-running task
            task = asyncio.create_task(runner.invoke({"num_frames": 1000}))
            
            # Cancel it (delivered immediately, even if workers are busy)
            await runner.cancel()
        """
        import zmq

        if not self.is_alive():
            raise RuntimeError("Distributed processes are not running.")

        control_socket = self.get_zmq_control_socket()
        # Send via control socket for immediate delivery (bypasses task queue)
        print("[debug] Sending CANCEL signal via control socket")
        await control_socket.send(b"CANCEL")
        
        # Small delay for PUB/SUB propagation to all workers
        await asyncio.sleep(0.05)

    async def stream(
        self,
        payload: dict[str, Any] = {},
        timeout: Optional[int] = None,
        streaming_timeout: Optional[int] = None,
        as_text_events: bool = False,
    ) -> AsyncIterator[Any]:
        """
        Streams the result from the distributed worker.
        :param payload: The payload to send to the worker.
        :param timeout: The timeout for the overall operation.
        :param streaming_timeout: The timeout in-between streamed results.
        :param as_text_events: Whether to yield results as text events.
        :return: An async iterator that yields the result from the worker.
        """
        import zmq

        self.ensure_alive()
        self.maybe_cancel_keepalive()  # Cancel until the streaming is done
        socket = self.get_zmq_socket()
        payload_serialized = distributed_serialize(payload, is_final=True)
        await socket.send_multipart(
            [b"0", b"1" + (b"1" if as_text_events else b"0") + payload_serialized]
        )

        start_time = time.perf_counter()
        last_yield_time = start_time
        yielded_once = False

        while True:
            iter_start_time = time.perf_counter()
            try:
                rank, response = await socket.recv_multipart(flags=zmq.NOBLOCK)  # type: ignore[misc]
            except zmq.Again:
                if timeout is not None and iter_start_time - start_time > timeout:
                    raise TimeoutError(f"Streaming timed out after {timeout} seconds.")
                if (
                    streaming_timeout is not None
                    and iter_start_time - last_yield_time > streaming_timeout
                ):
                    raise TimeoutError(
                        f"Streaming timed out after {streaming_timeout} "
                        f"seconds of inactivity."
                    )

                await asyncio.sleep(0.1)
                continue

            assert rank == b"0", "Expected response from worker with rank 0"

            if response == b"DONE":
                if not yielded_once:
                    raise RuntimeError("No data was yielded from the worker.")
                break

            if as_text_events:
                yield response
            else:
                yield distributed_deserialize(response)

            yielded_once = True
            last_yield_time = iter_start_time

        self.maybe_start_keepalive()  # Restart the keepalive timer

    async def invoke(
        self,
        payload: dict[str, Any] = {},
        timeout: Optional[int] = None,
    ) -> Any:
        """
        Invokes the distributed worker with the given payload.
        :param payload: The payload to send to the worker.
        :param timeout: The timeout for the overall operation.
        :return: The result from the worker.
        """
        import zmq

        self.ensure_alive()
        self.maybe_cancel_keepalive()  # Cancel until the invocation is done
        socket = self.get_zmq_socket()
        payload_serialized = distributed_serialize(payload, is_final=True)

        await socket.send_multipart([b"0", b"00" + payload_serialized])

        # Wait for the response from the worker
        start_time = time.perf_counter()
        while True:
            try:
                rank, response = await socket.recv_multipart(flags=zmq.NOBLOCK)  # type: ignore[misc]
                break  # Exit the loop if we received a response
            except zmq.Again:
                elapsed = time.perf_counter() - start_time
                if timeout is not None and elapsed > timeout:
                    raise TimeoutError(f"Invocation timed out after {timeout} seconds.")

                await asyncio.sleep(0.1)
                self.ensure_alive()

        self.maybe_start_keepalive()  # Restart the keepalive timer
        assert rank == b"0", "Expected response from worker with rank 0"
        return distributed_deserialize(response)

    async def __aenter__(self) -> DistributedRunner:
        """
        Enter the context manager.
        :return: The DistributedRunner instance.
        """
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        exc_traceback: Optional[traceback.StackSummary],
    ) -> None:
        """
        Exit the context manager.
        :param exc_type: The type of the exception raised, if any.
        :param exc_value: The value of the exception raised, if any.
        :param traceback: The traceback of the exception raised, if any.
        """
        try:
            await self.stop()
        except Exception as e:
            print(f"[debug] Error during cleanup: {e}\n{traceback.format_exc()}")
