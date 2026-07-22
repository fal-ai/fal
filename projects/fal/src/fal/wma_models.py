from __future__ import annotations

import asyncio
import inspect
import logging
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from fractions import Fraction
from itertools import islice
from typing import Any, AsyncIterable, Awaitable, Callable, Literal, Optional

from fal.wma import AiortcPeer, Session

try:
    import av
    from aiortc import MediaStreamTrack
    from aiortc.mediastreams import MediaStreamError
except ImportError as error:
    _AIORTC_IMPORT_ERROR = error
    av = None

    class MediaStreamTrack:  # type: ignore[no-redef]
        kind = "video"

        def __init__(self) -> None:
            raise RuntimeError(
                "Video processor support requires aiortc; install fal[wma]"
            ) from _AIORTC_IMPORT_ERROR

    class MediaStreamError(Exception):  # type: ignore[no-redef]
        pass


OverflowPolicy = Literal["drop_oldest", "drop_newest"]
ExecutionPolicy = Literal["worker", "event_loop"]
FrameProcessor = Callable[[list[Any]], Any | Awaitable[Any]]
_END = object()
logger = logging.getLogger(__name__)
VIDEO_CLOCK_RATE = 90_000


@dataclass(frozen=True)
class VideoProcessorPolicy:
    """Latency and execution policy for decoded-frame processing."""

    batch_size: int = 1
    max_queue_size: int = 1
    max_batch_wait_ms: float = 10
    overflow: OverflowPolicy = "drop_oldest"
    execution: ExecutionPolicy = "worker"
    output_format: str = "bgr24"
    max_output_frames: Optional[int] = None
    processor_timeout_ms: Optional[float] = 30_000
    shutdown_timeout_ms: float = 5_000

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.max_queue_size < 1:
            raise ValueError("max_queue_size must be at least 1")
        if self.max_batch_wait_ms < 0:
            raise ValueError("max_batch_wait_ms cannot be negative")
        if self.overflow not in {"drop_oldest", "drop_newest"}:
            raise ValueError("overflow must be drop_oldest or drop_newest")
        if self.execution not in {"worker", "event_loop"}:
            raise ValueError("execution must be worker or event_loop")
        if self.max_output_frames is not None and self.max_output_frames < 1:
            raise ValueError("max_output_frames must be at least 1")
        if self.processor_timeout_ms is not None and self.processor_timeout_ms <= 0:
            raise ValueError("processor_timeout_ms must be positive")
        if self.shutdown_timeout_ms <= 0:
            raise ValueError("shutdown_timeout_ms must be positive")


@dataclass(frozen=True)
class VideoProcessorStats:
    input_frames: int
    output_frames: int
    dropped_frames: int
    batches: int


class VideoProcessorTrack(MediaStreamTrack):
    """A bounded, batched aiortc video transform track."""

    kind = "video"

    def __init__(
        self,
        source: Any,
        processor: FrameProcessor,
        *,
        policy: Optional[VideoProcessorPolicy] = None,
        on_error: Optional[Callable[[BaseException], Any]] = None,
    ) -> None:
        super().__init__()
        self.source = source
        self.processor = processor
        self.policy = policy or VideoProcessorPolicy()
        self._on_error = on_error
        self._queue: asyncio.Queue[Any] = asyncio.Queue(
            maxsize=self.policy.max_queue_size + 1
        )
        self._ready: deque[Any] = deque()
        self._reader: Optional[asyncio.Task] = None
        self._recv_tasks: set[asyncio.Task] = set()
        self._worker_futures: set[asyncio.Future] = set()
        self._source_error: Optional[BaseException] = None
        self._source_ended = False
        self._stopped = False
        self._failure: Optional[BaseException] = None
        self._input_frames = 0
        self._output_frames = 0
        self._dropped_frames = 0
        self._batches = 0

    @property
    def stats(self) -> VideoProcessorStats:
        return VideoProcessorStats(
            input_frames=self._input_frames,
            output_frames=self._output_frames,
            dropped_frames=self._dropped_frames,
            batches=self._batches,
        )

    async def recv(self) -> Any:
        task = asyncio.current_task()
        if task is not None:
            self._recv_tasks.add(task)
        try:
            if self._stopped:
                raise MediaStreamError
            self._ensure_reader()
            while not self._ready:
                batch = await self._next_batch()
                if self._stopped:
                    raise MediaStreamError
                outputs = await self._process(batch)
                if self._stopped:
                    raise MediaStreamError
                self._ready.extend(outputs)
            return self._ready.popleft()
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            self._notify_failure(error)
            raise
        finally:
            if task is not None:
                self._recv_tasks.discard(task)

    async def aclose(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.policy.shutdown_timeout_ms / 1000
        self.stop()
        current = asyncio.current_task()
        recv_tasks = [task for task in self._recv_tasks if task is not current]
        for task in recv_tasks:
            task.cancel()
        await self._wait_for_shutdown(*recv_tasks, deadline=deadline)
        reader = self._reader
        if reader is not None:
            await self._wait_for_shutdown(reader, deadline=deadline)
        futures = list(self._worker_futures)
        if futures:
            await self._wait_for_shutdown(*futures, deadline=deadline)

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        reader = self._reader
        if reader is not None:
            reader.cancel()
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        for task in list(self._recv_tasks):
            if task is not current:
                task.cancel()
        self.source.stop()
        super().stop()

    def _ensure_reader(self) -> None:
        if self._reader is None:
            self._reader = asyncio.create_task(self._read_source())

    async def _read_source(self) -> None:
        try:
            while not self._stopped:
                frame = await self.source.recv()
                self._input_frames += 1
                self._enqueue(frame)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            self._source_error = error
        finally:
            self._queue.put_nowait(_END)

    def _enqueue(self, frame: Any) -> None:
        if self._queue.qsize() < self.policy.max_queue_size:
            self._queue.put_nowait(frame)
            return
        self._dropped_frames += 1
        if self.policy.overflow == "drop_newest":
            return
        self._queue.get_nowait()
        self._queue.put_nowait(frame)

    async def _next_batch(self) -> list[Any]:
        if self._source_ended:
            self._raise_source_error()

        first = await self._queue.get()
        if first is _END:
            self._source_ended = True
            self._raise_source_error()

        batch = [first]
        if self.policy.batch_size == 1:
            return batch
        if self.policy.max_batch_wait_ms == 0:
            while len(batch) < self.policy.batch_size and not self._queue.empty():
                item = self._queue.get_nowait()
                if item is _END:
                    self._source_ended = True
                    break
                batch.append(item)
            return batch

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.policy.max_batch_wait_ms / 1000
        while len(batch) < self.policy.batch_size:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=remaining,
                )
            except TimeoutError:
                break
            if item is _END:
                self._source_ended = True
                break
            batch.append(item)
        return batch

    async def _process(self, batch: list[Any]) -> list[Any]:
        self._batches += 1
        deadline = self._processor_deadline()
        if self.policy.execution == "event_loop":
            result = self.processor(batch)
            if inspect.isawaitable(result):
                result = await self._wait_for_processor(
                    result,
                    cancel_on_timeout=True,
                    deadline=deadline,
                )
            frames = self._prepare_outputs(result, batch)
        else:
            result = await self._run_worker(
                self._invoke_and_prepare,
                batch,
                deadline=deadline,
            )
            if inspect.isawaitable(result):
                result = await self._wait_for_processor(
                    result,
                    cancel_on_timeout=True,
                    deadline=deadline,
                )
                frames = await self._run_worker(
                    self._prepare_outputs,
                    result,
                    batch,
                    deadline=deadline,
                )
            else:
                frames = result
        self._output_frames += len(frames)
        return frames

    def _invoke_and_prepare(self, batch: list[Any]) -> Any:
        result = self.processor(batch)
        if inspect.isawaitable(result):
            return result
        return self._prepare_outputs(result, batch)

    def _prepare_outputs(self, result: Any, batch: list[Any]) -> list[Any]:
        max_outputs = self.policy.max_output_frames or self.policy.batch_size
        outputs = self._normalize_outputs(result, max_outputs)
        return [
            self._to_video_frame(output, batch, index)
            for index, output in enumerate(outputs)
        ]

    async def _run_worker(
        self,
        function: Callable,
        *args: Any,
        deadline: Optional[float],
    ) -> Any:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, function, *args)
        self._worker_futures.add(future)
        future.add_done_callback(self._worker_done)
        return await self._wait_for_processor(
            future,
            cancel_on_timeout=False,
            deadline=deadline,
        )

    def _normalize_outputs(self, outputs: Any, limit: int) -> list[Any]:
        if outputs is None:
            return []
        if hasattr(outputs, "pts") or getattr(outputs, "ndim", None) == 3:
            return [outputs]
        try:
            bounded = list(islice(iter(outputs), limit + 1))
        except TypeError:
            return [outputs]
        if len(bounded) > limit:
            raise ValueError(f"processor returned more than {limit} output frames")
        return bounded

    def _to_video_frame(
        self,
        output: Any,
        inputs: list[Any],
        index: int,
    ) -> Any:
        frame = output
        if not hasattr(frame, "pts"):
            if av is None:
                raise RuntimeError("Array frame conversion requires aiortc and PyAV")
            frame = av.VideoFrame.from_ndarray(
                output,
                format=self.policy.output_format,
            )
        if frame.pts is None:
            if index >= len(inputs):
                raise ValueError(
                    "processor outputs beyond the input batch need explicit timestamps"
                )
            if inputs[index].pts is None:
                raise ValueError("input frame is missing pts")
            frame.pts = inputs[index].pts
        if frame.time_base is None:
            if index >= len(inputs):
                raise ValueError(
                    "processor outputs beyond the input batch need explicit time_base"
                )
            if inputs[index].time_base is None:
                raise ValueError("input frame is missing time_base")
            frame.time_base = inputs[index].time_base
        return frame

    async def _wait_for_processor(
        self,
        awaitable: Awaitable[Any],
        *,
        cancel_on_timeout: bool,
        deadline: Optional[float],
    ) -> Any:
        protected = awaitable if cancel_on_timeout else asyncio.shield(awaitable)
        timeout_ms = self.policy.processor_timeout_ms
        if deadline is None:
            return await protected
        remaining = max(0, deadline - asyncio.get_running_loop().time())
        try:
            return await asyncio.wait_for(
                protected,
                timeout=remaining,
            )
        except TimeoutError as error:
            assert timeout_ms is not None
            raise TimeoutError(f"video processor exceeded {timeout_ms:g} ms") from error

    def _processor_deadline(self) -> Optional[float]:
        timeout_ms = self.policy.processor_timeout_ms
        if timeout_ms is None:
            return None
        return asyncio.get_running_loop().time() + timeout_ms / 1000

    async def _wait_for_shutdown(
        self,
        *awaitables: Awaitable[Any],
        deadline: float,
    ) -> None:
        if not awaitables:
            return
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return
        combined = asyncio.gather(*awaitables, return_exceptions=True)
        with suppress(TimeoutError):
            await asyncio.wait_for(
                asyncio.shield(combined),
                timeout=remaining,
            )

    def _worker_done(self, future: asyncio.Future) -> None:
        self._worker_futures.discard(future)
        if not future.cancelled():
            with suppress(Exception):
                future.exception()

    def _notify_failure(self, error: BaseException) -> None:
        if self._failure is not None:
            return
        self._failure = error
        if self._on_error is not None:
            try:
                self._on_error(error)
            except Exception:
                logger.exception("WMA video processor error callback failed")

    def _raise_source_error(self) -> None:
        if self._source_error is not None:
            raise self._source_error
        raise RuntimeError("video source ended")


class VideoProcessorBinding:
    """Attaches processed video and optional audio passthrough to an aiortc peer."""

    def __init__(
        self,
        peer_connection: Any,
        processor: FrameProcessor,
        *,
        policy: Optional[VideoProcessorPolicy] = None,
        passthrough_audio: bool = False,
        on_error: Optional[Callable[[BaseException], Any]] = None,
    ) -> None:
        self.peer_connection = peer_connection
        self.processor = processor
        self.policy = policy or VideoProcessorPolicy()
        self.passthrough_audio = passthrough_audio
        self._on_error = on_error
        self.tracks: list[VideoProcessorTrack] = []
        self._closed = False
        self.error: Optional[BaseException] = None
        self.failed = asyncio.Event()

        @peer_connection.on("track")
        def on_track(track: Any) -> None:
            if self._closed:
                track.stop()
                return
            if track.kind == "video":
                processed = VideoProcessorTrack(
                    track,
                    self.processor,
                    policy=self.policy,
                    on_error=self._handle_error,
                )
                self.tracks.append(processed)
                peer_connection.addTrack(processed)
            elif track.kind == "audio" and self.passthrough_audio:
                peer_connection.addTrack(track)

    def _handle_error(self, error: BaseException) -> None:
        if self.error is not None:
            return
        self.error = error
        self.failed.set()
        if self._on_error is not None:
            try:
                self._on_error(error)
            except Exception:
                logger.exception("WMA video binding error callback failed")

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.tracks:
            await asyncio.gather(
                *(track.aclose() for track in self.tracks),
                return_exceptions=True,
            )


def attach_video_processor(
    peer_connection: Any,
    processor: FrameProcessor,
    *,
    policy: Optional[VideoProcessorPolicy] = None,
    passthrough_audio: bool = False,
    session: Optional[Session] = None,
    on_error: Optional[Callable[[BaseException], Any]] = None,
) -> VideoProcessorBinding:
    """Attach a decoded-frame processor to incoming aiortc video tracks."""

    binding = VideoProcessorBinding(
        peer_connection,
        processor,
        policy=policy,
        passthrough_audio=passthrough_audio,
        on_error=on_error,
    )
    if session is not None:
        session.defer(binding.close)
    return binding


class VideoProcessorPeer(AiortcPeer):
    """High-level aiortc backend for decoded-frame world models."""

    def __init__(
        self,
        session: Session,
        processor: FrameProcessor,
        *,
        policy: Optional[VideoProcessorPolicy] = None,
        passthrough_audio: bool = False,
        configure_peer: Optional[Callable[[Any, VideoProcessorBinding], Any]] = None,
        create_default_channel: bool = False,
        rtc_configuration: Any = None,
        peer_connection_factory: Optional[Callable[[], Any]] = None,
        disconnected_grace_seconds: Optional[float] = 0,
    ) -> None:
        self.binding: Optional[VideoProcessorBinding] = None

        async def connect(peer_connection: Any) -> None:
            self.binding = attach_video_processor(
                peer_connection,
                processor,
                policy=policy,
                passthrough_audio=passthrough_audio,
                on_error=lambda _error: self._closed.set(),
            )
            if configure_peer is not None:
                result = configure_peer(peer_connection, self.binding)
                if inspect.isawaitable(result):
                    await result

        super().__init__(
            session,
            connect,
            create_default_channel=create_default_channel,
            rtc_configuration=rtc_configuration,
            peer_connection_factory=peer_connection_factory,
            disconnected_grace_seconds=disconnected_grace_seconds,
        )

    async def close(self) -> None:
        try:
            await super().close()
        finally:
            if self.binding is not None:
                await self.binding.close()


@dataclass(frozen=True)
class VideoSourcePolicy:
    """Queueing, pacing, and conversion policy for generated video frames."""

    fps: float = 30
    max_queue_size: int = 1
    overflow: OverflowPolicy = "drop_oldest"
    output_format: str = "rgb24"
    execution: ExecutionPolicy = "worker"
    start_immediately: bool = True
    initial_prefetch_frames: int = 1
    shutdown_timeout_ms: float = 5_000

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.max_queue_size < 1:
            raise ValueError("max_queue_size must be at least 1")
        if self.overflow not in {"drop_oldest", "drop_newest"}:
            raise ValueError("overflow must be drop_oldest or drop_newest")
        if self.execution not in {"worker", "event_loop"}:
            raise ValueError("execution must be worker or event_loop")
        if not 1 <= self.initial_prefetch_frames <= self.max_queue_size:
            raise ValueError(
                "initial_prefetch_frames must be between 1 and max_queue_size"
            )
        if self.shutdown_timeout_ms <= 0:
            raise ValueError("shutdown_timeout_ms must be positive")


@dataclass(frozen=True)
class VideoSourceStats:
    produced_frames: int
    output_frames: int
    dropped_frames: int
    queue_depth: int
    last_queue_age_ms: float
    last_pace_sleep_ms: float


class VideoSourceTrack(MediaStreamTrack):
    """An fps-paced, bounded outgoing track backed by an async frame source."""

    kind = "video"

    def __init__(
        self,
        source: AsyncIterable[Any],
        *,
        policy: Optional[VideoSourcePolicy] = None,
        on_error: Optional[Callable[[BaseException], Any]] = None,
    ) -> None:
        super().__init__()
        self.source = source
        self.policy = policy or VideoSourcePolicy()
        self._on_error = on_error
        self._queue: asyncio.Queue[Any] = asyncio.Queue(
            maxsize=self.policy.max_queue_size + 1
        )
        self._reader: Optional[asyncio.Task] = None
        self._recv_tasks: set[asyncio.Task] = set()
        self._worker_futures: set[asyncio.Future] = set()
        self._consumer_started = asyncio.Event()
        self._source_error: Optional[BaseException] = None
        self._stopped = False
        self._failure: Optional[BaseException] = None
        self._source_closed = False
        self._source_close_task: Optional[asyncio.Task] = None
        self._next_emit_at: Optional[float] = None
        self._frame_index = 0
        self._produced_frames = 0
        self._output_frames = 0
        self._dropped_frames = 0
        self._queued_frames = 0
        self._last_queue_age_ms = 0.0
        self._last_pace_sleep_ms = 0.0
        self._time_base = Fraction(1, VIDEO_CLOCK_RATE)
        self._frame_step = Fraction(VIDEO_CLOCK_RATE, 1) / Fraction(
            str(self.policy.fps)
        )
        if self.policy.start_immediately:
            self._ensure_reader()

    @property
    def stats(self) -> VideoSourceStats:
        return VideoSourceStats(
            produced_frames=self._produced_frames,
            output_frames=self._output_frames,
            dropped_frames=self._dropped_frames,
            queue_depth=self._queued_frames,
            last_queue_age_ms=self._last_queue_age_ms,
            last_pace_sleep_ms=self._last_pace_sleep_ms,
        )

    async def recv(self) -> Any:
        task = asyncio.current_task()
        if task is not None:
            self._recv_tasks.add(task)
        try:
            if self._stopped:
                raise MediaStreamError
            self._consumer_started.set()
            self._ensure_reader()
            pts, self._last_pace_sleep_ms = await self._pace()
            if self._stopped:
                raise MediaStreamError
            item = await self._queue.get()
            if item is _END:
                self._raise_source_end()

            self._queued_frames -= 1
            queued_at, value = item
            self._last_queue_age_ms = max(
                0,
                (asyncio.get_running_loop().time() - queued_at) * 1_000,
            )
            frame = await self._prepare_frame(value)
            if self._stopped:
                raise MediaStreamError
            frame.pts = pts
            frame.time_base = self._time_base
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            self._notify_failure(error)
            raise
        finally:
            if task is not None:
                self._recv_tasks.discard(task)

        self._output_frames += 1
        return frame

    async def aclose(self) -> None:
        deadline = (
            asyncio.get_running_loop().time() + self.policy.shutdown_timeout_ms / 1_000
        )
        self.stop()
        current = asyncio.current_task()
        recv_tasks = [task for task in self._recv_tasks if task is not current]
        for task in recv_tasks:
            task.cancel()
        await self._wait_for_shutdown(*recv_tasks, deadline=deadline)
        reader = self._reader
        if reader is not None:
            await self._wait_for_shutdown(reader, deadline=deadline)
        futures = list(self._worker_futures)
        if futures:
            await self._wait_for_shutdown(*futures, deadline=deadline)
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining > 0:
            close_task = self._source_close_task
            if close_task is None:
                close_task = asyncio.create_task(self._close_source())
                self._source_close_task = close_task
                close_task.add_done_callback(self._source_close_done)
            with suppress(TimeoutError):
                await asyncio.wait_for(
                    asyncio.shield(close_task),
                    timeout=remaining,
                )

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._consumer_started.set()
        if self._reader is not None:
            self._reader.cancel()
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        for task in list(self._recv_tasks):
            if task is not current:
                task.cancel()
        super().stop()

    def _ensure_reader(self) -> None:
        if self._reader is None:
            self._reader = asyncio.create_task(self._read_source())

    async def _read_source(self) -> None:
        iterator = self.source.__aiter__()
        try:
            while not self._stopped:
                await self._wait_for_prefetch_capacity()
                frame = await anext(iterator)
                self._produced_frames += 1
                self._enqueue(frame)
        except StopAsyncIteration:
            pass
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            self._source_error = error
            self._notify_failure(error)
        finally:
            self._queue.put_nowait(_END)

    async def _wait_for_prefetch_capacity(self) -> None:
        if self._consumer_started.is_set():
            return
        if self._queued_frames < self.policy.initial_prefetch_frames:
            return
        await self._consumer_started.wait()

    def _enqueue(self, frame: Any) -> None:
        if self._queue.qsize() < self.policy.max_queue_size:
            self._queue.put_nowait((asyncio.get_running_loop().time(), frame))
            self._queued_frames += 1
            return
        self._dropped_frames += 1
        if self.policy.overflow == "drop_newest":
            return
        self._queue.get_nowait()
        self._queue.put_nowait((asyncio.get_running_loop().time(), frame))

    async def _prepare_frame(self, value: Any) -> Any:
        if hasattr(value, "pts"):
            return value
        if av is None:
            raise RuntimeError("Array frame conversion requires aiortc and PyAV")
        if self.policy.execution == "event_loop":
            return av.VideoFrame.from_ndarray(
                value,
                format=self.policy.output_format,
            )
        future = asyncio.get_running_loop().run_in_executor(
            None,
            av.VideoFrame.from_ndarray,
            value,
            self.policy.output_format,
        )
        self._worker_futures.add(future)
        future.add_done_callback(self._worker_done)
        return await asyncio.shield(future)

    async def _pace(self) -> tuple[int, float]:
        loop = asyncio.get_running_loop()
        now = loop.time()
        sleep_seconds = 0.0
        if self._next_emit_at is None:
            emitted_at = now
        else:
            if now < self._next_emit_at:
                sleep_seconds = self._next_emit_at - now
                await asyncio.sleep(sleep_seconds)
                now = loop.time()
            emitted_at = max(now, self._next_emit_at)
        self._next_emit_at = emitted_at + 1 / self.policy.fps
        pts = round(self._frame_index * self._frame_step)
        self._frame_index += 1
        return pts, sleep_seconds * 1_000

    async def _close_source(self) -> None:
        if self._source_closed:
            return
        self._source_closed = True
        close = getattr(self.source, "aclose", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            with suppress(Exception):
                await result

    async def _wait_for_shutdown(
        self,
        *awaitables: Awaitable[Any],
        deadline: float,
    ) -> None:
        if not awaitables:
            return
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return
        combined = asyncio.gather(*awaitables, return_exceptions=True)
        with suppress(TimeoutError):
            await asyncio.wait_for(
                asyncio.shield(combined),
                timeout=remaining,
            )

    def _worker_done(self, future: asyncio.Future) -> None:
        self._worker_futures.discard(future)
        if not future.cancelled():
            with suppress(Exception):
                future.exception()

    def _source_close_done(self, task: asyncio.Task) -> None:
        if not task.cancelled():
            with suppress(Exception):
                task.exception()

    def _notify_failure(self, error: BaseException) -> None:
        if self._failure is not None:
            return
        self._failure = error
        if self._on_error is not None:
            try:
                self._on_error(error)
            except Exception:
                logger.exception("WMA video source error callback failed")

    def _raise_source_end(self) -> None:
        if self._source_error is not None:
            raise self._source_error
        error = MediaStreamError()
        self._notify_failure(error)
        raise error


class VideoSourcePeer(AiortcPeer):
    """High-level aiortc backend for source-generating world models."""

    def __init__(
        self,
        session: Session,
        source: AsyncIterable[Any],
        *,
        policy: Optional[VideoSourcePolicy] = None,
        configure_peer: Optional[Callable[[Any, VideoSourceTrack], Any]] = None,
        create_default_channel: bool = False,
        rtc_configuration: Any = None,
        peer_connection_factory: Optional[Callable[[], Any]] = None,
        disconnected_grace_seconds: Optional[float] = 0,
    ) -> None:
        self.track = VideoSourceTrack(
            source,
            policy=policy,
            on_error=lambda _error: self._closed.set(),
        )

        async def connect(peer_connection: Any) -> None:
            peer_connection.addTrack(self.track)
            if configure_peer is not None:
                result = configure_peer(peer_connection, self.track)
                if inspect.isawaitable(result):
                    await result

        super().__init__(
            session,
            connect,
            create_default_channel=create_default_channel,
            rtc_configuration=rtc_configuration,
            peer_connection_factory=peer_connection_factory,
            disconnected_grace_seconds=disconnected_grace_seconds,
        )

    async def close(self) -> None:
        try:
            await super().close()
        finally:
            await self.track.aclose()
