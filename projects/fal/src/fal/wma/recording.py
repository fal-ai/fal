"""Bounded video-only recording for request-bound WMA final SSE results."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import queue
import tempfile
import threading
import time
from concurrent.futures import Future
from contextvars import copy_context
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

from fastapi import Request

from fal.compat import run_in_thread
from fal.toolkit import File
from fal.wma._raw import sse_event

logger = logging.getLogger(__name__)
MiB = 1024**2


@dataclass(frozen=True)
class RecordingConfig:
    """Opt-in video recording limits; finalization keeps the request open."""

    segment_seconds: float = 30
    finalization_timeout: float = 45
    max_active_recordings: int = 2
    max_pending_frames: int = 8
    max_frame_bytes: int = 12 * MiB
    max_pending_uploads: int = 2
    max_segment_bytes: int = 64 * MiB
    max_total_bytes: int = 1024 * MiB
    max_segments: int = 240

    def __post_init__(self) -> None:
        for name in ("segment_seconds", "finalization_timeout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "max_active_recordings",
            "max_pending_frames",
            "max_frame_bytes",
            "max_pending_uploads",
            "max_segment_bytes",
            "max_total_bytes",
            "max_segments",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


class RecordingPool:
    """Bound recording resources across sessions, including finalization."""

    def __init__(self) -> None:
        self.active: set[SessionRecording] = set()

    def create(
        self,
        session_id: str | None,
        request_id: str | None,
        request: Request | None,
        config: RecordingConfig,
    ) -> SessionRecording | None:
        if len(self.active) >= config.max_active_recordings:
            return None
        recording = SessionRecording(session_id, request_id, request, config=config)
        recording.on_release = lambda: self.active.discard(recording)
        self.active.add(recording)
        return recording


class SessionRecording:
    """Own immutable frame bytes; encode and upload with bounded backlogs."""

    def __init__(
        self,
        session_id: str | None,
        request_id: str | None,
        request: Request | None,
        *,
        upload: Callable[..., Any] | None = None,
        segment_seconds: float | None = None,
        config: RecordingConfig | None = None,
        fps: float = 30,
    ):
        self.session_id, self.request_id, self.request = session_id, request_id, request
        self.config = config or RecordingConfig()
        self.on_release: Callable[[], None] = lambda: None
        self.track_attached = False
        self.loop = asyncio.get_running_loop()
        self.context = copy_context()
        self.directory = tempfile.TemporaryDirectory(prefix="wma-recording-")
        self.queue: queue.Queue[Any] = queue.Queue(
            maxsize=self.config.max_pending_frames
        )
        self.aborted = threading.Event()
        self.sealed = threading.Event()
        self.encoder_done = threading.Event()
        self.thread: threading.Thread | None = None
        self.upload = upload or self._upload
        self.segment_seconds = (
            self.config.segment_seconds if segment_seconds is None else segment_seconds
        )
        self.fps = fps
        self.uploads: list[Future[Any]] = []
        self.files: list[dict[str, Any]] = []
        self.reason: str | None = None
        self.frames = 0
        self.bytes = 0
        self._upload_tasks: set[asyncio.Task] = set()
        self._released = False

    def seal(self) -> None:
        self.sealed.set()
        if self.thread is None:
            self.encoder_done.set()

    def abort(self) -> None:
        if self.aborted.is_set():
            self._release_if_finished()
            return
        self.aborted.set()
        self.seal()
        for task in tuple(self._upload_tasks):
            task.cancel()
        self._release_if_finished()

    def _release_if_finished(self) -> None:
        if (
            self.aborted.is_set()
            and self.encoder_done.is_set()
            and not self._upload_tasks
            and all(task.done() for task in self.uploads)
            and not self._released
        ):
            self._released = True
            self.directory.cleanup()
            self.on_release()

    def _snapshot(self, frame: Any) -> tuple[bytes, int, int, float]:
        if frame.width * frame.height * 3 // 2 > self.config.max_frame_bytes:
            raise ValueError("frame_limit")
        pixels = frame.to_ndarray(format="yuv420p")
        # Own the bytes before returning the original frame to aiortc.
        data = pixels.tobytes()
        if len(data) > self.config.max_frame_bytes:
            raise ValueError("frame_limit")
        return data, frame.width, frame.height, float(frame.pts * frame.time_base)

    async def offer(self, frame: Any) -> None:
        if self.sealed.is_set():
            return
        try:
            snapshot = await run_in_thread(self._snapshot, frame)
            # Closure/cancellation can happen while the worker copies pixels.
            # Keep queue/thread lifecycle mutations on the event loop.
            if self.sealed.is_set():
                return
            self.queue.put_nowait(snapshot)
            if self.thread is None:
                self.thread = threading.Thread(target=self._encode, daemon=True)
                self.thread.start()
        except Exception:
            self.reason = "recording_backlog_or_frame_limit"
            self.seal()

    async def _upload(self, path: Path, request: Request | None) -> dict[str, Any]:
        if request is None:
            raise ValueError("recording request unavailable")
        # Shield the toolkit's blocking upload: cancellation cannot stop its
        # worker thread, so its source file must survive until that worker exits.
        upload = asyncio.create_task(
            File.from_path_async(
                path,
                content_type="video/mp4",
                request=request,
                multipart=True,
                fallback_repository=None,
            )
        )
        try:
            file = await asyncio.shield(upload)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await asyncio.shield(upload)
            raise

        return {
            "url": file.url,
            "file_name": path.name,
            "content_type": "video/mp4",
            "file_size": path.stat().st_size,
        }

    async def _upload_segment(self, path: Path) -> None:
        task = asyncio.current_task()
        assert task is not None
        self._upload_tasks.add(task)
        try:
            if self.aborted.is_set():
                return
            metadata = await self.upload(path, self.request)
            self.files.append(
                {
                    **metadata,
                    "segment_index": int(path.stem.split("-")[-1]),
                }
            )
        except Exception:
            logger.exception("WMA recording upload failed")
            self.reason = "recording_upload_failed"
            self.seal()
        finally:
            path.unlink(missing_ok=True)
            self._upload_tasks.discard(task)
            self.loop.call_soon(self._release_if_finished)

    def _encode(self) -> None:
        container = None
        stream = None
        segment = 0
        segment_start = 0.0
        shape = None
        path = None

        def finish_segment() -> None:
            nonlocal container, stream
            if container is None:
                return
            assert stream is not None and path is not None
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            container, stream = None, None
            if self.aborted.is_set():
                path.unlink(missing_ok=True)
                return
            self.bytes += path.stat().st_size
            active = sum(not task.done() for task in self.uploads)
            if (
                active >= self.config.max_pending_uploads
                or self.bytes > self.config.max_total_bytes
            ):
                path.unlink(missing_ok=True)
                self.reason = "recording_spool_limit"
                self.seal()
                return
            upload = self.context.run(
                asyncio.run_coroutine_threadsafe, self._upload_segment(path), self.loop
            )
            self.uploads.append(upload)
            upload.add_done_callback(
                lambda _: self.loop.call_soon_threadsafe(self._release_if_finished)
            )

        try:
            import av
            import numpy as np

            while not self.aborted.is_set() and (
                not self.sealed.is_set() or not self.queue.empty()
            ):
                try:
                    data, width, height, stamp = self.queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if container is not None and (
                    stamp - segment_start >= self.segment_seconds
                    or shape != (width, height)
                    or (
                        path is not None
                        and path.stat().st_size >= self.config.max_segment_bytes
                    )
                ):
                    finish_segment()
                if self.reason:
                    break
                if container is None:
                    segment += 1
                    if segment > self.config.max_segments:
                        self.reason = "recording_segment_limit"
                        self.seal()
                        break
                    path = Path(self.directory.name) / f"recording-{segment:05d}.mp4"
                    container = av.open(
                        str(path), "w", format="mp4", options={"movflags": "+faststart"}
                    )
                    stream = container.add_stream(
                        "libx264", rate=Fraction(str(self.fps))
                    )
                    stream.width, stream.height, stream.pix_fmt = (
                        width,
                        height,
                        "yuv420p",
                    )
                    stream.options = {
                        "preset": "ultrafast",
                        "tune": "zerolatency",
                        "crf": "23",
                    }
                    stream.time_base = Fraction(1, 90000)
                    stream.codec_context.time_base = Fraction(1, 90000)
                    stream.codec_context.thread_count = 1
                    stream.codec_context.max_b_frames = 0
                    segment_start, shape = stamp, (width, height)
                frame = av.VideoFrame.from_ndarray(
                    np.frombuffer(data, dtype=np.uint8).reshape(height * 3 // 2, width),
                    format="yuv420p",
                )
                frame.pts, frame.time_base = (
                    round((stamp - segment_start) * 90000),
                    Fraction(1, 90000),
                )
                assert stream is not None
                for packet in stream.encode(frame):
                    container.mux(packet)
                self.frames += 1
            finish_segment()
        except Exception:
            logger.exception("WMA recording encode failed")
            self.reason = "recording_encode_failed"
            self.seal()
        finally:
            if container is not None:
                with contextlib.suppress(Exception):
                    container.close()
            while not self.queue.empty():
                with contextlib.suppress(queue.Empty):
                    self.queue.get_nowait()
            self.encoder_done.set()
            if self.aborted.is_set():
                self.loop.call_soon_threadsafe(self._release_if_finished)

    async def result_events(self, timeout: float | None = None):
        self.seal()
        deadline = time.monotonic() + (
            self.config.finalization_timeout if timeout is None else timeout
        )
        try:
            while not self.encoder_done.is_set() or any(
                not task.done() for task in self.uploads
            ):
                if time.monotonic() >= deadline:
                    self.reason = "recording_upload_timeout"
                    break
                yield ": finishing recording\n\n"
                await asyncio.sleep(0.1)
            files = sorted(self.files, key=lambda file: file["segment_index"])
            status = (
                "partial"
                if files and self.reason
                else "complete"
                if files
                else "failed"
                if self.reason
                else "empty"
            )
            output = {
                "session_id": self.session_id,
                "request_id": self.request_id,
                "recording_status": status,
                "publication_transport": "final_sse",
                "video": files,
            }
            if self.reason:
                output["error"] = self.reason
            yield sse_event(output, event="result")
        finally:
            self.abort()

    def wrap_track(self, track: Any, *, fps: float) -> Any:
        """Record one selected outgoing track without changing its delivered frames."""
        if self.track_attached:
            raise ValueError("only one video track can be recorded per session")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("recording fps must be finite and positive")
        from aiortc import VideoStreamTrack

        self.fps = fps
        self.track_attached = True
        recording = self

        class RecordingTrack(VideoStreamTrack):
            async def recv(self):
                frame = await track.recv()
                await recording.offer(frame)
                return frame

            def stop(self):
                recording.seal()
                track.stop()
                super().stop()

        return RecordingTrack()
