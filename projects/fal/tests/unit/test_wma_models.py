from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import deque
from fractions import Fraction

import pytest

pytest.importorskip("aiortc")

from aiortc import (  # noqa: E402
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiortc.mediastreams import MediaStreamError  # noqa: E402
from av import VideoFrame  # noqa: E402

import fal.wma as wma  # noqa: E402
from fal.wma import Session, StartSessionRequest  # noqa: E402
from fal.wma_models import (  # noqa: E402
    VideoProcessorBinding,
    VideoProcessorPeer,
    VideoProcessorPolicy,
    VideoProcessorTrack,
    VideoSourcePeer,
    VideoSourcePolicy,
    VideoSourceTrack,
    attach_video_processor,
)


class FakeFrame:
    def __init__(self, pts, time_base=Fraction(1, 90_000)):
        self.source_id = pts
        self.pts = pts
        self.time_base = time_base


class FakeSource:
    kind = "video"

    def __init__(self, frames):
        self.frames = deque(frames)
        self.stopped = False

    async def recv(self):
        await asyncio.sleep(0)
        if not self.frames:
            raise MediaStreamError
        return self.frames.popleft()

    def stop(self):
        self.stopped = True


class FakePeerConnection:
    def __init__(self):
        self.handlers = {}
        self.added_tracks = []

    def on(self, event):
        def register(handler):
            self.handlers[event] = handler
            return handler

        return register

    def emit(self, event, value):
        self.handlers[event](value)

    def addTrack(self, track):  # noqa: N802
        self.added_tracks.append(track)
        return track


def test_video_processor_policy_validation():
    with pytest.raises(ValueError, match="batch_size"):
        VideoProcessorPolicy(batch_size=0)
    with pytest.raises(ValueError, match="max_queue_size"):
        VideoProcessorPolicy(max_queue_size=0)
    with pytest.raises(ValueError, match="max_batch_wait_ms"):
        VideoProcessorPolicy(max_batch_wait_ms=-1)
    with pytest.raises(ValueError, match="overflow"):
        VideoProcessorPolicy(overflow="block")
    with pytest.raises(ValueError, match="execution"):
        VideoProcessorPolicy(execution="process")
    with pytest.raises(ValueError, match="max_output_frames"):
        VideoProcessorPolicy(max_output_frames=0)
    with pytest.raises(ValueError, match="processor_timeout_ms"):
        VideoProcessorPolicy(processor_timeout_ms=0)
    with pytest.raises(ValueError, match="shutdown_timeout_ms"):
        VideoProcessorPolicy(shutdown_timeout_ms=0)


def test_model_adapter_is_available_from_wma_namespace():
    assert wma.VideoProcessorPeer is VideoProcessorPeer
    assert wma.VideoProcessorPolicy is VideoProcessorPolicy
    assert wma.attach_video_processor is attach_video_processor
    assert "VideoProcessorPeer" in dir(wma)
    assert wma.VideoSourcePeer is VideoSourcePeer
    assert wma.VideoSourcePolicy is VideoSourcePolicy


def test_video_source_policy_validation():
    with pytest.raises(ValueError, match="fps"):
        VideoSourcePolicy(fps=0)
    with pytest.raises(ValueError, match="max_queue_size"):
        VideoSourcePolicy(max_queue_size=0)
    with pytest.raises(ValueError, match="overflow"):
        VideoSourcePolicy(overflow="block")
    with pytest.raises(ValueError, match="execution"):
        VideoSourcePolicy(execution="process")
    with pytest.raises(ValueError, match="shutdown_timeout_ms"):
        VideoSourcePolicy(shutdown_timeout_ms=0)


def test_video_source_track_paces_and_assigns_timestamps():
    async def frames():
        for _ in range(3):
            yield FakeFrame(None, time_base=None)

    async def scenario():
        track = VideoSourceTrack(
            frames(),
            policy=VideoSourcePolicy(
                fps=1_000,
                max_queue_size=3,
            ),
        )
        outputs = [await track.recv() for _ in range(3)]
        with pytest.raises(MediaStreamError):
            await track.recv()
        await track.aclose()

        assert [frame.pts for frame in outputs] == [0, 90, 180]
        assert all(frame.time_base == Fraction(1, 90_000) for frame in outputs)
        assert track.stats.produced_frames == 3
        assert track.stats.output_frames == 3

    asyncio.run(scenario())


def test_video_source_track_starts_source_before_first_recv():
    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()

        async def frames():
            started.set()
            await release.wait()
            yield FakeFrame(1)

        track = VideoSourceTrack(frames())
        await asyncio.wait_for(started.wait(), timeout=1)
        assert track.stats.output_frames == 0

        release.set()
        await track.recv()
        await track.aclose()

    asyncio.run(scenario())


def test_video_source_track_limits_prefetch_before_first_consumer():
    async def scenario():
        requested = 0

        async def frames():
            nonlocal requested
            for index in range(10):
                requested += 1
                yield FakeFrame(index)

        track = VideoSourceTrack(
            frames(),
            policy=VideoSourcePolicy(
                fps=1_000,
                max_queue_size=4,
                initial_prefetch_frames=1,
            ),
        )
        await asyncio.sleep(0.01)

        assert requested == 1
        assert track.stats.produced_frames == 1
        assert track.stats.queue_depth == 1

        output = await track.recv()
        assert output.source_id == 0
        await track.aclose()

    asyncio.run(scenario())


def test_video_source_shutdown_timeout_bounds_source_close():
    class StuckSource:
        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.Future()

        async def aclose(self):
            await asyncio.Future()

    async def scenario():
        track = VideoSourceTrack(
            StuckSource(),
            policy=VideoSourcePolicy(
                start_immediately=False,
                shutdown_timeout_ms=10,
            ),
        )
        await asyncio.wait_for(track.aclose(), timeout=0.05)

    asyncio.run(scenario())


def test_video_source_shutdown_does_not_wait_for_resistant_source_close():
    async def scenario():
        release = asyncio.Event()

        class ResistantCloseSource:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

            async def aclose(self):
                await release.wait()

        track = VideoSourceTrack(
            ResistantCloseSource(),
            policy=VideoSourcePolicy(
                start_immediately=False,
                shutdown_timeout_ms=10,
            ),
        )
        await asyncio.wait_for(track.aclose(), timeout=0.05)

        assert track._source_close_task is not None
        assert not track._source_close_task.done()
        release.set()
        await asyncio.wait_for(track._source_close_task, timeout=0.05)

    asyncio.run(scenario())


def test_video_source_shutdown_is_bounded_when_reader_resists_cancellation():
    async def scenario():
        release = asyncio.Event()

        class ResistantSource:
            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    await release.wait()
                    raise StopAsyncIteration

        track = VideoSourceTrack(
            ResistantSource(),
            policy=VideoSourcePolicy(shutdown_timeout_ms=10),
        )
        await asyncio.sleep(0)
        await asyncio.wait_for(track.aclose(), timeout=0.05)
        assert track._reader is not None
        assert not track._reader.done()

        release.set()
        await asyncio.wait_for(track._reader, timeout=0.05)

    asyncio.run(scenario())


def test_video_source_close_cancels_recv_during_pacing():
    async def frames():
        for index in range(3):
            yield FakeFrame(index)

    async def scenario():
        track = VideoSourceTrack(
            frames(),
            policy=VideoSourcePolicy(
                fps=5,
                max_queue_size=3,
            ),
        )
        await track.recv()
        recv_task = asyncio.create_task(track.recv())
        await asyncio.sleep(0.01)
        assert not recv_task.done()

        await track.aclose()

        assert recv_task.cancelled()

    asyncio.run(scenario())


def test_video_source_track_drops_oldest_generated_frames():
    async def frames():
        for index in range(5):
            yield FakeFrame(index)

    async def scenario():
        track = VideoSourceTrack(
            frames(),
            policy=VideoSourcePolicy(
                fps=1_000,
                max_queue_size=2,
                overflow="drop_oldest",
            ),
        )
        track._consumer_started.set()
        track._ensure_reader()
        await asyncio.sleep(0.01)

        outputs = [await track.recv() for _ in range(2)]
        with pytest.raises(MediaStreamError):
            await track.recv()
        await track.aclose()

        assert track.stats.dropped_frames == 3
        assert [frame.source_id for frame in outputs] == [3, 4]

    asyncio.run(scenario())


def test_video_source_track_can_drop_newest_generated_frames():
    async def frames():
        for index in range(5):
            yield FakeFrame(index)

    async def scenario():
        track = VideoSourceTrack(
            frames(),
            policy=VideoSourcePolicy(
                fps=1_000,
                max_queue_size=2,
                overflow="drop_newest",
            ),
        )
        track._consumer_started.set()
        track._ensure_reader()
        await asyncio.sleep(0.01)

        outputs = [await track.recv() for _ in range(2)]
        with pytest.raises(MediaStreamError):
            await track.recv()
        await track.aclose()

        assert track.stats.dropped_frames == 3
        assert [frame.source_id for frame in outputs] == [0, 1]

    asyncio.run(scenario())


def test_video_source_track_converts_numpy_arrays():
    numpy = pytest.importorskip("numpy")

    async def frames():
        yield numpy.zeros((2, 3, 3), dtype=numpy.uint8)

    async def scenario():
        track = VideoSourceTrack(
            frames(),
            policy=VideoSourcePolicy(
                fps=1_000,
                output_format="rgb24",
            ),
        )
        frame = await track.recv()
        await track.aclose()

        assert isinstance(frame, VideoFrame)
        assert frame.width == 3
        assert frame.height == 2
        assert frame.pts == 0
        assert frame.time_base == Fraction(1, 90_000)

    asyncio.run(scenario())


def test_video_source_peer_failure_ends_session():
    async def frames():
        yield FakeFrame(1)
        raise RuntimeError("generation failed")

    async def scenario():
        session = Session(StartSessionRequest(sdp="offer"))
        peer = VideoSourcePeer(
            session,
            frames(),
            policy=VideoSourcePolicy(fps=1_000, max_queue_size=2),
        )
        peer_connection = FakePeerConnection()
        await peer._on_connect(peer_connection)

        track = peer_connection.added_tracks[0]
        await track.recv()
        await asyncio.wait_for(peer.wait_closed(), timeout=1)
        assert track._failure is not None
        await peer.close()

    asyncio.run(scenario())


def test_video_source_peer_configures_peer_and_closes_source():
    closed = asyncio.Event()

    async def frames():
        try:
            while True:
                yield FakeFrame(1)
                await asyncio.sleep(0)
        finally:
            closed.set()

    async def scenario():
        configured = []
        session = Session(StartSessionRequest(sdp="offer"))

        async def configure(peer_connection, track):
            configured.append((peer_connection, track))

        peer = VideoSourcePeer(
            session,
            frames(),
            configure_peer=configure,
        )
        peer_connection = FakePeerConnection()
        await peer._on_connect(peer_connection)
        track = peer_connection.added_tracks[0]
        track._ensure_reader()
        await asyncio.sleep(0)
        await peer.close()

        assert configured == [(peer_connection, track)]
        assert closed.is_set()
        assert not peer._create_default_channel

    asyncio.run(scenario())


def test_processor_batches_frames_off_loop_and_preserves_timestamps():
    async def scenario():
        inputs = [FakeFrame(pts) for pts in (0, 3_000, 6_000, 9_000)]
        source = FakeSource(inputs)
        batches = []
        worker_threads = []
        loop_thread = threading.get_ident()

        def process(frames):
            batches.append([frame.pts for frame in frames])
            worker_threads.append(threading.get_ident())
            return [FakeFrame(None) for _ in frames]

        track = VideoProcessorTrack(
            source,
            process,
            policy=VideoProcessorPolicy(
                batch_size=2,
                max_queue_size=8,
                max_batch_wait_ms=50,
            ),
        )
        outputs = [await track.recv() for _ in range(4)]
        await track.aclose()

        assert batches == [[0, 3_000], [6_000, 9_000]]
        assert [frame.pts for frame in outputs] == [0, 3_000, 6_000, 9_000]
        assert worker_threads
        assert all(thread != loop_thread for thread in worker_threads)
        assert track.stats.input_frames == 4
        assert track.stats.output_frames == 4
        assert track.stats.batches == 2
        assert source.stopped

    asyncio.run(scenario())


def test_processor_supports_async_event_loop_execution():
    async def scenario():
        loop_thread = threading.get_ident()
        processor_threads = []

        async def process(frames):
            processor_threads.append(threading.get_ident())
            return frames

        track = VideoProcessorTrack(
            FakeSource([FakeFrame(1)]),
            process,
            policy=VideoProcessorPolicy(execution="event_loop"),
        )
        frame = await track.recv()
        await track.aclose()

        assert frame.pts == 1
        assert processor_threads == [loop_thread]

    asyncio.run(scenario())


def test_processor_converts_numpy_arrays_and_copies_timing():
    numpy = pytest.importorskip("numpy")

    async def scenario():
        source_frame = FakeFrame(12_000)
        track = VideoProcessorTrack(
            FakeSource([source_frame]),
            lambda _frames: numpy.zeros((2, 3, 3), dtype=numpy.uint8),
            policy=VideoProcessorPolicy(output_format="rgb24"),
        )
        frame = await track.recv()
        await track.aclose()

        assert isinstance(frame, VideoFrame)
        assert frame.width == 3
        assert frame.height == 2
        assert frame.pts == source_frame.pts
        assert frame.time_base == source_frame.time_base

    asyncio.run(scenario())


def test_processor_fills_time_base_independently_from_pts():
    async def scenario():
        output = FakeFrame(42, time_base=None)
        source = FakeFrame(12_000)
        track = VideoProcessorTrack(
            FakeSource([source]),
            lambda _frames: output,
        )
        frame = await track.recv()
        await track.aclose()

        assert frame.pts == 42
        assert frame.time_base == source.time_base

    asyncio.run(scenario())


def test_processor_rejects_missing_input_timing():
    async def scenario():
        track = VideoProcessorTrack(
            FakeSource([FakeFrame(None, time_base=None)]),
            lambda _frames: FakeFrame(None, time_base=None),
        )
        with pytest.raises(ValueError, match="missing pts"):
            await track.recv()
        await track.aclose()

    asyncio.run(scenario())


def test_processor_rejects_excess_output_without_materializing_everything():
    generated = 0

    def process(_frames):
        def outputs():
            nonlocal generated
            while True:
                generated += 1
                yield FakeFrame(generated)

        return outputs()

    async def scenario():
        track = VideoProcessorTrack(
            FakeSource([FakeFrame(1)]),
            process,
            policy=VideoProcessorPolicy(max_output_frames=2),
        )
        with pytest.raises(ValueError, match="more than 2"):
            await track.recv()
        await track.aclose()

    asyncio.run(scenario())
    assert generated == 3


def test_bounded_queue_drops_oldest_frames():
    async def scenario():
        track = VideoProcessorTrack(
            FakeSource([FakeFrame(index) for index in range(5)]),
            lambda frames: frames,
            policy=VideoProcessorPolicy(
                max_queue_size=2,
                overflow="drop_oldest",
            ),
        )
        track._ensure_reader()
        await asyncio.sleep(0.01)

        first = await track.recv()
        second = await track.recv()
        await track.aclose()

        assert [first.pts, second.pts] == [3, 4]
        assert track.stats.dropped_frames == 3

    asyncio.run(scenario())


def test_bounded_queue_can_drop_newest_frames():
    async def scenario():
        track = VideoProcessorTrack(
            FakeSource([FakeFrame(index) for index in range(5)]),
            lambda frames: frames,
            policy=VideoProcessorPolicy(
                max_queue_size=2,
                overflow="drop_newest",
            ),
        )
        track._ensure_reader()
        await asyncio.sleep(0.01)

        first = await track.recv()
        second = await track.recv()
        with pytest.raises(MediaStreamError):
            await track.recv()
        await track.aclose()

        assert [first.pts, second.pts] == [0, 1]
        assert track.stats.dropped_frames == 3

    asyncio.run(scenario())


def test_close_drains_worker_left_by_cancelled_recv():
    async def scenario():
        started = threading.Event()
        release = threading.Event()

        def process(frames):
            started.set()
            release.wait(timeout=1)
            return frames

        track = VideoProcessorTrack(
            FakeSource([FakeFrame(1)]),
            process,
        )
        recv_task = asyncio.create_task(track.recv())
        await asyncio.to_thread(started.wait, 1)
        recv_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await recv_task

        close_task = asyncio.create_task(track.aclose())
        await asyncio.sleep(0)
        assert not close_task.done()

        release.set()
        await close_task

    asyncio.run(scenario())


def test_close_cancels_and_drains_async_processor():
    async def scenario():
        started = asyncio.Event()
        finished = asyncio.Event()

        async def process(_frames):
            started.set()
            try:
                await asyncio.Future()
            finally:
                finished.set()

        track = VideoProcessorTrack(
            FakeSource([FakeFrame(1)]),
            process,
            policy=VideoProcessorPolicy(
                execution="event_loop",
                processor_timeout_ms=None,
            ),
        )
        recv_task = asyncio.create_task(track.recv())
        await started.wait()
        await track.aclose()

        assert finished.is_set()
        assert recv_task.done()
        with pytest.raises(asyncio.CancelledError):
            await recv_task

    asyncio.run(scenario())


def test_close_cancels_partial_batch_before_processing():
    class PartialSource:
        kind = "video"

        def __init__(self):
            self.sent = False
            self.stopped = False

        async def recv(self):
            if not self.sent:
                self.sent = True
                return FakeFrame(1)
            await asyncio.Future()

        def stop(self):
            self.stopped = True

    async def scenario():
        processed = False

        def process(frames):
            nonlocal processed
            processed = True
            return frames

        source = PartialSource()
        track = VideoProcessorTrack(
            source,
            process,
            policy=VideoProcessorPolicy(
                batch_size=2,
                max_batch_wait_ms=10_000,
            ),
        )
        recv_task = asyncio.create_task(track.recv())
        while track.stats.input_frames < 1:
            await asyncio.sleep(0)
        await track.aclose()

        assert recv_task.done()
        assert not processed
        assert source.stopped

    asyncio.run(scenario())


def test_processor_timeout_bounds_recv_and_shutdown():
    async def scenario():
        release = threading.Event()

        def process(frames):
            release.wait(timeout=1)
            return frames

        track = VideoProcessorTrack(
            FakeSource([FakeFrame(1)]),
            process,
            policy=VideoProcessorPolicy(
                processor_timeout_ms=10,
                shutdown_timeout_ms=10,
            ),
        )
        try:
            with pytest.raises(TimeoutError, match="exceeded 10 ms"):
                await track.recv()
            await asyncio.wait_for(track.aclose(), timeout=0.1)
        finally:
            release.set()

    asyncio.run(scenario())


def test_shutdown_timeout_is_one_end_to_end_deadline():
    async def scenario():
        started = threading.Event()
        release = threading.Event()

        def process(frames):
            started.set()
            release.wait(timeout=1)
            return frames

        track = VideoProcessorTrack(
            FakeSource([FakeFrame(1)]),
            process,
            policy=VideoProcessorPolicy(
                processor_timeout_ms=None,
                shutdown_timeout_ms=50,
            ),
        )
        recv_task = asyncio.create_task(track.recv())
        await asyncio.to_thread(started.wait, 1)
        started_at = asyncio.get_running_loop().time()
        try:
            await track.aclose()
            elapsed = asyncio.get_running_loop().time() - started_at
            assert elapsed < 0.085
            assert recv_task.cancelled()
        finally:
            release.set()

    asyncio.run(scenario())


def test_async_worker_processor_uses_one_end_to_end_deadline():
    async def scenario():
        async def process(_frames):
            await asyncio.sleep(0.03)

            def outputs():
                time.sleep(0.03)
                yield FakeFrame(1)

            return outputs()

        track = VideoProcessorTrack(
            FakeSource([FakeFrame(1)]),
            process,
            policy=VideoProcessorPolicy(
                processor_timeout_ms=45,
                shutdown_timeout_ms=50,
            ),
        )
        with pytest.raises(TimeoutError, match="exceeded 45 ms"):
            await track.recv()
        await track.aclose()

    asyncio.run(scenario())


def test_lazy_output_iteration_runs_in_worker_and_obeys_timeout():
    async def scenario():
        release = threading.Event()
        iterator_threads = []

        def process(_frames):
            def outputs():
                iterator_threads.append(threading.get_ident())
                release.wait(timeout=1)
                yield FakeFrame(1)

            return outputs()

        loop_thread = threading.get_ident()
        track = VideoProcessorTrack(
            FakeSource([FakeFrame(1)]),
            process,
            policy=VideoProcessorPolicy(
                processor_timeout_ms=10,
                shutdown_timeout_ms=10,
            ),
        )
        try:
            with pytest.raises(TimeoutError, match="exceeded 10 ms"):
                await track.recv()
            assert iterator_threads
            assert iterator_threads[0] != loop_thread
            await asyncio.wait_for(track.aclose(), timeout=0.1)
        finally:
            release.set()

    asyncio.run(scenario())


def test_async_processor_timeout_cancels_processor():
    async def scenario():
        cancelled = asyncio.Event()

        async def process(_frames):
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        track = VideoProcessorTrack(
            FakeSource([FakeFrame(1)]),
            process,
            policy=VideoProcessorPolicy(
                execution="event_loop",
                processor_timeout_ms=10,
            ),
        )
        with pytest.raises(TimeoutError, match="exceeded 10 ms"):
            await track.recv()
        assert cancelled.is_set()
        await track.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure_kind", ["sync", "async", "conversion"])
def test_processor_failure_closes_peer(failure_kind):
    async def scenario():
        session = Session(StartSessionRequest(sdp="offer"))
        peer_connection = FakePeerConnection()

        if failure_kind == "sync":

            def processor(_frames):
                raise RuntimeError("sync failure")

            policy = VideoProcessorPolicy()
        elif failure_kind == "async":

            async def processor(_frames):
                raise RuntimeError("async failure")

            policy = VideoProcessorPolicy(execution="event_loop")
        else:

            def processor(_frames):
                return object()

            policy = VideoProcessorPolicy()

        peer = VideoProcessorPeer(session, processor, policy=policy)
        await peer._on_connect(peer_connection)
        peer_connection.emit("track", FakeSource([FakeFrame(1)]))
        processed = peer_connection.added_tracks[0]

        with pytest.raises(Exception):
            await processed.recv()
        await asyncio.wait_for(peer.wait_closed(), timeout=1)
        assert peer.binding is not None
        assert peer.binding.error is not None
        await peer.close()

    asyncio.run(scenario())


def test_binding_attaches_video_and_optional_audio():
    async def scenario():
        peer_connection = FakePeerConnection()
        binding = attach_video_processor(
            peer_connection,
            lambda frames: frames,
            passthrough_audio=True,
        )
        video = FakeSource([FakeFrame(1)])
        audio = FakeSource([])
        audio.kind = "audio"

        peer_connection.emit("track", video)
        peer_connection.emit("track", audio)

        assert isinstance(binding, VideoProcessorBinding)
        assert isinstance(peer_connection.added_tracks[0], VideoProcessorTrack)
        assert peer_connection.added_tracks[1] is audio
        await binding.close()
        assert video.stopped

    asyncio.run(scenario())


def test_peer_closes_binding_when_peer_connection_close_fails():
    class FailingClosePeer:
        async def close(self):
            raise RuntimeError("close failed")

    async def scenario():
        session = Session(StartSessionRequest(sdp="offer"))
        peer_connection = FakePeerConnection()
        peer = VideoProcessorPeer(session, lambda frames: frames)
        await peer._on_connect(peer_connection)
        video = FakeSource([FakeFrame(1)])
        peer_connection.emit("track", video)
        peer._pc = FailingClosePeer()

        with pytest.raises(RuntimeError, match="close failed"):
            await peer.close()
        assert video.stopped

    asyncio.run(scenario())


def test_video_processor_peer_registers_binding_with_session_cleanup():
    async def scenario():
        session = Session(StartSessionRequest(sdp="offer"))
        peer_connection = FakePeerConnection()
        peer = VideoProcessorPeer(
            session,
            lambda frames: frames,
        )
        assert not peer._create_default_channel
        session.bind_backend(peer)

        await peer._on_connect(peer_connection)
        assert isinstance(peer.binding, VideoProcessorBinding)

        video = FakeSource([FakeFrame(1)])
        peer_connection.emit("track", video)
        await session.close()
        assert video.stopped

    asyncio.run(scenario())


def test_video_processor_peer_negotiates_real_aiortc_video():
    class SourceTrack(VideoStreamTrack):
        async def recv(self):
            pts, time_base = await self.next_timestamp()
            frame = VideoFrame(width=4, height=2, format="rgb24")
            frame.planes[0].update(bytes(frame.planes[0].buffer_size))
            frame.pts = pts
            frame.time_base = time_base
            return frame

    async def scenario():
        client = RTCPeerConnection()
        client.addTrack(SourceTrack())
        received = asyncio.Event()

        @client.on("track")
        def on_track(track):
            async def receive():
                frame = await track.recv()
                if frame.width == 4 and frame.height == 2:
                    received.set()

            asyncio.create_task(receive())

        offer = await client.createOffer()
        await client.setLocalDescription(offer)
        request = StartSessionRequest(
            sdp=client.localDescription.sdp,
            type=client.localDescription.type,
            session_id="model-loopback",
        )
        session = Session(request)
        backend = VideoProcessorPeer(
            session,
            lambda frames: frames,
        )
        session.bind_backend(backend)

        try:
            answer = await backend.negotiate(request)
            await client.setRemoteDescription(
                RTCSessionDescription(sdp=answer.sdp, type=answer.type)
            )
            await asyncio.wait_for(received.wait(), timeout=5)
        finally:
            await session.close()
            await client.close()

    asyncio.run(scenario())


def test_video_source_peer_negotiates_real_aiortc_video():
    async def frames():
        while True:
            frame = VideoFrame(width=4, height=2, format="rgb24")
            frame.planes[0].update(bytes(frame.planes[0].buffer_size))
            yield frame
            await asyncio.sleep(0)

    async def scenario():
        client = RTCPeerConnection()
        client.addTransceiver("video", direction="recvonly")
        channel = client.createDataChannel("control")
        received = asyncio.Event()
        channel_open = asyncio.Event()
        session_info = asyncio.Event()
        controls_received = asyncio.Event()

        @channel.on("open")
        def on_open():
            channel_open.set()

        @channel.on("message")
        def on_message(raw):
            if json.loads(raw).get("type") == "session_info":
                session_info.set()

        @client.on("track")
        def on_track(track):
            async def receive():
                frame = await track.recv()
                if frame.width == 4 and frame.height == 2:
                    received.set()

            asyncio.create_task(receive())

        offer = await client.createOffer()
        await client.setLocalDescription(offer)
        request = StartSessionRequest(
            sdp=client.localDescription.sdp,
            type=client.localDescription.type,
            session_id="source-loopback",
        )
        session = Session(request)
        session.on_channel_open(
            lambda: session.send({"type": "session_info", "fps": 60})
        )
        session.on_message(
            "controls",
            lambda _message: controls_received.set(),
        )
        backend = VideoSourcePeer(
            session,
            frames(),
            policy=VideoSourcePolicy(fps=60),
        )
        session.bind_backend(backend)

        try:
            answer = await backend.negotiate(request)
            await client.setRemoteDescription(
                RTCSessionDescription(sdp=answer.sdp, type=answer.type)
            )
            await asyncio.wait_for(received.wait(), timeout=5)
            await asyncio.wait_for(channel_open.wait(), timeout=5)
            await asyncio.wait_for(session_info.wait(), timeout=5)
            channel.send(json.dumps({"type": "controls", "state": {"W": True}}))
            await asyncio.wait_for(controls_received.wait(), timeout=5)
        finally:
            await session.close()
            await client.close()

    asyncio.run(scenario())
