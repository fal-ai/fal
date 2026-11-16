import asyncio
import json
import time
from fractions import Fraction
from pathlib import Path

import pytest
from fastapi import Request

from fal.wma.recording import SessionRecording

av = pytest.importorskip("av")
np = pytest.importorskip("numpy")


@pytest.mark.asyncio
@pytest.mark.parametrize("fps", [16, 48])
async def test_recording_encodes_delivered_frames_and_uploads_before_close(
    tmp_path, fps
):
    uploaded = []

    async def upload(path, request):
        target = tmp_path / path.name
        target.write_bytes(path.read_bytes())
        uploaded.append(target)
        return {
            "url": f"https://v3.fal.media/{path.name}",
            "file_size": target.stat().st_size,
        }

    request = Request({"type": "http", "headers": []})
    rec = SessionRecording(
        "session-test",
        "request-test",
        request,
        upload=upload,
        fps=fps,
        segment_seconds=2 / fps,
    )
    for index in range(8):
        frame = av.VideoFrame.from_ndarray(
            np.full((48, 64, 3), index * 20, dtype=np.uint8), format="rgb24"
        )
        frame.pts, frame.time_base = round(index * 90000 / fps), Fraction(1, 90000)
        await rec.offer(frame)
        await asyncio.sleep(0.02)
    deadline = time.monotonic() + 5
    while not uploaded and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    assert uploaded and not rec.sealed.is_set()
    rec.seal()
    events = [event async for event in rec.result_events(timeout=5)]
    result = json.loads(events[-1].split("data: ")[1])
    assert result["recording_status"] == "complete"
    assert result["session_id"] == "session-test"
    assert len(result["video"]) >= 2
    frames = []
    duration = 0
    for path in uploaded:
        with av.open(str(path)) as container:
            frames.extend(container.decode(video=0))
            duration += container.duration / av.time_base
    assert duration == pytest.approx(8 / fps, abs=0.01)
    assert len(frames) == 8
    assert all(frame.width == 64 and frame.height == 48 for frame in frames)


@pytest.mark.asyncio
async def test_empty_recording_never_uploads():
    async def upload(*args):
        raise AssertionError("empty recording uploaded")

    rec = SessionRecording("session-test", None, None, upload=upload)
    rec.seal()
    events = [event async for event in rec.result_events(timeout=5)]
    assert '"recording_status": "empty"' in events[-1]


@pytest.mark.asyncio
async def test_upload_failure_is_safe_and_does_not_raise_into_stream():
    async def upload(*args):
        raise RuntimeError("private internal detail")

    rec = SessionRecording("session-test", None, None, upload=upload)
    frame = av.VideoFrame.from_ndarray(
        np.zeros((48, 64, 3), dtype=np.uint8), format="rgb24"
    )
    frame.pts, frame.time_base = 0, Fraction(1, 90000)
    await rec.offer(frame)
    rec.seal()
    events = [event async for event in rec.result_events(timeout=5)]
    assert '"recording_status": "failed"' in events[-1]
    assert "private internal detail" not in events[-1]


class Backend:
    def __init__(self):
        self.closed = asyncio.Event()
        self.close_calls = 0

    async def negotiate(self, offer):
        from fal.wma import SessionAnswer

        return SessionAnswer(sdp="transport-only")

    async def wait_closed(self):
        await self.closed.wait()

    async def close(self):
        self.close_calls += 1
        self.closed.set()


def app_with_recording(config):
    from fal.wma import App

    class TestApp(App):
        recording = config
        secrets = []

        async def create_backend(self, session):
            self.sessions.append(session)
            backend = Backend()
            self.backends.append(backend)
            return backend

    app = TestApp(_allow_init=True)
    app.sessions, app.backends = [], []
    return app


async def start(app):
    from fal.wma import StartSessionRequest

    return await app.start_session(
        StartSessionRequest(session_id="test-session", sdp="offer", type="offer"),
        "caller",
        None,
        None,
    )


@pytest.mark.asyncio
async def test_final_sse_preserves_signaling_and_publishes_safe_terminal_output():
    from fal.wma import RecordingConfig

    app = app_with_recording(RecordingConfig())
    response = await start(app)
    app.backends[0].closed.set()
    chunks = [chunk async for chunk in response.body_iterator]
    assert chunks[0].startswith("data: {")
    assert '"sdp": "transport-only"' in chunks[0]
    assert '"recording_status": "uploading"' in chunks[0]
    assert '"recording_status": "empty"' in chunks[-1]
    assert "transport-only" not in chunks[-1]
    assert response.headers["x-fal-sse-last-event-as-body"] == "true"
    assert app.backends[0].close_calls == 1
    assert not app._recording_pool.active


@pytest.mark.asyncio
async def test_disconnect_aborts_recording_and_closes_session():
    from fal.wma import RecordingConfig

    app = app_with_recording(RecordingConfig())
    response = await start(app)
    rec = app.sessions[0]._recording
    await anext(response.body_iterator)
    await response.body_iterator.aclose()
    assert rec.aborted.is_set()
    assert not Path(rec.directory.name).exists()
    assert not app._recording_pool.active
    assert app.backends[0].close_calls == 1


@pytest.mark.asyncio
async def test_disabled_recording_preserves_original_track_and_signaling_only():
    app = app_with_recording(None)
    response = await start(app)
    track = object()
    assert app.sessions[0].record_video(track, fps=16) is track
    app.backends[0].closed.set()
    chunks = [chunk async for chunk in response.body_iterator]
    assert len(chunks) == 1
    assert "recording_status" not in chunks[0]
    assert "x-fal-sse-last-event-as-body" not in response.headers
    assert not hasattr(app, "_recording_pool")


@pytest.mark.asyncio
async def test_capacity_is_per_recording_not_client_session_id_and_recovers():
    from fal.wma import RecordingConfig

    app = app_with_recording(RecordingConfig(max_active_recordings=1))
    first = await start(app)
    second = await start(app)
    assert app.sessions[0].id == app.sessions[1].id
    assert app.sessions[0]._recording is not None
    assert app.sessions[1]._recording is None
    app.backends[1].closed.set()
    chunks = [chunk async for chunk in second.body_iterator]
    assert '"error": "recording_capacity"' in chunks[-1]
    assert '"recording_status": "uploading"' not in "".join(chunks)
    assert len(app._recording_pool.active) == 1
    app.backends[0].closed.set()
    _ = [chunk async for chunk in first.body_iterator]
    third = await start(app)
    assert app.sessions[2]._recording is not None
    await third.background()
    assert not app._recording_pool.active


@pytest.mark.asyncio
async def test_failed_negotiation_releases_recording(monkeypatch):
    from fal.wma import ClientOfferError, RecordingConfig
    from fal.wma._errors import InputValueError

    async def fail(*args):
        raise ClientOfferError("invalid offer")

    monkeypatch.setattr(Backend, "negotiate", fail)
    app = app_with_recording(RecordingConfig())
    with pytest.raises(InputValueError):
        await start(app)
    assert app.backends[0].close_calls == 1
    assert not app._recording_pool.active
    assert app.sessions[0]._recording.aborted.is_set()


@pytest.mark.asyncio
async def test_stream_watchdog_releases_recording_without_iteration(monkeypatch):
    from fal.wma import RecordingConfig, sdk

    monkeypatch.setattr(sdk, "STREAM_START_TIMEOUT_SECONDS", 0.01)
    app = app_with_recording(RecordingConfig())
    await start(app)
    await asyncio.sleep(0.1)
    assert app.backends[0].close_calls == 1
    assert not app._recording_pool.active


@pytest.mark.asyncio
async def test_session_closes_before_upload_finishes_and_final_sse_waits(monkeypatch):
    import sys
    from types import SimpleNamespace

    from fal.wma import RecordingConfig

    class TrackBase:
        def stop(self):
            pass

    monkeypatch.setitem(
        sys.modules, "aiortc", SimpleNamespace(VideoStreamTrack=TrackBase)
    )
    app = app_with_recording(RecordingConfig())
    response = await start(app)
    session = app.sessions[0]
    rec = session._recording
    uploading, finish = asyncio.Event(), asyncio.Event()

    async def upload(path, request):
        assert session.closed.is_set()
        uploading.set()
        await finish.wait()
        return {"url": "https://v3.fal.media/recording.mp4"}

    rec.upload = upload
    frame = av.VideoFrame.from_ndarray(
        np.zeros((48, 64, 3), dtype=np.uint8), format="rgb24"
    )
    frame.pts, frame.time_base = 0, Fraction(1, 90000)

    class Source:
        async def recv(self):
            return frame

        def stop(self):
            pass

    source = Source()
    track = session.record_video(source, fps=16)
    assert await track.recv() is frame
    with pytest.raises(ValueError, match="one video track"):
        session.record_video(source, fps=16)
    app.backends[0].closed.set()

    async def consume():
        return [chunk async for chunk in response.body_iterator]

    task = asyncio.create_task(consume())
    await asyncio.wait_for(uploading.wait(), 5)
    assert not task.done()
    assert app.backends[0].close_calls == 1
    finish.set()
    chunks = await asyncio.wait_for(task, 5)
    assert '"recording_status": "complete"' in chunks[-1]
    assert '"url": "https://v3.fal.media/recording.mp4"' in chunks[-1]
    assert not app._recording_pool.active


@pytest.mark.parametrize(
    "values",
    [
        {"segment_seconds": float("nan")},
        {"finalization_timeout": 0},
        {"max_pending_frames": 0},
        {"max_pending_uploads": 1.5},
        {"max_active_recordings": True},
    ],
)
def test_recording_limits_must_be_positive_and_bounded(values):
    from fal.wma import RecordingConfig

    with pytest.raises(ValueError):
        RecordingConfig(**values)


@pytest.mark.asyncio
async def test_cancellation_during_finalization_cancels_upload_and_releases_capacity():
    from fal.wma import RecordingConfig

    app = app_with_recording(RecordingConfig())
    response = await start(app)
    rec = app.sessions[0]._recording
    uploading, canceled = asyncio.Event(), asyncio.Event()

    async def upload(path, request):
        uploading.set()
        try:
            await asyncio.Event().wait()
        finally:
            canceled.set()

    rec.upload = upload
    frame = av.VideoFrame.from_ndarray(
        np.zeros((48, 64, 3), dtype=np.uint8), format="rgb24"
    )
    frame.pts, frame.time_base = 0, Fraction(1, 90000)
    await rec.offer(frame)
    app.backends[0].closed.set()

    async def consume():
        return [chunk async for chunk in response.body_iterator]

    task = asyncio.create_task(consume())
    await asyncio.wait_for(uploading.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(canceled.wait(), 5)
    assert rec.aborted.is_set()

    async def released():
        while app._recording_pool.active:
            await asyncio.sleep(0)

    await asyncio.wait_for(released(), 2)
    assert not app._recording_pool.active
    assert not Path(rec.directory.name).exists()
    assert app.backends[0].close_calls == 1


def test_app_metadata_uses_single_shared_endpoint_with_recording_enabled():
    from fal.wma import App, RecordingConfig

    app = app_with_recording(RecordingConfig())
    assert type(app).start_session is App.start_session
    metadata = type(app).build_metadata()
    assert "/start-session" in metadata["openapi"]["paths"]
    assert list(metadata["openapi"]["paths"]["/start-session"]) == ["post"]


@pytest.mark.asyncio
async def test_recorder_initialization_failure_does_not_fail_live_session(monkeypatch):
    from fal.wma import RecordingConfig
    from fal.wma.recording import RecordingPool

    def fail(*args):
        raise OSError("private disk path")

    monkeypatch.setattr(RecordingPool, "create", fail)
    app = app_with_recording(RecordingConfig())
    response = await start(app)
    app.backends[0].closed.set()
    chunks = [chunk async for chunk in response.body_iterator]
    assert '"sdp": "transport-only"' in chunks[0]
    assert '"error": "recording_initialization_failed"' in chunks[-1]
    assert "private disk path" not in chunks[-1]
    assert not app._recording_pool.active


@pytest.mark.asyncio
async def test_encoder_upload_retains_originating_request_context():
    from contextvars import ContextVar

    caller = ContextVar("recording_test_caller", default=None)
    token = caller.set("session-caller")
    seen = []

    async def upload(path, request):
        seen.append(caller.get())
        return {"url": "https://example.com/video.mp4"}

    try:
        rec = SessionRecording("session", None, None, upload=upload)
    finally:
        caller.reset(token)
    frame = av.VideoFrame.from_ndarray(
        np.zeros((48, 64, 3), dtype=np.uint8), format="rgb24"
    )
    frame.pts, frame.time_base = 0, Fraction(1, 90000)
    await rec.offer(frame)
    events = [event async for event in rec.result_events(timeout=5)]
    assert seen == ["session-caller"]
    assert '"recording_status": "complete"' in events[-1]


@pytest.mark.asyncio
async def test_timeout_retains_file_and_capacity_until_blocking_upload_finishes(
    monkeypatch,
):
    import threading
    from types import SimpleNamespace

    from fal.toolkit import File
    from fal.wma import RecordingConfig

    started, finish = threading.Event(), threading.Event()
    observed = {}

    def upload(path, **kwargs):
        observed.update(path=path, kwargs=kwargs)
        started.set()
        assert finish.wait(5)
        assert path.exists()
        return SimpleNamespace(url="https://example.com/video.mp4")

    monkeypatch.setattr(File, "from_path", upload)
    app = app_with_recording(RecordingConfig(finalization_timeout=0.1))
    response = await start(app)
    rec = app.sessions[0]._recording
    rec.request = Request({"type": "http", "headers": []})
    frame = av.VideoFrame.from_ndarray(
        np.zeros((48, 64, 3), dtype=np.uint8), format="rgb24"
    )
    frame.pts, frame.time_base = 0, Fraction(1, 90000)
    await rec.offer(frame)
    app.backends[0].closed.set()
    try:
        chunks = await asyncio.wait_for(collect(response.body_iterator), timeout=2)
        assert started.is_set()
        assert '"recording_upload_timeout"' in chunks[-1]
        assert observed["path"].exists()
        assert app._recording_pool.active
        assert observed["kwargs"]["request"] is rec.request
        assert observed["kwargs"]["content_type"] == "video/mp4"
        assert observed["kwargs"]["fallback_repository"] is None
    finally:
        finish.set()

        async def released():
            while app._recording_pool.active:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(released(), 5)
    assert not Path(rec.directory.name).exists()


async def collect(events):
    return [event async for event in events]


@pytest.mark.asyncio
async def test_recording_setup_http_error_preserves_headers_without_sse(monkeypatch):
    from fastapi import HTTPException

    from fal.wma import RecordingConfig

    async def fail(*args):
        raise HTTPException(429, "busy", headers={"Retry-After": "5"})

    monkeypatch.setattr(Backend, "negotiate", fail)
    app = app_with_recording(RecordingConfig())
    with pytest.raises(HTTPException) as error:
        await start(app)
    assert error.value.detail == "busy"
    assert error.value.headers["Retry-After"] == "5"
    assert error.value.headers["x-fal-billable-units"] == "0"
    assert not any(key.startswith("x-fal-sse-") for key in error.value.headers)
    assert not app._recording_pool.active


@pytest.mark.asyncio
async def test_recording_init_failure_does_not_claim_uploading(monkeypatch):
    from fal.wma import RecordingConfig
    from fal.wma.recording import RecordingPool

    def fail(*args):
        raise RuntimeError("initialization failed")

    monkeypatch.setattr(RecordingPool, "create", fail)
    app = app_with_recording(RecordingConfig())
    response = await start(app)
    app.backends[0].closed.set()
    chunks = [chunk async for chunk in response.body_iterator]
    assert '"recording_status": "uploading"' not in "".join(chunks)
    assert '"error": "recording_initialization_failed"' in chunks[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("close_during_snapshot", [False, True])
async def test_snapshot_leaves_loop_responsive_and_respects_closure(
    monkeypatch, close_during_snapshot
):
    import threading

    rec = SessionRecording("snapshot", None, None)
    entered, release = threading.Event(), threading.Event()
    loop_thread = threading.get_ident()

    def snapshot(frame):
        assert threading.get_ident() != loop_thread
        entered.set()
        assert release.wait(5)
        return bytes(64 * 48 * 3 // 2), 64, 48, 0.0

    monkeypatch.setattr(rec, "_snapshot", snapshot)
    task = asyncio.create_task(rec.offer(object()))
    try:

        async def wait_entered():
            while not entered.is_set():
                await asyncio.sleep(0.001)

        await asyncio.wait_for(wait_entered(), 2)
        # A sibling session can run while snapshot work is still blocked.
        assert not task.done()
        if close_during_snapshot:
            rec.abort()
        release.set()
        await asyncio.wait_for(task, 2)
        if close_during_snapshot:
            assert rec.thread is None
            assert rec.queue.empty()
        else:
            assert rec.thread is not None
    finally:
        release.set()
        await task
        rec.seal()
        _ = [event async for event in rec.result_events(timeout=5)]
