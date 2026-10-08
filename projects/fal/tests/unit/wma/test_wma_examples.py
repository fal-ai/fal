"""Exercise the public examples' media and control paths with real local peers."""

import asyncio
import json
import runpy
from pathlib import Path

import pydantic
import pytest

if not hasattr(pydantic, "TypeAdapter"):
    pytest.skip("playground examples require Pydantic 2", allow_module_level=True)

import fal.wma

EXAMPLES = Path(__file__).parents[3] / "examples"


def load(name):
    return runpy.run_path(str(EXAMPLES / ("wma_" + name + ".py")))


@pytest.mark.parametrize(
    "name,send,receive", [("video", 0, 1), ("filter", 1, 1), ("motion", 1, 0)]
)
def test_example_metadata_matches_media_direction(name, send, receive):
    from fal.app import wrap_app

    app = load(name)[name.title() + "App"]
    metadata = wrap_app(app).options.host["metadata"]
    assert (
        metadata["openapi"]["paths"]["/start-session"]["x-fal-realtime"]["transport"][
            "sessionProtocol"
        ]
        == "wma"
    )
    assert metadata["asyncapi"]["channels"]["control"]["messages"]
    assert len(app.realtime_contract.media.send) == send
    assert len(app.realtime_contract.media.receive) == receive


def test_scene_animates_and_filter_strength_is_respected():
    np = pytest.importorskip("numpy")
    render = load("video")["render_scene"]
    assert not np.array_equal(render(0, "ocean"), render(1, "ocean"))
    assert not np.array_equal(render(0, "ocean"), render(0, "sunset"))
    transform = load("filter")["apply_effect"]
    pixels = np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8)
    assert np.array_equal(transform(pixels, "mirror", 1), pixels[:, ::-1])
    assert np.array_equal(transform(pixels, "edges", 0), pixels)
    gray = transform(pixels, "monochrome", 1)
    assert np.array_equal(gray[..., 0], gray[..., 1])


def test_motion_measurements_detect_change_and_validate_settings():
    np = pytest.importorskip("numpy")
    module = load("motion")
    dark, light = np.zeros((90, 160)), np.ones((90, 160))
    measure = module["measure_motion"]
    assert measure(dark, None, 0.1).changed_fraction == 0
    assert measure(light, dark, 0.1).changed_fraction == 1
    assert measure(light, light, 0.1).brightness == 1
    with pytest.raises(pydantic.ValidationError):
        module["Sensitivity"](threshold=-1)


@pytest.mark.allow_real_sleep
@pytest.mark.parametrize("name", ["video", "filter", "motion"])
def test_example_real_media_controls_and_cleanup(name, monkeypatch):
    pytest.importorskip("aiortc")
    from aioice import ice
    from aiortc import (
        RTCConfiguration,
        RTCPeerConnection,
        RTCSessionDescription,
        VideoStreamTrack,
    )

    from fal.wma import Session, StartSessionRequest, _raw

    async def no_ice(*args, **kwargs):
        return [], False

    monkeypatch.setattr(fal.wma.RunnerIceConfig, "build_ice_servers_async", no_ice)
    monkeypatch.setattr(
        ice, "get_host_addresses", lambda use_ipv4, use_ipv6: ["127.0.0.1"]
    )
    monkeypatch.setattr(_raw, "is_globally_routable_ip", lambda ip: True)

    async def scenario():
        client = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        channel = client.createDataChannel("control")
        opened = asyncio.Event()
        messages = asyncio.Queue()
        tracks = asyncio.Queue()
        channel.on("open", opened.set)
        channel.on("message", lambda raw: messages.put_nowait(json.loads(raw)))
        client.on("track", tracks.put_nowait)
        source = None
        if name == "video":
            client.addTransceiver("video", direction="recvonly")
        else:
            source = VideoStreamTrack()
            client.addTrack(source)
        session = None
        backend = None
        server = None
        try:
            await client.setLocalDescription(await client.createOffer())
            offer = StartSessionRequest(sdp=client.localDescription.sdp, type="offer")
            session = Session(offer)
            app = load(name)[name.title() + "App"](_allow_init=True)
            backend = await app.create_backend(session)
            session.bind_backend(backend)
            answer = await backend.negotiate(offer)
            server = backend._pc
            await client.setRemoteDescription(
                RTCSessionDescription(sdp=answer.sdp, type=answer.type)
            )
            await asyncio.wait_for(opened.wait(), 10)
            command = {
                "video": {"type": "scene", "palette": "sunset", "speed": 2},
                "filter": {"type": "effect", "mode": "edges", "strength": 1},
                "motion": {"type": "sensitivity", "threshold": 0.2},
            }[name]
            channel.send(json.dumps(command))
            if name == "motion":
                message = await asyncio.wait_for(messages.get(), 10)
                assert message["type"] == "motion"
                assert 0 <= message["brightness"] <= 1
                assert 0 <= message["changed_fraction"] <= 1
                assert tracks.empty()
            else:
                message = await asyncio.wait_for(messages.get(), 10)
                assert message == command
                track = await asyncio.wait_for(tracks.get(), 10)
                frame = await asyncio.wait_for(track.recv(), 10)
                assert (frame.width, frame.height) == (480, 270)
        finally:
            if source is not None:
                source.stop()
            await asyncio.wait_for(
                asyncio.gather(client.close(), *([session.close()] if session else [])),
                10,
            )
        assert session is not None and session._is_closed
        assert all(task.done() for task in session._tasks)
        assert server.connectionState == "closed"
        assert all(
            sender.track is None or sender.track.readyState == "ended"
            for sender in server.getSenders()
        )

    asyncio.run(asyncio.wait_for(scenario(), 30))
