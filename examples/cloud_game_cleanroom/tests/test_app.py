from __future__ import annotations

import asyncio
import json
import threading
import unittest

from fal.wma import Session, StartSessionRequest

from app import (
    CloudGame,
    _bind_game_messages,
    _create_game_video_track,
    _cuda_conversion_enabled,
    _native_streaming_enabled,
)
from doom import OrbitBackend
from engine import HEIGHT, WIDTH


class ProtocolTests(unittest.TestCase):
    def test_start_session_is_a_json_body_endpoint(self) -> None:
        operation = CloudGame(_allow_init=True).openapi()["paths"]["/start-session"][
            "post"
        ]

        self.assertIn("requestBody", operation)
        self.assertNotIn("parameters", operation)

    def test_runner_uses_native_transport_unless_overridden(self) -> None:
        from unittest.mock import patch

        with patch.dict("os.environ", {"IS_ISOLATE_AGENT": "1"}, clear=True):
            self.assertTrue(_native_streaming_enabled())
        with patch.dict(
            "os.environ",
            {"IS_ISOLATE_AGENT": "1", "FAL_CLOUD_GAME_TRANSPORT": "aiortc"},
            clear=True,
        ):
            self.assertFalse(_native_streaming_enabled())

    def test_cuda_conversion_is_opt_in(self) -> None:
        from unittest.mock import patch

        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(_cuda_conversion_enabled())
            self.assertTrue(_cuda_conversion_enabled(default=True))
        with patch.dict(
            "os.environ", {"FAL_CLOUD_GAME_CUDA_CONVERT": "true"}, clear=True
        ):
            self.assertTrue(_cuda_conversion_enabled())
        with patch.dict(
            "os.environ", {"FAL_CLOUD_GAME_CUDA_CONVERT": "invalid"}, clear=True
        ):
            with self.assertRaisesRegex(ValueError, "boolean"):
                _cuda_conversion_enabled()


class VideoTrackTests(unittest.TestCase):
    def test_track_produces_timed_rgb_frame(self) -> None:
        async def receive_frame():
            track = _create_game_video_track(OrbitBackend())
            frame = await track.recv()
            track.stop()
            return frame

        frame = asyncio.run(receive_frame())

        self.assertEqual(frame.width, WIDTH)
        self.assertEqual(frame.height, HEIGHT)
        self.assertEqual(frame.format.name, "rgb24")
        self.assertIsNotNone(frame.pts)
        self.assertIsNotNone(frame.time_base)


class SessionLifecycleTests(unittest.TestCase):
    def test_close_waits_for_restart_worker_before_game_cleanup(self) -> None:
        class BlockingGame:
            def __init__(self) -> None:
                self.restart_started = threading.Event()
                self.restart_release = threading.Event()
                self.restart_finished = threading.Event()
                self.closed = False

            def apply_input(self, _message) -> None:
                pass

            def restart(self) -> None:
                self.restart_started.set()
                self.restart_release.wait(timeout=1)
                self.restart_finished.set()

            def close(self) -> None:
                self.assert_restart_finished()
                self.closed = True

            def assert_restart_finished(self) -> None:
                if not self.restart_finished.is_set():
                    raise AssertionError("game closed while restart was running")

        async def scenario() -> None:
            request = StartSessionRequest(sdp="offer")
            session = Session(request)
            game = BlockingGame()
            _bind_game_messages(session, game)
            session.defer(game.close)

            session.receive({"type": "restart"})
            await asyncio.to_thread(game.restart_started.wait, 1)
            close_task = asyncio.create_task(session.close())
            await asyncio.sleep(0)
            self.assertFalse(game.closed)

            game.restart_release.set()
            await close_task
            self.assertTrue(game.restart_finished.is_set())
            self.assertTrue(game.closed)

        asyncio.run(scenario())


class LocalWebRtcSessionTests(unittest.TestCase):
    def test_offer_receives_answer_and_video(self) -> None:
        async def run_session() -> None:
            from aiortc import RTCPeerConnection, RTCSessionDescription

            client = RTCPeerConnection()
            stream = None
            client.addTransceiver("video", direction="recvonly")
            channel = client.createDataChannel("input", ordered=False, maxRetransmits=0)
            frame_received = asyncio.Event()
            telemetry_received = asyncio.Event()
            pong_received = asyncio.Event()

            @channel.on("open")
            def on_open() -> None:
                channel.send(
                    json.dumps(
                        {
                            "type": "input",
                            "seq": 1,
                            "keys": ["ArrowRight"],
                            "gamepad": None,
                        }
                    )
                )
                channel.send(json.dumps({"type": "ping", "client_ts": 42}))

            @channel.on("message")
            def on_message(raw) -> None:
                message = json.loads(raw)
                if message.get("type") == "pong" and message.get("client_ts") == 42:
                    pong_received.set()
                if (
                    message.get("type") == "game_state"
                    and message.get("paddle_x", 0) > 414
                ):
                    telemetry_received.set()

            @client.on("track")
            def on_track(track) -> None:
                async def receive() -> None:
                    frame = await track.recv()
                    if frame.width == WIDTH and frame.height == HEIGHT:
                        frame_received.set()

                asyncio.create_task(receive())

            try:
                offer = await client.createOffer()
                await client.setLocalDescription(offer)
                if "a=candidate:" not in client.localDescription.sdp:
                    raise unittest.SkipTest(
                        "local UDP sockets are unavailable in this environment"
                    )

                app = CloudGame(_allow_init=True)
                response = await app.start_session(
                    StartSessionRequest(
                        sdp=client.localDescription.sdp,
                        type=client.localDescription.type,
                        session_id="local-test",
                    )
                )
                stream = response.body_iterator
                first_event = await anext(stream)
                answer = json.loads(first_event.removeprefix("data: ").strip())
                self.assertIn("H264/90000", answer["sdp"])
                self.assertNotIn("VP8/90000", answer["sdp"])
                await client.setRemoteDescription(
                    RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
                )

                try:
                    await asyncio.wait_for(
                        asyncio.gather(
                            frame_received.wait(),
                            telemetry_received.wait(),
                            pong_received.wait(),
                        ),
                        timeout=5,
                    )
                except TimeoutError:
                    self.fail(
                        "session timed out: "
                        f"frame={frame_received.is_set()}, "
                        f"telemetry={telemetry_received.is_set()}, "
                        f"pong={pong_received.is_set()}"
                    )
            finally:
                await client.close()
                if stream is not None:
                    await stream.aclose()

        asyncio.run(run_session())


if __name__ == "__main__":
    unittest.main()
