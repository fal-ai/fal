from __future__ import annotations

import asyncio
import json
import unittest

from aiortc import RTCPeerConnection, RTCSessionDescription
from fal.wma import GStreamerPeer, PipelineSpec, Session, StartSessionRequest

from doom import DoomSession
from gstreamer import offer_formats, pipeline_description


async def _wait_for_ice(connection: RTCPeerConnection) -> None:
    if connection.iceGatheringState == "complete":
        return
    complete = asyncio.Event()

    @connection.on("icegatheringstatechange")
    def on_ice_gathering_state_change() -> None:
        if connection.iceGatheringState == "complete":
            complete.set()

    await asyncio.wait_for(complete.wait(), timeout=7)


class NativeRuntimeSmokeTests(unittest.TestCase):
    def test_nvenc_webrtc_audio_video_input_and_teardown(self) -> None:
        asyncio.run(self._run_session())

    async def _run_session(self) -> None:
        DoomSession.require_available()
        GStreamerPeer.require_available(
            [
                "ximagesrc",
                "nvh264enc",
                "h264parse",
                "rtph264pay",
                "pulsesrc",
                "opusenc",
                "rtpopuspay",
            ]
        )
        game = DoomSession()
        client = RTCPeerConnection()
        session = None
        video_received = asyncio.Event()
        audio_received = asyncio.Event()
        channel_open = asyncio.Event()
        input_received = asyncio.Event()
        receive_tasks: list[asyncio.Task[None]] = []

        def on_message(message: dict[str, object]) -> None:
            if message.get("type") != "input":
                return
            game.apply_input(message)
            if game.snapshot()["input_seq"] == 1:
                input_received.set()

        @client.on("track")
        def on_track(track) -> None:
            async def receive() -> None:
                await track.recv()
                if track.kind == "video":
                    video_received.set()
                elif track.kind == "audio":
                    audio_received.set()

            receive_tasks.append(asyncio.create_task(receive()))

        channel = client.createDataChannel(
            "input", ordered=False, maxRetransmits=0
        )

        @channel.on("open")
        def on_channel_open() -> None:
            channel_open.set()
            channel.send(
                json.dumps(
                    {
                        "type": "input",
                        "seq": 1,
                        "keys": ["KeyW"],
                        "gamepad": None,
                        "mouse": {"dx": 4, "dy": -2, "buttons": [1]},
                    }
                )
            )

        try:
            await asyncio.to_thread(game.start, capture=False, audio=True)
            assert game.pulse_server is not None
            client.addTransceiver("video", direction="recvonly")
            client.addTransceiver("audio", direction="recvonly")
            offer = await client.createOffer()
            await client.setLocalDescription(offer)
            await _wait_for_ice(client)
            request = StartSessionRequest(
                sdp=client.localDescription.sdp,
                type=client.localDescription.type,
                session_id="native-smoke",
            )
            session = Session(request)
            session.defer(lambda: asyncio.to_thread(game.close))
            session.on_message("input", on_message, inline=True)

            def build_pipeline(offer_request: StartSessionRequest) -> PipelineSpec:
                video, audio_payload = offer_formats(offer_request.sdp)
                return PipelineSpec(
                    pipeline_description(
                        display_name=game.display_name,
                        pulse_server=game.pulse_server,
                        pulse_monitor="fal_game.monitor",
                        video_payload=video.payload,
                        audio_payload=audio_payload,
                        h264_profile=video.profile,
                        h264_level=video.level,
                        video_fps=video.fps,
                    )
                )

            transport = GStreamerPeer(
                session,
                build_pipeline,
                channel_labels={"input"},
            )
            session.bind_backend(transport)
            answer = await transport.negotiate(request)
            self.assertIn("profile-level-id=", answer.sdp)
            await client.setRemoteDescription(
                RTCSessionDescription(sdp=answer.sdp, type=answer.type)
            )
            await asyncio.wait_for(
                asyncio.gather(
                    video_received.wait(),
                    audio_received.wait(),
                    channel_open.wait(),
                    input_received.wait(),
                ),
                timeout=15,
            )
        finally:
            await client.close()
            await asyncio.gather(*receive_tasks, return_exceptions=True)
            if session is not None:
                await session.close()
            else:
                await asyncio.to_thread(game.close)

        self.assertIsNone(game._game)
        self.assertIsNone(game._xvfb)
        self.assertIsNone(game._audio)


if __name__ == "__main__":
    unittest.main()
