"""An animated color field controlled live from the WMA playground."""

import asyncio
import math
from typing import Literal

from pydantic import BaseModel, Field

import fal.wma


class Scene(BaseModel):
    type: Literal["scene"] = "scene"
    palette: Literal["ocean", "sunset", "forest"] = "ocean"
    speed: float = Field(
        default=1.0, ge=0, le=4, description="Animation speed; zero pauses."
    )


def render_scene(phase, palette, width=480, height=270):
    import numpy as np

    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    wave = (np.sin(x / 35 + phase) + np.cos(y / 28 - phase) + 2) / 4
    colors = {
        "ocean": ((10, 20, 65), (40, 230, 250)),
        "sunset": ((65, 10, 80), (255, 175, 60)),
        "forest": ((10, 35, 30), (175, 240, 75)),
    }
    low, high = (np.array(color) for color in colors[palette])
    pixels = low + wave[..., None] * (high - low)
    cx = width / 2 + math.cos(phase) * width / 3
    cy = height / 2 + math.sin(phase * 1.3) * height / 3
    pixels[(x - cx) ** 2 + (y - cy) ** 2 < 18**2] = (255, 255, 240)
    return pixels.astype(np.uint8)


class VideoApp(fal.wma.App):
    machine_type = "XS"
    requirements = ["pydantic>=2,<3", "numpy>=1.24,<3"]
    min_concurrency = 0
    max_concurrency = 1
    max_multiplexing = 2
    keep_alive = 30
    realtime_contract = fal.wma.RealtimeContract(
        media=fal.wma.MediaContract(
            receive=(fal.wma.Track(kind="video", width=480, height=270, frame_rate=30),)
        ),
        client_messages=Scene,
        server_messages=Scene,
        client_message_examples=(
            fal.wma.MessageExample(
                name="Ocean", payload={"type": "scene", "palette": "ocean", "speed": 1}
            ),
            fal.wma.MessageExample(
                name="Fast sunset",
                payload={"type": "scene", "palette": "sunset", "speed": 3},
            ),
            fal.wma.MessageExample(
                name="Pause in the forest",
                payload={"type": "scene", "palette": "forest", "speed": 0},
            ),
        ),
    )

    async def create_backend(self, session):
        from aiortc import RTCConfiguration, VideoStreamTrack
        from av import VideoFrame

        (
            servers,
            _,
        ) = await fal.wma.RunnerIceConfig.from_bridge().build_ice_servers_async(
            session.offer.ice_servers, forwarded_status=session.offer.ice_status
        )
        state = Scene()

        def configure_scene(message):
            nonlocal state
            state = Scene.model_validate(message)
            session.send(state.model_dump())

        session.on_message("scene", configure_scene)

        class SceneTrack(VideoStreamTrack):
            phase = 0.0

            async def recv(self):
                pts, time_base = await self.next_timestamp()
                self.phase += state.speed / 30
                pixels = render_scene(self.phase, state.palette)
                frame = VideoFrame.from_ndarray(pixels, format="rgb24")
                frame.pts, frame.time_base = pts, time_base
                return frame

        async def expire():
            await asyncio.sleep(120)
            await session.close()

        session.create_task(expire())
        return fal.wma.AiortcPeer(
            session,
            lambda pc: pc.addTrack(SceneTrack()),
            rtc_configuration=RTCConfiguration(iceServers=servers),
            disconnected_grace_seconds=10,
        )
