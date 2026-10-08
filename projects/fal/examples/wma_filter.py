"""Send camera video and receive a live mirror, monochrome, or edge effect."""

import asyncio
from typing import Literal

from pydantic import BaseModel, Field

import fal.wma


class Effect(BaseModel):
    type: Literal["effect"] = "effect"
    mode: Literal["mirror", "monochrome", "edges"] = "mirror"
    strength: float = Field(
        default=1.0, ge=0, le=1, description="Blend effect with original video."
    )


def apply_effect(pixels, mode, strength):
    import numpy as np

    original = pixels.astype(np.float32)
    if mode == "mirror":
        effect = original[:, ::-1]
    else:
        gray = original.mean(axis=2)
        if mode == "edges":
            dx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
            dy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
            gray = np.clip((dx + dy) * 3, 0, 255)
        effect = np.repeat(gray[..., None], 3, axis=2)
    return np.clip(original * (1 - strength) + effect * strength, 0, 255).astype(
        np.uint8
    )


class FilterApp(fal.wma.App):
    machine_type = "XS"
    requirements = ["pydantic>=2,<3", "numpy>=1.24,<3"]
    min_concurrency = 0
    max_concurrency = 1
    max_multiplexing = 2
    keep_alive = 30
    realtime_contract = fal.wma.RealtimeContract(
        media=fal.wma.MediaContract(
            send=(
                fal.wma.Track(
                    kind="video",
                    source="camera",
                    required=True,
                    width=fal.wma.Constraint(ideal=480),
                    height=fal.wma.Constraint(ideal=270),
                ),
            ),
            receive=(fal.wma.Track(kind="video", width=480, height=270),),
        ),
        client_messages=Effect,
        server_messages=Effect,
        client_message_examples=tuple(
            fal.wma.MessageExample(
                name=mode.title(),
                payload={"type": "effect", "mode": mode, "strength": 1},
            )
            for mode in ("mirror", "monochrome", "edges")
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
        state = Effect()

        def configure_effect(message):
            nonlocal state
            state = Effect.model_validate(message)
            session.send(state.model_dump())

        session.on_message("effect", configure_effect)

        class FilterTrack(VideoStreamTrack):
            def __init__(self, source):
                super().__init__()
                self.source = source

            async def recv(self):
                source = await self.source.recv()
                # Bound processing cost even if the camera ignores ideal constraints.
                pixels = source.reformat(
                    width=480, height=270, format="rgb24"
                ).to_ndarray()
                frame = VideoFrame.from_ndarray(
                    apply_effect(pixels, state.mode, state.strength), format="rgb24"
                )
                frame.pts, frame.time_base = source.pts, source.time_base
                return frame

            def stop(self):
                self.source.stop()
                super().stop()

        def connect(pc):
            attached = False

            @pc.on("track")
            def incoming(track):
                nonlocal attached
                if track.kind == "video" and not attached:
                    attached = True
                    pc.addTrack(FilterTrack(track))

        async def expire():
            await asyncio.sleep(120)
            await session.close()

        session.create_task(expire())
        return fal.wma.AiortcPeer(
            session,
            connect,
            rtc_configuration=RTCConfiguration(iceServers=servers),
            disconnected_grace_seconds=10,
        )
