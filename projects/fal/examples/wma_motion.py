"""Camera in, motion measurements out over the data channel; no video returned."""

import asyncio
from typing import Literal

from pydantic import BaseModel, Field

import fal.wma


class Sensitivity(BaseModel):
    type: Literal["sensitivity"] = "sensitivity"
    threshold: float = Field(
        default=0.08,
        ge=0.01,
        le=1,
        description="Per-pixel change needed to count as motion.",
    )


class Motion(BaseModel):
    type: Literal["motion"] = "motion"
    brightness: float = Field(ge=0, le=1)
    changed_fraction: float = Field(ge=0, le=1)


def measure_motion(current, previous, threshold):
    import numpy as np

    changed = (
        0.0
        if previous is None
        else float(np.mean(np.abs(current - previous) >= threshold))
    )
    return Motion(brightness=float(current.mean()), changed_fraction=changed)


class MotionApp(fal.wma.App):
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
                    width=fal.wma.Constraint(ideal=320),
                    height=fal.wma.Constraint(ideal=180),
                ),
            )
        ),
        client_messages=Sensitivity,
        server_messages=Motion,
        client_message_examples=(
            fal.wma.MessageExample(
                name="Subtle movement",
                payload={"type": "sensitivity", "threshold": 0.04},
            ),
            fal.wma.MessageExample(
                name="Large movement", payload={"type": "sensitivity", "threshold": 0.2}
            ),
        ),
    )

    async def create_backend(self, session):
        import numpy as np
        from aiortc import RTCConfiguration
        from aiortc.mediastreams import MediaStreamError

        (
            servers,
            _,
        ) = await fal.wma.RunnerIceConfig.from_bridge().build_ice_servers_async(
            session.offer.ice_servers, forwarded_status=session.offer.ice_status
        )
        state = Sensitivity()

        def configure_sensitivity(message):
            nonlocal state
            state = Sensitivity.model_validate(message)

        session.on_message("sensitivity", configure_sensitivity)

        async def consume(track):
            previous = None
            next_report = 0.0
            loop = asyncio.get_running_loop()
            try:
                while True:
                    frame = await track.recv()
                    # Drain every frame; never sleep on the receiving queue. Sample
                    # at most five times/second and publish only current measurements.
                    now = loop.time()
                    if now < next_report:
                        continue
                    next_report = now + 0.2
                    current = (
                        frame.reformat(width=160, height=90, format="gray")
                        .to_ndarray()
                        .astype(np.float32)
                        / 255
                    )
                    session.send(
                        measure_motion(current, previous, state.threshold).model_dump()
                    )
                    previous = current
            except MediaStreamError:
                await session.close()

        def connect(pc):
            attached = False

            @pc.on("track")
            def incoming(track):
                nonlocal attached
                if track.kind == "video" and not attached:
                    attached = True
                    session.defer(track.stop)
                    session.create_task(consume(track))

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
