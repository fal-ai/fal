import os
from typing import Any

import fal.wma as wma
from aiortc import RTCConfiguration, RTCIceServer


class GrayscaleProcessor:
    def __init__(self) -> None:
        self._neutral_planes: dict[int, bytes] = {}

    def __call__(self, frames: list[Any]) -> list[Any]:
        outputs = []
        for frame in frames:
            output = frame.reformat(format="yuv420p")
            for plane in output.planes[1:]:
                neutral = self._neutral_planes.get(plane.buffer_size)
                if neutral is None:
                    neutral = bytes([128]) * plane.buffer_size
                    self._neutral_planes[plane.buffer_size] = neutral
                plane.update(neutral)
            outputs.append(output)
        return outputs


def rtc_configuration() -> RTCConfiguration | None:
    ice_servers = []
    stun_url = os.getenv("FAL_WMA_STUN_SERVER", "")
    if stun_url:
        ice_servers.append(RTCIceServer(urls=stun_url))

    turn_url = os.getenv("FAL_WMA_TURN_SERVER")
    if turn_url:
        ice_servers.append(
            RTCIceServer(
                urls=turn_url,
                username=os.getenv("FAL_WMA_TURN_USERNAME"),
                credential=os.getenv("FAL_WMA_TURN_CREDENTIAL"),
            )
        )

    if not ice_servers:
        return None
    return RTCConfiguration(iceServers=ice_servers)


class GrayscaleApp(
    wma.App,
    name="wma-grayscale",
    max_concurrency=1,
    max_multiplexing=1,
):
    requirements = ["aiortc==1.15.0"]
    keep_alive = 300

    def setup(self) -> None:
        self.processor = GrayscaleProcessor()

    async def create_backend(
        self,
        session: wma.Session,
    ) -> wma.PeerBackend:
        session.answer_metadata.update(
            {
                "effect": "grayscale",
                "queue_size": 1,
            }
        )
        session.set_response_header("x-fal-billable-units", "0")

        return wma.VideoProcessorPeer(
            session,
            self.processor,
            policy=wma.VideoProcessorPolicy(
                batch_size=1,
                max_queue_size=1,
                max_batch_wait_ms=0,
                overflow="drop_oldest",
                execution="worker",
                max_output_frames=1,
                processor_timeout_ms=1_000,
                shutdown_timeout_ms=1_000,
            ),
            rtc_configuration=rtc_configuration(),
            create_default_channel=False,
            disconnected_grace_seconds=5,
        )
