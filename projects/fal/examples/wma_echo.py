"""Standalone WMA control-channel app with a discoverable playground."""

from typing import Literal

from pydantic import BaseModel, Field

import fal.wma


class EchoMessage(BaseModel):
    type: Literal["echo"] = "echo"
    text: str = Field(default="Hello, WMA!", description="Text to echo back.")


class EchoApp(fal.wma.App):
    machine_type = "XS"
    requirements = ["pydantic>=2,<3"]
    max_multiplexing = 4
    secrets = ["FAL_KEY"]
    realtime_contract = fal.wma.RealtimeContract(
        # Track directions are from the client's perspective. This app only
        # exchanges control messages, so it has no audio or video tracks.
        media=fal.wma.MediaContract(),
        client_messages=EchoMessage,
        server_messages=EchoMessage,
        client_message_examples=(
            fal.wma.MessageExample(
                name="Say hello",
                payload={"type": "echo", "text": "Hello, WMA!"},
            ),
        ),
    )

    async def create_backend(self, session):
        from aiortc import RTCConfiguration

        ice = fal.wma.RunnerIceConfig.from_bridge()
        servers, _turn_available = await ice.build_ice_servers_async(
            session.offer.ice_servers, forwarded_status=session.offer.ice_status
        )
        session.on_message("echo", lambda message: session.send(message))
        return fal.wma.AiortcPeer(
            session,
            lambda _pc: None,
            rtc_configuration=RTCConfiguration(iceServers=servers),
        )
