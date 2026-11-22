"""Two independent control-channel experiences in one CPU deployment."""

from typing import Literal, Optional

from fastapi import Body, Header, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import fal
from fal import wma


class Echo(BaseModel):
    type: Literal["echo"] = "echo"
    text: str = "Hello!"


class Move(BaseModel):
    type: Literal["move"] = "move"
    direction: Literal["left", "right"] = "left"


class MultiEndpointApp(wma.App):
    machine_type = "XS"
    requirements = ["pydantic>=2,<3"]
    max_multiplexing = 4
    secrets = ["FAL_KEY"]
    realtime_contract = wma.RealtimeContract(
        client_messages=Echo,
        server_messages=Echo,
        client_message_examples=(
            wma.MessageExample(name="Echo", payload={"type": "echo", "text": "Hello!"}),
        ),
    )
    realtime_contracts = {
        "/game/start-session": wma.RealtimeContract(
            client_messages=Move,
            server_messages=Move,
            client_message_examples=(
                wma.MessageExample(name="Move left", payload={"type": "move", "direction": "left"}),
            ),
        ),
    }

    @fal.endpoint("/game/start-session")
    async def game(
        self,
        request: wma.StartSessionRequest = Body(...),
        x_fal_caller_user_id: Optional[str] = Header(None),
        x_fal_request_id: Optional[str] = Header(None),
        http_request: Request = None,
    ) -> StreamingResponse:
        # Reuse the same negotiation, cleanup, and billing lifecycle.
        return await self.serve_session(
            request,
            create_backend=self.create_game_backend,
            x_fal_caller_user_id=x_fal_caller_user_id,
            x_fal_request_id=x_fal_request_id,
            http_request=http_request,
        )

    async def create_backend(self, session):
        return await self.create_control_backend(session, "echo")

    async def create_game_backend(self, session):
        return await self.create_control_backend(session, "move")

    async def create_control_backend(self, session, message_type):
        from aiortc import RTCConfiguration

        session.on_message(message_type, lambda message: session.send(message))
        ice = wma.RunnerIceConfig.from_bridge()
        servers, _turn_available = await ice.build_ice_servers_async(
            session.offer.ice_servers, forwarded_status=session.offer.ice_status
        )
        return wma.AiortcPeer(
            session,
            lambda _pc: None,
            rtc_configuration=RTCConfiguration(iceServers=servers),
        )
