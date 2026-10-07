"""Minimal standalone WMA control-channel app."""

import fal.wma


class EchoApp(fal.wma.App):
    machine_type = "XS"
    max_multiplexing = 4
    secrets = ["FAL_KEY"]

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
            rtc_configuration=RTCConfiguration(
                iceServers=servers
            ),
        )
