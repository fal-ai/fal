import sys
from types import ModuleType

from fal.wma import IceServer, StartSessionRequest, _aiortc_configuration
from fal.wma_gstreamer import GStreamerPeer, _gstreamer_ice_servers


def test_start_session_accepts_forwarded_ice_servers():
    request = StartSessionRequest(
        sdp="v=0",
        ice_servers=[
            {"urls": "stun:relay.example.com:80"},
            {
                "urls": [
                    "turn:relay.example.com:80",
                    "turns:relay.example.com:443?transport=tcp",
                ],
                "username": "runner user",
                "credential": "secret/value",
            },
        ],
        ice_status="ready",
        credential_age_seconds=12.5,
    )

    assert request.ice_status == "ready"
    assert request.credential_age_seconds == 12.5
    assert request.ice_servers[1].username == "runner user"


def test_aiortc_configuration_uses_forwarded_ice_servers(monkeypatch):
    aiortc = ModuleType("aiortc")

    class RTCIceServer:
        def __init__(self, *, urls, username, credential):
            self.urls = urls
            self.username = username
            self.credential = credential

    class RTCConfiguration:
        def __init__(self, *, iceServers):
            self.iceServers = iceServers

    aiortc.RTCIceServer = RTCIceServer
    aiortc.RTCConfiguration = RTCConfiguration
    monkeypatch.setitem(sys.modules, "aiortc", aiortc)

    configuration = _aiortc_configuration(
        [
            IceServer(urls="stun:relay.example.com:80"),
            IceServer(
                urls="turn:relay.example.com:443?transport=tcp",
                username="runner",
                credential="secret",
            ),
        ]
    )

    assert configuration.iceServers[0].urls == "stun:relay.example.com:80"
    assert configuration.iceServers[1].username == "runner"
    assert configuration.iceServers[1].credential == "secret"


def test_gstreamer_ice_servers_embed_encoded_turn_credentials():
    stun_servers, turn_servers = _gstreamer_ice_servers(
        [
            IceServer(urls="stun:relay.example.com:80"),
            IceServer(
                urls=[
                    "turn:relay.example.com:80",
                    "turns:relay.example.com:443?transport=tcp",
                ],
                username="runner user",
                credential="secret/value",
            ),
        ]
    )

    assert stun_servers == ["stun://relay.example.com:80"]
    assert turn_servers == [
        "turn://runner%20user:secret%2Fvalue@relay.example.com:80",
        "turns://runner%20user:secret%2Fvalue@relay.example.com:443?transport=tcp",
    ]


def test_gstreamer_configures_every_forwarded_turn_server():
    class WebRtcElement:
        def __init__(self):
            self.properties = []
            self.signals = []

        def set_property(self, name, value):
            self.properties.append((name, value))

        def emit(self, name, value):
            self.signals.append((name, value))

    peer = object.__new__(GStreamerPeer)
    peer._stun_server = None
    peer._turn_server = None
    peer._webrtc = WebRtcElement()
    peer._configure_ice_servers(
        [
            IceServer(urls="stun:relay.example.com:80"),
            IceServer(
                urls=[
                    "turn:relay.example.com:80",
                    "turns:relay.example.com:443?transport=tcp",
                ],
                username="runner",
                credential="secret",
            ),
        ]
    )

    assert peer._webrtc.properties == [
        ("stun-server", "stun://relay.example.com:80"),
        ("turn-server", "turn://runner:secret@relay.example.com:80"),
    ]
    assert peer._webrtc.signals == [
        (
            "add-turn-server",
            "turns://runner:secret@relay.example.com:443?transport=tcp",
        )
    ]
