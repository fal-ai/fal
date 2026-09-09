"""Connection-oriented WebRTC session apps (WMA) for fal.

.. warning::
    This is an **experimental** API: it may change in a minor release. Import
    it explicitly (``import fal.wma``); it is deliberately not part of the
    top-level ``fal`` namespace while experimental.

A WMA app serves live browser-to-runner sessions — world models, realtime
video transforms, avatars — over one WebRTC peer connection. Signaling runs
through the fal WMA bridge: the browser posts a complete SDP offer, the
bridge forwards it to the app's ``POST /start-session`` endpoint, and the
app answers with an SSE stream whose first event carries the SDP answer.
That HTTP response is then held open for the whole session; when it ends,
teardown runs. Media and control data flow peer-to-peer (or via TURN),
never through the bridge.

Subclass :class:`fal.wma.App` and implement ``create_backend()``::

    import fal
    import fal.wma

    class MyTransform(fal.wma.App):
        machine_type = "GPU-H100"

        async def create_backend(self, session):
            def on_connect(pc):
                ...  # add tracks / configure the RTCPeerConnection
            return fal.wma.AiortcPeer(session, on_connect)

The browser side is the ``wma()`` extension for ``fal.realtime.open()`` in
the ``@fal-ai/client`` JavaScript package.

``aiortc`` is required at session time (not import time); deployed WMA apps
receive it automatically in their runner environment.
"""

from fal.wma._errors import (
    AppError,
    Error,
    InputValueError,
    InternalServerError,
)
from fal.wma._raw import (
    INITIAL_CONNECT_TIMEOUT_SECONDS,
    SSE_KEEPALIVE,
    VIDEO_CLOCK_RATE,
    ClientOfferError,
    SessionSlot,
    close_peer_connection,
    filter_sdp_ice_candidates,
    make_video_queue_track,
    negotiate_answer,
    queue_put_drop_oldest,
    sse_event,
    wait_for_initial_connect,
    watch_connection_state,
    wma_session_stream,
)
from fal.wma.contract import (
    ASYNCAPI_DOCUMENT_VERSION,
    ASYNCAPI_SPEC_VERSION,
    ASYNCAPI_URL,
    CONTRACT_EXTENSION,
    CONTRACT_VERSION,
    OPENAPI_URL,
    SESSION_PROTOCOL,
    SESSION_PROTOCOL_VERSION,
    TRANSPORT_PROTOCOL,
    Constraint,
    MediaContract,
    RealtimeContract,
    Track,
    apply_contract,
    message_types,
    render_asyncapi,
)
from fal.wma.ice import (
    DEFAULT_STUN_URL,
    ICE_STATUS_BRIDGE_MANAGED,
    ICE_STATUS_MISCONFIGURED,
    ICE_STATUS_SERVER_MANAGED,
    ICE_STATUS_STUN_ONLY,
    ICE_STATUS_TURN,
    ICE_STATUS_UNREACHABLE,
    IceServerConfigError,
    IceServerProvider,
    RunnerIceConfig,
    build_rtc_ice_servers,
    ice_candidate_type_counts,
    ice_servers_for_aiortc,
    stun_only_ice_servers,
    validate_server_ice_servers,
)
from fal.wma.metered import (
    MeteredConfigError,
    MeteredError,
    MeteredFetchError,
    MeteredIceProvider,
    fetch_metered_credential,
    fetch_metered_ice_array,
    mint_ice_servers,
    parse_metered_ice_array,
    sanitize_metered_domain,
    synthesize_ice_servers,
    validate_forwarded_ice_servers,
)
from fal.wma.sdk import (
    DATA_CHANNEL_LABEL,
    START_SESSION_PATH,
    AiortcPeer,
    App,
    PeerBackend,
    Session,
    SessionAnswer,
    SessionParams,
    StartSessionRequest,
)
from fal.wma.telemetry import (
    CONNECTION_REPORT_VERSION,
    ConnectionReport,
    ConnectionReportObserver,
    observe_peer_connection,
)

__all__ = [
    # wire-shaped session errors (raise these from create_backend/handlers)
    "AppError",
    "Error",
    "InputValueError",
    "InternalServerError",
    # connection-oriented application surface
    "AiortcPeer",
    "App",
    "DATA_CHANNEL_LABEL",
    "PeerBackend",
    "START_SESSION_PATH",
    "Session",
    "SessionAnswer",
    "SessionParams",
    "StartSessionRequest",
    # ICE configuration and provider-neutral runner helpers
    "DEFAULT_STUN_URL",
    "ICE_STATUS_BRIDGE_MANAGED",
    "ICE_STATUS_MISCONFIGURED",
    "ICE_STATUS_SERVER_MANAGED",
    "ICE_STATUS_STUN_ONLY",
    "ICE_STATUS_TURN",
    "ICE_STATUS_UNREACHABLE",
    "IceServerConfigError",
    "IceServerProvider",
    "RunnerIceConfig",
    "build_rtc_ice_servers",
    "ice_candidate_type_counts",
    "ice_servers_for_aiortc",
    "stun_only_ice_servers",
    "validate_server_ice_servers",
    # Metered bridge validation and app-owned minting helpers
    "MeteredConfigError",
    "MeteredError",
    "MeteredFetchError",
    "MeteredIceProvider",
    "fetch_metered_credential",
    "fetch_metered_ice_array",
    "mint_ice_servers",
    "parse_metered_ice_array",
    "sanitize_metered_domain",
    "synthesize_ice_servers",
    "validate_forwarded_ice_servers",
    # linked OpenAPI / AsyncAPI contract declarations
    "ASYNCAPI_DOCUMENT_VERSION",
    "ASYNCAPI_SPEC_VERSION",
    "ASYNCAPI_URL",
    "CONTRACT_EXTENSION",
    "CONTRACT_VERSION",
    "OPENAPI_URL",
    "SESSION_PROTOCOL",
    "SESSION_PROTOCOL_VERSION",
    "TRANSPORT_PROTOCOL",
    "Constraint",
    "MediaContract",
    "RealtimeContract",
    "Track",
    "apply_contract",
    "message_types",
    "render_asyncapi",
    # runner connection telemetry
    "CONNECTION_REPORT_VERSION",
    "ConnectionReport",
    "ConnectionReportObserver",
    "observe_peer_connection",
    # raw (/start-session SSE path)
    "INITIAL_CONNECT_TIMEOUT_SECONDS",
    "SSE_KEEPALIVE",
    "VIDEO_CLOCK_RATE",
    "ClientOfferError",
    "SessionSlot",
    "close_peer_connection",
    "filter_sdp_ice_candidates",
    "make_video_queue_track",
    "negotiate_answer",
    "queue_put_drop_oldest",
    "sse_event",
    "wait_for_initial_connect",
    "watch_connection_state",
    "wma_session_stream",
]
