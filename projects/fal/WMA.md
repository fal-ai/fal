# WebRTC session apps with `fal.wma`

`fal.wma` provides WebRTC session APIs for Python applications. Import it
explicitly. This API is experimental; pin your SDK version when deploying.

Install the optional local WebRTC and schema dependencies:

```sh
pip install 'fal[wma]'
```

Deploy [the CPU echo example](examples/wma_echo.py) with
`fal deploy examples/wma_echo.py::EchoApp` from this directory. WMA subclasses
automatically receive `aiortc` in their runner requirements; an explicit named app pin
takes precedence. Contract rendering, typed control messages and UI hints require
Pydantic 2. Core sessions, ICE and billing remain importable with Pydantic 1.

For a custom build, use a named requirement such as
`aiortc @ git+https://example.com/team/aiortc.git@revision`. When using an
unnamed local path or archive, also include `"aiortc"` in the app's requirements
alongside that path. This declares the package without imposing a version range
and suppresses the SDK's default requirement. The SDK does not inspect arbitrary
paths or run build backends to discover distribution names.

## Session lifecycle

Subclass `fal.wma.App` and implement `create_backend(session)`. Return an
`AiortcPeer` for runner-hosted WebRTC, or a `PeerBackend` implementing
`negotiate`, `wait_closed` and `close` for a custom transport. Bind any backend
resources to the session before work that can fail so setup failures clean up.

The WMA bridge forwards one complete offer to `POST /start-session`. The response
is SSE: an answer, optional bounded connection telemetry, then keepalives. That
HTTP connection owns the session lifetime. Closing it closes the backend and
settles deferred usage. Media and data flow over WebRTC. Do not add REST heartbeat
or close routes to this protocol. `AiortcPeer` uses the `control` data channel,
has a 35-second initial connection watchdog, and treats a disconnected state as
transient unless `disconnected_grace_seconds` is set. Failed/closed peers end the
session. Use the `wma()` extension for `fal.realtime.open()` in `@fal-ai/client` on
the browser side; SDK installation alone does not configure bridge access,
TURN service, endpoint pricing, or model deployment.

## Public modules

| Module | Purpose |
| --- | --- |
| `fal.wma.sdk` | App, Session, SessionParams, PeerBackend and AiortcPeer |
| `fal.wma.raw` | SSE lifecycle, SDP filtering, connection watchdogs and video queues |
| `fal.wma.protocol` | Typed control messages, semantic commands and key-state sampling |
| `fal.wma.contract` | Linked OpenAPI/AsyncAPI, validated examples and message presentation |
| `fal.wma.profile` | Configured-session and command-world interaction declarations |
| `fal.wma.ui` | Optional FieldUI and ExampleUI schema hints |
| `fal.wma.ice` / `fal.wma.metered` | Bridge-forwarded, environment or app-owned ICE configuration |
| `fal.wma.relay` | Ordered WebSocket signaling/control relay for partner-hosted media |
| `fal.wma.duration_billing` | Duration billing for `@fal.realtime` WebSocket handlers |
| `fal.wma.telemetry` | Bounded connection reports without addresses or credentials |
| `fal.wma.app` | Deprecated REST-lifecycle compatibility helpers; not root exports |

## ICE, billing and application responsibilities

Await `RunnerIceConfig.from_bridge().build_ice_servers_async(...)` with the
offer’s `ice_servers` and `ice_status` for bridge-provisioned TURN. It returns
`(rtc_ice_servers, turn_available)`; pass the server list into an aiortc
`RTCConfiguration`. The SDK validates forwarded ICE servers and strips
unsafe SDP targets; do not bypass either check in a deployment.

Call `session.add_billable_units(n)` for delivered usage. If cleanup determines
that some accumulated usage was undelivered, `session.cap_billable_units(n)` may
reduce it before finalization. `minimum_billable_units` is an optional per-app
floor on a successfully activated deferred settlement, not on setup failures.
The gateway request ID identifies the report; caller headers are not proof of
authorization. Keep any service credential server-side and authenticate custom
ingress before trusting forwarded identity. The base app declares `FAL_KEY` for
the billing reporter; subclasses overriding `secrets` must retain it alongside
their own required secrets. Failures before activation bill zero by default.
After close, adding/capping units raises an error. An exhausted billing report
retry is logged and needs operational follow-up; a unit test is not proof of
production settlement.

`session.http_request` exposes the original FastAPI request for app-specific
authentication. `billing_debug=True` enables diagnostic billing event logs;
leave it off unless needed. Configure multiplexing and model state isolation for
your workload, validate control inputs, and stop model work in backend cleanup.

## Deployment validation

Before release, verify a deployed session through the real bridge with the
intended client: connect, exchange controls/media, disconnect/reconnect, release
runner resources, and confirm the settled billing amount. Repeat from a clean
installation of the published SDK version before customer handoff.
