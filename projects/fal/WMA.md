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

## Playground and realtime documentation

Declare `realtime_contract` on every app intended for the WMA playground. The
SDK publishes an AsyncAPI document and links it from the `/start-session`
OpenAPI path through `x-fal-realtime`. The playground uses this discovery
metadata to show Connect/Disconnect, control-message forms, examples and media
instead of the raw SDP request form. The echo example above includes a contract.

Use `RealtimeContract(client_messages=..., server_messages=..., media=...)` to
describe the messages and tracks your backend actually supports. Message
schemas are Pydantic models; `MessageExample` supplies sample payloads. Media
directions are from the client's perspective: `MediaContract.send` describes
tracks the browser sends, while `receive` describes tracks the app produces.
For example, a camera echo app declares a required `Track(kind="video",
source="camera", required=True)` in `send` and `Track(kind="video")` in `receive`.
A control-only app uses an empty `MediaContract()`.

After adding or changing a contract, redeploy the app to update its metadata,
then refresh the playground. Verify that the AsyncAPI link and WMA controls
appear and exercise a real session. WebRTC transport can run without a contract,
but the SDK cannot infer arbitrary message handlers or media tracks; omitting
the contract leaves the generic HTTP playground. Contract declarations describe
the API and do not validate incoming messages at runtime.

## Examples

These standalone CPU apps cover three media patterns as well as basic control
messages. Install `fal[wma]` locally before deploying; the examples with typed
contracts require Pydantic 2. Each app generates its own playground metadata.

| Example | Try in the playground | Media flow |
| --- | --- | --- |
| [EchoApp](examples/wma_echo.py) | Send a text message and receive it back | Control only |
| [VideoApp](examples/wma_video.py) | Change the palette and speed of an animated color field; pause and resume | Video from app to browser, controls both ways |
| [FilterApp](examples/wma_filter.py) | Switch your camera between mirror, monochrome and edge effects; adjust the blend | Camera to app, processed video back |
| [MotionApp](examples/wma_motion.py) | Move in front of the camera and watch brightness and changed-pixel fractions in the message log; adjust sensitivity | Camera to app, measurements back as data |

For example, from this directory:

```sh
fal deploy examples/wma_video.py::VideoApp --auth private
fal deploy examples/wma_filter.py::FilterApp --auth private
fal deploy examples/wma_motion.py::MotionApp --auth private
```

Open the deployed app's `/start-session` playground and select **Connect**.
The camera examples request browser camera permission; the animated scene does
not need camera or microphone access. Use the example selector and **Send
message** to update live settings. Motion measurements appear in the server
message log, with no return video. These measurements use pixel differences,
not an object detector or learned model.

The three media examples close sessions after two minutes, allow at most two
sessions per runner, and scale down when idle. Their processing state belongs
to each session. `session.create_task()` owns the motion reader and expiration
tasks; `AiortcPeer` closes media resources on teardown. Frame conversion uses
fixed resolutions, and the motion reader drains every incoming frame while
reporting at most five times per second to avoid a growing input queue. Runner
compute may incur charges; disconnect when finished.

## Session lifecycle

Subclass `fal.wma.App` and implement `create_backend(session)`. Return an
`AiortcPeer` for runner-hosted WebRTC, or a `PeerBackend` implementing
`negotiate`, `wait_closed` and `close` for a custom transport. Bind any backend
resources to the session before work that can fail so setup failures clean up.
Calling `session.bind_backend(backend)` before returning that same backend is
safe. `AiortcPeer` stops outbound source tracks attached to its peer on close;
attach a per-session relay subscription when sharing a source across sessions.

On the session event loop, `session.send(message)` returns `False` when the
control channel is unavailable or accepting the message would exceed its 1 MiB
outbound buffer limit. Handle that result by dropping stale updates or retrying
later. Worker threads use a bounded 64-message handoff and return `False` when
it is full or the session loop has closed. Their `True` result acknowledges
queue acceptance, not transport acceptance. Pending messages are discarded on
session close; delivery batches yield so producers cannot starve other loop work.

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
