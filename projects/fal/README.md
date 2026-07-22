[![PyPI](https://img.shields.io/pypi/v/fal.svg?logo=PyPI)](https://pypi.org/project/fal)
[![Tests](https://img.shields.io/github/actions/workflow/status/fal-ai/fal/fal-unit-tests.yml?label=Tests)](https://github.com/fal-ai/fal/actions)

# fal

fal is a serverless Python runtime that lets you run and scale code in the cloud with no infra management.

With fal, you can build pipelines, serve ML models and scale them up to many users. You scale down to 0 when you don't use any resources.

For full product and platform documentation, see [fal.ai/docs](https://fal.ai/docs/documentation).

## Quickstart

Install the package and authenticate:

```bash
pip install fal
fal auth login
```

Create a minimal app:

```python
import fal


class MyApp(fal.App):
    @fal.endpoint("/")
    def run(self) -> dict:
        return {"message": "Hello, World!"}
```

Run it on fal for testing:

```bash
fal run hello_world.py::MyApp
```

Deploy it to a persistent endpoint:

```bash
fal deploy hello_world.py::MyApp
```

## Next steps

If you want to go deeper, start with:

- [Quick start](https://fal.ai/docs/documentation/development/getting-started/quick-start)
- [Deploy to production](https://fal.ai/docs/documentation/deployment/deploy-to-production)
- [Serverless documentation](https://fal.ai/docs/documentation/serverless)

## World Model Accelerator

The hosted WMA connection flow has not changed. A browser or other WebRTC
client still connects through the WMA service using the fal application ID,
and WMA forwards the offer to the application's `/start-session` endpoint.
What changed is the application API: `fal.wma.App` now owns that endpoint,
answer streaming, keepalives, session lifetime, message dispatch, and cleanup.
Applications only select and configure a media backend.

Do not implement `/start-session` yourself. The canonical shape is:

```python
import fal.wma as wma


class MyWmaApp(wma.App):
    async def create_backend(self, session: wma.Session) -> wma.PeerBackend:
        return ...
```

Run or deploy it like any other fal app:

```bash
fal run app.py::MyWmaApp
fal deploy app.py::MyWmaApp
```

Choose the backend at the media-processing boundary:

| Backend | Use it when |
| --- | --- |
| `VideoSourcePeer` | A stateful Python world model generates video from prompts, controls, or internal state. |
| `VideoProcessorPeer` | A Python world model consumes decoded frames and may need bounded batching. |
| `AiortcPeer` | Python needs custom control over aiortc tracks, transceivers, or data channels. |
| `GStreamerPeer` | Media should remain native or encoded, such as NVENC cloud gaming and relaying. |

Install the decoded-frame extras for Python world models:

```bash
pip install 'fal[wma]'
```

### Source-generating world models

`VideoSourcePeer` turns an async frame generator into a paced WebRTC video
track. It is the high-level path for autoregressive models that produce frame
blocks from a seed and control state without receiving an input video track:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

import fal.wma as wma
from aiortc import RTCConfiguration, RTCIceServer


class GeneratedWorld(wma.App):
    requirements = ["aiortc==1.15.0", "numpy"]
    machine_type = "GPU-H100"
    max_multiplexing = 1

    def setup(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.model = load_model()

    async def create_backend(self, session):
        controls = {}
        loop = asyncio.get_running_loop()

        session.on_message(
            "controls",
            lambda message: controls.update(message["state"]),
            inline=True,
        )
        session.on_channel_open(
            lambda: session.send(
                {"type": "session_info", "fps": 16},
            )
        )
        session.set_response_header("x-fal-billable-units", "0")
        session.answer_metadata["model"] = "generated-world"

        async def frames():
            while not session.closed.is_set():
                block = await loop.run_in_executor(
                    self.executor,
                    self.model.next_block,
                    dict(controls),
                )
                for frame in block:
                    yield frame

        configuration = RTCConfiguration(
            iceServers=[
                RTCIceServer(urls="stun:stun.example.com:3478"),
            ]
        )
        return wma.VideoSourcePeer(
            session,
            frames(),
            policy=wma.VideoSourcePolicy(
                fps=16,
                max_queue_size=12,
                overflow="drop_oldest",
                output_format="rgb24",
                initial_prefetch_frames=1,
            ),
            rtc_configuration=configuration,
            disconnected_grace_seconds=30,
        )
```

`rtc_configuration` is passed directly to aiortc's
`RTCPeerConnection(configuration=...)`. It configures ICE servers and WebRTC
bundle/data-channel negotiation—not model execution, regions, FPS, codecs, or
GPU selection. Omit it to use aiortc defaults. TURN credentials should come
from per-session secrets or a short-lived credential provider, not source code.

The source is consumed continuously in a session-owned reader task. Generated
frames enter a bounded queue, stale frames follow the configured overflow
policy, and `recv()` emits them at `fps` with a 90 kHz RTP timestamp clock.
PyAV frames are used directly; numpy arrays are converted off the event loop by
default. `VideoSourceTrack.stats` reports produced, output, dropped, queued,
latest queue-age, and latest pacing-sleep values. Ending or raising from the
source closes the media session, and session teardown closes the async
generator.

Source consumption starts when `VideoSourcePeer` is created, before aiortc asks
for its first frame. This lets a model prepare its first block while WebRTC is
negotiating. Set `start_immediately=False` to defer generator consumption until
the sender first calls `recv()`.

Generating blocks and pacing frames are separate operations. A model can yield
all frames from block N immediately, resume generation of block N+1, and let
the source track pace block N concurrently. Size `max_queue_size` for the
intended frame reservoir; overflow remains bounded if generation outruns
encoding or the network.

### Incoming-frame world models

`VideoProcessorPeer` turns an incoming browser video track into bounded,
batched model calls and sends the processed frames back on the same peer:

```python
import fal
import fal.wma as wma


class VideoWorldModel(wma.App):
    requirements = ["aiortc==1.15.0", "numpy"]
    machine_type = "GPU-A100"
    max_multiplexing = 1

    def process(self, frames):
        return [run_model(frame) for frame in frames]

    async def create_backend(self, session):
        return wma.VideoProcessorPeer(
            session,
            self.process,
            policy=wma.VideoProcessorPolicy(
                batch_size=2,
                max_queue_size=4,
                max_batch_wait_ms=20,
                overflow="drop_oldest",
                execution="worker",
                output_format="rgb24",
                processor_timeout_ms=30_000,
                shutdown_timeout_ms=5_000,
            ),
        )
```

The processor receives a list of PyAV video frames. It may return PyAV frames,
numpy arrays, one frame, an iterable of frames, or an awaitable producing those
values. Frames without timestamps inherit `pts` and `time_base` from the input
at the same batch position.

Batch waiting adds directly to latency. At 60 fps, two consecutive frames are
about 17 ms apart, so the example's 20 ms window can form a two-frame temporal
batch. At 30 fps, use at least 34 ms to do the same. Shorter windows still batch
frames opportunistically when model processing has created queue backlog.

The real-time defaults favor bounded latency:

- the input queue is bounded;
- old frames are dropped rather than allowed to accumulate;
- a partial batch runs when `max_batch_wait_ms` expires;
- synchronous processors run on a worker thread;
- `execution="event_loop"` supports async processors and deliberately
  event-loop-safe synchronous functions;
- outputs are capped at `batch_size` unless `max_output_frames` is set;
- a processor that exceeds `processor_timeout_ms` fails the media session;
- active worker calls are drained for up to `shutdown_timeout_ms` before
  model/session cleanup.

Use `wma.attach_video_processor()` when an application needs to configure its
own `AiortcPeer`. `VideoProcessorTrack.stats` reports input, output, dropped
frame, and batch counts.

`VideoProcessorPeer` does not create a data channel by default, so a video-only
browser offer works without a silent unusable channel. Set
`create_default_channel=True` only when the browser creates a data channel in
its offer.

Python cannot forcibly terminate a running thread. After a processor timeout,
WMA waits up to `shutdown_timeout_ms` for cooperative completion before
continuing teardown. Long-running synchronous processors should implement
their own cancellation/deadline checks or use process isolation.
`execution="event_loop"` must only be used for async processors or synchronous
functions that return immediately; a blocking synchronous event-loop function
cannot be preempted by an asyncio timeout.
With the default worker policy, processor invocation, lazy output
materialization, and numpy-to-PyAV conversion all remain off the event loop and
inside the processor timeout.

Decoded Python frames are appropriate when the model consumes pixels. For
gaming, relaying, or already-encoded model output, use a native backend such as
`GStreamerPeer` so capture, encoding, RTP, and WebRTC remain outside Python.

### Custom aiortc backend

Use `AiortcPeer` when `VideoProcessorPeer` is too opinionated but the media path
still belongs in Python:

```python
from typing import Any

import fal.wma as wma


class CustomMediaApp(wma.App):
    requirements = ["aiortc==1.15.0"]

    async def create_backend(self, session: wma.Session) -> wma.PeerBackend:
        session.on_message("control", self.handle_control)
        session.defer(self.close_resources)

        async def configure(peer_connection: Any) -> None:
            peer_connection.addTrack(self.create_video_track())

        return wma.AiortcPeer(
            session,
            configure,
            create_default_channel=False,
        )
```

`Session` is shared by every backend. Use `session.on_message()` for typed
data-channel messages, `session.send()` for server messages,
`session.create_task()` for session-owned background work, and
`session.defer()` for cleanup. `session.on_channel_open()` can send a hello or
interaction contract after the selected channel is ready.
`session.set_response_header()` configures the inherited `/start-session`
response, while `session.answer_metadata` adds fields to its first SSE event.
Set `create_default_channel=False` when the browser creates the channel in its
offer; enable it only when the offer contains an application media section that
can accept the server-created channel.

`AiortcPeer`, `VideoSourcePeer`, and `VideoProcessorPeer` accept either an
aiortc `RTCConfiguration` or a custom `peer_connection_factory`. Set
`disconnected_grace_seconds` to tolerate temporary network transitions; `0`
closes immediately and `None` waits for a terminal failed or closed state.

### Migrating older WMA applications

The former `RealtimeApp`, `EventHandler`, and `BatchedFnTrack` API and manually
implemented WMA endpoints are intentionally unsupported. Migrate each concern
directly:

| Older application concern | Current SDK |
| --- | --- |
| Custom `/start-session` route and SSE generator | Subclass `wma.App` |
| Per-connection state and cleanup lists | `wma.Session` |
| Message handlers | `session.on_message()` |
| Source-generated video | `wma.VideoSourcePeer` |
| Batched decoded-frame transform | `wma.VideoProcessorPeer` |
| Custom aiortc setup | `wma.AiortcPeer` |
| Native GStreamer/WebRTC pipeline | `wma.GStreamerPeer` |

The browser-side WMA integration does not need a compatibility mode: it still
sends an SDP offer and receives an SDP answer. Only the application-side
implementation changes.

## Install from source

From the repository root:

```bash
pip install -e 'projects/fal[dev]'
```

## Contributing

### Running tests

Use the smallest relevant scope first:

```bash
pytest -n auto -v projects/fal/tests/unit
```

### Pre-commit

Run the repository hooks before opening or finishing work:

```bash
pre-commit run --all-files
```

### Commit format

Please follow the [Conventional Commits](https://www.conventionalcommits.org/) specification for commit messages.
