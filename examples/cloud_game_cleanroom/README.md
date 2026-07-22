# Freedoom Cloud Game

A clean-room WMA cloud-gaming reference running a real, freely redistributable
game. Each session launches Crispy Doom 7.1 with Freedoom 0.13.0 inside an
isolated Xvfb display. On fal, GStreamer captures the display, encodes H.264
with NVENC, and sends video plus Opus audio directly through WebRTC. Browser
keyboard, Pointer Lock mouse, and controller input return over an unordered
data channel.

The app uses the transport-neutral `fal.wma.App` session runtime. The SDK owns
the `/start-session` body contract, SDP answer stream, SSE keepalives,
cancellation, message dispatch, task ownership, and idempotent cleanup.
There is no hand-written FastAPI endpoint or application-level SSE loop; the
existing hosted WMA client flow reaches the endpoint inherited from
`fal.wma.App`.
`fal.wma.GStreamerPeer` owns native WebRTC negotiation and the GStreamer
lifecycle, while this example supplies only the game process, offer-specific
media pipeline, controls, telemetry, and metadata. Local development can select
the independent `fal.wma.AiortcPeer` backend through the same session API.
Pixel-processing world models can instead use `fal.wma.VideoProcessorPeer` for
bounded queues and batching. This game intentionally uses `GStreamerPeer`
because decoding frames into Python would add copies and another encode step.

The runtime path is:

```text
Crispy Doom + Freedoom
  -> Xvfb
  -> GStreamer ximagesrc
  -> nvh264enc (60 fps, no B-frames, bounded queues)
  -> webrtcbin H.264 + Opus

browser key/mouse/gamepad events
  -> unordered WebRTC data channel
  -> XTest key, button, and relative-motion transitions
  -> Crispy Doom
```

## Run

From this directory:

```bash
FAL_URL_OUTPUT=all \
uvx --no-cache --from ../../projects/fal \
  fal run --no-cache app.py::CloudGame
```

The first `--no-cache` rebuilds the local fal SDK tool, including changes below
`projects/fal/src`; uv otherwise caches local directory tools based primarily
on project metadata. The second `--no-cache` prevents reuse of an older runner
image. The first run builds a custom GPU container with Crispy Doom, Freedoom,
GStreamer, PulseAudio, Xvfb, and XTest support. Both flags can be removed once
the SDK and image are stable.

`fal run` prints an ephemeral sync URL. Its path contains the application ID
that WMA needs. If the URL is:

```text
https://fal.run/example/cloud-game-cleanroom
```

use `example/cloud-game-cleanroom` as the application ID.

Serve the browser client in another terminal:

```bash
python3 -m http.server 4173 --directory web
```

Open `http://localhost:4173`, then provide:

- WMA endpoint: `https://wma.fal.run`
- application ID from the run output
- a fal key

The key remains in the page's memory and is not persisted. A production client
must exchange an authenticated application session for a short-lived token
instead of accepting a long-lived fal key in the browser.

The client requires HTTPS for remote signaling endpoints and asks for explicit
confirmation before sending a key to a non-`fal.run` host.

## Controls

- W/S or up/down: move forward/backward
- Mouse or left/right arrows: turn
- A/D: strafe
- Left mouse, Ctrl, or primary controller button: fire
- Space or secondary controller button: use
- Shift or controller shoulder button: run
- Escape: menu
- 1–7: weapon selection

Click **Capture controls** to enable Pointer Lock. Press Escape to release the
mouse. Held keys and mouse buttons are released automatically when the browser
loses focus or the session closes.

## Test

```bash
uvx --from pytest==9.1.1 \
  --with-editable ../../projects/fal \
  --with aiortc==1.15.0 \
  pytest -q tests

bun test tests/session.test.js
```

The real Linux runtime has a separate container smoke test:

```bash
docker build -t fal-freedoom-smoke .
docker run --rm \
  -v "$PWD:/work:ro" \
  -w /work \
  -e PYTHONPATH=/work \
  fal-freedoom-smoke \
  python tests/smoke_doom.py
```

It starts Xvfb and Crispy Doom, captures changing RGB frames, injects held input,
and verifies every child process exits during teardown.

`tests/smoke_native.py` is the GPU integration harness. Run it inside the final
fal image on a GPU runner; it creates a real aiortc browser peer and verifies
NVENC video, PulseAudio/Opus audio, ICE, the data channel, input injection, and
complete teardown. It intentionally is not part of ordinary CPU test
collection.

```bash
python tests/smoke_native.py
```

On a development machine without the Linux game runtime, tests and direct
imports automatically use the deterministic Orbit Breaker fallback. Force a
backend with:

```bash
FAL_CLOUD_GAME_BACKEND=orbit       # deterministic local fallback
FAL_CLOUD_GAME_BACKEND=doom        # require the real game runtime
FAL_CLOUD_GAME_TRANSPORT=aiortc    # CPU development path
FAL_CLOUD_GAME_TRANSPORT=gstreamer # require native NVENC WebRTC
FAL_CLOUD_GAME_STUN_SERVER=stun://stun.example.com:3478
FAL_CLOUD_GAME_TURN_SERVER=turns://user:secret@turn.example.com:5349
```

The fal runner defaults to `gstreamer`; local development defaults to `aiortc`.
The server uses `stun://stun.l.google.com:19302` by default. An empty
`FAL_CLOUD_GAME_STUN_SERVER` disables it. Keep TURN credentials in fal secrets,
not source control.
The optimized app prefers a `GPU-A6000` runner and falls back to
`GPU-RTXPRO6000`. Startup fails explicitly if NVENC, WebRTC, PulseAudio, or
required GStreamer elements are missing.

## Session isolation and teardown

`max_multiplexing = 1` gives each active session its own runner. Inside that
runner, every game uses a private temporary home directory for configuration,
audio runtime state, and saves. Xvfb listens only on the container's local Unix
socket. When the WebRTC connection or WMA stream closes, the app releases held
controls, stops GStreamer and PulseAudio, terminates Crispy Doom and Xvfb, and
deletes the temporary directory.

## Licensing

The container builds Crispy Doom 7.1 from its GPL-2.0 source release and
installs Freedoom 0.13.0 from Debian. Freedoom's game content is BSD-3-Clause.
The original proprietary Doom WADs are not downloaded or included.

The image preserves Crispy Doom's GPL license and the exact corresponding
source archive under `/opt/crispy-doom/share`. If you redistribute the image or
binary, preserve those files and satisfy the GPL-2.0 corresponding-source
requirements. Preserve Freedoom's copyright and BSD license when redistributing
its game assets.

## Performance boundary

```text
default: game -> Xvfb -> ximagesrc BGRx -> NVENC -> WebRTC
optional: ximagesrc BGRx -> CUDA upload/convert -> CUDA NV12 -> NVENC
ideal:   game -> Gamescope -> DMA-BUF/CUDA memory -> NVENC -> WebRTC
```

The default path lets NVENC consume the X11 BGRx frame directly, avoiding the
CPU `videoconvert` pass. Set `CloudGame.cuda_conversion = True` to instead
upload BGRx once and perform the NV12 conversion in CUDA. A runner environment
can override that class default with `FAL_CLOUD_GAME_CUDA_CONVERT=0` or `1`.
The image installs NVRTC and its unversioned loader link so GStreamer's
`cudaconvert` factory is available when this option is enabled.
The native path removes Python frame copies and software H.264 encoding, bounds
every live-video queue, sends input on changes, captures relative mouse motion,
and adds low-buffer Opus audio.

Gamescope/DMA-BUF and `/dev/uinput` are deliberately not required because
ordinary fal containers do not promise a DRM seat or the kernel input device.
Those interfaces remain the next zero-copy optimization for a runner profile
that exposes both devices. Network round trip is independent of this pipeline:
a 211 ms RTT still imposes roughly 105 ms before an input can reach the runner.

## Deployment

Once the ephemeral run is working:

```bash
uvx --from ../../projects/fal \
  fal deploy app.py::CloudGame
```

The app currently allows one active session. Increase `max_concurrency` to
allow the platform to create more one-session runners. Keep
`max_multiplexing = 1` for per-session isolation.
