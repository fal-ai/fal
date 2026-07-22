# WMA grayscale

A minimal end-to-end incoming-video transform using the current `fal.wma`
runtime. The browser sends a camera track, the runner neutralizes the YUV chroma
planes, and aiortc sends the grayscale video track back.

## Run

From this directory:

```bash
FAL_URL_OUTPUT=all \
uvx --no-cache --from ../../projects/fal \
  fal run --no-cache app.py::GrayscaleApp
```

Use the application ID printed by `fal run` with the existing WMA browser
client and `https://wma.fal.run`. The browser offer must make its camera video
transceiver `sendrecv` so it can receive the processed track.

The app uses aiortc's default ICE configuration. Add an explicit STUN server
only when the deployment requires one:

```bash
FAL_WMA_STUN_SERVER=stun:stun.example.com:3478
```

TURN is optional:

```bash
FAL_WMA_TURN_SERVER=turns:turn.example.com:5349
FAL_WMA_TURN_USERNAME=user
FAL_WMA_TURN_CREDENTIAL=secret
```

Keep TURN credentials in fal secrets rather than source code.

## Performance

The transform stays in YUV420P, preserves the luma plane, and replaces only the
two chroma planes. It avoids RGB/BGR and numpy round trips. The input queue has
capacity one and drops the oldest frame, so processing cannot build a stale
latency backlog.
