import asyncio
import os
from pathlib import Path
from typing import Any, ClassVar

import fal
import fal.wma as wma
from fal.wma import GStreamerPeer, PipelineSpec

from doom import DoomSession, create_game_backend
from gstreamer import DEFAULT_STUN_SERVER, offer_formats, pipeline_description

APP_DIR = Path(__file__).resolve().parent


class CloudGame(wma.App, name="cloud-game-cleanroom"):
    cuda_conversion: ClassVar[bool] = False
    local_python_modules = ["doom", "engine", "gstreamer"]
    image = fal.ContainerImage.from_dockerfile(
        str(APP_DIR / "Dockerfile"), context_dir=APP_DIR
    )
    machine_type = ["GPU-A6000", "GPU-RTXPRO6000"]
    min_concurrency = 0
    max_concurrency = 1
    max_multiplexing = 1
    keep_alive = 300
    request_timeout = 3660
    regions = ["eu-north", "eu-west"]

    def setup(self) -> None:
        if os.getenv("FAL_CLOUD_GAME_BACKEND", "doom").lower() != "orbit":
            DoomSession.require_available()
        if _native_streaming_enabled():
            required = [
                "ximagesrc",
                "nvh264enc",
                "h264parse",
                "rtph264pay",
                "pulsesrc",
                "opusenc",
                "rtpopuspay",
            ]
            if _cuda_conversion_enabled(self.cuda_conversion):
                required.extend(("cudaupload", "cudaconvert"))
            GStreamerPeer.require_available(required)
        else:
            try:
                import aiortc  # noqa: F401, PLC0415
            except ImportError as error:
                raise RuntimeError(
                    "The optional aiortc development backend is unavailable; "
                    "install aiortc or select FAL_CLOUD_GAME_TRANSPORT=gstreamer"
                ) from error

    async def create_backend(self, session: wma.Session) -> wma.PeerBackend:
        if _native_streaming_enabled():
            return await _create_native_backend(
                session,
                cuda_conversion=_cuda_conversion_enabled(self.cuda_conversion),
            )
        return await _create_aiortc_backend(session)


async def _create_aiortc_backend(session: wma.Session) -> wma.AiortcPeer:
    game = await asyncio.to_thread(create_game_backend)
    session.defer(lambda: asyncio.to_thread(game.close))
    _bind_game_messages(session, game)
    session.create_task(_send_telemetry(session, game))

    async def configure(peer_connection: Any) -> None:
        transceiver = peer_connection.addTransceiver(
            "video",
            direction="sendonly",
        )
        transceiver.sender.replaceTrack(_create_game_video_track(game))
        _prefer_h264(transceiver)

    return wma.AiortcPeer(
        session,
        configure,
        create_default_channel=False,
    )


async def _create_native_backend(
    session: wma.Session,
    *,
    cuda_conversion: bool,
) -> GStreamerPeer:
    game = await asyncio.to_thread(
        create_game_backend,
        capture=False,
        audio=True,
    )
    if not isinstance(game, DoomSession) or game.pulse_server is None:
        game.close()
        raise RuntimeError(
            "GStreamer transport requires the Doom backend with session audio"
        )

    session.defer(lambda: asyncio.to_thread(game.close))
    _bind_game_messages(session, game)
    session.create_task(_send_telemetry(session, game))

    def build_pipeline(offer: wma.StartSessionRequest) -> PipelineSpec:
        video, audio_payload = offer_formats(offer.sdp)
        description = pipeline_description(
            display_name=game.display_name,
            pulse_server=game.pulse_server,
            pulse_monitor="fal_game.monitor",
            video_payload=video.payload,
            audio_payload=audio_payload,
            h264_profile=video.profile,
            h264_level=video.level,
            video_fps=video.fps,
            cuda_conversion=cuda_conversion,
        )
        metadata: dict[str, Any] = {
            "video": {
                "width": game.width,
                "height": game.height,
                "fps": video.fps,
                "encoder": "nvenc",
                "conversion": "cuda" if cuda_conversion else "direct-bgrx",
            }
        }
        if audio_payload is not None:
            metadata["audio"] = {"codec": "opus"}
        diagnostic_pads = [
            ("source", "src"),
            ("encoder", "sink"),
            ("encoder", "src"),
            ("parser", "src"),
            ("payloader", "src"),
        ]
        if cuda_conversion:
            diagnostic_pads[1:1] = [
                ("uploader", "src"),
                ("converter", "src"),
            ]
        return PipelineSpec(
            description,
            metadata=metadata,
            diagnostic_pads=tuple(diagnostic_pads),
        )

    return GStreamerPeer(
        session,
        build_pipeline,
        channel_labels={"input"},
        stun_server=(
            os.getenv("FAL_CLOUD_GAME_STUN_SERVER", DEFAULT_STUN_SERVER) or None
        ),
        turn_server=os.getenv("FAL_CLOUD_GAME_TURN_SERVER") or None,
    )


def _bind_game_messages(session: wma.Session, game: Any) -> None:
    session.on_message("input", game.apply_input, inline=True)

    async def restart(_message: dict) -> None:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, game.restart)
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError:
            try:
                await asyncio.shield(future)
            except Exception:
                pass
            raise

    session.on_message("restart", restart)


async def _send_telemetry(session: wma.Session, game: Any) -> None:
    while not session.closed.is_set():
        session.send({"type": "game_state", **game.snapshot()})
        try:
            await asyncio.wait_for(session.closed.wait(), timeout=0.25)
        except TimeoutError:
            continue


def _prefer_h264(transceiver: Any) -> None:
    from aiortc import RTCRtpSender

    codecs = [
        codec
        for codec in RTCRtpSender.getCapabilities("video").codecs
        if codec.mimeType.lower() == "video/h264"
    ]
    if codecs:
        transceiver.setCodecPreferences(codecs)


def _create_game_video_track(game: Any) -> Any:
    from aiortc import VideoStreamTrack

    class GameVideoTrack(VideoStreamTrack):
        kind = "video"

        async def recv(self) -> Any:
            import av

            pts, time_base = await self.next_timestamp()
            frame = av.VideoFrame(
                width=game.width,
                height=game.height,
                format="rgb24",
            )
            frame.planes[0].update(await asyncio.to_thread(game.read_rgb))
            frame.pts = pts
            frame.time_base = time_base
            return frame

    return GameVideoTrack()


def _native_streaming_enabled() -> bool:
    selected = os.getenv("FAL_CLOUD_GAME_TRANSPORT", "auto").lower()
    if selected not in {"auto", "aiortc", "gstreamer"}:
        raise ValueError(
            "FAL_CLOUD_GAME_TRANSPORT must be auto, aiortc, or gstreamer"
        )
    if selected == "gstreamer":
        return True
    if selected == "aiortc":
        return False
    return bool(os.getenv("IS_ISOLATE_AGENT")) and os.getenv(
        "FAL_CLOUD_GAME_BACKEND",
        "doom",
    ).lower() != "orbit"


def _cuda_conversion_enabled(default: bool = False) -> bool:
    configured = os.getenv("FAL_CLOUD_GAME_CUDA_CONVERT")
    if configured is None:
        return default
    value = configured.strip().lower()
    if value in {"0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    raise ValueError("FAL_CLOUD_GAME_CUDA_CONVERT must be a boolean value")
