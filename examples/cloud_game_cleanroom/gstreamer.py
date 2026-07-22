from __future__ import annotations

import re
from dataclasses import dataclass

from engine import HEIGHT, WIDTH

DEFAULT_STUN_SERVER = "stun://stun.l.google.com:19302"
VIDEO_BITRATE_KBPS = 8_000
VIDEO_FPS = 60
MIN_VIDEO_FPS = 30
PROFILE_LEVEL_ID = re.compile(r"^[0-9a-fA-F]{6}$")
ENCODER_LEVEL = "3.2"
FRAME_MACROBLOCKS = ((WIDTH + 15) // 16) * ((HEIGHT + 15) // 16)
H264_LEVELS: dict[int, tuple[str, int]] = {
    0x1F: ("3.1", 108_000),
    0x20: ("3.2", 216_000),
    0x28: ("4", 245_760),
    0x29: ("4.1", 245_760),
    0x2A: ("4.2", 522_240),
    0x32: ("5", 589_824),
    0x33: ("5.1", 983_040),
    0x34: ("5.2", 2_073_600),
}
H264_PROFILE_PATTERNS = (
    (0x42, 0x4F, 0x40, "constrained-baseline"),
    (0x4D, 0x8F, 0x80, "constrained-baseline"),
    (0x58, 0xCF, 0xC0, "constrained-baseline"),
    (0x42, 0x4F, 0x00, "baseline"),
    (0x58, 0xCF, 0x80, "baseline"),
    (0x4D, 0xAF, 0x00, "main"),
    (0x64, 0xFF, 0x00, "high"),
)


@dataclass(frozen=True)
class H264Format:
    payload: int
    packetization_mode: str
    offered_profile_level_id: str
    profile_level_id: str
    level_asymmetry_allowed: str
    profile: str
    level: str
    fps: int


def payload_types(sdp: str) -> tuple[int, int | None]:
    video, audio = offer_formats(sdp)
    return video.payload, audio


def offer_formats(sdp: str) -> tuple[H264Format, int | None]:
    sections = sdp.replace("\r\n", "\n").split("\nm=")
    video_payload = _h264_format(sections)
    audio_payload = _payload_for_encoding(sections, "audio", "opus")
    if video_payload is None:
        raise RuntimeError(
            "The browser offer does not contain a compatible "
            "packetization-mode=1 H.264 format"
        )
    return video_payload, audio_payload


def _h264_format(sections: list[str]) -> H264Format | None:
    for index, raw_section in enumerate(sections):
        section = raw_section if index == 0 else "m=" + raw_section
        if not section.startswith("m=video "):
            continue
        codecs: dict[int, str] = {}
        parameters: dict[int, dict[str, str]] = {}
        for line in section.splitlines():
            if line.startswith("a=rtpmap:"):
                payload, _, codec = line.removeprefix("a=rtpmap:").partition(" ")
                try:
                    codecs[int(payload)] = codec.split("/", 1)[0]
                except ValueError:
                    continue
            elif line.startswith("a=fmtp:"):
                payload, _, raw_parameters = line.removeprefix("a=fmtp:").partition(
                    " "
                )
                try:
                    payload_number = int(payload)
                except ValueError:
                    continue
                parameters[payload_number] = _fmtp_parameters(raw_parameters)

        for payload, codec in codecs.items():
            fmtp = parameters.get(payload, {})
            profile_level_id = fmtp.get("profile-level-id", "").lower()
            if (
                codec.lower() != "h264"
                or fmtp.get("packetization-mode") != "1"
                or PROFILE_LEVEL_ID.fullmatch(profile_level_id) is None
            ):
                continue
            profile = _h264_profile(profile_level_id)
            if profile is None:
                continue
            asymmetry_allowed = (
                "1" if fmtp.get("level-asymmetry-allowed") == "1" else "0"
            )
            encoder_settings = _encoder_settings(profile_level_id)
            if encoder_settings is None:
                continue
            level, fps = encoder_settings
            return H264Format(
                payload=payload,
                packetization_mode="1",
                offered_profile_level_id=profile_level_id,
                profile_level_id=profile_level_id,
                level_asymmetry_allowed=asymmetry_allowed,
                profile=profile,
                level=level,
                fps=fps,
            )
    return None


def _fmtp_parameters(value: str) -> dict[str, str]:
    result = {}
    for item in value.split(";"):
        name, separator, parameter = item.strip().partition("=")
        if separator:
            result[name.lower()] = parameter.strip()
    return result


def _h264_profile(profile_level_id: str) -> str | None:
    profile_idc = int(profile_level_id[:2], 16)
    profile_iop = int(profile_level_id[2:4], 16)
    for expected_idc, mask, expected_iop, profile in H264_PROFILE_PATTERNS:
        if profile_idc == expected_idc and profile_iop & mask == expected_iop:
            return profile
    return None


def _encoder_settings(offered_profile_level_id: str) -> tuple[str, int] | None:
    offered_level_idc = int(offered_profile_level_id[4:], 16)
    level_limits = H264_LEVELS.get(offered_level_idc)
    if level_limits is None:
        return None
    level, max_macroblocks_per_second = level_limits
    # GStreamer 1.26 intersects the exact RTP level before applying WebRTC
    # level asymmetry, so the encoded stream must fit the offered level.
    fps = min(VIDEO_FPS, max_macroblocks_per_second // FRAME_MACROBLOCKS)
    if fps < MIN_VIDEO_FPS:
        return None
    return level, fps


def _payload_for_encoding(
    sections: list[str], media_kind: str, encoding: str
) -> int | None:
    prefix = f"m={media_kind} "
    for index, raw_section in enumerate(sections):
        section = raw_section if index == 0 else "m=" + raw_section
        if not section.startswith(prefix):
            continue
        mappings: dict[int, str] = {}
        packetization_mode_one: set[int] = set()
        for line in section.splitlines():
            if line.startswith("a=rtpmap:"):
                payload, _, codec = line.removeprefix("a=rtpmap:").partition(" ")
                try:
                    mappings[int(payload)] = codec.split("/", 1)[0]
                except ValueError:
                    continue
            elif line.startswith("a=fmtp:") and "packetization-mode=1" in line:
                payload = line.removeprefix("a=fmtp:").split(" ", 1)[0]
                try:
                    packetization_mode_one.add(int(payload))
                except ValueError:
                    continue
        matches = [
            payload
            for payload, codec in mappings.items()
            if codec.lower() == encoding.lower()
        ]
        if media_kind == "video":
            matches.sort(key=lambda payload: payload not in packetization_mode_one)
        return matches[0] if matches else None
    return None


def pipeline_description(
    *,
    display_name: str,
    pulse_server: str,
    pulse_monitor: str,
    video_payload: int,
    audio_payload: int | None,
    h264_profile: str = "constrained-baseline",
    h264_level: str = ENCODER_LEVEL,
    video_fps: int = VIDEO_FPS,
    cuda_conversion: bool = False,
) -> str:
    conversion = ""
    if cuda_conversion:
        conversion = """
            cudaupload name=uploader !
            cudaconvert name=converter !
            video/x-raw(memory:CUDAMemory),format=NV12 !
        """
    video = f"""
        ximagesrc name=source display-name={display_name}
            use-damage=false show-pointer=false !
        video/x-raw,format=BGRx,framerate={video_fps}/1 !
        queue max-size-buffers=1 leaky=downstream !
        {conversion}
        nvh264enc name=encoder zerolatency=true bframes=0 rc-lookahead=0
            bitrate={VIDEO_BITRATE_KBPS} gop-size={video_fps}
            vbv-buffer-size=133 repeat-sequence-header=true !
        h264parse name=parser config-interval=-1 !
        video/x-h264,profile={h264_profile},level=(string){h264_level} !
        rtph264pay name=payloader aggregate-mode=zero-latency config-interval=-1
            pt={video_payload} !
        queue max-size-buffers=2 leaky=downstream !
        application/x-rtp,media=video,encoding-name=H264,payload={video_payload} !
        peer.
    """
    audio = ""
    if audio_payload is not None:
        audio = f"""
            pulsesrc server={pulse_server} device={pulse_monitor} do-timestamp=true
                buffer-time=20000 latency-time=10000 !
            queue max-size-time=20000000 leaky=downstream !
            audioconvert ! audioresample !
            audio/x-raw,rate=48000,channels=2 !
            opusenc bitrate=128000 frame-size=10 perfect-timestamp=true
                inband-fec=true !
            rtpopuspay pt={audio_payload} !
            application/x-rtp,media=audio,encoding-name=OPUS,payload={audio_payload} !
            peer.
        """
    return f"""
        webrtcbin name=peer latency=0 bundle-policy=max-bundle
        {video}
        {audio}
    """
