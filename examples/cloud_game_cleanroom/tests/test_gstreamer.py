from __future__ import annotations

import unittest

from gstreamer import (
    VIDEO_FPS,
    offer_formats,
    payload_types,
    pipeline_description,
)


BROWSER_OFFER = """v=0
m=video 9 UDP/TLS/RTP/SAVPF 96 102 103
a=rtpmap:96 VP8/90000
a=rtpmap:102 H264/90000
a=fmtp:102 packetization-mode=0;profile-level-id=42001f
a=rtpmap:103 H264/90000
a=fmtp:103 packetization-mode=1;profile-level-id=42e01f;level-asymmetry-allowed=1
m=audio 9 UDP/TLS/RTP/SAVPF 111 0
a=rtpmap:111 opus/48000/2
a=rtpmap:0 PCMU/8000
m=application 9 UDP/DTLS/SCTP webrtc-datachannel
"""


class PayloadTypeTests(unittest.TestCase):
    def test_prefers_packetization_mode_one_h264_and_finds_opus(self) -> None:
        self.assertEqual(payload_types(BROWSER_OFFER), (103, 111))
        video, _ = offer_formats(BROWSER_OFFER)
        self.assertEqual(video.offered_profile_level_id, "42e01f")
        self.assertEqual(video.profile_level_id, "42e01f")
        self.assertEqual(video.profile, "constrained-baseline")
        self.assertEqual(video.level, "3.1")
        self.assertEqual(video.fps, 52)
        self.assertEqual(video.level_asymmetry_allowed, "1")

    def test_audio_is_optional(self) -> None:
        offer = BROWSER_OFFER.replace(
            "m=audio 9 UDP/TLS/RTP/SAVPF 111 0\n"
            "a=rtpmap:111 opus/48000/2\n"
            "a=rtpmap:0 PCMU/8000\n",
            "",
        )

        self.assertEqual(payload_types(offer), (103, None))

    def test_h264_is_required(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "H.264"):
            payload_types(BROWSER_OFFER.replace("H264", "VP9"))

    def test_rejects_unsupported_profile(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "H.264"):
            offer_formats(BROWSER_OFFER.replace("42e01f", "6e001f"))

    def test_preserves_baseline_instead_of_relabeling_it(self) -> None:
        video, _ = offer_formats(BROWSER_OFFER.replace("42e01f", "42001f"))

        self.assertEqual(video.profile, "baseline")
        self.assertEqual(video.profile_level_id, "42001f")

    def test_rejects_a_level_too_low_for_real_time_capture(self) -> None:
        offer = BROWSER_OFFER.replace(
            "profile-level-id=42e01f;level-asymmetry-allowed=1",
            "profile-level-id=42e01e;level-asymmetry-allowed=1",
        )

        with self.assertRaisesRegex(RuntimeError, "H.264"):
            offer_formats(offer)

    def test_uses_offered_compatible_level_without_asymmetry(self) -> None:
        offer = BROWSER_OFFER.replace(
            "profile-level-id=42e01f;level-asymmetry-allowed=1",
            "profile-level-id=42e029;level-asymmetry-allowed=0",
        )

        video, _ = offer_formats(offer)
        self.assertEqual(video.profile_level_id, "42e029")
        self.assertEqual(video.level, "4.1")
        self.assertEqual(video.fps, VIDEO_FPS)


class PipelineTests(unittest.TestCase):
    def test_native_pipeline_is_hardware_encoded_and_bounded(self) -> None:
        description = pipeline_description(
            display_name=":99",
            pulse_server="unix:/tmp/pulse.sock",
            pulse_monitor="fal_game.monitor",
            video_payload=103,
            audio_payload=111,
        )

        self.assertIn(
            f"video/x-raw,format=BGRx,framerate={VIDEO_FPS}/1", description
        )
        self.assertIn("max-size-buffers=1 leaky=downstream", description)
        self.assertNotIn("videoconvert", description)
        self.assertNotIn("cudaupload", description)
        self.assertNotIn("cudaconvert", description)
        self.assertNotIn("memory:CUDAMemory", description)
        self.assertIn(
            "nvh264enc name=encoder zerolatency=true bframes=0 rc-lookahead=0",
            description,
        )
        self.assertIn("aggregate-mode=zero-latency", description)
        self.assertIn(
            "video/x-h264,profile=constrained-baseline,level=(string)3.2",
            description,
        )
        self.assertIn(
            "application/x-rtp,media=video,encoding-name=H264,payload=103",
            description,
        )
        self.assertNotIn("profile-level-id=(string)", description)
        self.assertNotIn("level-asymmetry-allowed=(string)", description)
        self.assertIn(
            "pulsesrc server=unix:/tmp/pulse.sock device=fal_game.monitor",
            description,
        )
        self.assertIn("opusenc bitrate=128000 frame-size=10", description)
        self.assertIn("webrtcbin name=peer latency=0", description)

    def test_audio_branch_is_omitted_when_offer_has_no_opus(self) -> None:
        description = pipeline_description(
            display_name=":99",
            pulse_server="unix:/tmp/pulse.sock",
            pulse_monitor="fal_game.monitor",
            video_payload=103,
            audio_payload=None,
        )

        self.assertNotIn("pulsesrc", description)
        self.assertNotIn("opusenc", description)

    def test_cuda_conversion_is_an_explicit_option(self) -> None:
        description = pipeline_description(
            display_name=":99",
            pulse_server="unix:/tmp/pulse.sock",
            pulse_monitor="fal_game.monitor",
            video_payload=103,
            audio_payload=111,
            cuda_conversion=True,
        )

        self.assertIn("video/x-raw,format=BGRx", description)
        self.assertIn("cudaupload name=uploader", description)
        self.assertIn("cudaconvert name=converter", description)
        self.assertIn(
            "video/x-raw(memory:CUDAMemory),format=NV12", description
        )

    def test_level_31_pipeline_uses_compatible_capture_rate_and_caps(self) -> None:
        description = pipeline_description(
            display_name=":99",
            pulse_server="unix:/tmp/pulse.sock",
            pulse_monitor="fal_game.monitor",
            video_payload=103,
            audio_payload=111,
            h264_level="3.1",
            video_fps=52,
        )

        self.assertIn("video/x-raw,format=BGRx,framerate=52/1", description)
        self.assertIn("gop-size=52", description)
        self.assertIn("level=(string)3.1", description)
        self.assertNotIn("profile-level-id=(string)", description)


if __name__ == "__main__":
    unittest.main()
