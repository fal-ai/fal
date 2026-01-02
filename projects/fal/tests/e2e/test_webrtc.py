"""
WebRTC tests using the new @fal.webrtc decorator interface.

This demonstrates the improved API that:
- Handles signaling automatically
- Exposes RTCPeerConnection directly
- Separates control messages from signaling
- Provides convenient STUN configuration
"""

import asyncio
import json
import math
from fractions import Fraction

import numpy as np
import websockets
from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from av import AudioFrame, VideoFrame
from fastapi import WebSocket

import fal  # type: ignore[import-not-found]
from fal.app import AppClient


# Reusable track classes
class SineAudioTrack(MediaStreamTrack):
    """Simple synthetic audio track producing a sine wave."""

    kind = "audio"

    def __init__(self, freq=440, rate=48000):
        super().__init__()
        self.freq = freq
        self.rate = rate
        self._t = 0

    async def recv(self):
        samples = 960
        t = (np.arange(samples) + self._t) / self.rate
        waveform = 0.2 * np.sin(2 * math.pi * self.freq * t)
        pcm = np.int16(waveform * 32767)

        frame = AudioFrame(format="s16", layout="mono", samples=samples)
        frame.sample_rate = self.rate
        frame.pts = self._t
        frame.time_base = Fraction(1, self.rate)
        frame.planes[0].update(pcm.tobytes())

        self._t += samples
        return frame


class ColorVideoTrack(MediaStreamTrack):
    """Simple synthetic video track producing colored frames."""

    kind = "video"

    def __init__(self, width=640, height=480, color=(255, 0, 0), fps=30):
        super().__init__()
        self.width = width
        self.height = height
        self.color = color
        self.fps = fps
        self._frame_count = 0

    async def recv(self):
        pts = self._frame_count * (1 / self.fps)
        time_base = Fraction(1, self.fps)

        frame = VideoFrame(width=self.width, height=self.height, format="rgb24")
        frame.pts = int(pts / float(time_base))
        frame.time_base = time_base

        arr = np.frombuffer(frame.planes[0], dtype=np.uint8)
        arr[:] = np.tile(self.color, self.width * self.height)

        self._frame_count += 1
        await asyncio.sleep(1 / self.fps)
        return frame


# Test 1: Unidirectional Audio
class UnidirectionalAudioApp(fal.App):
    """Remote peer sends audio using the new @fal.webrtc interface."""

    requirements = ["aiortc", "av", "numpy"]

    @fal.webrtc("/audio")
    async def audio_session(self, pc: RTCPeerConnection, ws: WebSocket):
        """
        With @fal.webrtc:
        - pc is pre-configured with STUN
        - Signaling is handled automatically
        - ws only receives control messages (not offer/answer/ICE)
        """
        # Add outgoing audio track
        pc.addTrack(SineAudioTrack(freq=880))

        # Handle incoming tracks
        received = asyncio.Event()

        @pc.on("track")
        async def on_track(track):
            try:
                await track.recv()
                received.set()
            except Exception:
                pass

        # Wait for media exchange
        await asyncio.wait_for(received.wait(), timeout=10)


def test_unidirectional_audio():
    """Test unidirectional audio with new @fal.webrtc interface."""

    with AppClient.connect(UnidirectionalAudioApp) as client:
        ws_url = client.url.replace("http", "ws") + "/audio"

        async def run():
            ice_servers = [RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
            config = RTCConfiguration(iceServers=ice_servers)
            pc = RTCPeerConnection(configuration=config)

            pc.addTrack(SineAudioTrack(freq=440))

            received_track = asyncio.Queue()

            @pc.on("track")
            async def on_track(track):
                received_track.put_nowait(track)

            # ICE handling
            candidates_to_send = []

            @pc.on("icecandidate")
            def on_ice(candidate):
                if candidate:
                    candidates_to_send.append(candidate)

            done = asyncio.Event()

            @pc.on("icegatheringstatechange")
            def on_gathering_state_change():
                if pc.iceGatheringState == "complete":
                    done.set()

            async with websockets.connect(ws_url) as ws:
                # Signaling
                offer = await pc.createOffer()
                await pc.setLocalDescription(offer)
                await asyncio.wait_for(done.wait(), timeout=10)

                await ws.send(
                    json.dumps({"type": "offer", "sdp": pc.localDescription.sdp})
                )

                for cand in candidates_to_send:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "icecandidate",
                                "candidate": {
                                    "candidate": cand.candidate,
                                    "sdpMid": cand.sdpMid,
                                    "sdpMLineIndex": cand.sdpMLineIndex,
                                },
                            }
                        )
                    )

                msg = json.loads(await ws.recv())
                assert msg.get("type") == "answer"

                await pc.setRemoteDescription(
                    RTCSessionDescription(sdp=msg["sdp"], type="answer")
                )

                # Verify media
                track = await asyncio.wait_for(received_track.get(), timeout=15)
                assert track.kind == "audio"

                frame = await asyncio.wait_for(track.recv(), timeout=10)
                arr = frame.to_ndarray()
                assert arr.shape[0] > 0

                await ws.send(json.dumps({"type": "close"}))
                await asyncio.sleep(0.5)

            await pc.close()

        asyncio.run(run())


# Test 2: Unidirectional Video
class UnidirectionalVideoApp(fal.App):
    """Remote peer sends video."""

    requirements = ["aiortc", "av", "numpy"]

    @fal.webrtc("/video")
    async def video_session(self, pc: RTCPeerConnection, ws: WebSocket):
        # Add blue video
        pc.addTrack(ColorVideoTrack(width=320, height=240, color=(0, 0, 255)))

        received = asyncio.Event()

        @pc.on("track")
        async def on_track(track):
            try:
                await track.recv()
                received.set()
            except Exception:
                pass

        await asyncio.wait_for(received.wait(), timeout=10)

        # Keep connection alive until client sends close
        try:
            async for _ in ws.iter_text():
                pass
        except Exception:
            pass


def test_unidirectional_video():
    """Test unidirectional video."""

    with AppClient.connect(UnidirectionalVideoApp) as client:
        ws_url = client.url.replace("http", "ws") + "/video"

        async def run():
            ice_servers = [RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
            config = RTCConfiguration(iceServers=ice_servers)
            pc = RTCPeerConnection(configuration=config)

            pc.addTrack(ColorVideoTrack(width=320, height=240, color=(255, 0, 0)))

            received_track = asyncio.Queue()

            @pc.on("track")
            async def on_track(track):
                received_track.put_nowait(track)

            candidates_to_send = []

            @pc.on("icecandidate")
            def on_ice(candidate):
                if candidate:
                    candidates_to_send.append(candidate)

            done = asyncio.Event()

            @pc.on("icegatheringstatechange")
            def on_gathering_state_change():
                if pc.iceGatheringState == "complete":
                    done.set()

            async with websockets.connect(ws_url) as ws:
                offer = await pc.createOffer()
                await pc.setLocalDescription(offer)
                await asyncio.wait_for(done.wait(), timeout=10)

                await ws.send(
                    json.dumps({"type": "offer", "sdp": pc.localDescription.sdp})
                )

                for cand in candidates_to_send:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "icecandidate",
                                "candidate": {
                                    "candidate": cand.candidate,
                                    "sdpMid": cand.sdpMid,
                                    "sdpMLineIndex": cand.sdpMLineIndex,
                                },
                            }
                        )
                    )

                msg = json.loads(await ws.recv())
                assert msg.get("type") == "answer"

                await pc.setRemoteDescription(
                    RTCSessionDescription(sdp=msg["sdp"], type="answer")
                )

                track = await asyncio.wait_for(received_track.get(), timeout=15)
                assert track.kind == "video"

                frame = await asyncio.wait_for(track.recv(), timeout=10)
                arr = frame.to_ndarray(format="rgb24")
                assert arr.shape == (240, 320, 3)
                assert arr[0, 0, 2] > 250  # Blue

                await ws.send(json.dumps({"type": "close"}))
                await asyncio.sleep(0.5)

            await pc.close()

        asyncio.run(run())


# Test 3: Bidirectional Audio (Phone Call)
class BidirectionalAudioApp(fal.App):
    """Bidirectional audio session."""

    requirements = ["aiortc", "av", "numpy"]

    @fal.webrtc("/audio_bidi")
    async def audio_session(self, pc: RTCPeerConnection, ws: WebSocket):
        # Both send and receive audio
        pc.addTrack(SineAudioTrack(freq=1000))

        received = asyncio.Event()

        @pc.on("track")
        async def on_track(track):
            try:
                frame = await track.recv()
                arr = np.frombuffer(frame.planes[0].to_bytes(), dtype=np.int16)
                if arr.shape[0] > 0:
                    received.set()
            except Exception:
                pass

        await asyncio.wait_for(received.wait(), timeout=15)


def test_bidirectional_audio():
    """Test bidirectional audio (phone call)."""

    with AppClient.connect(BidirectionalAudioApp) as client:
        ws_url = client.url.replace("http", "ws") + "/audio_bidi"

        async def run():
            ice_servers = [RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
            config = RTCConfiguration(iceServers=ice_servers)
            pc = RTCPeerConnection(configuration=config)

            pc.addTrack(SineAudioTrack(freq=440))

            received_track = asyncio.Queue()

            @pc.on("track")
            async def on_track(track):
                received_track.put_nowait(track)

            candidates_to_send = []

            @pc.on("icecandidate")
            def on_ice(candidate):
                if candidate:
                    candidates_to_send.append(candidate)

            done = asyncio.Event()

            @pc.on("icegatheringstatechange")
            def on_gathering_state_change():
                if pc.iceGatheringState == "complete":
                    done.set()

            async with websockets.connect(ws_url) as ws:
                offer = await pc.createOffer()
                await pc.setLocalDescription(offer)
                await asyncio.wait_for(done.wait(), timeout=10)

                await ws.send(
                    json.dumps({"type": "offer", "sdp": pc.localDescription.sdp})
                )

                for cand in candidates_to_send:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "icecandidate",
                                "candidate": {
                                    "candidate": cand.candidate,
                                    "sdpMid": cand.sdpMid,
                                    "sdpMLineIndex": cand.sdpMLineIndex,
                                },
                            }
                        )
                    )

                msg = json.loads(await ws.recv())
                assert msg.get("type") == "answer"

                await pc.setRemoteDescription(
                    RTCSessionDescription(sdp=msg["sdp"], type="answer")
                )

                track = await asyncio.wait_for(received_track.get(), timeout=15)
                assert track.kind == "audio"

                # Verify multiple frames
                for _ in range(3):
                    frame = await asyncio.wait_for(track.recv(), timeout=10)
                    arr = frame.to_ndarray()
                    assert arr.shape[0] > 0

                await ws.send(json.dumps({"type": "close"}))
                await asyncio.sleep(0.5)

            await pc.close()

        asyncio.run(run())


# Test 4: Video with Controls
class VideoWithControlsApp(fal.App):
    """Video with control messages over websocket."""

    requirements = ["aiortc", "av", "numpy"]

    @fal.webrtc("/video_control")
    async def video_session(self, pc: RTCPeerConnection, ws: WebSocket):
        """
        Demonstrates control messages flowing through ws while
        signaling is handled automatically by the decorator.
        """
        # Shared color state
        current_color = [(255, 0, 0)]

        class ControllableVideoTrack(MediaStreamTrack):
            kind = "video"

            def __init__(self):
                super().__init__()
                self.width = 320
                self.height = 240
                self.fps = 30
                self._frame_count = 0

            async def recv(self):
                pts = self._frame_count * (1 / self.fps)
                time_base = Fraction(1, self.fps)

                frame = VideoFrame(width=self.width, height=self.height, format="rgb24")
                frame.pts = int(pts / float(time_base))
                frame.time_base = time_base

                arr = np.frombuffer(frame.planes[0], dtype=np.uint8)
                arr[:] = np.tile(current_color[0], self.width * self.height)

                self._frame_count += 1
                await asyncio.sleep(1 / self.fps)
                return frame

        pc.addTrack(ControllableVideoTrack())

        # Handle control messages via ws (signaling messages are filtered out)
        async def handle_messages():
            try:
                async for message in ws.iter_text():
                    msg = json.loads(message)
                    if msg.get("cmd") == "toggle_color":
                        # Toggle color
                        new_color = (
                            (0, 255, 0)
                            if current_color[0] == (255, 0, 0)
                            else (255, 0, 0)
                        )
                        current_color[0] = new_color

                        # Send response
                        await ws.send_text(
                            json.dumps(
                                {
                                    "type": "control_response",
                                    "status": "ok",
                                    "new_color": list(new_color),
                                }
                            )
                        )
            except Exception:
                pass

        await handle_messages()


def test_video_with_controls():
    """Test video with control commands."""

    with AppClient.connect(VideoWithControlsApp) as client:
        ws_url = client.url.replace("http", "ws") + "/video_control"

        async def run():
            ice_servers = [RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
            config = RTCConfiguration(iceServers=ice_servers)
            pc = RTCPeerConnection(configuration=config)

            pc.addTrack(ColorVideoTrack(width=160, height=120, color=(0, 0, 0), fps=1))

            received_track = asyncio.Queue()

            @pc.on("track")
            async def on_track(track):
                received_track.put_nowait(track)

            candidates_to_send = []

            @pc.on("icecandidate")
            def on_ice(candidate):
                if candidate:
                    candidates_to_send.append(candidate)

            done = asyncio.Event()

            @pc.on("icegatheringstatechange")
            def on_gathering_state_change():
                if pc.iceGatheringState == "complete":
                    done.set()

            async with websockets.connect(ws_url) as ws:
                offer = await pc.createOffer()
                await pc.setLocalDescription(offer)
                await asyncio.wait_for(done.wait(), timeout=10)

                await ws.send(
                    json.dumps({"type": "offer", "sdp": pc.localDescription.sdp})
                )

                for cand in candidates_to_send:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "icecandidate",
                                "candidate": {
                                    "candidate": cand.candidate,
                                    "sdpMid": cand.sdpMid,
                                    "sdpMLineIndex": cand.sdpMLineIndex,
                                },
                            }
                        )
                    )

                msg = json.loads(await ws.recv())
                assert msg.get("type") == "answer"

                await pc.setRemoteDescription(
                    RTCSessionDescription(sdp=msg["sdp"], type="answer")
                )

                track = await asyncio.wait_for(received_track.get(), timeout=15)
                assert track.kind == "video"

                # First frame - red
                frame1 = await asyncio.wait_for(track.recv(), timeout=10)
                arr1 = frame1.to_ndarray(format="rgb24")
                assert arr1[0, 0, 0] > 250  # Red

                # Send control command
                await ws.send(json.dumps({"cmd": "toggle_color"}))

                # Receive response
                control_msg = json.loads(await ws.recv())
                assert control_msg.get("status") == "ok"
                assert control_msg.get("new_color") == [0, 255, 0]

                # Wait for color change
                await asyncio.sleep(3.0)

                # Verify green frames
                green_found = False
                for _ in range(10):
                    frame2 = await asyncio.wait_for(track.recv(), timeout=10)
                    arr2 = frame2.to_ndarray(format="rgb24")
                    if arr2[0, 0, 1] > 200 and arr2[0, 0, 1] > arr2[0, 0, 0] + 50:
                        green_found = True
                        break

                assert green_found

                await ws.send(json.dumps({"type": "close"}))
                await asyncio.sleep(0.5)

            await pc.close()

        asyncio.run(run())


# Test 5: Bidirectional Audio + Video (Video Call)
class VideoCallApp(fal.App):
    """Full video call with audio and video in both directions."""

    requirements = ["aiortc", "av", "numpy"]

    @fal.webrtc("/video_call")
    async def video_call_session(self, pc: RTCPeerConnection, ws: WebSocket):
        """
        Clean interface: just add tracks and handle events.
        No manual signaling needed!
        """
        # Add both audio and video
        pc.addTrack(SineAudioTrack(freq=2000))
        pc.addTrack(ColorVideoTrack(width=320, height=240, color=(0, 0, 255)))

        # Handle incoming media
        received_audio = asyncio.Event()
        received_video = asyncio.Event()

        @pc.on("track")
        async def on_track(track):
            try:
                await track.recv()
                if track.kind == "audio":
                    received_audio.set()
                elif track.kind == "video":
                    received_video.set()
            except Exception:
                pass

        # Wait for both media types
        await asyncio.wait_for(
            asyncio.gather(received_audio.wait(), received_video.wait()),
            timeout=15,
        )

        # Keep connection alive until client sends close
        try:
            async for _ in ws.iter_text():
                pass
        except Exception:
            pass


def test_bidirectional_av():
    """Test full video call (audio + video both ways)."""

    with AppClient.connect(VideoCallApp) as client:
        ws_url = client.url.replace("http", "ws") + "/video_call"

        async def run():
            ice_servers = [RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
            config = RTCConfiguration(iceServers=ice_servers)
            pc = RTCPeerConnection(configuration=config)

            pc.addTrack(SineAudioTrack(freq=440))
            pc.addTrack(ColorVideoTrack(width=320, height=240, color=(255, 0, 0)))

            received_tracks = {"audio": None, "video": None}
            track_queue = asyncio.Queue()

            @pc.on("track")
            async def on_track(track):
                track_queue.put_nowait(track)

            candidates_to_send = []

            @pc.on("icecandidate")
            def on_ice(candidate):
                if candidate:
                    candidates_to_send.append(candidate)

            done = asyncio.Event()

            @pc.on("icegatheringstatechange")
            def on_gathering_state_change():
                if pc.iceGatheringState == "complete":
                    done.set()

            async with websockets.connect(ws_url) as ws:
                offer = await pc.createOffer()
                await pc.setLocalDescription(offer)
                await asyncio.wait_for(done.wait(), timeout=10)

                await ws.send(
                    json.dumps({"type": "offer", "sdp": pc.localDescription.sdp})
                )

                for cand in candidates_to_send:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "icecandidate",
                                "candidate": {
                                    "candidate": cand.candidate,
                                    "sdpMid": cand.sdpMid,
                                    "sdpMLineIndex": cand.sdpMLineIndex,
                                },
                            }
                        )
                    )

                msg = json.loads(await ws.recv())
                assert msg.get("type") == "answer"

                await pc.setRemoteDescription(
                    RTCSessionDescription(sdp=msg["sdp"], type="answer")
                )

                # Receive both tracks
                for _ in range(2):
                    track = await asyncio.wait_for(track_queue.get(), timeout=15)
                    received_tracks[track.kind] = track

                assert received_tracks["audio"] is not None
                assert received_tracks["video"] is not None

                # Verify audio
                for _ in range(2):
                    audio_frame = await asyncio.wait_for(
                        received_tracks["audio"].recv(), timeout=10
                    )
                    arr = audio_frame.to_ndarray()
                    assert arr.shape[0] > 0

                # Verify video
                video_frame = await asyncio.wait_for(
                    received_tracks["video"].recv(), timeout=10
                )
                arr = video_frame.to_ndarray(format="rgb24")
                assert arr.shape == (240, 320, 3)
                assert arr[0, 0, 2] > 250  # Blue

                await ws.send(json.dumps({"type": "close"}))
                await asyncio.sleep(0.5)

            await pc.close()

        asyncio.run(run())
