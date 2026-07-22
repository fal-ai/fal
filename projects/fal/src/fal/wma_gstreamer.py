from __future__ import annotations

import asyncio
import json
import site
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from fal.wma import PeerBackend, Session, SessionAnswer, StartSessionRequest

GI_DIST_PACKAGES = Path("/usr/lib/python3/dist-packages")
ICE_GATHERING_TIMEOUT_SECONDS = 7
PROMISE_TIMEOUT_SECONDS = 7


@dataclass(frozen=True)
class PipelineSpec:
    description: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    diagnostic_pads: tuple[tuple[str, str], ...] = ()


def load_gstreamer() -> tuple[Any, Any, Any]:
    if GI_DIST_PACKAGES.is_dir():
        site.addsitedir(str(GI_DIST_PACKAGES))

    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstSdp", "1.0")
    gi.require_version("GstWebRTC", "1.0")
    from gi.repository import Gst, GstSdp, GstWebRTC

    Gst.init(None)
    return Gst, GstSdp, GstWebRTC


class GStreamerPeer(PeerBackend):
    """Native WMA peer backed by a GStreamer ``webrtcbin`` pipeline."""

    def __init__(
        self,
        session: Session,
        pipeline: Callable[[StartSessionRequest], PipelineSpec | str],
        *,
        peer_element: str = "peer",
        channel_labels: Iterable[str] | None = None,
        stun_server: str | None = None,
        turn_server: str | None = None,
    ) -> None:
        self._session = session
        self._pipeline_factory = pipeline
        self._peer_element = peer_element
        self._channel_labels = (
            None if channel_labels is None else frozenset(channel_labels)
        )
        self._stun_server = stun_server
        self._turn_server = turn_server
        self._event_loop = asyncio.get_running_loop()
        self._closed = asyncio.Event()
        self._close_lock = asyncio.Lock()
        self._stop = threading.Event()
        self._negotiate_future: asyncio.Future[str] | None = None
        self._gst: Any = None
        self._gst_sdp: Any = None
        self._gst_webrtc: Any = None
        self._pipeline: Any = None
        self._webrtc: Any = None
        self._channel: Any = None
        self._bus_thread: threading.Thread | None = None
        self._bus_error: str | None = None
        self._metadata: dict[str, Any] = {}
        self._diagnostic_pads: tuple[tuple[str, str], ...] = ()

    @classmethod
    def availability_issues(
        cls,
        required_elements: Iterable[str] = (),
    ) -> list[str]:
        try:
            Gst, _, _ = load_gstreamer()
        except (ImportError, ValueError) as error:
            return [f"GStreamer Python bindings unavailable: {error}"]

        elements = {"webrtcbin", "nicesink", *required_elements}
        registry = Gst.Registry.get()
        return [
            f"missing GStreamer element: {name}"
            for name in sorted(elements)
            if registry.find_feature(name, Gst.ElementFactory.__gtype__) is None
        ]

    @classmethod
    def require_available(
        cls,
        required_elements: Iterable[str] = (),
    ) -> None:
        issues = cls.availability_issues(required_elements)
        if issues:
            raise RuntimeError(
                "Native WebRTC runtime is unavailable: " + "; ".join(issues)
            )

    async def negotiate(self, offer: StartSessionRequest) -> SessionAnswer:
        spec = self._pipeline_factory(offer)
        if isinstance(spec, str):
            spec = PipelineSpec(spec)
        self._metadata = dict(spec.metadata)
        self._diagnostic_pads = spec.diagnostic_pads
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None,
            self._negotiate_sync,
            offer.sdp,
            spec.description,
        )
        self._negotiate_future = future
        try:
            sdp = await asyncio.shield(future)
        except BaseException:
            await self.close()
            raise
        finally:
            if self._negotiate_future is future:
                self._negotiate_future = None
        return SessionAnswer(sdp=sdp, metadata=self._metadata)

    async def wait_closed(self) -> None:
        await self._closed.wait()

    async def close(self) -> None:
        async with self._close_lock:
            self._closed.set()
            self._stop.set()
            future = self._negotiate_future
            if future is not None and not future.done():
                with suppress(Exception):
                    await asyncio.shield(future)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._close_sync)

    def _negotiate_sync(self, offer_sdp: str, description: str) -> str:
        self._gst, self._gst_sdp, self._gst_webrtc = load_gstreamer()
        self._pipeline = self._gst.parse_launch(description)
        self._webrtc = self._pipeline.get_by_name(self._peer_element)
        if self._webrtc is None:
            raise RuntimeError(
                f"GStreamer pipeline has no {self._peer_element!r} element"
            )
        self._configure_ice_servers()
        self._webrtc.connect("on-data-channel", self._on_data_channel)
        self._webrtc.connect(
            "notify::connection-state", self._on_connection_state_changed
        )
        state_result = self._pipeline.set_state(self._gst.State.READY)
        if state_result == self._gst.StateChangeReturn.FAILURE:
            raise RuntimeError("GStreamer pipeline could not enter READY")

        result, sdp_message = self._gst_sdp.SDPMessage.new_from_text(offer_sdp)
        if result != self._gst_sdp.SDPResult.OK:
            raise RuntimeError(f"GStreamer rejected the browser SDP ({result})")
        offer = self._gst_webrtc.WebRTCSessionDescription.new(
            self._gst_webrtc.WebRTCSDPType.OFFER,
            sdp_message,
        )
        self._wait_for_promise("set-remote-description", offer)
        answer_reply = self._wait_for_promise(
            "create-answer",
            None,
            expect_reply=True,
        )
        answer = answer_reply.get_value("answer")
        self._wait_for_promise("set-local-description", answer)

        state_result = self._pipeline.set_state(self._gst.State.PLAYING)
        if state_result == self._gst.StateChangeReturn.FAILURE:
            raise RuntimeError(
                "GStreamer pipeline could not enter PLAYING: "
                + self._caps_diagnostics()
            )
        self._start_bus_monitor()
        self._wait_for_ice()
        local_description = self._webrtc.get_property("local-description")
        if local_description is None:
            raise RuntimeError("GStreamer did not produce a local description")
        return local_description.sdp.as_text()

    def _configure_ice_servers(self) -> None:
        if self._stun_server:
            self._webrtc.set_property("stun-server", self._stun_server)
        if self._turn_server:
            self._webrtc.set_property("turn-server", self._turn_server)

    def _wait_for_promise(
        self,
        signal: str,
        value: Any,
        *,
        expect_reply: bool = False,
    ) -> Any:
        completed = threading.Event()

        def on_promise_changed(
            _promise: Any,
            _user_data: Any,
            _notify: Any,
        ) -> None:
            completed.set()

        promise = self._gst.Promise.new_with_change_func(
            on_promise_changed,
            None,
            None,
        )
        self._webrtc.emit(signal, value, promise)
        deadline = time.monotonic() + PROMISE_TIMEOUT_SECONDS
        while not completed.wait(0.02):
            self._raise_bus_error()
            if self._stop.is_set():
                promise.interrupt()
                raise RuntimeError(f"GStreamer {signal} cancelled")
            if time.monotonic() >= deadline:
                promise.interrupt()
                raise TimeoutError(f"GStreamer {signal} timed out")
        result = promise.wait()
        if result != self._gst.PromiseResult.REPLIED:
            raise RuntimeError(f"GStreamer {signal} failed ({result})")
        reply = promise.get_reply()
        if reply is None:
            if expect_reply:
                raise RuntimeError(f"GStreamer {signal} returned no result")
            return None
        error = reply.get_value("error")
        if error is not None:
            raise RuntimeError(f"GStreamer {signal} failed: {error}")
        return reply

    def _wait_for_ice(self) -> None:
        complete = self._gst_webrtc.WebRTCICEGatheringState.COMPLETE
        deadline = time.monotonic() + ICE_GATHERING_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self._stop.is_set():
                raise RuntimeError("GStreamer ICE gathering cancelled")
            if self._webrtc.get_property("ice-gathering-state") == complete:
                return
            self._raise_bus_error()
            time.sleep(0.02)
        raise TimeoutError("GStreamer ICE gathering timed out")

    def _on_data_channel(self, _webrtc: Any, channel: Any) -> None:
        label = channel.get_property("label")
        if self._channel_labels is not None and label not in self._channel_labels:
            return
        self._channel = channel
        self._session.bind_sender(self.send, thread_safe=True)
        channel.connect("on-message-string", self._on_channel_message)
        channel.connect("on-open", self._on_channel_open)
        channel.connect("on-close", self._on_channel_close)

    def _on_channel_message(self, _channel: Any, raw: str) -> None:
        try:
            message = json.loads(raw)
        except (TypeError, ValueError):
            return
        if isinstance(message, dict):
            self._session.receive(message)

    def _on_channel_open(self, _channel: Any) -> None:
        self._event_loop.call_soon_threadsafe(self._session.channel_opened)

    def _on_channel_close(self, channel: Any) -> None:
        if channel is self._channel:
            self._signal_closed()

    def send(self, message: dict) -> bool:
        channel = self._channel
        if channel is None:
            return False
        state = channel.get_property("ready-state")
        if state is None or state.value_nick != "open":
            return False
        channel.emit("send-string", json.dumps(message))
        return True

    def _on_connection_state_changed(self, webrtc: Any, _spec: Any) -> None:
        state = webrtc.get_property("connection-state").value_nick
        if state in {"failed", "closed"}:
            self._signal_closed()

    def _signal_closed(self) -> None:
        self._event_loop.call_soon_threadsafe(self._closed.set)

    def _start_bus_monitor(self) -> None:
        bus = self._pipeline.get_bus()
        self._bus_thread = threading.Thread(
            target=self._monitor_bus,
            args=(bus,),
            name="wma-gstreamer-bus",
            daemon=True,
        )
        self._bus_thread.start()

    def _monitor_bus(self, bus: Any) -> None:
        mask = self._gst.MessageType.ERROR | self._gst.MessageType.EOS
        while not self._stop.is_set():
            message = bus.timed_pop_filtered(250 * self._gst.MSECOND, mask)
            if message is None:
                continue
            if message.type == self._gst.MessageType.ERROR:
                error, debug = message.parse_error()
                self._bus_error = f"{error} ({debug})"
            self._signal_closed()
            return

    def _raise_bus_error(self) -> None:
        if self._bus_error is not None:
            raise RuntimeError(
                f"GStreamer pipeline failed: {self._bus_error}; "
                + self._caps_diagnostics()
            )
        message = self._pipeline.get_bus().pop_filtered(self._gst.MessageType.ERROR)
        if message is None:
            return
        error, debug = message.parse_error()
        self._bus_error = f"{error} ({debug})"
        raise RuntimeError(
            f"GStreamer pipeline failed: {error} ({debug}); " + self._caps_diagnostics()
        )

    def _caps_diagnostics(self) -> str:
        diagnostics = []
        for element_name, pad_name in self._diagnostic_pads:
            element = self._pipeline.get_by_name(element_name)
            pad = element.get_static_pad(pad_name) if element is not None else None
            if pad is None:
                diagnostics.append(f"{element_name}.{pad_name}=missing")
                continue
            current = pad.get_current_caps()
            accepted = current if current is not None else pad.query_caps(None)
            caps = "none" if accepted is None else accepted.to_string()
            diagnostics.append(f"{element_name}.{pad_name}={caps[:500]}")
        return "caps[" + " | ".join(diagnostics) + "]"

    def _close_sync(self) -> None:
        self._stop.set()
        channel, self._channel = self._channel, None
        if channel is not None:
            channel.emit("close")
        pipeline, self._pipeline = self._pipeline, None
        if pipeline is not None and self._gst is not None:
            pipeline.set_state(self._gst.State.NULL)
        self._webrtc = None
        thread, self._bus_thread = self._bus_thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1)
