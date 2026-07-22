from __future__ import annotations

import asyncio
import json
import threading
from unittest.mock import Mock, call

import pytest

from fal import wma
from fal.wma import Session, StartSessionRequest
from fal.wma_gstreamer import GStreamerPeer


def make_peer() -> tuple[GStreamerPeer, Session]:
    async def create():
        session = Session(StartSessionRequest(sdp="offer", session_id="test"))
        return GStreamerPeer(session, lambda _offer: "pipeline"), session

    return asyncio.run(create())


def test_gstreamer_backend_is_available_from_wma_namespace():
    assert wma.GStreamerPeer is GStreamerPeer


def configure_promise(peer, reply):
    promise = Mock()
    replied = object()
    promise.wait.return_value = replied
    promise.get_reply.return_value = reply
    gst = Mock()
    gst.PromiseResult.REPLIED = replied

    def create_promise(callback, user_data, notify):
        callback(promise, user_data, notify)
        return promise

    gst.Promise.new_with_change_func.side_effect = create_promise
    peer._gst = gst
    peer._webrtc = Mock()


def test_setter_promise_accepts_empty_success_reply():
    peer, _ = make_peer()
    configure_promise(peer, None)

    assert peer._wait_for_promise("set-remote-description", Mock()) is None


def test_create_answer_requires_reply_structure():
    peer, _ = make_peer()
    configure_promise(peer, None)

    with pytest.raises(RuntimeError, match="returned no result"):
        peer._wait_for_promise("create-answer", None, expect_reply=True)


def test_configures_stun_and_turn_before_negotiation():
    peer, _ = make_peer()
    peer._webrtc = Mock()
    peer._stun_server = "stun://stun.example.com:3478"
    peer._turn_server = "turns://user:secret@turn.example.com:5349"

    peer._configure_ice_servers()

    assert peer._webrtc.set_property.call_args_list == [
        call("stun-server", "stun://stun.example.com:3478"),
        call("turn-server", "turns://user:secret@turn.example.com:5349"),
    ]


def test_data_channel_dispatches_to_transport_neutral_session():
    async def scenario():
        session = Session(StartSessionRequest(sdp="offer"))
        peer = GStreamerPeer(session, lambda _offer: "pipeline")
        received = []
        session.on_message("input", received.append)
        channel = Mock()
        channel.get_property.return_value = "input"
        callbacks = {}
        channel.connect.side_effect = lambda event, callback: callbacks.update(
            {event: callback}
        )

        peer._on_data_channel(None, channel)
        callbacks["on-message-string"](
            channel,
            json.dumps({"type": "input", "seq": 3}),
        )

        assert received == [{"type": "input", "seq": 3}]

    asyncio.run(scenario())


def test_channel_filter_rejects_unregistered_label():
    async def create():
        session = Session(StartSessionRequest(sdp="offer"))
        return GStreamerPeer(
            session,
            lambda _offer: "pipeline",
            channel_labels={"input"},
        )

    peer = asyncio.run(create())
    channel = Mock()
    channel.get_property.return_value = "other"

    peer._on_data_channel(None, channel)

    channel.connect.assert_not_called()


def test_only_current_data_channel_close_ends_session():
    async def scenario():
        session = Session(StartSessionRequest(sdp="offer"))
        peer = GStreamerPeer(session, lambda _offer: "pipeline")

        first = Mock()
        first.get_property.return_value = "input"
        second = Mock()
        second.get_property.return_value = "input"
        peer._on_data_channel(None, first)
        peer._on_data_channel(None, second)

        peer._on_channel_close(first)
        await asyncio.sleep(0)
        assert not peer._closed.is_set()

        peer._on_channel_close(second)
        await asyncio.sleep(0)
        assert peer._closed.is_set()

    asyncio.run(scenario())


def test_cancelling_negotiation_waits_for_worker_before_teardown():
    async def scenario():
        session = Session(StartSessionRequest(sdp="offer"))
        peer = GStreamerPeer(session, lambda _offer: "pipeline")
        started = threading.Event()
        worker_finished = threading.Event()
        close_order = []

        def negotiate(_sdp, _description):
            started.set()
            peer._stop.wait(timeout=1)
            worker_finished.set()
            raise RuntimeError("cancelled")

        def close():
            close_order.append(worker_finished.is_set())

        peer._negotiate_sync = negotiate
        peer._close_sync = close
        task = asyncio.create_task(peer.negotiate(StartSessionRequest(sdp="offer")))
        await asyncio.to_thread(started.wait, 1)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert close_order == [True]

    asyncio.run(scenario())
