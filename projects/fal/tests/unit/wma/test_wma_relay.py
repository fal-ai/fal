"""Unit tests for the WebSocket signaling relay (``fal.wma.relay``).

The relay is driven end-to-end with a fake upstream WebSocket and a
queue-fed client input iterator; items and outputs are plain dicts so the
tests exercise only the relay machinery, not any app's translation hooks.
"""

import asyncio
import json
from typing import Any, List, Union

from fal.wma import ControlRejected, WebSocketSignalingRelay

_EOF = object()


class FakeUpstream:
    """Duck-typed websockets connection: send/close plus async iteration."""

    def __init__(self) -> None:
        self.sent: List[str] = []
        self.closed = False
        self.fail_next_send = False
        self.hang_next_send = False
        self._incoming: asyncio.Queue = asyncio.Queue()

    async def push(self, payload: str) -> None:
        await self._incoming.put(payload)

    async def send(self, payload: str) -> None:
        if self.fail_next_send:
            self.fail_next_send = False
            raise RuntimeError("send failed")
        if self.hang_next_send:
            self.hang_next_send = False
            await asyncio.sleep(3600)
        self.sent.append(payload)

    async def close(self) -> None:
        if not self.closed:
            self.closed = True
            await self._incoming.put(_EOF)

    def __aiter__(self) -> "FakeUpstream":
        return self

    async def __anext__(self) -> Any:
        item = await self._incoming.get()
        if item is _EOF:
            raise StopAsyncIteration
        return item


class Harness:
    """One running relay session with hooks over plain dicts."""

    def __init__(self, connect_error: bool = False, send_timeout: float = 30.0) -> None:
        self.upstream = FakeUpstream()
        self.outputs: List[dict] = []
        self._input_queue: asyncio.Queue = asyncio.Queue()

        async def connect() -> FakeUpstream:
            if connect_error:
                raise RuntimeError("no route to partner")
            return self.upstream

        async def control_payload(item: dict) -> Union[str, None]:
            if item.get("prompt") == "reject-me":
                raise ControlRejected("bad reference image")
            if item.get("prompt") == "explode":
                raise RuntimeError("boom")
            return json.dumps({"type": "prompt", "prompt": item["prompt"]})

        self.relay = WebSocketSignalingRelay(
            upstream_url="wss://partner.example/stream",
            connect_upstream=connect,
            signaling_payload=lambda item: json.dumps(
                {"type": item["type"], "sdp": item.get("sdp")}
            ),
            control_payload=control_payload,
            from_upstream=lambda raw: (
                None if json.loads(raw)["type"] == "internal" else json.loads(raw)
            ),
            is_signaling=lambda item: item.get("type") in ("offer", "icecandidate"),
            has_control=lambda item: "prompt" in item,
            is_session_ready=lambda out: out["type"] == "answer",
            make_error=lambda msg: {"type": "error", "error": msg},
            bootstrap=[{"type": "ready"}, {"type": "iceServers", "iceServers": []}],
            connect_timeout=0.5,
            send_timeout=send_timeout,
            label="test-relay",
        )
        self._consumer: Union[asyncio.Task, None] = None

    async def _inputs(self):
        while True:
            item = await self._input_queue.get()
            if item is None:
                return
            yield item

    def start(self) -> None:
        async def consume() -> None:
            async for out in self.relay.run(self._inputs()):
                self.outputs.append(out)

        self._consumer = asyncio.create_task(consume())

    async def send_input(self, item: dict) -> None:
        await self._input_queue.put(item)

    async def end_inputs(self) -> None:
        await self._input_queue.put(None)

    async def finish(self) -> List[dict]:
        await self.end_inputs()
        assert self._consumer is not None
        await asyncio.wait_for(self._consumer, timeout=2)
        return self.outputs

    async def wait_for(self, predicate, what: str) -> None:
        for _ in range(200):
            if predicate():
                return
            await asyncio.sleep(0.005)
        raise AssertionError(f"timed out waiting for {what}")


def run(coro):
    return asyncio.run(coro)


class TestConnectAndBootstrap:
    def test_connect_failure_yields_error_and_ends(self):
        async def scenario():
            harness = Harness(connect_error=True)
            harness.start()
            return await harness.finish()

        outputs = run(scenario())
        assert outputs == [{"type": "error", "error": harness_connect_error_message()}]

    def test_bootstrap_events_precede_everything(self):
        async def scenario():
            harness = Harness()
            harness.start()
            await harness.wait_for(lambda: len(harness.outputs) >= 2, "bootstrap")
            return await harness.finish()

        outputs = run(scenario())
        assert outputs[0] == {"type": "ready"}
        assert outputs[1]["type"] == "iceServers"


def harness_connect_error_message() -> str:
    return WebSocketSignalingRelay(
        upstream_url="wss://x",
        signaling_payload=lambda i: None,
        control_payload=_none_control,
        from_upstream=lambda r: None,
        is_signaling=lambda i: False,
        has_control=lambda i: False,
        is_session_ready=lambda o: False,
        make_error=lambda m: m,
    ).connect_error


async def _none_control(item: Any) -> None:
    return None


class TestSignalingAndControlFlow:
    def test_signaling_forwarded_inline_and_answer_relayed(self):
        async def scenario():
            harness = Harness()
            harness.start()
            await harness.send_input({"type": "offer", "sdp": "v=0 offer"})
            await harness.wait_for(
                lambda: len(harness.upstream.sent) == 1, "offer upstream"
            )
            await harness.upstream.push(json.dumps({"type": "answer", "sdp": "v=0"}))
            await harness.wait_for(
                lambda: any(o.get("type") == "answer" for o in harness.outputs),
                "answer output",
            )
            outputs = await harness.finish()
            return outputs, harness.upstream.sent

        outputs, sent = run(scenario())
        assert json.loads(sent[0]) == {"type": "offer", "sdp": "v=0 offer"}
        assert {"type": "answer", "sdp": "v=0"} in outputs

    def test_controls_are_gated_on_answer_and_ordered(self):
        async def scenario():
            harness = Harness()
            harness.start()
            # Controls sent before the answer must be buffered...
            await harness.send_input({"prompt": "first"})
            await harness.send_input({"prompt": "second"})
            await asyncio.sleep(0.05)
            assert harness.upstream.sent == []
            # ...and released, in order, once the upstream answers.
            await harness.upstream.push(json.dumps({"type": "answer", "sdp": "v=0"}))
            await harness.wait_for(
                lambda: len(harness.upstream.sent) == 2, "controls released"
            )
            await harness.finish()
            return harness.upstream.sent

        sent = run(scenario())
        assert [json.loads(p)["prompt"] for p in sent] == ["first", "second"]

    def test_message_with_signaling_and_control_handles_both(self):
        async def scenario():
            harness = Harness()
            harness.start()
            await harness.upstream.push(json.dumps({"type": "answer", "sdp": "v=0"}))
            await harness.send_input(
                {"type": "icecandidate", "sdp": None, "prompt": "hello"}
            )
            await harness.wait_for(
                lambda: len(harness.upstream.sent) == 2, "both payloads"
            )
            await harness.finish()
            return harness.upstream.sent

        sent = run(scenario())
        types = [json.loads(p)["type"] for p in sent]
        assert types == ["icecandidate", "prompt"]

    def test_from_upstream_none_drops_message(self):
        async def scenario():
            harness = Harness()
            harness.start()
            await harness.upstream.push(json.dumps({"type": "internal"}))
            await harness.upstream.push(json.dumps({"type": "answer", "sdp": "v=0"}))
            await harness.wait_for(
                lambda: any(o.get("type") == "answer" for o in harness.outputs),
                "answer output",
            )
            return await harness.finish()

        outputs = run(scenario())
        assert all(o.get("type") != "internal" for o in outputs)


class TestErrorPaths:
    def test_control_rejected_surfaces_error_and_session_continues(self):
        async def scenario():
            harness = Harness()
            harness.start()
            await harness.upstream.push(json.dumps({"type": "answer", "sdp": "v=0"}))
            await harness.send_input({"prompt": "reject-me"})
            await harness.wait_for(
                lambda: any(o.get("type") == "error" for o in harness.outputs),
                "rejection error",
            )
            await harness.send_input({"prompt": "still-works"})
            await harness.wait_for(
                lambda: len(harness.upstream.sent) == 1, "later control"
            )
            outputs = await harness.finish()
            return outputs, harness.upstream.sent

        outputs, sent = run(scenario())
        assert {"type": "error", "error": "bad reference image"} in outputs
        assert json.loads(sent[0])["prompt"] == "still-works"

    def test_control_crash_reports_and_keeps_worker_alive(self):
        async def scenario():
            harness = Harness()
            harness.start()
            await harness.upstream.push(json.dumps({"type": "answer", "sdp": "v=0"}))
            await harness.send_input({"prompt": "explode"})
            await harness.wait_for(
                lambda: any(o.get("type") == "error" for o in harness.outputs),
                "crash error",
            )
            await harness.send_input({"prompt": "recovered"})
            await harness.wait_for(
                lambda: len(harness.upstream.sent) == 1, "later control"
            )
            outputs = await harness.finish()
            return outputs, harness.upstream.sent

        outputs, sent = run(scenario())
        assert any(o.get("error") == harness_control_error_message() for o in outputs)
        assert json.loads(sent[0])["prompt"] == "recovered"

    def test_upstream_send_hang_times_out_and_reports_forward_error(self):
        async def scenario():
            harness = Harness(send_timeout=0.05)
            harness.start()
            harness.upstream.hang_next_send = True
            await harness.send_input({"type": "offer", "sdp": "v=0"})
            await harness.wait_for(
                lambda: any(o.get("type") == "error" for o in harness.outputs),
                "send timeout error",
            )
            return await harness.finish()

        outputs = run(scenario())
        assert any(o.get("error") == harness_forward_error_message() for o in outputs)

    def test_upstream_send_failure_reports_forward_error(self):
        async def scenario():
            harness = Harness()
            harness.start()
            harness.upstream.fail_next_send = True
            await harness.send_input({"type": "offer", "sdp": "v=0"})
            await harness.wait_for(
                lambda: any(o.get("type") == "error" for o in harness.outputs),
                "forward error",
            )
            return await harness.finish()

        outputs = run(scenario())
        assert any(o.get("error") == harness_forward_error_message() for o in outputs)


def harness_control_error_message() -> str:
    return "Failed to apply the update."


def harness_forward_error_message() -> str:
    return "Failed to forward the update."


class TestTeardown:
    def test_client_end_closes_upstream_and_ends_session(self):
        async def scenario():
            harness = Harness()
            harness.start()
            await harness.wait_for(lambda: len(harness.outputs) >= 2, "bootstrap")
            await harness.finish()
            return harness.upstream.closed

        assert run(scenario()) is True

    def test_upstream_end_ends_session(self):
        async def scenario():
            harness = Harness()
            harness.start()
            await harness.wait_for(lambda: len(harness.outputs) >= 2, "bootstrap")
            await harness.upstream.close()
            assert harness._consumer is not None
            await asyncio.wait_for(harness._consumer, timeout=2)
            return harness.upstream.closed

        assert run(scenario()) is True
