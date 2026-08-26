from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from fal.wma import telemetry
from fal.wma.telemetry import (
    CONNECTION_REPORT_VERSION,
    build_connection_report,
    observe_peer_connection,
)


class FakePeerConnection:
    def __init__(
        self,
        runner_candidate: str = "host",
        browser_candidate: str = "host",
        protocol: str = "udp",
    ) -> None:
        self.connectionState = "new"
        self._handlers: dict[str, list] = {}
        pair = SimpleNamespace(
            local_candidate=SimpleNamespace(
                type=runner_candidate,
                transport=protocol,
                host="203.0.113.10",
                port=41000,
            ),
            remote_candidate=SimpleNamespace(
                type=browser_candidate,
                transport=protocol,
                host="198.51.100.20",
                port=42000,
            ),
        )
        connection = SimpleNamespace(_nominated={1: pair})
        ice_transport = SimpleNamespace(_connection=connection)
        dtls_transport = SimpleNamespace(transport=ice_transport)
        self.sctp = SimpleNamespace(transport=dtls_transport)

    def on(self, event: str):
        def register(handler):
            self._handlers.setdefault(event, []).append(handler)
            return handler

        return register

    def set_state(self, state: str) -> None:
        self.connectionState = state
        for handler in self._handlers.get("connectionstatechange", []):
            handler()


@pytest.mark.parametrize(
    ("runner_candidate", "browser_candidate"),
    [
        ("host", "srflx"),
        ("relay", "host"),
        ("host", "relay"),
        ("relay", "relay"),
        ("prflx", "srflx"),
    ],
)
def test_report_uses_only_bounded_selected_pair_fields(
    monkeypatch, runner_candidate, browser_candidate
):
    monkeypatch.setattr(telemetry.time, "monotonic", lambda: 10.842)
    pc = FakePeerConnection(runner_candidate, browser_candidate, "UDP")

    payload = build_connection_report(pc, started_at=10.0).as_payload()

    assert payload == {
        "version": CONNECTION_REPORT_VERSION,
        "runner_candidate": runner_candidate,
        "browser_candidate": browser_candidate,
        "ice_protocol": "udp",
        "setup_ms": 842,
    }
    assert set(payload) == {
        "version",
        "runner_candidate",
        "browser_candidate",
        "ice_protocol",
        "setup_ms",
    }


def test_missing_or_changed_aiortc_internals_report_unknown(monkeypatch):
    monkeypatch.setattr(telemetry.time, "monotonic", lambda: 2.25)
    pc = SimpleNamespace(sctp=None, getTransceivers=list)

    payload = build_connection_report(pc, started_at=2.0).as_payload()

    assert payload == {
        "version": CONNECTION_REPORT_VERSION,
        "runner_candidate": "unknown",
        "browser_candidate": "unknown",
        "ice_protocol": "unknown",
        "setup_ms": 250,
    }


def test_observer_resolves_only_the_first_connected_path(monkeypatch):
    async def scenario():
        monkeypatch.setattr(telemetry.time, "monotonic", lambda: 5.5)
        pc = FakePeerConnection("srflx", "relay", "tcp")
        observer = observe_peer_connection(pc, started_at=5.0)
        report_waiter = asyncio.create_task(observer.wait())

        pc.set_state("connecting")
        await asyncio.sleep(0)
        assert not report_waiter.done()
        pc.set_state("connected")
        first = await report_waiter

        pair = pc.sctp.transport.transport._connection._nominated[1]
        pair.local_candidate.type = "relay"
        pc.set_state("disconnected")
        pc.set_state("connected")
        second = await observer.wait()
        return first, second

    first, second = asyncio.run(scenario())
    assert first == second
    assert first["runner_candidate"] == "srflx"
    assert first["browser_candidate"] == "relay"
    assert first["ice_protocol"] == "tcp"
    assert first["setup_ms"] == 500
