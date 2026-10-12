"""Privacy-safe, one-shot WebRTC connection reports for the WMA bridge."""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Mapping, Union, cast

CONNECTION_REPORT_VERSION = 1

CandidateType = Literal["host", "srflx", "prflx", "relay", "unknown"]
IceProtocol = Literal["udp", "tcp", "unknown"]

_CANDIDATE_TYPES = frozenset({"host", "srflx", "prflx", "relay"})
_ICE_PROTOCOLS = frozenset({"udp", "tcp"})
_REPORTED_CANDIDATE_TYPES = _CANDIDATE_TYPES | {"unknown"}
_REPORTED_ICE_PROTOCOLS = _ICE_PROTOCOLS | {"unknown"}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConnectionReport:
    """The bounded report accepted by the WMA bridge's version-1 contract."""

    runner_candidate: CandidateType
    browser_candidate: CandidateType
    ice_protocol: IceProtocol
    setup_ms: int

    def as_payload(self) -> Dict[str, Union[str, int]]:
        return {
            "version": CONNECTION_REPORT_VERSION,
            "runner_candidate": self.runner_candidate,
            "browser_candidate": self.browser_candidate,
            "ice_protocol": self.ice_protocol,
            "setup_ms": self.setup_ms,
        }


def sanitize_connection_report(
    payload: Any,
) -> Union[Dict[str, Union[str, int]], None]:
    """Return one privacy-safe version-1 report, or ``None`` when invalid.

    ``wait_connection_report`` is an extension boundary, so filtering only its
    keys is insufficient: a custom backend could put an address or credential
    inside an allowed string field. Validate the complete bounded value schema
    here and use the result at every SSE serialization boundary.
    """

    if not isinstance(payload, Mapping):
        return None

    version = payload.get("version")
    runner_candidate = payload.get("runner_candidate")
    browser_candidate = payload.get("browser_candidate")
    ice_protocol = payload.get("ice_protocol")
    setup_ms = payload.get("setup_ms")

    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != CONNECTION_REPORT_VERSION
    ):
        return None
    if not isinstance(runner_candidate, str) or (
        runner_candidate not in _REPORTED_CANDIDATE_TYPES
    ):
        return None
    if not isinstance(browser_candidate, str) or (
        browser_candidate not in _REPORTED_CANDIDATE_TYPES
    ):
        return None
    if not isinstance(ice_protocol, str) or ice_protocol not in _REPORTED_ICE_PROTOCOLS:
        return None
    if isinstance(setup_ms, bool) or not isinstance(setup_ms, int) or setup_ms < 0:
        return None

    return {
        "version": version,
        "runner_candidate": runner_candidate,
        "browser_candidate": browser_candidate,
        "ice_protocol": ice_protocol,
        "setup_ms": setup_ms,
    }


def _bounded_candidate_type(value: Any) -> CandidateType:
    normalized = value.lower() if isinstance(value, str) else ""
    if normalized in _CANDIDATE_TYPES:
        return cast(CandidateType, normalized)
    return "unknown"


def _bounded_ice_protocol(*values: Any) -> IceProtocol:
    for value in values:
        normalized = value.lower() if isinstance(value, str) else ""
        if normalized in _ICE_PROTOCOLS:
            return cast(IceProtocol, normalized)
    return "unknown"


def _selected_candidate_pair(pc: Any) -> Union[Any, None]:
    """Return aiortc's nominated pair without reading candidate addresses.

    aiortc 1.15 does not include candidate-pair records in ``getStats()``. Its
    RTCIceTransport delegates nomination to aioice, where the selected pair is
    stored in ``Connection._nominated``. Keep that compatibility access in this
    one helper and fail closed to an unknown path if the internal shape changes.

    WMA always carries a data channel, so prefer the SCTP transport. The RTP
    fallback keeps this useful for a custom media-only peer using the same SDK.
    """

    ice_transports: List[Any] = []
    sctp = getattr(pc, "sctp", None)
    dtls_transport = getattr(sctp, "transport", None)
    ice_transport = getattr(dtls_transport, "transport", None)
    if ice_transport is not None:
        ice_transports.append(ice_transport)

    get_transceivers = getattr(pc, "getTransceivers", None)
    if callable(get_transceivers):
        for transceiver in get_transceivers():
            for endpoint_name in ("sender", "receiver"):
                endpoint = getattr(transceiver, endpoint_name, None)
                dtls_transport = getattr(endpoint, "transport", None)
                ice_transport = getattr(dtls_transport, "transport", None)
                if ice_transport is not None and all(
                    ice_transport is not existing for existing in ice_transports
                ):
                    ice_transports.append(ice_transport)

    for transport in ice_transports:
        connection = getattr(transport, "_connection", None)
        nominated = getattr(connection, "_nominated", None)
        if not isinstance(nominated, dict) or not nominated:
            continue
        pair = nominated.get(1)
        if pair is not None:
            return pair
        return next(iter(nominated.values()), None)
    return None


def build_connection_report(pc: Any, *, started_at: float) -> ConnectionReport:
    """Describe the selected path without retaining or returning addresses."""

    setup_ms = max(0, round((time.monotonic() - started_at) * 1000))
    try:
        pair = _selected_candidate_pair(pc)
        local = getattr(pair, "local_candidate", None)
        remote = getattr(pair, "remote_candidate", None)
        return ConnectionReport(
            runner_candidate=_bounded_candidate_type(getattr(local, "type", None)),
            browser_candidate=_bounded_candidate_type(getattr(remote, "type", None)),
            ice_protocol=_bounded_ice_protocol(
                getattr(local, "transport", None),
                getattr(remote, "transport", None),
            ),
            setup_ms=setup_ms,
        )
    except Exception:
        logger.warning("WMA could not inspect the selected ICE pair", exc_info=True)
        return ConnectionReport(
            runner_candidate="unknown",
            browser_candidate="unknown",
            ice_protocol="unknown",
            setup_ms=setup_ms,
        )


class ConnectionReportObserver:
    """Resolve one sanitized report when a peer first reaches ``connected``."""

    version = CONNECTION_REPORT_VERSION

    def __init__(self, pc: Any, *, started_at: Union[float, None] = None) -> None:
        self._pc = pc
        self._started_at = time.monotonic() if started_at is None else started_at
        self._report: asyncio.Future = asyncio.get_running_loop().create_future()
        pc.on("connectionstatechange")(self._on_connection_state_change)
        self._on_connection_state_change()

    def _on_connection_state_change(self) -> None:
        if self._pc.connectionState != "connected" or self._report.done():
            return
        self._report.set_result(
            build_connection_report(self._pc, started_at=self._started_at)
        )

    async def wait(self) -> Dict[str, Union[str, int]]:
        return (await self._report).as_payload()


def observe_peer_connection(
    pc: Any, *, started_at: Union[float, None] = None
) -> ConnectionReportObserver:
    """Begin observing the first successful connection for ``pc``."""

    return ConnectionReportObserver(pc, started_at=started_at)
