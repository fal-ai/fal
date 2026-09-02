"""Pinning tests for the security-critical WMA helpers.

These pin behaviors whose loss would not otherwise fail the suite: the
request-id canonicalization guarding the FAL_KEY-bearing billing path
(CWE-88), the SSRF classification of encapsulated IPv4 forms in the ICE
candidate filter, and the ICE URL shapes that would crash or silently
degrade aiortc if a validator ever re-admitted them.
"""

from __future__ import annotations

import pytest

from fal.wma import (
    DEFAULT_STUN_URL,
    IceServerConfigError,
    MeteredFetchError,
    RunnerIceConfig,
    filter_sdp_ice_candidates,
    validate_forwarded_ice_servers,
    validate_server_ice_servers,
)
from fal.wma._errors import format_billable_units
from fal.wma._request_id import valid_fal_request_id

CANONICAL = "2f9c8f6a-0d1e-4b7a-9c3d-5e6f7a8b9c0d"


class TestValidFalRequestId:
    @pytest.mark.parametrize(
        "given",
        [
            CANONICAL,
            CANONICAL.upper(),
            CANONICAL.replace("-", ""),
            f"urn:uuid:{CANONICAL}",
        ],
    )
    def test_uuid_spellings_canonicalize(self, given):
        assert valid_fal_request_id(given) == CANONICAL

    @pytest.mark.parametrize(
        "payload",
        [
            "../other-request",
            f"{CANONICAL}/..%2f..%2fadmin",
            f"{CANONICAL}?units=0",
            f"{CANONICAL}&x=1",
            f"{CANONICAL}#frag",
            "not-a-uuid",
            "",
            None,
        ],
    )
    def test_injection_payloads_are_rejected(self, payload):
        assert valid_fal_request_id(payload) is None

    def test_output_can_never_escape_a_path_segment(self):
        # The returned value is interpolated into a FAL_KEY-bearing REST path;
        # a canonical UUID contains only [0-9a-f-].
        result = valid_fal_request_id(CANONICAL.upper())
        assert result is not None
        assert set(result) <= set("0123456789abcdef-")


class TestIceCandidateSsrfPins:
    """The encapsulated-IPv4 forms that reach the cloud metadata service.

    ``filter_sdp_ice_candidates`` delegates to the toolkit SSRF classifier;
    these pin the delegation end to end so a future "simplify to
    ``ip.is_global``" refactor of either side fails loudly here.
    """

    @staticmethod
    def _sdp_with(address: str) -> str:
        return (
            "v=0\r\n"
            f"a=candidate:1 1 UDP 2130706431 {address} 51000 typ host\r\n"
            "a=candidate:2 1 UDP 2130706431 8.8.8.8 52000 typ host\r\n"
        )

    @pytest.mark.parametrize(
        "address",
        [
            "::ffff:169.254.169.254",  # IPv4-mapped metadata endpoint
            "::ffff:100.64.0.1",  # IPv4-mapped CGNAT
            "2002:a9fe:a9fe::",  # 6to4 -> 169.254.169.254
            "64:ff9b::a9fe:a9fe",  # NAT64 -> 169.254.169.254
        ],
    )
    def test_encapsulated_internal_targets_are_stripped(self, address):
        filtered = filter_sdp_ice_candidates(self._sdp_with(address))
        assert address not in filtered
        assert "8.8.8.8" in filtered


class TestIceUrlShapePins:
    """URI shapes that crash or silently degrade aiortc must stay rejected."""

    def test_forwarded_stun_with_transport_is_rejected(self):
        # aiortc raises "stun must not contain transport" at negotiation time,
        # which would turn this request field into a 500 instead of STUN-only.
        with pytest.raises(MeteredFetchError):
            validate_forwarded_ice_servers(
                [{"urls": "stun:global.relay.metered.ca?transport=udp"}]
            )

    def test_forwarded_turns_over_udp_is_rejected(self):
        # aiortc silently discards TURNS-over-UDP; accepting it would report
        # the session as TURN-capable with no relay candidate able to gather.
        with pytest.raises(MeteredFetchError):
            validate_forwarded_ice_servers(
                [
                    {
                        "urls": "turns:global.relay.metered.ca:443?transport=udp",
                        "username": "u",
                        "credential": "p",
                    }
                ]
            )

    @pytest.mark.parametrize(
        "url",
        [
            "turn:global.relay.metered.ca:²",  # non-decimal Unicode digit
            "turn:global.relay.metered.ca:٠٠٠",  # Arabic-Indic
        ],
    )
    def test_non_decimal_ports_raise_the_module_error_not_valueerror(self, url):
        # '²'.isdigit() is True but int('²') raises; a bare ValueError would
        # escape the except-clauses and 500 the session.
        with pytest.raises(MeteredFetchError):
            validate_forwarded_ice_servers(
                [{"urls": url, "username": "u", "credential": "p"}]
            )
        with pytest.raises(IceServerConfigError):
            validate_server_ice_servers(
                [{"urls": url, "username": "u", "credential": "p"}]
            )

    @pytest.mark.asyncio
    async def test_hostile_forwarded_port_falls_back_to_stun_only(self, monkeypatch):
        import sys
        import types

        fake = types.ModuleType("aiortc")
        fake.RTCIceServer = lambda **kw: kw  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "aiortc", fake)
        rc = RunnerIceConfig.from_bridge()
        servers, turn = await rc.build_ice_servers_async(
            [
                {
                    "urls": "turn:global.relay.metered.ca:²",
                    "username": "u",
                    "credential": "p",
                }
            ],
            forwarded_status="turn",
        )
        assert turn is False
        assert servers == [{"urls": DEFAULT_STUN_URL}]


class TestBillableUnitsWireFormat:
    def test_small_floats_render_fixed_point_not_scientific(self):
        assert format_billable_units(1.23e-05) == "0.00001230"
        assert str(1.23e-05) == "1.23e-05"  # the form this exists to avoid

    def test_integers_render_without_decimals(self):
        assert format_billable_units(0) == "0"
        assert format_billable_units(3) == "3"

    def test_matches_the_sdk_renderer(self):
        # Mirror of the AppException handler's formatting in fal.api.api.
        for units in (0, 3, 0.5, 1.23e-05, 42.0):
            expected = format(float(units), ".0f" if isinstance(units, int) else ".8f")
            assert format_billable_units(units) == expected
