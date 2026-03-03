from __future__ import annotations

import grpc
import pytest

from fal.api.api import (
    FalServerlessError,
    _classify_unavailable_error,
    _handle_grpc_error,
)


class _FakeRpcError(grpc.RpcError):
    """A raisable gRPC error for testing."""

    def __init__(self, code, details, debug_error_string=None):
        self._code = code
        self._details = details
        self._debug_error_string = debug_error_string

    def code(self):
        return self._code

    def details(self):
        return self._details

    def debug_error_string(self):
        if self._debug_error_string is None:
            raise AttributeError("no debug info")
        return self._debug_error_string


class TestClassifyUnavailableError:
    def test_dns_resolution_failed(self):
        cause, guidance = _classify_unavailable_error("DNS resolution failed", None)
        assert cause == "DNS resolution failed"

    def test_name_resolution_failure(self):
        cause, _ = _classify_unavailable_error(
            None, '{"description":"Name resolution failure"}'
        )
        assert cause == "DNS resolution failed"

    def test_connection_refused(self):
        cause, _ = _classify_unavailable_error("Connection refused", None)
        assert cause == "Connection refused"

    def test_econnrefused(self):
        cause, _ = _classify_unavailable_error(None, "ECONNREFUSED")
        assert cause == "Connection refused"

    def test_deadline_exceeded(self):
        cause, _ = _classify_unavailable_error("Deadline Exceeded", None)
        assert cause == "Connection timed out"

    def test_context_deadline_exceeded(self):
        cause, _ = _classify_unavailable_error("context deadline exceeded", None)
        assert cause == "Connection timed out"

    def test_ssl(self):
        cause, _ = _classify_unavailable_error("SSL handshake failed", None)
        assert cause == "TLS/SSL handshake failed"

    def test_tls(self):
        cause, _ = _classify_unavailable_error("TLS error", None)
        assert cause == "TLS/SSL handshake failed"

    def test_certificate(self):
        cause, _ = _classify_unavailable_error(None, "certificate verify failed")
        assert cause == "TLS/SSL handshake failed"

    def test_connection_reset(self):
        cause, _ = _classify_unavailable_error("Connection reset by peer", None)
        assert cause == "Connection reset by server"

    def test_econnreset(self):
        cause, _ = _classify_unavailable_error(None, "ECONNRESET")
        assert cause == "Connection reset by server"

    def test_no_route_to_host(self):
        cause, _ = _classify_unavailable_error("No route to host", None)
        assert cause == "No route to host"

    def test_ehostunreach(self):
        cause, _ = _classify_unavailable_error(None, "EHOSTUNREACH")
        assert cause == "No route to host"

    def test_network_unreachable(self):
        cause, _ = _classify_unavailable_error("Network is unreachable", None)
        assert cause == "Network unreachable"

    def test_enetunreach(self):
        cause, _ = _classify_unavailable_error(None, "ENETUNREACH")
        assert cause == "Network unreachable"

    def test_socket_closed(self):
        cause, _ = _classify_unavailable_error("Socket closed", None)
        assert cause == "Connection closed unexpectedly"

    def test_goaway(self):
        cause, _ = _classify_unavailable_error(None, "received GOAWAY")
        assert cause == "Server sent GOAWAY"

    def test_unknown_error(self):
        assert _classify_unavailable_error("something weird", None) is None

    def test_none_inputs(self):
        assert _classify_unavailable_error(None, None) is None

    def test_empty_inputs(self):
        assert _classify_unavailable_error("", "") is None

    def test_case_insensitive(self):
        cause, _ = _classify_unavailable_error("DNS RESOLUTION FAILED", None)
        assert cause == "DNS resolution failed"

    def test_first_match_wins(self):
        # DNS match should take priority over socket closed
        cause, _ = _classify_unavailable_error(
            "DNS resolution failed and socket closed", None
        )
        assert cause == "DNS resolution failed"


class TestHandleGrpcErrorUnavailable:
    def _call_decorated(self, error):
        @_handle_grpc_error()
        def failing_func():
            raise error

        failing_func()

    def test_includes_cause_in_message(self):
        error = _FakeRpcError(
            grpc.StatusCode.UNAVAILABLE,
            "DNS resolution failed",
            '{"description":"DNS resolution failed"}',
        )
        with pytest.raises(FalServerlessError, match="Cause: DNS resolution failed"):
            self._call_decorated(error)

    def test_includes_debug_info(self):
        debug = '{"description":"Connection refused"}'
        error = _FakeRpcError(grpc.StatusCode.UNAVAILABLE, "Connection refused", debug)
        with pytest.raises(FalServerlessError, match=r"debug: .+Connection refused"):
            self._call_decorated(error)

    def test_message_starts_with_could_not_reach(self):
        error = _FakeRpcError(grpc.StatusCode.UNAVAILABLE, "Socket closed", None)
        with pytest.raises(FalServerlessError, match="^Could not reach fal host"):
            self._call_decorated(error)

    def test_debug_error_string_failure_handled(self):
        error = _FakeRpcError(grpc.StatusCode.UNAVAILABLE, "Socket closed", None)
        with pytest.raises(FalServerlessError, match="Cause: Connection closed"):
            self._call_decorated(error)

    def test_no_debug_suffix_when_debug_unavailable(self):
        error = _FakeRpcError(grpc.StatusCode.UNAVAILABLE, "Socket closed", None)
        with pytest.raises(FalServerlessError) as exc_info:
            self._call_decorated(error)
        assert "debug:" not in exc_info.value.message

    def test_unclassified_uses_original_message(self):
        error = _FakeRpcError(grpc.StatusCode.UNAVAILABLE, "something unexpected", None)
        with pytest.raises(FalServerlessError, match="transient problem"):
            self._call_decorated(error)

    def test_unclassified_no_cause_label(self):
        error = _FakeRpcError(grpc.StatusCode.UNAVAILABLE, "something unexpected", None)
        with pytest.raises(FalServerlessError) as exc_info:
            self._call_decorated(error)
        assert "Cause:" not in exc_info.value.message

    def test_non_unavailable_error_unchanged(self):
        error = _FakeRpcError(grpc.StatusCode.INTERNAL, "internal server error", None)
        with pytest.raises(FalServerlessError, match="internal server error"):
            self._call_decorated(error)
