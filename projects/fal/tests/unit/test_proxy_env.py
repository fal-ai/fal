from __future__ import annotations

import httpx
import pytest

from fal import _proxy
from fal.dev import transport
from fal.dev.cli import parser as dev_parser

PROXY_VARS = (
    "HTTPS_PROXY",
    "https_proxy",
    "NO_PROXY",
    "no_proxy",
    "grpc_proxy",
    "FAL_DEV_CONNECT_TIMEOUT",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in PROXY_VARS:
        monkeypatch.delenv(name, raising=False)


def test_grpc_gets_proxy_from_uppercase_var(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8898")
    assert _proxy.grpc_proxy_options("api.alpha.fal.ai") == [
        ("grpc.http_proxy", "http://127.0.0.1:8898")
    ]


def test_grpc_proxy_from_lowercase_var(monkeypatch):
    monkeypatch.setenv("https_proxy", "http://p:1")
    assert _proxy.grpc_proxy_options("api.alpha.fal.ai") == [
        ("grpc.http_proxy", "http://p:1")
    ]


def test_grpc_no_proxy_without_env():
    assert _proxy.grpc_proxy_options("api.alpha.fal.ai") == []


def test_grpc_honors_no_proxy(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://p:1")
    monkeypatch.setenv("NO_PROXY", "localhost,.fal.ai")
    assert _proxy.grpc_proxy_options("api.alpha.fal.ai") == [
        ("grpc.enable_http_proxy", 0)
    ]


def test_grpc_no_proxy_disables_lowercase_proxy(monkeypatch):
    monkeypatch.setenv("https_proxy", "http://p:1")
    monkeypatch.setenv("NO_PROXY", ".fal.ai")
    assert _proxy.grpc_proxy_options("api.alpha.fal.ai:443") == [
        ("grpc.enable_http_proxy", 0)
    ]


def test_explicit_grpc_proxy_wins(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://p:1")
    monkeypatch.setenv("grpc_proxy", "http://other:2")
    assert _proxy.grpc_proxy_options("api.alpha.fal.ai") == []


def test_redact_hides_credentials():
    assert _proxy.redact("http://u:pw@corp:3128") == "http://***@corp:3128"
    assert _proxy.redact("http://corp:3128") == "http://corp:3128"


def test_connect_failed_names_host_and_proxy(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://u:pw@127.0.0.1:1")
    with pytest.raises(transport.ConnectFailed) as info:
        transport.retry_connect(
            lambda: (_ for _ in ()).throw(httpx.ConnectError("refused")),
            host="fal.run",
            step="probing",
            tries=1,
        )
    message = str(info.value)
    assert "fal.run" in message
    assert "http://***@127.0.0.1:1" in message
    assert "pw" not in message
    assert "hint:" in message


def test_connect_failed_without_proxy_has_no_hint():
    with pytest.raises(transport.ConnectFailed) as info:
        transport.retry_connect(
            lambda: (_ for _ in ()).throw(httpx.ConnectError("refused")),
            host="fal.run",
            step="probing",
            tries=1,
        )
    assert "proxy" not in str(info.value)


def test_connect_timeout_env(monkeypatch):
    assert transport._connect_timeout() == 3.0
    monkeypatch.setenv("FAL_DEV_CONNECT_TIMEOUT", "12.5")
    assert transport._connect_timeout() == 12.5
    monkeypatch.setenv("FAL_DEV_CONNECT_TIMEOUT", "inf")
    assert transport._connect_timeout() == 3.0
    monkeypatch.setenv("FAL_DEV_CONNECT_TIMEOUT", "nope")
    assert transport._connect_timeout() == 3.0


def test_unreachable_ignores_other_errors():
    assert dev_parser._unreachable(ValueError("x")) is None


def test_unreachable_wraps_httpx_connect_error(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    request = httpx.Request("GET", "https://rest.alpha.fal.ai/x")
    err = dev_parser._unreachable(httpx.ConnectError("refused", request=request))
    assert err is not None
    assert "rest.alpha.fal.ai through proxy http://127.0.0.1:1" in str(err)
    assert dev_parser._unreachable(httpx.ReadTimeout("slow", request=request)) is None
