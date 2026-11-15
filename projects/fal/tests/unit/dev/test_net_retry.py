"""Flaky connects: retried in the transport, never leave a box billing."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import grpc
import httpx
import pytest

from fal.dev import box as box_module
from fal.dev import transport
from fal.dev.errors import DevError
from fal.sdk import RunnerState

from .fakes import make_box, make_runner

URL = "https://fal.run/me/dev-x/__fal_dev/status"


@pytest.fixture(autouse=True)
def sleeps(monkeypatch):
    """Record the sleeps instead of waiting (time is one module for all callers)."""
    slept = []
    monkeypatch.setattr(transport.time, "sleep", slept.append)
    return slept


def use_handler(monkeypatch, handler):
    """Route post_json through `handler`; returns the list of requests it saw."""
    seen = []

    def wrapped(request):
        seen.append(request)
        return handler(len(seen), request)

    monkeypatch.setattr(
        transport,
        "_pool",
        [httpx.Client(transport=httpx.MockTransport(wrapped))],
    )
    return seen


def post():
    return transport.post_json(URL, {}, {}, step="testing")


def test_connect_errors_then_success(monkeypatch):
    def handler(n, request):
        if n <= 3:
            raise httpx.ConnectError("Errno 60", request=request)
        return httpx.Response(200, json={"ok": True})

    seen = use_handler(monkeypatch, handler)

    assert post().json() == {"ok": True}
    assert len(seen) == 4


def test_connect_timeouts_are_retried_too(monkeypatch):
    def handler(n, request):
        if n == 1:
            raise httpx.ConnectTimeout("slow", request=request)
        return httpx.Response(200)

    seen = use_handler(monkeypatch, handler)

    assert post().status_code == 200
    assert len(seen) == 2


def test_gives_up_naming_host_and_step(monkeypatch, sleeps):
    def handler(n, request):
        raise httpx.ConnectError("Errno 60", request=request)

    seen = use_handler(monkeypatch, handler)

    with pytest.raises(DevError) as caught:
        post()
    assert len(seen) == transport.CONNECT_TRIES
    assert sleeps == [1.0, 2.0, 3.0]
    message = str(caught.value)
    assert "could not reach fal.run while testing after 4 tries" in message


def test_requests_use_a_short_connect_and_the_given_read_timeout(monkeypatch):
    seen = use_handler(monkeypatch, lambda n, request: httpx.Response(200))

    post()

    timeout = seen[0].extensions["timeout"]
    assert timeout["connect"] == transport.CONNECT_TIMEOUT
    assert timeout["read"] == 60


def test_read_timeout_is_not_retried(monkeypatch):
    def handler(n, request):
        raise httpx.ReadTimeout("slow", request=request)

    seen = use_handler(monkeypatch, handler)

    with pytest.raises(httpx.ReadTimeout):
        post()
    assert len(seen) == 1


@pytest.mark.parametrize(
    "error", [httpx.ReadError, httpx.RemoteProtocolError, httpx.WriteError]
)
def test_mid_request_failures_are_not_retried(monkeypatch, error):
    def handler(n, request):
        raise error("dropped", request=request)

    seen = use_handler(monkeypatch, handler)

    with pytest.raises(error):
        post()
    assert len(seen) == 1


@pytest.mark.parametrize("status", [500, 502, 503])
def test_server_errors_are_not_retried(monkeypatch, status):
    seen = use_handler(monkeypatch, lambda n, request: httpx.Response(status))

    assert post().status_code == status
    assert len(seen) == 1


def test_one_pooled_client_is_reused(monkeypatch):
    monkeypatch.setattr(transport, "_pool", [])

    assert transport.client() is transport.client()


def test_volume_create_retries_connect_errors():
    box = make_box()
    request = httpx.Request("POST", "https://rest.alpha.fal.ai/volumes/")
    box.client._rest_url = "https://rest.alpha.fal.ai"
    box.client.volumes.create.side_effect = [
        httpx.ConnectError("Errno 60", request=request),
        httpx.ConnectTimeout("slow", request=request),
        "dev-x",
    ]

    assert box.ensure_volume()
    assert box.client.volumes.create.call_count == 3


def test_volume_create_uses_the_short_connect_timeout():
    box = make_box()
    box.client._rest_url = "https://rest.alpha.fal.ai"
    box.client.volumes.create.return_value = "dev-x"

    box.ensure_volume()

    kwargs = box.client.volumes.create.call_args.kwargs
    assert kwargs == {"connect_timeout": transport.CONNECT_TIMEOUT}


def test_the_volumes_namespace_passes_the_connect_timeout_on(monkeypatch):
    from fal.api import client as api_client

    seen = {}

    def fake_create(sdk, name, **kwargs):
        seen.update(name=name, **kwargs)
        return name

    monkeypatch.setattr(api_client.volumes_api, "create_volume", fake_create)

    api_client.VolumesNamespace(SimpleNamespace()).create("v", connect_timeout=3.0)

    assert seen == {"name": "v", "connect_timeout": 3.0}


def test_create_volume_applies_the_connect_timeout(monkeypatch):
    from fal.api import volumes

    seen = {}

    class FakeClient:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, path, json):
            return httpx.Response(201, json={"name": "dev-x"})

    rest = SimpleNamespace(
        base_url="https://rest.alpha.fal.ai",
        get_headers=lambda: {},
        get_timeout=lambda: 30,
        verify_ssl=True,
        follow_redirects=True,
    )
    sdk = SimpleNamespace(_create_rest_client=lambda: rest)
    monkeypatch.setattr(volumes.httpx, "Client", FakeClient)

    assert volumes.create_volume(sdk, "dev-x", connect_timeout=3.0) == "dev-x"
    assert seen["timeout"].connect == 3.0
    assert seen["timeout"].read == 30

    volumes.create_volume(sdk, "dev-x")
    assert seen["timeout"] == 30


def test_volume_create_names_the_step_when_retries_run_out():
    box = make_box()
    request = httpx.Request("POST", "https://rest.alpha.fal.ai/volumes/")
    box.client.volumes.create.side_effect = httpx.ConnectError(
        "Errno 60", request=request
    )
    box.client._rest_url = "https://rest.alpha.fal.ai"

    with pytest.raises(DevError, match="while creating the volume after 4 tries"):
        box.ensure_volume()
    assert box.client.volumes.create.call_count == 4


def test_readiness_poll_survives_connect_failures():
    box = make_box()
    request = httpx.Request("GET", "https://rest.alpha.fal.ai/x")
    ready = make_runner("r1", RunnerState.RUNNING)
    box.client.apps.runners.side_effect = [
        httpx.ConnectError("Errno 60", request=request)
    ] * 5 + [[ready]]

    assert box.wait_until_ready(timeout=3600) is ready
    # Each poll tries three times; the first gave up, the second got through.
    assert box.poll_failures == 1


def test_start_stops_the_box_and_says_so_when_never_reachable(monkeypatch, capsys):
    box = make_box(min_concurrency=0)
    request = httpx.Request("GET", "https://rest.alpha.fal.ai/x")
    error = httpx.ConnectError("Errno 60", request=request)
    # The first listing (was it live?) works, then the network is gone.
    box.client.apps.runners.side_effect = [[]] + [error] * 100
    clock = iter(range(0, 10_000, 20))
    monkeypatch.setattr(box_module.time, "time", lambda: next(clock))
    box.stop = MagicMock(return_value=True)

    with pytest.raises(DevError, match="could not connect"):
        box.start(timeout=100)

    box.stop.assert_called_once()
    assert box.start_cancelled is True
    err = capsys.readouterr().err
    assert "stopped box devbox-demo: could not reach it after" in err
    assert "failed polls" in err


def test_readiness_poll_raises_errors_that_are_not_the_network():
    box = make_box()
    box.client.apps.runners.side_effect = ValueError("bad credentials")

    with pytest.raises(ValueError, match="bad credentials"):
        box.wait_until_ready(timeout=3600)
    assert box.poll_failures == 0


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (httpx.ConnectError("x"), True),
        (httpx.ReadTimeout("x"), True),
        (TimeoutError("Operation timed out"), True),
        (ConnectionResetError("reset"), True),
        (httpx.RemoteProtocolError("closed"), True),
        (FileNotFoundError("no credentials"), False),
        (httpx.UnsupportedProtocol("bad scheme"), False),
        (ValueError("x"), False),
        (KeyError("x"), False),
    ],
)
def test_is_network_error(error, expected):
    assert box_module.is_network_error(error) is expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("UNAVAILABLE", True),
        ("DEADLINE_EXCEEDED", True),
        ("PERMISSION_DENIED", False),
        ("NOT_FOUND", False),
    ],
)
def test_is_network_error_for_grpc(name, expected):
    class Error(grpc.RpcError):
        def code(self):
            return getattr(grpc.StatusCode, name)

    assert box_module.is_network_error(Error()) is expected


def test_a_bare_rpc_error_without_a_code_is_not_a_network_error():
    assert box_module.is_network_error(grpc.RpcError()) is False


def test_readiness_timeout_after_a_recovered_poll_does_not_blame_the_network(
    monkeypatch, capsys
):
    box = make_box(min_concurrency=0)
    request = httpx.Request("GET", "https://rest.alpha.fal.ai/x")
    error = httpx.ConnectError("Errno 60", request=request)
    # Listed once (was it live?), one blip, then polls that work but find no runner.
    box.client.apps.runners.side_effect = [[], error, error, error] + [[]] * 100
    clock = iter(range(0, 10_000, 20))
    monkeypatch.setattr(box_module.time, "time", lambda: next(clock))
    box.stop = MagicMock(return_value=True)

    with pytest.raises(DevError) as caught:
        box.start(timeout=100)

    assert "could not connect" not in str(caught.value)
    assert "could not reach it" not in capsys.readouterr().err


def test_an_interrupted_start_does_not_blame_the_network(monkeypatch, capsys):
    box = make_box(min_concurrency=0)
    request = httpx.Request("GET", "https://rest.alpha.fal.ai/x")
    error = httpx.ConnectError("Errno 60", request=request)
    # A first listing, a failed poll, then the user interrupts.
    box.client.apps.runners.side_effect = [[], error, error, error, KeyboardInterrupt]
    monkeypatch.setattr(box_module.time, "time", lambda: 0)
    box.stop = MagicMock(return_value=True)

    with pytest.raises(KeyboardInterrupt):
        box.start(timeout=100)

    box.stop.assert_called_once()
    assert "could not reach it" not in capsys.readouterr().err


def test_a_second_start_forgets_the_first_ones_poll_failures(monkeypatch):
    box = make_box(min_concurrency=0)
    box.poll_failures = 5
    box.poll_error = httpx.ConnectError("old")
    box.client.apps.runners.side_effect = [[], [make_runner("r1")]]
    monkeypatch.setattr(box_module.time, "time", lambda: 0)

    box.start(timeout=100)

    assert box.poll_failures == 0
    assert box.poll_error is None
