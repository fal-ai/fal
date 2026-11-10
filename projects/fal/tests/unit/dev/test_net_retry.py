"""Flaky connects: retried in the transport, never leave a box billing."""

import io
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import grpc
import httpx
import pytest
from rich.console import Console

from fal.dev import box as box_module
from fal.dev import transport
from fal.dev.app import client as app_client
from fal.dev.cli import app as cli_app
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


def test_wait_ready_keeps_polling_through_connect_failures(monkeypatch):
    replies = [
        transport.ConnectFailed("down"),
        transport.ConnectFailed("down"),
        {"lines": [], "total": 3},
        {"phase": "ready", "detail": ""},
    ]

    def fake_call(session, action, payload=None, **kwargs):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(app_client, "call", fake_call)

    state = app_client.wait_ready(SimpleNamespace(), after=0, timeout=3600)

    assert state["phase"] == "ready"


def test_wait_ready_raises_the_connect_error_if_it_never_clears(monkeypatch):
    def fake_call(session, action, payload=None, **kwargs):
        raise transport.ConnectFailed("could not reach x")

    monkeypatch.setattr(app_client, "call", fake_call)
    clock = iter(range(0, 1000, 10))
    monkeypatch.setattr(app_client.time, "monotonic", lambda: next(clock))

    with pytest.raises(transport.ConnectFailed):
        app_client.wait_ready(SimpleNamespace(), after=0, timeout=50)


def scripted_call(monkeypatch, replies, fallback):
    """Replace `call`: each reply (or raised error) in turn, then `fallback(action)`."""
    queue = list(replies)

    def fake_call(session, action, payload=None, **kwargs):
        reply = queue.pop(0) if queue else fallback(action)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(app_client, "call", fake_call)


def test_wait_ready_keeps_polling_through_gateway_errors(monkeypatch):
    scripted_call(
        monkeypatch,
        [
            transport.BoxUnreachable("logs failed (504): gateway timeout"),
            {"lines": [], "total": 1},
            {"phase": "ready", "detail": ""},
        ],
        lambda action: None,
    )

    assert app_client.wait_ready(SimpleNamespace(), after=0)["phase"] == "ready"


def test_wait_ready_timeout_after_a_blip_is_a_timeout_not_the_stale_error(
    monkeypatch,
):
    scripted_call(
        monkeypatch,
        [transport.ConnectFailed("down")],
        lambda action: (
            {"lines": [], "total": 1}
            if action == "logs"
            else {"phase": "starting", "detail": ""}
        ),
    )
    clock = iter(range(0, 1000, 10))
    monkeypatch.setattr(app_client.time, "monotonic", lambda: next(clock))

    with pytest.raises(DevError, match="Timed out") as caught:
        app_client.wait_ready(SimpleNamespace(), after=0, timeout=50)
    assert not isinstance(caught.value, transport.ConnectFailed)


@pytest.mark.parametrize(
    "message", ["logs failed (401): bad token", "status failed (500)"]
)
def test_wait_ready_fails_fast_on_an_answer_that_is_not_unreachable(
    monkeypatch, message
):
    calls = []

    def fake_call(session, action, payload=None, **kwargs):
        calls.append(action)
        assert len(calls) < 4, "kept polling after an answer that is not unreachable"
        raise DevError(message)

    monkeypatch.setattr(app_client, "call", fake_call)

    with pytest.raises(DevError, match="failed") as caught:
        app_client.wait_ready(SimpleNamespace(), after=0)
    assert not isinstance(caught.value, transport.BoxUnreachable)
    assert calls == ["logs"]


def test_wait_ready_idle_is_not_an_app_failure(monkeypatch):
    scripted_call(
        monkeypatch,
        [],
        lambda action: (
            {"lines": [], "total": 1}
            if action == "logs"
            else {"phase": "idle", "detail": "", "module": None}
        ),
    )

    with pytest.raises(DevError, match="no app running") as caught:
        app_client.wait_ready(SimpleNamespace(), after=0)
    assert not isinstance(caught.value, app_client.AppFailed)
    assert "fal dev app start" in str(caught.value)


def test_app_failure_says_the_runner_is_still_billing(monkeypatch):
    scripted_call(
        monkeypatch,
        [],
        lambda action: (
            {"lines": [], "total": 1}
            if action == "logs"
            else {"phase": "failed", "detail": "ImportError"}
        ),
    )

    with pytest.raises(app_client.AppFailed, match="still up and billing"):
        app_client.wait_ready(SimpleNamespace(), after=0)


def start_args():
    return SimpleNamespace(
        session_env=[],
        app_ref=("app.py", None),
        root=None,
        name=None,
        team=None,
        host=None,
        yes=True,
        replace=False,
        keep_alive=None,
        machine_type=None,
        env=None,
    )


def run_start(monkeypatch, tmp_path, boot):
    out = io.StringIO()
    monkeypatch.setattr(cli_app, "console", Console(file=out, width=200))
    file = tmp_path / "app.py"
    file.touch()
    session = SimpleNamespace(
        name="dev-x", root=str(tmp_path), delete=MagicMock(), account=None
    )
    stop_box = MagicMock(return_value=(["r1"], True))
    monkeypatch.setattr(
        cli_app.deploy,
        "resolve_app",
        lambda ref: (str(file), "App", SimpleNamespace(team=None)),
    )
    monkeypatch.setattr(cli_app.deploy, "package_root", lambda f: tmp_path)
    monkeypatch.setattr(cli_app.deploy, "start_box", lambda *a, **k: session)
    monkeypatch.setattr(cli_app.deploy, "stop_box", stop_box)
    monkeypatch.setattr(
        cli_app,
        "resolve_account",
        lambda *a: SimpleNamespace(team=None, runs_as="me", fal_team=None),
    )
    monkeypatch.setattr(cli_app, "announce", lambda account: None)
    monkeypatch.setattr(cli_app, "confirm_deploy", lambda *a, **k: None)
    monkeypatch.setattr(cli_app, "SyncServerlessClient", MagicMock())
    monkeypatch.setattr(cli_app.live, "set_phase", lambda *a: None)
    monkeypatch.setattr(cli_app.live, "clear_phase", lambda *a: None)
    monkeypatch.setattr(cli_app.live, "clear_sync", lambda *a: None)
    monkeypatch.setattr(cli_app.client, "boot", boot)
    monkeypatch.setattr(cli_app.time, "sleep", lambda _: None)
    return session, stop_box, out


def test_failed_start_stops_the_box_and_prints_why(monkeypatch, tmp_path):
    def boot(session):
        raise transport.ConnectFailed(
            "could not reach fal.run while calling the dev box (files) after 4 tries"
        )

    session, stop_box, out = run_start(monkeypatch, tmp_path, boot)

    with pytest.raises(transport.ConnectFailed):
        cli_app.cmd_start(start_args())

    stop_box.assert_called_once_with(session)
    session.delete.assert_called_once()
    assert "stopped box dev-x: could not reach fal.run" in out.getvalue()


def test_interrupted_start_stops_the_box(monkeypatch, tmp_path):
    def boot(session):
        raise KeyboardInterrupt

    session, stop_box, out = run_start(monkeypatch, tmp_path, boot)

    with pytest.raises(KeyboardInterrupt):
        cli_app.cmd_start(start_args())

    stop_box.assert_called_once_with(session)
    assert "stopped box dev-x: interrupted" in out.getvalue()


def test_a_second_interrupt_during_the_stop_still_reports(monkeypatch, tmp_path):
    def boot(session):
        raise transport.ConnectFailed("could not reach fal.run after 4 tries")

    session, stop_box, out = run_start(monkeypatch, tmp_path, boot)
    stop_box.side_effect = KeyboardInterrupt

    with pytest.raises(transport.ConnectFailed):
        cli_app.cmd_start(start_args())

    assert "may still be billing" in out.getvalue()


def test_failed_start_retries_a_flaky_stop(monkeypatch, tmp_path):
    def boot(session):
        raise transport.ConnectFailed("could not reach fal.run after 4 tries")

    session, stop_box, out = run_start(monkeypatch, tmp_path, boot)
    stop_box.side_effect = [OSError("Errno 60"), (["r1"], True)]

    with pytest.raises(transport.ConnectFailed):
        cli_app.cmd_start(start_args())

    assert stop_box.call_count == 2
    assert "stopped box dev-x" in out.getvalue()


def test_an_interrupt_while_waiting_to_stop_again_still_reports(monkeypatch, tmp_path):
    def boot(session):
        raise transport.ConnectFailed("could not reach fal.run after 4 tries")

    session, stop_box, out = run_start(monkeypatch, tmp_path, boot)
    stop_box.side_effect = OSError("Errno 60")

    def interrupted(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_app.time, "sleep", interrupted)

    with pytest.raises(transport.ConnectFailed):
        cli_app.cmd_start(start_args())

    assert stop_box.call_count == 1
    assert "may still be billing" in out.getvalue()


def test_an_interrupted_stop_names_the_error_type(monkeypatch, tmp_path):
    def boot(session):
        raise transport.ConnectFailed("could not reach fal.run after 4 tries")

    session, stop_box, out = run_start(monkeypatch, tmp_path, boot)
    stop_box.side_effect = KeyboardInterrupt

    with pytest.raises(transport.ConnectFailed):
        cli_app.cmd_start(start_args())

    assert "(KeyboardInterrupt)" in out.getvalue()


def test_failed_stop_says_the_box_may_still_bill(monkeypatch, tmp_path):
    def boot(session):
        raise transport.ConnectFailed("could not reach fal.run after 4 tries")

    session, stop_box, out = run_start(monkeypatch, tmp_path, boot)
    stop_box.side_effect = OSError("Errno 60")

    with pytest.raises(transport.ConnectFailed):
        cli_app.cmd_start(start_args())

    assert stop_box.call_count == cli_app.STOP_TRIES
    assert "may still be billing" in out.getvalue()
    session.delete.assert_not_called()


def test_app_code_failure_keeps_the_box_for_restart(monkeypatch, tmp_path):
    def boot(session):
        raise app_client.AppFailed("The app did not start (failed). Fix the code")

    session, stop_box, out = run_start(monkeypatch, tmp_path, boot)

    with pytest.raises(app_client.AppFailed):
        cli_app.cmd_start(start_args())

    stop_box.assert_not_called()


def run_status(monkeypatch, capsys, remote):
    session = SimpleNamespace(
        name="dev-x",
        file_path="app.py",
        class_name="App",
        run_url="https://fal.run/me/dev-x",
        account=SimpleNamespace(team="me"),
    )
    monkeypatch.setattr(cli_app, "_session", lambda args: session)
    if isinstance(remote, Exception):
        monkeypatch.setattr(cli_app.client, "status", MagicMock(side_effect=remote))
    else:
        monkeypatch.setattr(cli_app.client, "status", lambda s: remote)
    cli_app.cmd_status(SimpleNamespace())
    return json.loads(capsys.readouterr().out)


def test_status_flags_an_idle_box_with_no_app_as_billing(monkeypatch, capsys):
    info = run_status(monkeypatch, capsys, {"phase": "idle", "module": None})

    assert "billing" in info["billing"]
    assert "fal dev app restart" in info["billing"]


@pytest.mark.parametrize("phase", ["exited", "failed", "stopped"])
def test_status_flags_a_box_whose_app_stopped_as_billing(monkeypatch, capsys, phase):
    info = run_status(monkeypatch, capsys, {"phase": phase, "module": "app"})

    assert "billing" in info["billing"]


def test_status_has_no_billing_flag_for_a_ready_box(monkeypatch, capsys):
    info = run_status(monkeypatch, capsys, {"phase": "ready", "module": "app"})

    assert "billing" not in info


def test_status_has_no_billing_flag_when_the_box_is_unreachable(monkeypatch, capsys):
    info = run_status(monkeypatch, capsys, DevError("could not reach"))

    assert "billing" not in info
    assert info["remote"]["error"] == "could not reach"
