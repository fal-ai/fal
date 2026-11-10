import json
from types import SimpleNamespace

import pytest

from dev.app.fakes import make_tree_session
from fal.dev.app import session as app_session
from fal.dev.app.session import NAME_PREFIX, session_name
from fal.dev.errors import DevError
from fal.dev.transport import BoxUnreachable

pytestmark = pytest.mark.usefixtures("dev_account")


def test_session_names_are_always_dev_boxes():
    assert session_name("my-prod-app", "Demo").startswith(NAME_PREFIX)
    assert session_name("dev-x", "Demo") == "dev-x"
    assert "/" not in session_name("../../etc", "Demo")
    assert session_name(None, "Demo").startswith("dev-demo-")


def test_calls_to_the_box_switch_request_io_storage_off(tmp_path, monkeypatch):
    import httpx

    import fal.sdk

    session = make_tree_session(tmp_path, "a.py")
    sent = {}

    def handler(request):
        sent.update(headers=request.headers, json=json.loads(request.content))
        return httpx.Response(200, json={})

    monkeypatch.setattr(
        "fal.dev.transport._pool",
        [httpx.Client(transport=httpx.MockTransport(handler))],
    )
    monkeypatch.setattr(
        fal.sdk,
        "get_credentials",
        lambda team: SimpleNamespace(to_headers=lambda: {"Authorization": "Key k"}),
    )
    app_session.call(session, "status")

    assert sent["headers"]["x-fal-store-io"] == "0"
    assert sent["json"]["token"] == "t"


def test_a_box_that_cannot_be_reached_names_the_step(tmp_path, monkeypatch):
    import httpx

    import fal.sdk

    def handler(request):
        raise httpx.ConnectError("Errno 60", request=request)

    monkeypatch.setattr(
        "fal.dev.transport._pool",
        [httpx.Client(transport=httpx.MockTransport(handler))],
    )
    monkeypatch.setattr("fal.dev.transport.time.sleep", lambda _: None)
    monkeypatch.setattr(
        fal.sdk,
        "get_credentials",
        lambda team: SimpleNamespace(to_headers=lambda: {}),
    )

    with pytest.raises(DevError, match=r"calling the dev box \(status\) after 4"):
        app_session.call(make_tree_session(tmp_path, "a.py"), "status")


def test_a_read_timeout_is_unreachable_and_not_retried(tmp_path, monkeypatch):
    import httpx

    import fal.sdk

    seen = []

    def handler(request):
        seen.append(request)
        raise httpx.ReadTimeout("slow", request=request)

    monkeypatch.setattr(
        "fal.dev.transport._pool",
        [httpx.Client(transport=httpx.MockTransport(handler))],
    )
    monkeypatch.setattr(
        fal.sdk,
        "get_credentials",
        lambda team: SimpleNamespace(to_headers=lambda: {}),
    )

    with pytest.raises(BoxUnreachable, match="Could not reach the dev box"):
        app_session.call(make_tree_session(tmp_path, "a.py"), "status")
    assert len(seen) == 1


@pytest.mark.parametrize(
    ("status", "unreachable"),
    [(502, True), (503, True), (504, True), (500, False), (404, False), (400, False)],
)
def test_gateway_errors_are_unreachable_not_the_boxs_answer(
    tmp_path, monkeypatch, status, unreachable
):
    import httpx

    import fal.sdk

    monkeypatch.setattr(
        "fal.dev.transport._pool",
        [httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(status)))],
    )
    monkeypatch.setattr(
        fal.sdk,
        "get_credentials",
        lambda team: SimpleNamespace(to_headers=lambda: {}),
    )

    with pytest.raises(DevError) as caught:
        app_session.call(make_tree_session(tmp_path, "a.py"), "status")
    assert isinstance(caught.value, BoxUnreachable) is unreachable


def test_an_old_session_file_is_called_out(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(app_session, "SESSIONS_DIR", tmp_path)
    (tmp_path / "dev-old.json").write_text('{"name": "dev-old", "app_name": "x"}')
    assert app_session.load_sessions() == []
    err = capsys.readouterr().err
    assert "dev-old.json" in err and "fal apps delete dev-old" in err
    assert err.count("\n") == 1
