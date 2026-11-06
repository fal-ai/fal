from types import SimpleNamespace

import pytest

from dev.app.fakes import make_tree_session
from fal.dev.app import session as app_session
from fal.dev.app.session import NAME_PREFIX, session_name

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

    def post(url, **kwargs):
        sent.update(kwargs)
        return httpx.Response(200, json={})

    monkeypatch.setattr("fal.dev.transport.httpx.post", post)
    monkeypatch.setattr(
        fal.sdk,
        "get_credentials",
        lambda team: SimpleNamespace(to_headers=lambda: {"Authorization": "Key k"}),
    )
    app_session.call(session, "status")

    assert sent["headers"]["X-Fal-Store-IO"] == "0"
    assert sent["json"]["token"] == "t"


def test_an_old_session_file_is_called_out(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(app_session, "SESSIONS_DIR", tmp_path)
    (tmp_path / "dev-old.json").write_text('{"name": "dev-old", "app_name": "x"}')
    assert app_session.load_sessions() == []
    err = capsys.readouterr().err
    assert "dev-old.json" in err and "fal apps delete dev-old" in err
    assert err.count("\n") == 1
