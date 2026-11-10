import gzip
import json

import pytest

from fal.dev.account import Account

pytestmark = pytest.mark.usefixtures("dev_account")


def _fake_profile_box(monkeypatch, trace_bytes):
    import base64

    from fal.dev.app import client

    trace = gzip.compress(json.dumps({"traceEvents": [{"name": "k"}]}).encode())
    files = {"trace.json.gz": trace, "ops.txt": b"ops table\n"}
    calls = []

    def call(_session, action, payload=None, **_):
        calls.append((action, payload))
        if action == "status":
            return {"phase": "ready", "module": "demo_app"}
        if action == "profile":
            return {
                "id": "e" * 32,
                "status": "ok",
                "elapsed_seconds": 2.5,
                "trace_bytes": trace_bytes or len(trace),
                "top_ops": "TOP OPS",
                "request": {"status_code": 200},
            }
        data = files[payload["name"]]
        piece = data[payload["offset"] : payload["offset"] + 7]
        return {"size": len(data), "data": base64.b64encode(piece).decode()}

    monkeypatch.setattr(client, "call", call)
    return calls, trace


def _profile_session(tmp_path, monkeypatch):
    from fal.dev.app import session as app_session

    monkeypatch.setattr(app_session, "SESSIONS_DIR", tmp_path / "sessions")
    app_session.Session(
        name="dev-demo-me",
        file_path="/src/demo_app.py",
        root="/src",
        module="demo_app",
        class_name="Demo",
        run_url="https://fal.run/me/dev-demo-me",
        token="t",
        account=Account("me", True),
    ).save()


def test_profile_command_downloads_the_trace(tmp_path, monkeypatch, capsys):
    from fal.cli.main import parse_args

    _profile_session(tmp_path, monkeypatch)
    calls, trace = _fake_profile_box(monkeypatch, None)
    out = tmp_path / "profiles"
    args = parse_args(["dev", "app", "profile", "/", "--output-dir", str(out)])
    args.func(args)

    (folder,) = out.iterdir()
    assert (folder / "trace.json.gz").read_bytes() == trace
    assert (folder / "ops.txt").read_text() == "ops table\n"
    printed = capsys.readouterr().out
    assert "TOP OPS" in printed
    assert "ui.perfetto.dev" in printed
    # Without --data the request body is empty.
    (sent,) = (p for action, p in calls if action == "profile")
    assert sent["body"] == {}


def test_profile_command_sends_the_given_body(tmp_path, monkeypatch):
    from fal.cli.main import parse_args

    _profile_session(tmp_path, monkeypatch)
    calls, _ = _fake_profile_box(monkeypatch, None)
    args = parse_args(
        ["dev", "app", "profile", "/", "--data", '{"prompt": "x"}', "--light"]
        + ["--output-dir", str(tmp_path / "profiles")]
    )
    args.func(args)
    (sent,) = (p for action, p in calls if action == "profile")
    assert sent["body"] == {"prompt": "x"}
    assert sent["mode"] == "light"


def test_profile_command_skips_traces_over_the_cap(tmp_path, monkeypatch, capsys):
    from fal.cli.main import parse_args

    _profile_session(tmp_path, monkeypatch)
    calls, _ = _fake_profile_box(monkeypatch, 300 * 1024 * 1024)
    out = tmp_path / "profiles"
    args = parse_args(["dev", "app", "profile", "/", "--output-dir", str(out)])
    args.func(args)

    (folder,) = out.iterdir()
    assert not (folder / "trace.json.gz").exists()
    assert (folder / "ops.txt").exists()
    assert "Trace not downloaded" in capsys.readouterr().out
