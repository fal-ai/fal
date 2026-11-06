import base64
import io
import tarfile

import pytest

from dev.app.fakes import FakeDepsBox, make_cli_session, make_tree_session
from fal.cli.main import parse_args
from fal.dev.app import client
from fal.dev.app import session as app_session
from fal.dev.errors import DevError

pytestmark = pytest.mark.usefixtures("dev_account")


def _uploaded(archive: str) -> list[str]:
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(archive))) as tar:
        return sorted(tar.getnames())


def test_sync_uploads_only_what_the_store_lacks(tmp_path, monkeypatch):
    session = make_tree_session(tmp_path, "a.py", "b.py", "c.py")
    monkeypatch.setattr(app_session, "SESSIONS_DIR", tmp_path / "sessions")
    calls = []

    def call(session, action, payload=None, *, timeout=60):
        calls.append((action, payload))
        if action == "place":
            return {"placed": sorted(set(payload["files"]) - {"c.py"})}
        return {}

    monkeypatch.setattr(client, "call", call)
    result = client.sync_files(session, full=True, reload=False)

    assert result.placed == 2
    (files_call,) = (p for action, p in calls if action == "files")
    assert _uploaded(files_call["archive"]) == ["c.py"]
    assert files_call["all_files"] == ["a.py", "b.py", "c.py"]
    assert session.manifest.keys() == {"a.py", "b.py", "c.py"}

    # A small edit is uploaded directly: no extra round trip to the store.
    calls.clear()
    (tmp_path / "tree" / "a.py").write_text("edited")
    result = client.sync_files(session)
    assert [action for action, _ in calls] == ["files"]
    assert _uploaded(calls[0][1]["archive"]) == ["a.py"]
    assert result.changed == ["a.py"] and result.placed == 0


def test_a_big_incremental_sync_asks_the_store(tmp_path, monkeypatch):
    session = make_tree_session(tmp_path, "a.py", "b.py", "c.py")
    monkeypatch.setattr(app_session, "SESSIONS_DIR", tmp_path / "sessions")
    calls = []

    def call(session, action, payload=None, *, timeout=60):
        calls.append(action)
        return {"placed": []} if action == "place" else {}

    monkeypatch.setattr(client, "call", call)
    client.sync_files(session, reload=False)
    calls.clear()
    monkeypatch.setattr(client, "PLACE_MIN_FILES", 1)
    for name in ("a.py", "b.py"):
        (tmp_path / "tree" / name).write_text("edit " + name)
    client.sync_files(session, reload=False)
    assert calls == ["place", "files"]
    # Bytes count too.
    calls.clear()
    monkeypatch.setattr(client, "PLACE_MIN_FILES", 100)
    monkeypatch.setattr(client, "PLACE_MIN_BYTES", 5)
    (tmp_path / "tree" / "c.py").write_text("a longer edit")
    client.sync_files(session, reload=False)
    assert calls == ["place", "files"]


def test_sync_falls_back_to_uploading(tmp_path, monkeypatch):
    """No store, a failed place, or a damaged object: those files are sent."""
    session = make_tree_session(tmp_path, "a.py", "b.py", "c.py")
    monkeypatch.setattr(app_session, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(client, "MAX_PLACE_FILES", 1)
    places = []

    def call(session, action, payload=None, *, timeout=60):
        if action == "place":
            places.append(sorted(payload["files"]))
            (rel,) = payload["files"]
            if rel == "b.py":
                raise DevError("place failed (400): disk error")
            return {"placed": [] if rel == "c.py" else [rel]}
        return {}

    monkeypatch.setattr(client, "call", call)
    result = client.sync_files(session, full=True, reload=False)
    # One file per place request, as MAX_PLACE_FILES says.
    assert places == [["a.py"], ["b.py"], ["c.py"]]
    assert result.placed == 1
    assert result.changed == ["a.py", "b.py", "c.py"]

    # A box without a store places nothing, and gets the whole tree.
    monkeypatch.setattr(
        client, "call", lambda s, a, p=None, **k: {"placed": []} if a == "place" else {}
    )
    result = client.sync_files(session, full=True, reload=False)
    assert result.placed == 0


def test_sync_announces_withheld_files_once_per_change(tmp_path, monkeypatch, capsys):
    session = make_tree_session(tmp_path, "a.py", "server.pem")
    monkeypatch.setattr(app_session, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(client, "call", lambda *a, **k: {"placed": []})

    def announced() -> bool:
        return "server.pem" in capsys.readouterr().out

    client.sync_files(session, reload=False)
    assert announced()
    client.sync_files(session, reload=False)
    assert not announced()
    (tmp_path / "tree" / "tls.key").write_text("k")
    client.sync_files(session, reload=False)
    assert announced()
    # A full sync (a new runner) says it again.
    client.sync_files(session, full=True, reload=False)
    assert announced()


def test_a_sync_that_needs_a_rebuild_is_repeated(tmp_path, monkeypatch):
    session = make_tree_session(tmp_path, "a.py")
    monkeypatch.setattr(app_session, "SESSIONS_DIR", tmp_path / "sessions")

    def call(session, action, payload=None, *, timeout=60):
        if action == "place":
            return {"placed": []}
        return {"reload": {"action": "rebuild", "reason": "requirements changed"}}

    monkeypatch.setattr(client, "call", call)
    client.sync_files(session)
    assert session.manifest == {}


def test_boot_ships_the_sdk_and_waits_for_setup(tmp_path, monkeypatch):
    session = make_tree_session(tmp_path, "a.py")
    monkeypatch.setattr(app_session, "SESSIONS_DIR", tmp_path / "sessions")
    sent = {}

    def call(session, action, payload=None, *, timeout=60):
        sent[action] = payload
        if action == "logs":
            return {"lines": [], "total": 0}
        if action == "status":
            return {"phase": "ready", "detail": "", "module": "a"}
        return {"placed": []}

    monkeypatch.setattr(client, "call", call)
    assert client.boot(session)["phase"] == "ready"
    start = sent["start"]
    assert (start["module"], start["class_name"]) == ("a", "A")
    assert start["layout"] == {"paths": ["."], "workdir": "."}
    sdk = _uploaded(start["sdk_archive"])
    assert "fal/dev/remote/child.py" in sdk and "fal/app.py" in sdk
    assert not any("__pycache__" in name for name in sdk)
    # The shipped SDK's own dependencies, minus optional extras.
    assert start["sdk_requirements"]
    assert not any("extra ==" in req for req in start["sdk_requirements"])


def test_a_box_on_a_new_runner_is_booted_again(tmp_path, monkeypatch):
    session = make_tree_session(tmp_path, "a.py")
    booted = []
    monkeypatch.setattr(client, "boot", lambda session: booted.append(1))
    status = {"phase": "idle", "module": ""}
    monkeypatch.setattr(client, "call", lambda *a, **k: status)
    assert client.ensure_booted(session) is True
    status["module"] = "a"
    assert client.ensure_booted(session) is False
    assert booted == [1]


def test_cli_remembers_the_overlay_and_replays_it_on_a_new_runner(
    tmp_path, monkeypatch, capsys
):
    from fal.dev.app import bench as app_bench

    box = FakeDepsBox()
    make_cli_session(tmp_path, monkeypatch, box)
    args = parse_args(
        ["dev", "app", "deps", "install", "diffusers==0.31.0", "--no-restart"]
        + ["--index-url", "https://u:secret@pypi.example/simple"]
    )
    args.func(args)
    install = next(p for a, p in box.sent if a == "deps")
    assert install["packages"] == ["diffusers==0.31.0"]
    assert install["restart"] is False
    assert "Installed diffusers==0.31.0, safetensors==0.5.0" in capsys.readouterr().out

    session = app_session.find_session(None)
    assert session.overlay["packages"] == ["diffusers==0.31.0", "safetensors==0.5.0"]
    run = app_bench.record(
        name="x",
        session=session,
        path="/",
        body={},
        result={"runs": [{"elapsed_seconds": 1.0}], "warmup_runs": []},
        warmups=0,
    )
    assert run["overlay"] == ["diffusers==0.31.0", "safetensors==0.5.0"]

    # A new runner: the box has no overlay; replay installs it, pinned.
    box.state = {"packages": []}
    box.sent.clear()
    client.replay_overlay(session)
    installs = [p for a, p in box.sent if a == "deps" and p["action"] == "install"]
    assert len(installs) == 1
    assert installs[0]["no_deps"] is True
    assert installs[0]["packages"] == ["diffusers==0.31.0", "safetensors==0.5.0"]
    assert installs[0]["index_url"] == "https://u:secret@pypi.example/simple"

    # Same runner, overlay already there: nothing is installed again.
    box.sent.clear()
    client.replay_overlay(session)
    assert [p["action"] for a, p in box.sent if a == "deps"] == ["status"]


def test_replay_drops_an_overlay_that_no_longer_installs(tmp_path, monkeypatch, capsys):
    box = FakeDepsBox(fail_on="broken")
    make_cli_session(tmp_path, monkeypatch, box)
    session = app_session.find_session(None)
    session.overlay = {"packages": ["broken @ https://u:tok@x/b.whl"]}
    client.replay_overlay(session)
    out = capsys.readouterr().out
    assert "Could not install the dependency overlay" in out
    assert "tok" not in out
    # Kept, so the next boot tries again.
    assert session.overlay == {"packages": ["broken @ https://u:tok@x/b.whl"]}


def test_replay_failure_from_a_refusing_index_explains_the_grant(
    tmp_path, monkeypatch, capsys
):
    refusal = "deps failed (400): HTTP status client error (401 Unauthorized)"
    box = FakeDepsBox(fail_on="private", error=refusal)
    make_cli_session(tmp_path, monkeypatch, box)
    session = app_session.find_session(None)
    session.overlay = {"packages": ["private-pkg==1"]}
    client.replay_overlay(session)
    out = " ".join(capsys.readouterr().out.split())
    assert "fal_wheels_access" in out and "--team" in out


def test_a_failed_replay_survives_status_reads_and_is_retried(tmp_path, monkeypatch):
    box = FakeDepsBox(fail_on="broken")
    make_cli_session(tmp_path, monkeypatch, box)
    session = app_session.find_session(None)
    session.overlay = {"packages": ["broken==1"]}
    session.save()
    client.replay_overlay(session)
    args = parse_args(["dev", "app", "deps", "status"])
    args.func(args)
    assert app_session.find_session(None).overlay == {"packages": ["broken==1"]}

    # The box recovers: the next boot installs the original pins.
    box.fail_on = None
    box.sent.clear()
    client.replay_overlay(app_session.find_session(None))
    (install,) = (p for a, p in box.sent if a == "deps" and p["action"] == "install")
    assert install["packages"] == ["broken==1"]
