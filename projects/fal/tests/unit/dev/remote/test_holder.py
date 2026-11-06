import asyncio
import threading
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from dev.remote import fakes
from dev.remote.fakes import FakeInstaller, bare_holder
from fal.dev.errors import DevError
from fal.dev.remote import deps as app_deps
from fal.dev.remote.holder import DevAppHolder, make_holder
from fal.dev.remote.process import ChildProcess, LogBuffer

project = fakes.project

pytestmark = pytest.mark.usefixtures("dev_account")


def test_make_holder_inherits_environment_but_not_routes_or_scaling(project):
    import demo_app

    holder = make_holder(demo_app.Demo, name="dev-demo-me", token="t", keep_alive=900)
    assert issubclass(holder, demo_app.Demo)
    assert holder.app_name == "dev-demo-me"
    assert holder.machine_type == "GPU-H100"
    assert holder.requirements == ["torch"]
    assert holder.host_kwargs["keep_alive"] == 900
    assert holder.host_kwargs["max_concurrency"] == 1
    assert holder.host_kwargs["min_concurrency"] == 0
    assert holder.host_kwargs["health_check_config"] is None
    assert holder.setup is DevAppHolder.setup

    routes = holder(_allow_init=True).collect_routes()
    paths = {signature.path for signature in routes}
    assert "/" not in paths
    assert "/__fal_dev/request" in paths

    overridden = make_holder(
        demo_app.Demo, name="x", token="t", keep_alive=1, machine_type="GPU-A100"
    )
    assert overridden.machine_type == "GPU-A100"


def test_holder_routes_check_the_token_and_report_failures_as_400(project):
    import demo_app

    holder = make_holder(demo_app.Demo, name="x", token="secret", keep_alive=1)
    instance = holder(_allow_init=True)

    def run(payload, handler):
        return asyncio.run(instance._run(payload, handler))

    for bad in ("wrong", "", "\u00e9"):
        with pytest.raises(HTTPException) as exc:
            run({"token": bad}, lambda payload: {})
        assert exc.value.status_code == 403
    with pytest.raises(HTTPException) as exc:
        run(["not", "an", "object"], lambda payload: {})
    assert exc.value.status_code == 422
    assert run({"token": "secret", "a": 1}, lambda payload: {"a": payload["a"]}) == {
        "a": 1
    }

    def refuse(payload):
        raise DevError("no way")

    with pytest.raises(HTTPException) as exc:
        run({"token": "secret"}, refuse)
    assert (exc.value.status_code, exc.value.detail) == (400, "no way")


def test_holder_hides_index_credentials_in_errors(project):
    import demo_app

    holder = make_holder(demo_app.Demo, name="x", token="t", keep_alive=1)
    instance = holder(_allow_init=True)

    def refuse(payload):
        raise DevError("failed for https://user:secret@pypi.example/simple")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(instance._run({"token": "t"}, refuse))
    assert "secret" not in exc.value.detail


def test_holder_refuses_a_bad_layout_without_changing_state(tmp_path):
    child = ChildProcess(tmp_path, LogBuffer())
    with pytest.raises(ValueError, match="refusing"):
        child.start("m", "C", layout={"paths": ["."], "workdir": "/etc"})
    assert (child.module, child.paths, child.workdir) == ("", ["."], ".")


def test_start_with_a_bad_layout_is_a_400_and_leaves_the_app_running(
    tmp_path, monkeypatch
):
    holder = bare_holder()
    holder._child = ChildProcess(tmp_path, holder._log_buffer)
    holder._child.phase = "ready"
    swapped = []
    monkeypatch.setattr(holder._child, "replace_sdk", swapped.append)
    for bad in (
        {"layout": "nope"},
        {"session_env": "LD_PRELOAD"},
        {"module": 5},
        {"class_name": "-c"},
    ):
        with pytest.raises(HTTPException) as info:
            asyncio.run(
                holder.dev_start(
                    {"token": "t", "sdk_archive": "x", "module": "m", **bad}
                )
            )
        assert info.value.status_code == 400
    assert swapped == [] and holder._child.phase == "ready"

    # A bug in a handler is a 400 too, and is logged.
    def broken(payload):
        raise KeyError("boom")

    with pytest.raises(HTTPException) as info:
        asyncio.run(holder._run({"token": "t"}, broken))
    assert info.value.status_code == 400
    assert "KeyError" in "\n".join(holder._log_buffer.lines)


def test_files_reload_includes_files_the_box_placed(tmp_path):
    reloaded = []
    answers = [{"action": "error"}, {"action": "reload"}, {"action": "reload"}]
    holder = bare_holder(
        placed=set(),
        placed_lock=threading.Lock(),
        tree=SimpleNamespace(
            place=lambda files: ["a.py"],
            sync=lambda **kwargs: (["b.py"], ["c.py"]),
        ),
        child=SimpleNamespace(
            reload=lambda files: reloaded.append(files) or answers.pop(0)
        ),
    )
    asyncio.run(holder.dev_place({"token": "t", "files": {"a.py": ["0" * 64, False]}}))
    for _ in range(3):
        asyncio.run(holder.dev_files({"token": "t", "reload": True}))
    # A failed reload keeps the placed file for the next one.
    assert reloaded == [["b.py", "c.py", "a.py"]] * 2 + [["b.py", "c.py"]]


def test_holder_logs_hide_credentials(tmp_path):
    holder = bare_holder()

    def broken(payload):
        raise KeyError("https://user:secret@pypi.example/simple")

    with pytest.raises(HTTPException):
        asyncio.run(holder._run({"token": "t"}, broken))
    assert "secret" not in "\n".join(holder._log_buffer.lines)


def _holder(tmp_path, monkeypatch):
    holder = bare_holder(child=ChildProcess(tmp_path, LogBuffer()))
    holder._child.phase = "ready"
    fake = FakeInstaller(["six==1.16.0"])
    monkeypatch.setattr(
        app_deps.Overlay, "_installer", lambda self: ["uv", "pip", "install"]
    )
    monkeypatch.setattr(app_deps.Overlay, "_run", lambda self, c: fake.run(c))

    def deps(**payload):
        return asyncio.run(holder.dev_deps({"token": "t", **payload}))

    return holder, fake, deps


def test_deps_route_restarts_the_app_only_when_asked_and_changed(tmp_path, monkeypatch):
    from fal.dev.remote import holder as holder_module

    holder, fake, deps = _holder(tmp_path, monkeypatch)
    starts = []
    monkeypatch.setattr(holder._child, "start", lambda: starts.append(1) or True)
    monkeypatch.setattr(
        holder_module, "child_call", lambda *a, **k: (200, {"six": {"version": "1"}})
    )

    result = deps(action="install", packages=["six==1.16.0"], restart=True)
    assert result["restarted"] and starts == [1]
    assert result["packages"][0]["name"] == "six"
    assert result["replay"]["packages"] == ["six==1.16.0"]

    fake.plan = []
    result = deps(action="install", packages=["six==1.16.0"], restart=True)
    assert not result["restarted"] and starts == [1]

    status = deps(action="status")
    assert status["app_versions"] == {"six": {"version": "1"}}
    assert deps(action="reset")["removed"] == 1
    with pytest.raises(Exception, match="unknown deps action"):
        deps(action="explode")
