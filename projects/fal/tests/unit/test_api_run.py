import sys
from unittest.mock import MagicMock, patch

import pytest

from fal import App, endpoint
from fal.api import FalServerlessError, IsolatedFunction, Options
from fal.api.api import _check_python_version_match
from fal.api.run import run as run_api


def _func():
    pass


def test_check_python_version_match_exempts_entrypoint(monkeypatch):
    # No callable is serialized (entrypoint mode, no setup_function) -> no check.
    monkeypatch.setattr(
        "isolate.backends.common.active_python", lambda: "3.11", raising=True
    )
    _check_python_version_match([None], "3.12")  # must not raise


def test_check_python_version_match_noop_when_target_is_none(monkeypatch):
    # target resolved server-side -> nothing to compare against.
    monkeypatch.setattr(
        "isolate.backends.common.active_python", lambda: "3.11", raising=True
    )
    _check_python_version_match([_func], None)  # must not raise


def test_check_python_version_match_allows_same_minor(monkeypatch):
    monkeypatch.setattr(
        "isolate.backends.common.active_python", lambda: "3.11", raising=True
    )
    # patch-level differences don't change the bytecode magic.
    _check_python_version_match([_func], "3.11")  # must not raise


def test_check_python_version_match_raises_on_minor_mismatch(monkeypatch):
    monkeypatch.setattr(
        "isolate.backends.common.active_python", lambda: "3.11", raising=True
    )
    with pytest.raises(FalServerlessError) as exc_info:
        _check_python_version_match([_func], "3.12")

    msg = str(exc_info.value)
    assert "3.11" in msg
    assert "3.12" in msg


def test_check_python_version_match_checks_setup_function_in_entrypoint(monkeypatch):
    # Entrypoint mode (func is None) but a setup_function is still pickled
    # and shipped -> the mismatch must still be caught.
    monkeypatch.setattr(
        "isolate.backends.common.active_python", lambda: "3.11", raising=True
    )
    with pytest.raises(FalServerlessError):
        _check_python_version_match([None, _func], "3.12")


class _FakeHost:
    pass


def test_options_get_exposed_port_prefers_configured_port_for_serve_true():
    options = Options(gateway={"serve": True, "exposed_port": 3000})

    assert options.get_exposed_port() == 3000


def test_run_local_with_entrypoint_resolves_user_symbol(monkeypatch):
    sentinel = MagicMock(name="user_module.UserApp.run_local")
    sentinel.return_value = "ran"

    fake_module = MagicMock()
    fake_module.UserApp.run_local = sentinel

    import importlib

    def fake_import_module(name):
        assert name == "user_module"
        return fake_module

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    iso = IsolatedFunction(
        host=_FakeHost(),
        options=Options(),
        entrypoint="user_module:UserApp",
    )

    result = run_api(iso, args=("a",), kwargs={"k": 1}, local=True)

    assert result == "ran"
    sentinel.assert_called_once_with("a", k=1)


def test_run_local_without_entrypoint_uses_raw_func():
    iso = IsolatedFunction(
        host=_FakeHost(),
        raw_func=lambda *a, **k: ("local", a, k),
        options=Options(),
    )

    assert run_api(iso, args=(1,), kwargs={"x": 2}, local=True) == (
        "local",
        (1,),
        {"x": 2},
    )


def test_run_local_with_app_files_configures_runtime_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def run():
        return "local"

    iso = IsolatedFunction(
        host=_FakeHost(),
        raw_func=run,
        options=Options(
            host={"app_files": ["assets"], "app_files_context_dir": "."},
        ),
    )

    with patch("fal.api.api.include_app_files_path") as include_app_files_path:
        assert run_api(iso, local=True) == "local"

    include_app_files_path.assert_called_once_with(".")


def test_run_local_container_config_includes_cwd_on_sys_path(monkeypatch):
    monkeypatch.setattr(sys, "path", [path for path in sys.path if path != ""])

    iso = IsolatedFunction(
        host=_FakeHost(),
        raw_func=lambda: "local",
        options=Options(environment={"kind": "container"}),
    )

    assert run_api(iso, local=True) == "local"
    assert sys.path[0] == ""


def test_run_local_forwards_exposed_ports_to_entrypoint_app(monkeypatch):
    sentinel = MagicMock(name="user_module.UserApp.run_local")
    sentinel.return_value = "ran"

    class UserApp(App):
        @endpoint("/")
        def run(self):
            return "ok"

    UserApp.run_local = sentinel

    fake_module = MagicMock()
    fake_module.UserApp = UserApp

    import importlib

    def fake_import_module(name):
        assert name == "user_module"
        return fake_module

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    iso = IsolatedFunction(
        host=_FakeHost(),
        options=Options(),
        entrypoint="user_module:UserApp",
    )

    result = run_api(
        iso,
        args=("a",),
        kwargs={"k": 1},
        local=True,
        exposed_port=3000,
        exposed_metrics_port=3001,
    )

    assert result == "ran"
    sentinel.assert_called_once_with(
        "a",
        k=1,
        exposed_port=3000,
        exposed_metrics_port=3001,
    )


def test_run_local_rejects_exposed_ports_for_plain_function():
    iso = IsolatedFunction(
        host=_FakeHost(),
        raw_func=lambda: "local",
        options=Options(),
    )

    with pytest.raises(FalServerlessError, match="Local exposed port options"):
        run_api(iso, local=True, exposed_port=3000)


def test_run_local_forwards_exposed_ports_to_served_function(monkeypatch):
    from fal.api.api import ServeWrapper

    called_port: int | None = None
    called_metrics_port: int | None = None

    async def fake_serve(
        self,
        *,
        port: int = 8080,
        metrics_port: int = 9090,
    ):
        nonlocal called_port, called_metrics_port
        called_port = port
        called_metrics_port = metrics_port

    monkeypatch.setattr(ServeWrapper, "serve", fake_serve)

    iso = IsolatedFunction(
        host=_FakeHost(),
        raw_func=lambda: "local",
        options=Options(gateway={"serve": True}),
    )

    run_api(iso, local=True, exposed_port=3000, exposed_metrics_port=3001)

    assert called_port == 3000
    assert called_metrics_port == 3001


def test_run_local_forwards_exposed_ports_to_entrypoint_served_function(monkeypatch):
    from fal.api.api import ServeWrapper

    called_port: int | None = None
    called_metrics_port: int | None = None

    async def fake_serve(
        self,
        *,
        port: int = 8080,
        metrics_port: int = 9090,
    ):
        nonlocal called_port, called_metrics_port
        called_port = port
        called_metrics_port = metrics_port

    monkeypatch.setattr(ServeWrapper, "serve", fake_serve)

    served_function = IsolatedFunction(
        host=_FakeHost(),
        raw_func=lambda: "local",
        options=Options(gateway={"serve": True}),
    )
    fake_module = MagicMock()
    fake_module.served_function = served_function

    import importlib

    def fake_import_module(name):
        assert name == "user_module"
        return fake_module

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    iso = IsolatedFunction(
        host=_FakeHost(),
        options=Options(gateway={"exposed_port": 8080}),
        entrypoint="user_module:served_function",
    )

    run_api(iso, local=True, exposed_port=3000, exposed_metrics_port=3001)

    assert called_port == 3000
    assert called_metrics_port == 3001


def test_run_local_entrypoint_served_function_keeps_configured_port(monkeypatch):
    from fal.api.api import ServeWrapper

    called_ports: list[int] = []

    async def fake_serve(
        self,
        *,
        port: int = 8080,
        metrics_port: int = 9090,
    ):
        called_ports.append(port)

    monkeypatch.setattr(ServeWrapper, "serve", fake_serve)

    served_function = IsolatedFunction(
        host=_FakeHost(),
        raw_func=lambda: "local",
        options=Options(gateway={"serve": True, "exposed_port": 9000}),
    )
    fake_module = MagicMock()
    fake_module.served_function = served_function

    import importlib

    def fake_import_module(name):
        assert name == "user_module"
        return fake_module

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    iso = IsolatedFunction(
        host=_FakeHost(),
        options=Options(gateway={"exposed_port": 8080}),
        entrypoint="user_module:served_function",
        _entrypoint_exposed_port_defaulted=True,
    )

    run_api(iso, local=True)

    assert called_ports == [9000]


def test_run_local_entrypoint_served_function_honors_explicit_default_port(
    monkeypatch,
):
    from fal.api.api import ServeWrapper

    called_ports: list[int] = []

    async def fake_serve(
        self,
        *,
        port: int = 8080,
        metrics_port: int = 9090,
    ):
        called_ports.append(port)

    monkeypatch.setattr(ServeWrapper, "serve", fake_serve)

    served_function = IsolatedFunction(
        host=_FakeHost(),
        raw_func=lambda: "local",
        options=Options(gateway={"serve": True, "exposed_port": 9000}),
    )
    fake_module = MagicMock()
    fake_module.served_function = served_function

    import importlib

    def fake_import_module(name):
        assert name == "user_module"
        return fake_module

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    iso = IsolatedFunction(
        host=_FakeHost(),
        options=Options(gateway={"exposed_port": 8080}),
        entrypoint="user_module:served_function",
    )

    run_api(iso, local=True)

    assert called_ports == [8080]


def test_run_local_exposed_port_override_does_not_mutate_options(monkeypatch):
    from fal.api.api import ServeWrapper

    called_ports: list[int] = []

    async def fake_serve(
        self,
        *,
        port: int = 8080,
        metrics_port: int = 9090,
    ):
        called_ports.append(port)

    monkeypatch.setattr(ServeWrapper, "serve", fake_serve)

    iso = IsolatedFunction(
        host=_FakeHost(),
        raw_func=lambda: "local",
        options=Options(gateway={"serve": True}),
    )

    run_api(iso, local=True, exposed_port=3000)
    run_api(iso, local=True)

    assert called_ports == [3000, 8080]
    assert iso.options.gateway == {"serve": True}
