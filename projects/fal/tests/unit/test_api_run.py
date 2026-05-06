from unittest.mock import MagicMock

from fal.api import IsolatedFunction, Options
from fal.api.run import run as run_api


class _FakeHost:
    pass


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
