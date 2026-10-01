"""How ``fal.wma`` plugs into the fal SDK itself.

These are the fal-side guarantees (not ported from the platform suites):
the experimental package stays out of ``import fal``, imports without its
optional WebRTC stack, gets that stack injected into deployed runner envs,
and survives the SDK's pickle-by-value serialization path.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
from contextvars import ContextVar

import pytest

import fal
import fal.wma


@pytest.fixture
def isolate_agent_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("IS_ISOLATE_AGENT", "1")


class EchoWmaApp(fal.wma.App):
    async def create_backend(self, session: fal.wma.Session) -> fal.wma.PeerBackend:
        return fal.wma.AiortcPeer(session, lambda _pc: None)


def test_wma_subclass_keeps_its_concrete_app_name():
    class NamedWmaApp(fal.wma.App, name="named-wma"):
        async def create_backend(self, session):  # pragma: no cover - not run
            raise NotImplementedError

    class DefaultNamedWmaApp(fal.wma.App):
        async def create_backend(self, session):  # pragma: no cover - not run
            raise NotImplementedError

    assert NamedWmaApp.app_name == "named-wma"
    assert DefaultNamedWmaApp.app_name == "default-named-wma-app"


def test_wrap_app_injects_wma_runner_requirements():
    from fal.app import WMA_APP_REQUIREMENTS, wrap_app

    fn = wrap_app(EchoWmaApp)
    requirements = fn.options.environment["requirements"]
    for requirement in WMA_APP_REQUIREMENTS:
        assert requirement in requirements


def test_wrap_app_leaves_plain_apps_without_wma_requirements():
    from fal.app import wrap_app

    class PlainApp(fal.App):
        @fal.endpoint("/")
        def hello(self) -> str:
            return "Hello, world!"

    fn = wrap_app(PlainApp)
    requirements = fn.options.environment.get("requirements", [])
    assert not any("aiortc" in requirement for requirement in requirements)


def test_wrap_app_respects_an_apps_own_aiortc_pin():
    from fal.app import wrap_app

    class PinnedWmaApp(fal.wma.App):
        requirements = ["aiortc==1.15.0"]

        async def create_backend(self, session):  # pragma: no cover - not run
            raise NotImplementedError

    fn = wrap_app(PinnedWmaApp)
    requirements = fn.options.environment["requirements"]
    assert "aiortc==1.15.0" in requirements
    # The injected range must not coexist with (or later override) the pin.
    assert not any("aiortc>" in requirement for requirement in requirements)


def test_wrap_app_does_not_mutate_class_state():
    import copy as copy_module

    from fal.app import wrap_app

    before_host_kwargs = copy_module.deepcopy(EchoWmaApp.host_kwargs)
    before_requirements = copy_module.deepcopy(EchoWmaApp.requirements)
    wrap_app(EchoWmaApp)
    wrap_app(EchoWmaApp)
    assert EchoWmaApp.host_kwargs == before_host_kwargs
    assert EchoWmaApp.requirements == before_requirements


def test_wma_base_class_is_not_auto_discovered():
    from fal.utils import _find_target

    class UserApp(fal.wma.App):
        async def create_backend(self, session):  # pragma: no cover - not run
            raise NotImplementedError

    # `from fal.wma import App` in a user file must not count as a second
    # deployable app.
    target, _, _, class_name = _find_target({"App": fal.wma.App, "UserApp": UserApp})
    assert target is UserApp
    assert class_name == "UserApp"


def test_session_close_releases_the_object_graph():
    import asyncio

    async def scenario():
        session = fal.wma.Session(fal.wma.StartSessionRequest(sdp="offer"))
        session.on_message("input", lambda message: None)
        session.bind_sender(lambda message: True)
        session.params["prompt"] = "kept"
        await session.close()
        # Private references are dropped so the peer/handler graph is
        # refcount-collectable; public state stays readable.
        assert session._backend is None
        assert session._sender is None
        assert session._handlers == {}
        assert session.params == {"prompt": "kept"}
        assert session.params._push is None

    asyncio.run(scenario())


def test_error_classes_are_publicly_exported():
    assert fal.wma.InputValueError is not None
    error = fal.wma.InputValueError.from_field_error(field="sdp", msg="bad")
    assert error.status_code == 422
    assert error.detail[0]["loc"] == ["body", "sdp"]
    assert error.detail[0]["url"] == "https://docs.fal.ai/errors#value_error"
    assert error.headers["x-fal-billable-units"] == "0"
    assert error.headers["X-Fal-needs-retry"] == "false"


def test_endpoint_annotations_survive_cloudpickle():
    # fal apps are cloudpickled to runners, and cloudpickle ships only the
    # globals referenced by CODE. Stringified annotations (a `from __future__
    # import annotations` in sdk.py) become unresolvable ForwardRefs there,
    # and FastAPI's dependency resolution 500s every /start-session at the
    # first request. Pin that every endpoint annotation is a def-time object.
    hints = fal.wma.App.start_session.__annotations__
    assert hints, "start_session must be annotated"
    stringified = {k: v for k, v in hints.items() if isinstance(v, str)}
    assert (
        not stringified
    ), f"stringified annotations would break on runners: {stringified}"


def test_import_fal_does_not_import_wma():
    # The experimental package must load only on explicit ``import fal.wma``:
    # ``import fal`` stays byte-identical for everyone else.
    code = "import sys, fal; assert 'fal.wma' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_wma_imports_without_aiortc():
    # ``aiortc`` is a session-time dependency, never an import-time one — the
    # SDK must import everywhere (CPU hosts, client machines) without it.
    code = (
        "import sys\n"
        "sys.modules['aiortc'] = None\n"
        "sys.modules['aioice'] = None\n"
        "import fal.wma\n"
        "assert fal.wma.App is not None\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_wma_app_is_picklable_with_request_context(isolate_agent_env):
    # fal serializes the whole app instance by value for the runner; the WMA
    # base class must survive the same round-trip as ``fal.App``.
    app = EchoWmaApp()
    app._current_request_context = ContextVar(  # type: ignore[assignment]
        "_current_request_context"
    )

    payload = pickle.dumps(app)
    loaded = pickle.loads(payload)

    assert loaded._current_request_context is None
    assert isinstance(loaded, fal.wma.App)


def test_wma_app_collects_the_session_route():
    routes = EchoWmaApp(_allow_init=True).collect_routes()
    assert fal.wma.START_SESSION_PATH in {signature.path for signature in routes}
