from __future__ import annotations

import os
import pickle
import signal
import socket
import subprocess
import sys
import textwrap
import time
from contextlib import ExitStack
from contextvars import ContextVar
from typing import AsyncIterator, Iterator
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from fastapi import WebSocket
from pydantic import BaseModel

import fal
from fal import App, endpoint
from fal.container import ContainerImage


class PickleApp(App):
    pass


class InputModel(BaseModel):
    prompt: str


class OutputModel(BaseModel):
    result: str


class RealtimeApp(fal.App):
    @fal.realtime("/realtime", buffering=2, session_timeout=1.5, max_batch_size=3)
    def generate(self, input: InputModel) -> OutputModel:
        return OutputModel(result=input.prompt)

    @fal.realtime("/realtime/server-streaming")
    async def generate_rt_server_streaming(
        self, input: InputModel
    ) -> AsyncIterator[OutputModel]:
        yield OutputModel(result=input.prompt)

    @fal.realtime("/realtime/server-streaming-sync")
    def generate_rt_server_streaming_sync(
        self, input: InputModel
    ) -> Iterator[OutputModel]:
        yield OutputModel(result=input.prompt)

    @fal.realtime("/realtime/client-streaming")
    async def generate_rt_client_streaming(
        self, inputs: AsyncIterator[InputModel]
    ) -> OutputModel:
        return OutputModel(result="ok")

    @fal.realtime("/realtime/bidi")
    async def generate_rt_bidi(
        self, inputs: AsyncIterator[InputModel]
    ) -> AsyncIterator[OutputModel]:
        async for item in inputs:
            yield OutputModel(result=item.prompt)

    @fal.realtime("/realtime/json", content_type="application/json")
    def generate_rt_json(self, input: InputModel) -> OutputModel:
        return OutputModel(result=input.prompt)

    @fal.endpoint("/ws", is_websocket=True)
    async def generate_ws(self, websocket: WebSocket) -> None:
        await websocket.close()


@pytest.fixture
def isolate_agent_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("IS_ISOLATE_AGENT", "1")


def test_app_regions_propagate_to_function_options():
    from fal.app import wrap_app

    class RegionsApp(App):
        regions = ["us-east", "eu-west"]

        @endpoint("/")
        def hello(self) -> str:
            return "Hello, world!"

    fn = wrap_app(RegionsApp)
    assert fn.options.host.get("regions") == ["us-east", "eu-west"]


def test_run_forwards_regions_to_machine_requirements():
    from fal.api.api import FalServerlessHost, Options
    from fal.sdk import HostedRunState

    host = FalServerlessHost()
    options = Options()
    options.host["regions"] = ["us-east"]

    connection = MagicMock()
    connection.define_environment.return_value = object()
    partial_result = MagicMock()
    partial_result.status.state = HostedRunState.SUCCESS
    partial_result.result = "ok"
    connection.run.return_value = iter([partial_result])

    with patch.object(
        FalServerlessHost, "_connection", new_callable=PropertyMock
    ) as mock_connection:
        mock_connection.return_value = connection
        result = host._run(
            lambda: "ok",
            options,
            args=(),
            kwargs={},
            result_handler=lambda _: None,
        )

    assert result == "ok"
    _, call_kwargs = connection.run.call_args
    assert call_kwargs["machine_requirements"].valid_regions == ["us-east"]


def test_register_leaves_private_logs_unset_by_default():
    from fal.api.api import FalServerlessHost, Options
    from fal.sdk import RegisterApplicationResult, RegisterApplicationResultType

    host = FalServerlessHost()
    options = Options()

    connection = MagicMock()
    connection.define_environment.return_value = object()
    partial_result = RegisterApplicationResult(
        result=RegisterApplicationResultType(application_id="app-id")
    )
    connection.register.return_value = iter([partial_result])

    with patch.object(
        FalServerlessHost, "_connection", new_callable=PropertyMock
    ) as mock_connection:
        mock_connection.return_value = connection
        result = host.register(
            lambda: "ok",
            options,
            application_name="example-app",
            deployment_strategy="recreate",
        )

    assert result == partial_result
    _, call_kwargs = connection.register.call_args
    assert call_kwargs["private_logs"] is None


def test_register_forwards_explicit_private_logs_false():
    from fal.api.api import FalServerlessHost, Options
    from fal.sdk import RegisterApplicationResult, RegisterApplicationResultType

    host = FalServerlessHost()
    options = Options()
    options.host["private_logs"] = False

    connection = MagicMock()
    connection.define_environment.return_value = object()
    partial_result = RegisterApplicationResult(
        result=RegisterApplicationResultType(application_id="app-id")
    )
    connection.register.return_value = iter([partial_result])

    with patch.object(
        FalServerlessHost, "_connection", new_callable=PropertyMock
    ) as mock_connection:
        mock_connection.return_value = connection
        result = host.register(
            lambda: "ok",
            options,
            application_name="example-app",
            deployment_strategy="recreate",
        )

    assert result == partial_result
    _, call_kwargs = connection.register.call_args
    assert call_kwargs["private_logs"] is False


def test_wrap_app_allows_resolver_with_container_kind():
    from fal.app import wrap_app

    class ContainerKindApp(App):
        image = ContainerImage.from_dockerfile_str("FROM python:3.11-slim")

        @endpoint("/")
        def hello(self) -> str:
            return "Hello, world!"

    fn = wrap_app(ContainerKindApp)
    assert fn.options.environment.get("resolver") == "uv"


def test_function_force_env_build_propagates_for_container_apps():
    image = ContainerImage.from_dockerfile_str("FROM python:3.11-slim")

    @fal.function(image=image, force_env_build=True)
    def container_function():
        return "ok"

    assert container_function.options.environment.get("force") is True


def test_wrap_app_limit_max_requests_propagates_to_serve(
    isolate_agent_env, monkeypatch: pytest.MonkeyPatch
):
    import asyncio

    from fal.app import wrap_app

    class SingleUseApp(App):
        @endpoint("/")
        def hello(self) -> str:
            return "Hello, world!"

    called_limit_max_requests: int | None = None

    async def fake_serve(self, *, limit_max_requests: int | None = None):
        nonlocal called_limit_max_requests
        called_limit_max_requests = limit_max_requests

    monkeypatch.setattr(SingleUseApp, "serve", fake_serve)

    fn = wrap_app(SingleUseApp, limit_max_requests=1)
    asyncio.run(fn.func())

    assert called_limit_max_requests == 1


def test_wrap_app_raises_for_virtualenv_only_keys_with_conda_kind():
    from fal.app import wrap_app

    class CondaKindApp(App):
        kind = "conda"

        @endpoint("/")
        def hello(self) -> str:
            return "Hello, world!"

    with pytest.raises(
        ValueError,
        match="Unsupported option 'requirements' for environment kind 'conda'",
    ):
        wrap_app(CondaKindApp)


def test_wrap_app_raises_for_unrecognised_option():
    from fal.app import wrap_app

    class UnknownOptionApp(App, totally_unknown_option=1):
        @endpoint("/")
        def hello(self) -> str:
            return "Hello, world!"

    with pytest.raises(
        ValueError,
        match="Unrecognised option 'totally_unknown_option'.",
    ):
        wrap_app(UnknownOptionApp)


def test_wrap_app_unrecognised_option_suggests_closest_match():
    from fal.app import wrap_app

    class TypoOptionApp(App, reqirements=["fastapi"]):
        @endpoint("/")
        def hello(self) -> str:
            return "Hello, world!"

    with pytest.raises(
        ValueError,
        match="Unrecognised option 'reqirements'. Did you mean 'requirements'\\?",
    ):
        wrap_app(TypoOptionApp)


def test_wrap_app_raises_for_unknown_environment_kind():
    from fal.app import wrap_app

    class UnknownKindApp(App):
        kind = "something-else"

        @endpoint("/")
        def hello(self) -> str:
            return "Hello, world!"

    with pytest.raises(
        ValueError,
        match="Unrecognised environment kind 'something-else'. "
        "Only virtualenv, container, conda are supported.",
    ):
        wrap_app(UnknownKindApp)


def test_app_default_app_name_is_generated_from_class_name():
    class MyCustomApp(App):
        pass

    assert MyCustomApp.app_name == "my-custom-app"


# ============================================================================
# Tests for _to_fal_app_name - the function that converts class names to app names
# ============================================================================


class TestToFalAppName:
    """Comprehensive tests for _to_fal_app_name conversion logic."""

    # --- Standard PascalCase conversions ---

    def test_simple_pascal_case(self):
        from fal.app import _to_fal_app_name

        assert _to_fal_app_name("MyApp") == "my-app"
        assert _to_fal_app_name("SimpleApp") == "simple-app"
        assert _to_fal_app_name("MyGoodApp") == "my-good-app"

    def test_single_word_pascal_case(self):
        from fal.app import _to_fal_app_name

        assert _to_fal_app_name("App") == "app"
        assert _to_fal_app_name("Model") == "model"

    def test_long_pascal_case(self):
        from fal.app import _to_fal_app_name

        assert (
            _to_fal_app_name("MyVeryLongApplicationName")
            == "my-very-long-application-name"
        )

    # --- Acronym handling (known quirky behavior) ---

    def test_all_caps_acronym_splits_each_letter(self):
        """Acronyms like SDXL get split into individual letters - known behavior."""
        from fal.app import _to_fal_app_name

        # Each uppercase letter becomes its own segment
        assert _to_fal_app_name("SDXLModel") == "s-d-x-l-model"
        assert _to_fal_app_name("ONNXRuntime") == "o-n-n-x-runtime"
        assert _to_fal_app_name("GPT4App") == "g-p-t-app"  # 4 is not captured by regex

    def test_mixed_acronym_and_words(self):
        from fal.app import _to_fal_app_name

        assert _to_fal_app_name("MySDXLApp") == "my-s-d-x-l-app"

    # --- snake_case fallback ---

    def test_snake_case_conversion(self):
        from fal.app import _to_fal_app_name

        assert _to_fal_app_name("mock_model") == "mock-model"
        assert _to_fal_app_name("my_cool_app") == "my-cool-app"
        assert _to_fal_app_name("image_generator") == "image-generator"

    def test_snake_case_with_numbers(self):
        from fal.app import _to_fal_app_name

        assert _to_fal_app_name("sdxl_v2") == "sdxl-v2"
        assert _to_fal_app_name("gpt_4_turbo") == "gpt-4-turbo"

    def test_snake_case_with_leading_trailing_underscores(self):
        from fal.app import _to_fal_app_name

        assert _to_fal_app_name("_private_app") == "private-app"
        assert _to_fal_app_name("app_") == "app"
        assert _to_fal_app_name("__dunder__") == "dunder"

    def test_snake_case_with_multiple_underscores(self):
        from fal.app import _to_fal_app_name

        assert _to_fal_app_name("my__app") == "my-app"

    # --- Lowercase fallback ---

    def test_simple_lowercase(self):
        from fal.app import _to_fal_app_name

        assert _to_fal_app_name("myapp") == "myapp"
        assert _to_fal_app_name("simple") == "simple"

    def test_lowercase_with_numbers(self):
        from fal.app import _to_fal_app_name

        assert _to_fal_app_name("app123") == "app123"
        assert _to_fal_app_name("v2model") == "v2model"

    # --- Edge cases ---

    def test_single_character(self):
        from fal.app import _to_fal_app_name

        assert _to_fal_app_name("A") == "a"
        assert _to_fal_app_name("x") == "x"

    def test_numbers_only_falls_through_to_lowercase(self):
        from fal.app import _to_fal_app_name

        # No uppercase letters, no underscores, so falls through to lowercase
        assert _to_fal_app_name("123") == "123"

    def test_mixed_case_not_pascal(self):
        """Names that start lowercase but have uppercase later."""
        from fal.app import _to_fal_app_name

        # Only captures uppercase-starting segments
        assert _to_fal_app_name("myApp") == "app"  # 'my' is ignored, only 'App' matches

    # --- Error cases ---

    def test_empty_string_raises_value_error(self):
        from fal.app import _to_fal_app_name

        with pytest.raises(ValueError, match="Cannot derive app name"):
            _to_fal_app_name("")

    # --- Real-world examples ---

    def test_realistic_model_names(self):
        from fal.app import _to_fal_app_name

        assert _to_fal_app_name("StableDiffusion") == "stable-diffusion"
        assert _to_fal_app_name("TextToImage") == "text-to-image"
        assert _to_fal_app_name("ImageUpscaler") == "image-upscaler"
        assert _to_fal_app_name("VideoGenerator") == "video-generator"

    def test_realistic_snake_case_names(self):
        from fal.app import _to_fal_app_name

        assert _to_fal_app_name("stable_diffusion") == "stable-diffusion"
        assert _to_fal_app_name("text_to_image") == "text-to-image"
        assert _to_fal_app_name("flux_schnell") == "flux-schnell"


def test_app_classvars_propagate_to_host_kwargs():
    class VarsApp(App):
        request_timeout = 11
        startup_timeout = 22
        min_concurrency = 2
        max_concurrency = 3
        concurrency_buffer = 4
        concurrency_buffer_perc = 50
        scaling_delay = 33
        max_multiplexing = 7
        kind = "container"
        image = ContainerImage.from_dockerfile_str(
            "FROM python:3.10-slim",
        )
        skip_retry_conditions = ["timeout", "connection_error"]

    hk = VarsApp.host_kwargs
    assert hk["request_timeout"] == 11
    assert hk["startup_timeout"] == 22
    assert hk["min_concurrency"] == 2
    assert hk["max_concurrency"] == 3
    assert hk["concurrency_buffer"] == 4
    assert hk["concurrency_buffer_perc"] == 50
    assert hk["scaling_delay"] == 33
    assert hk["max_multiplexing"] == 7
    assert hk["kind"] == "container"
    assert isinstance(hk["image"], ContainerImage)
    assert hk["skip_retry_conditions"] == ["timeout", "connection_error"]


def test_data_mounts_propagate_to_host_kwargs():
    class MountsApp(App):
        data_mounts = ["/data/.cache", "/data/my-app"]

    assert MountsApp.host_kwargs["data_mounts"] == ["/data/.cache", "/data/my-app"]


def test_data_mounts_not_in_host_kwargs_when_none():
    class NoMountsApp(App):
        pass

    assert "data_mounts" not in NoMountsApp.host_kwargs


def test_data_mounts_empty_list_propagates():
    class EmptyMountsApp(App):
        data_mounts = []

    assert EmptyMountsApp.host_kwargs["data_mounts"] == []


def test_data_mounts_propagate_through_wrap_app():
    from fal.app import wrap_app

    class MountsWrapApp(App):
        data_mounts = ["/data"]

        @endpoint("/")
        def hello(self) -> str:
            return "Hello, world!"

    fn = wrap_app(MountsWrapApp)
    assert fn.options.host.get("data_mounts") == ["/data"]


def test_data_mounts_register_sends_to_connection():
    from unittest.mock import PropertyMock

    from fal.api.api import FalServerlessHost, Options

    host = FalServerlessHost()
    options = Options()
    options.host["data_mounts"] = ["/data/.cache"]

    connection = MagicMock()
    connection.define_environment.return_value = object()
    connection.register.return_value = iter([])

    with patch.object(
        FalServerlessHost, "_connection", new_callable=PropertyMock
    ) as mock_connection:
        mock_connection.return_value = connection
        host.register(
            func=lambda: "ok",
            options=options,
            deployment_strategy="recreate",
        )

    _, call_kwargs = connection.register.call_args
    assert call_kwargs["data_mounts"] == ["/data/.cache"]


def test_data_mounts_run_sends_to_connection():
    from fal.api.api import FalServerlessHost, Options
    from fal.sdk import HostedRunState

    host = FalServerlessHost()
    options = Options()
    options.host["data_mounts"] = ["/data"]

    connection = MagicMock()
    connection.define_environment.return_value = object()
    partial_result = MagicMock()
    partial_result.status.state = HostedRunState.SUCCESS
    partial_result.result = "ok"
    connection.run.return_value = iter([partial_result])

    with patch.object(
        FalServerlessHost, "_connection", new_callable=PropertyMock
    ) as mock_connection:
        mock_connection.return_value = connection
        host._run(
            lambda: "ok",
            options,
            args=(),
            kwargs={},
            result_handler=lambda _: None,
        )

    _, call_kwargs = connection.run.call_args
    assert call_kwargs["data_mounts"] == ["/data"]


def test_secrets_propagate_to_host_kwargs():
    class SecretsApp(App):
        secrets = ["OPENAI_API_KEY", "HF_TOKEN"]

    assert SecretsApp.host_kwargs["secrets"] == ["OPENAI_API_KEY", "HF_TOKEN"]


def test_secrets_not_in_host_kwargs_when_none():
    class NoSecretsApp(App):
        pass

    assert "secrets" not in NoSecretsApp.host_kwargs


def test_secrets_empty_list_propagates():
    class EmptySecretsApp(App):
        secrets = []

    assert EmptySecretsApp.host_kwargs["secrets"] == []


def test_secrets_propagate_through_wrap_app():
    from fal.app import wrap_app

    class SecretsWrapApp(App):
        secrets = ["OPENAI_API_KEY"]

        @endpoint("/")
        def hello(self) -> str:
            return "Hello, world!"

    fn = wrap_app(SecretsWrapApp)
    assert fn.options.host.get("secrets") == ["OPENAI_API_KEY"]


def test_secrets_register_sends_to_connection():
    from fal.api.api import FalServerlessHost, Options

    host = FalServerlessHost()
    options = Options()
    options.host["secrets"] = ["OPENAI_API_KEY"]

    connection = MagicMock()
    connection.define_environment.return_value = object()
    connection.register.return_value = iter([])

    with patch.object(
        FalServerlessHost, "_connection", new_callable=PropertyMock
    ) as mock_connection:
        mock_connection.return_value = connection
        host.register(
            func=lambda: "ok",
            options=options,
            deployment_strategy="recreate",
        )

    _, call_kwargs = connection.register.call_args
    assert call_kwargs["secrets"] == ["OPENAI_API_KEY"]


def test_secrets_run_sends_to_connection():
    from fal.api.api import FalServerlessHost, Options
    from fal.sdk import HostedRunState

    host = FalServerlessHost()
    options = Options()
    options.host["secrets"] = ["HF_TOKEN"]

    connection = MagicMock()
    connection.define_environment.return_value = object()
    partial_result = MagicMock()
    partial_result.status.state = HostedRunState.SUCCESS
    partial_result.result = "ok"
    connection.run.return_value = iter([partial_result])

    with patch.object(
        FalServerlessHost, "_connection", new_callable=PropertyMock
    ) as mock_connection:
        mock_connection.return_value = connection
        host._run(
            lambda: "ok",
            options,
            args=(),
            kwargs={},
            result_handler=lambda _: None,
        )

    _, call_kwargs = connection.run.call_args
    assert call_kwargs["secrets"] == ["HF_TOKEN"]


def test_app_files_classvars_propagate_to_host_kwargs():
    class VarsApp(App):
        request_timeout = 11
        startup_timeout = 22
        app_files = ["a.py", "b.py"]
        app_files_ignore = [r"\\.venv/"]
        app_files_context_dir = "."
        min_concurrency = 2
        max_concurrency = 3
        concurrency_buffer = 4
        concurrency_buffer_perc = 50
        scaling_delay = 33
        max_multiplexing = 7

    hk = VarsApp.host_kwargs
    assert hk["request_timeout"] == 11
    assert hk["startup_timeout"] == 22
    assert hk["app_files"] == ["a.py", "b.py"]
    assert hk["app_files_ignore"] == [r"\\.venv/"]
    assert hk["app_files_context_dir"] == "."
    assert hk["min_concurrency"] == 2
    assert hk["max_concurrency"] == 3
    assert hk["concurrency_buffer"] == 4
    assert hk["concurrency_buffer_perc"] == 50
    assert hk["scaling_delay"] == 33
    assert hk["max_multiplexing"] == 7


def test_app_kwargs_merge_into_host_kwargs_and_override_defaults():
    class KwargsApp(App, keep_alive=123, resolver="pip", _scheduler="nomad"):
        pass

    hk = KwargsApp.host_kwargs
    assert hk["keep_alive"] == 123
    assert hk["resolver"] == "pip"
    assert hk["_scheduler"] == "nomad"
    # default keys should still exist
    assert "_scheduler_options" in hk


def test_non_host_classvars_do_not_leak_into_host_kwargs():
    class LeakCheckApp(App):
        requirements = ["fastapi"]
        local_python_modules = ["."]
        machine_type = "M"
        num_gpus = 1
        app_auth = "private"

    hk = LeakCheckApp.host_kwargs
    assert "requirements" not in hk
    assert "local_python_modules" not in hk
    assert "machine_type" not in hk
    assert "num_gpus" not in hk
    assert "app_auth" not in hk


@pytest.mark.asyncio
async def test_runner_state_lifecycle_complete(isolate_agent_env):
    """Test that FAL_RUNNER_STATE transitions through all phases correctly"""
    states = []

    class StateCheckApp(App):
        def setup(self):
            states.append(("setup", os.getenv("FAL_RUNNER_STATE")))

        def teardown(self):
            states.append(("teardown", os.getenv("FAL_RUNNER_STATE")))

    app = StateCheckApp()

    # Create a mock FastAPI app
    import fastapi

    fastapi_app = fastapi.FastAPI()

    async with app.lifespan(fastapi_app):
        states.append(("running", os.getenv("FAL_RUNNER_STATE")))

    # Verify the full lifecycle
    assert len(states) == 3
    assert states[0] == ("setup", "SETUP")
    assert states[1] == ("running", "RUNNING")
    assert states[2] == ("teardown", "TERMINATING")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("startup_timeout", "elapsed_seconds", "should_warn"),
    [
        (1, 2.5, True),
        (None, 601.0, True),
        (10, 5.0, False),
        (None, 300.0, False),
        (1, 1.0, False),
    ],
)
async def test_lifespan_startup_timeout_warning(
    isolate_agent_env,
    startup_timeout,
    elapsed_seconds,
    should_warn,
):
    import fastapi

    class TestApp(App):
        def setup(self):
            return None

    TestApp.startup_timeout = startup_timeout
    app = TestApp()
    fastapi_app = fastapi.FastAPI()

    with ExitStack() as stack:
        mock_print = stack.enter_context(patch("builtins.print"))
        stack.enter_context(patch("fal.app._print_python_packages"))
        stack.enter_context(
            patch("fal.app.time.perf_counter", side_effect=[0.0, elapsed_seconds])
        )
        async with app.lifespan(fastapi_app):
            pass

    printed_lines = ["".join(map(str, call.args)) for call in mock_print.call_args_list]
    has_warning = any(
        "app setup exceeded startup_timeout" in line for line in printed_lines
    )
    assert has_warning == should_warn, (
        f"Expected warning={'yes' if should_warn else 'no'} "
        f"(startup_timeout={startup_timeout}, elapsed={elapsed_seconds}s), "
        f"but got: {printed_lines}"
    )


def test_function_decorator_rejects_app_files_with_container_kind():
    """Test that app_files cannot be used with kind='container'."""
    image = ContainerImage.from_dockerfile_str("FROM python:3.11-slim")

    error_message = "app_files is not supported for container apps"
    with pytest.raises(ValueError, match=error_message):

        @fal.function("container", app_files=["a.py"])
        def container_because_kind_is_container():
            pass

    with pytest.raises(ValueError, match=error_message):

        @fal.function(image=image, app_files=["a.py"])
        def container_because_image_is_provided():
            pass


def test_app_classvars_propagate_to_host_kwargs_when_overriding_hidden_defaults():
    class VarsApp(App):
        _scheduler = "kubernetes"
        _scheduler_options = {
            "storage_region": "us-west",
        }
        keep_alive = 30
        resolver = "pip"
        _app_var = "example"

    hk = VarsApp.host_kwargs
    assert hk["_scheduler"] == "kubernetes"
    assert hk["_scheduler_options"] == {
        "storage_region": "us-west",
    }
    assert hk["keep_alive"] == 30
    assert hk["resolver"] == "pip"
    assert "_app_var" not in hk


def test_app_is_picklable_with_request_context(isolate_agent_env):
    app = PickleApp()
    app._current_request_context = ContextVar(  # type: ignore[assignment]
        "_current_request_context"
    )

    payload = pickle.dumps(app)
    loaded = pickle.loads(payload)

    assert loaded._current_request_context is None


def test_health_route_supports_async_health(isolate_agent_env):
    import fastapi
    from fastapi.testclient import TestClient

    class AsyncHealthApp(App):
        async def health(self):
            return {"status": "ok"}

    app = AsyncHealthApp()
    fastapi_app = fastapi.FastAPI()
    app._add_extra_routes(fastapi_app)

    client = TestClient(fastapi_app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_openapi_websocket_realtime_metadata_and_schemas(isolate_agent_env):
    app = RealtimeApp()
    spec = app.openapi()
    ws_spec = spec["paths"]["/realtime"]["x-fal-protocol"]

    assert ws_spec["type"] == "realtime"
    assert ws_spec["realtimeMode"] == "unary"
    assert ws_spec["contentType"] == "application/msgpack"
    assert ws_spec["config"] == {
        "buffering": 2,
        "sessionTimeout": 1.5,
        "maxBatchSize": 3,
    }
    assert "requestBody" not in ws_spec
    assert "responses" not in ws_spec
    assert "InputModel" in spec["components"]["schemas"]
    assert "OutputModel" in spec["components"]["schemas"]
    assert (
        spec["paths"]["/realtime"]["post"]["requestBody"]["content"][
            "application/json"
        ]["schema"]["$ref"]
        == "#/components/schemas/InputModel"
    )
    assert (
        spec["paths"]["/realtime"]["post"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]["$ref"]
        == "#/components/schemas/OutputModel"
    )


def test_openapi_websocket_barebones_has_no_realtime_marker(isolate_agent_env):
    app = RealtimeApp()
    spec = app.openapi()
    ws_spec = spec["paths"]["/ws"]["x-fal-protocol"]

    assert ws_spec["type"] == "websocket"
    assert "realtimeMode" not in ws_spec
    assert "contentType" not in ws_spec
    assert "requestBody" not in ws_spec
    assert "responses" not in ws_spec
    assert "post" not in spec["paths"]["/ws"]


def test_websocket_endpoint_requires_websocket_annotation(isolate_agent_env):
    class MissingWebSocketAnnotationApp(App):
        @endpoint("/ws", is_websocket=True)
        async def generate_ws(self, websocket) -> None:
            return None

    app = MissingWebSocketAnnotationApp()

    with patch("fal.api.api.fastapi_logger") as mock_logger:
        with pytest.raises(
            ValueError,
            match=(
                r"Websocket endpoint '.*generate_ws' at '/ws' must declare "
                r"a parameter annotated as fastapi\.WebSocket\."
            ),
        ) as exc_info:
            app._build_app()

    message = str(exc_info.value)
    assert "Got signature (self, websocket)" in message
    assert "async def generate_ws(self, websocket: WebSocket):" in message
    mock_logger.error.assert_called_once_with(message)


def test_websocket_endpoint_requires_async_handler(isolate_agent_env):
    class SyncWebSocketApp(App):
        @endpoint("/ws", is_websocket=True)
        def generate_ws(self, websocket: WebSocket) -> None:
            return None

    app = SyncWebSocketApp()

    with patch("fal.api.api.fastapi_logger") as mock_logger:
        with pytest.raises(
            ValueError,
            match=(
                r"Websocket endpoint '.*generate_ws' at '/ws' must be defined "
                r"with async def\."
            ),
        ) as exc_info:
            app._build_app()

    message = str(exc_info.value)
    assert "Got signature (self, websocket" in message
    assert (
        "For example: `async def generate_ws(self, websocket: WebSocket): ...`?"
        in message
    )
    mock_logger.error.assert_called_once_with(message)


def test_openapi_websocket_realtime_streaming_modes_marked(isolate_agent_env):
    app = RealtimeApp()
    spec = app.openapi()

    assert spec["paths"]["/realtime/json"]["x-fal-protocol"]["contentType"] == (
        "application/json"
    )
    assert (
        spec["paths"]["/realtime/server-streaming"]["x-fal-protocol"]["realtimeMode"]
        == "server_streaming"
    )
    assert (
        spec["paths"]["/realtime/server-streaming-sync"]["x-fal-protocol"][
            "realtimeMode"
        ]
        == "server_streaming"
    )
    assert (
        spec["paths"]["/realtime/client-streaming"]["x-fal-protocol"]["realtimeMode"]
        == "client_streaming"
    )
    assert spec["paths"]["/realtime/bidi"]["x-fal-protocol"]["realtimeMode"] == "bidi"

    for path in [
        "/realtime/server-streaming",
        "/realtime/server-streaming-sync",
        "/realtime/client-streaming",
        "/realtime/bidi",
    ]:
        ws_spec = spec["paths"][path]["x-fal-protocol"]
        assert ws_spec["type"] == "realtime"
        assert ws_spec["contentType"] == "application/msgpack"
        assert "requestBody" not in ws_spec
        assert "responses" not in ws_spec
        assert (
            spec["paths"][path]["post"]["requestBody"]["content"]["application/json"][
                "schema"
            ]["$ref"]
            == "#/components/schemas/InputModel"
        )
        assert (
            spec["paths"][path]["post"]["responses"]["200"]["content"][
                "application/json"
            ]["schema"]["$ref"]
            == "#/components/schemas/OutputModel"
        )


def test_openapi_does_not_duplicate_ws_paths_on_multiple_calls(isolate_agent_env):
    app = RealtimeApp()
    fal_app = app._build_app()
    first = fal_app.openapi()
    second = fal_app.openapi()

    expected_order = [
        "/realtime",
        "/realtime/bidi",
        "/realtime/client-streaming",
        "/realtime/json",
        "/realtime/server-streaming",
        "/realtime/server-streaming-sync",
        "/ws",
    ]
    assert [p for p in first["x-fal-order-paths"] if p != "/health"] == expected_order
    assert [p for p in second["x-fal-order-paths"] if p != "/health"] == expected_order


@pytest.mark.asyncio
async def test_serve_exits_with_exception_on_setup_failure(
    isolate_agent_env, monkeypatch: pytest.MonkeyPatch
):
    import asyncio

    from fal.api import api

    class FakeServer:
        def __init__(self, config):
            self.config = config
            self.started = False

        def set_handle_exit(self, handle_exit):
            self._handle_exit = handle_exit

        async def serve(self) -> None:
            app = self.config.app
            async with app.router.lifespan_context(app):
                self.started = True

    class FakeMetricsServer:
        def __init__(self, config):
            pass

        async def serve(self) -> None:
            await asyncio.Event().wait()

    monkeypatch.setattr(api, "FalServer", FakeServer)
    monkeypatch.setattr(api.uvicorn, "Server", FakeMetricsServer)

    class FailingSetupApp(App):
        def setup(self):
            raise Exception("TEST EXCEPTION")

        @endpoint("/")
        def run(self):
            return {"status": "ok"}

    app = FailingSetupApp()

    with pytest.raises(Exception, match="TEST EXCEPTION"):
        await app.serve()


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def test_app_lifecycle_callbacks_are_called():
    """SIGTERM triggers handle_exit, then teardown runs during shutdown."""
    port = _find_free_port()
    metrics_port = _find_free_port()

    script = textwrap.dedent(f"""\
        import os
        os.environ["IS_ISOLATE_AGENT"] = "1"

        import asyncio
        from fal import App, endpoint

        class LifecycleApp(App):
            def setup(self):
                print("setup")

            def handle_exit(self):
                print("handle_exit")

            def teardown(self):
                print("teardown")

            @endpoint("/")
            def run(self):
                return "ok"

        app = LifecycleApp()
        asyncio.run(app.serve(port={port}, metrics_port={metrics_port}))
    """)

    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert proc.stdout is not None

    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("localhost", port)) == 0:
                    break
            time.sleep(0.1)
        else:
            raise AssertionError("App server did not start in time")

        time.sleep(0.5)
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    lines = [line.decode("utf-8").strip() for line in proc.stdout.readlines()]

    assert "setup" in lines
    assert "handle_exit" in lines
    assert "teardown" in lines
