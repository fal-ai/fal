from __future__ import annotations

import os

import pytest

import fal
from fal import App, endpoint
from fal.container import ContainerImage


def test_app_regions_propagate_to_function_options():
    from fal.app import wrap_app

    class RegionsApp(App):
        regions = ["us-east", "eu-west"]

        @endpoint("/")
        def hello(self) -> str:
            return "Hello, world!"

    fn = wrap_app(RegionsApp)
    assert fn.options.host.get("regions") == ["us-east", "eu-west"]


def test_app_default_app_name_is_generated_from_class_name():
    class MyCustomApp(App):
        pass

    assert MyCustomApp.app_name == "my-custom-app"


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
async def test_runner_state_lifecycle_complete():
    """Test that FAL_RUNNER_STATE transitions through all phases correctly"""
    states = []

    class StateCheckApp(App):
        def setup(self):
            states.append(("setup", os.getenv("FAL_RUNNER_STATE")))

        def teardown(self):
            states.append(("teardown", os.getenv("FAL_RUNNER_STATE")))

    app = StateCheckApp(_allow_init=True)

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
