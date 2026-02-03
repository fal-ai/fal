from __future__ import annotations

import contextvars
import os
import pickle
from contextvars import ContextVar

import pytest

import fal
from fal import App, endpoint
from fal.app import LOG_CONTEXT_PREFIX, clear_contextvars, merge_contextvars
from fal.container import ContainerImage


class PickleApp(App):
    pass


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


def test_merge_context_vars():
    labels = {"fal_request_id": "123", "fal_endpoint": "/"}
    request_id_var = f"{LOG_CONTEXT_PREFIX}fal_request_id"
    endpoint_var = f"{LOG_CONTEXT_PREFIX}fal_endpoint"
    unrelated_var = "unrelated_key"
    contextvars.ContextVar(unrelated_var).set("value")

    # We have to convert to dict and lookup by name because each ContextVar
    # is a different object. Since merge_contextvars creates new ContextVars,
    # we can't just do direct lookups.
    vars = dict((k.name, v) for k, v in contextvars.copy_context().items())

    assert vars.get(unrelated_var) == "value"

    assert vars.get(request_id_var) is None
    assert vars.get(endpoint_var) is None

    merge_contextvars(labels)
    vars = dict((k.name, v) for k, v in contextvars.copy_context().items())

    assert vars.get(request_id_var) == "123"
    assert vars.get(endpoint_var) == "/"

    clear_contextvars()
    vars = dict((k.name, v) for k, v in contextvars.copy_context().items())

    # Cleared contextvars are set to Ellipsis
    assert vars.get(request_id_var) is Ellipsis
    assert vars.get(endpoint_var) is Ellipsis
    # Does not clear unrelated contextvars
    assert vars.get("unrelated_key") == "value"


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


def test_app_is_picklable_with_request_context():
    app = PickleApp(_allow_init=True)
    app._current_request_context = ContextVar(  # type: ignore[assignment]
        "_current_request_context"
    )

    payload = pickle.dumps(app)
    loaded = pickle.loads(payload)

    assert loaded._current_request_context is None
