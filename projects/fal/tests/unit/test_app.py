from __future__ import annotations

from fal import App


def test_app_default_app_name_is_generated_from_class_name():
    class MyCustomApp(App):
        pass

    assert MyCustomApp.app_name == "my-custom-app"


def test_app_classvars_propagate_to_host_kwargs():
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
