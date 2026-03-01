import pytest

import fal
from fal.api import IsolatedFunction, Options
from fal.api.api import merge_basic_config
from fal.api.client import SyncServerlessClient
from fal.utils import _find_target, load_function_from


class DummyHost:
    pass


def make_isolated_function(name: str = "func"):
    # Name is used only to label the symbol in the module dict
    return name, IsolatedFunction(
        host=DummyHost(), raw_func=lambda: None, options=Options()
    )


def test_find_target_by_name_missing():
    module = {}
    with pytest.raises(Exception) as exc:
        _find_target(module, "missing")
    assert "Function 'missing' not found in module" in str(exc.value)


def test_find_target_by_name_app_returns_class_and_metadata():
    class MyApp(fal.App):
        app_name = "my_app"
        app_auth = "public"

    module = {"MyApp": MyApp}
    target, app_name, app_auth, class_name = _find_target(module, "MyApp")

    assert target is MyApp
    assert app_name == MyApp.app_name
    assert app_auth == MyApp.app_auth
    assert class_name == "MyApp"


def test_find_target_by_name_isolated_function_returns_function_and_name():
    name, iso = make_isolated_function("the_func")
    module = {name: iso}

    target, returned_name, auth, class_name = _find_target(module, name)

    assert target is iso
    assert returned_name == name
    assert auth is None
    assert class_name == name


def test_find_target_by_name_invalid_type_raises():
    module = {"not_valid": 123}
    with pytest.raises(Exception) as exc:
        _find_target(module, "not_valid")
    assert "is not a fal.App or a fal.function" in str(exc.value)


def test_find_target_single_app_without_name():
    class OnlyApp(fal.App):
        app_name = "only_app"
        app_auth = "private"

    module = {"OnlyApp": OnlyApp}
    target, app_name, app_auth, class_name = _find_target(module)

    assert target is OnlyApp
    assert app_name == OnlyApp.app_name
    assert app_auth == OnlyApp.app_auth
    assert class_name == "OnlyApp"


def test_find_target_multiple_apps_without_name_raises():
    class AppOne(fal.App):
        pass

    class AppTwo(fal.App):
        pass

    module = {"AppOne": AppOne, "AppTwo": AppTwo}
    with pytest.raises(Exception) as exc:
        _find_target(module)
    assert "Multiple fal.Apps found in the module" in str(exc.value)


def test_find_target_no_apps_no_functions_raises():
    module = {"something": object()}
    with pytest.raises(Exception) as exc:
        _find_target(module)
    assert "No fal.App or fal.function found in the module" in str(exc.value)


def test_find_target_multiple_functions_without_name_raises():
    name1, iso1 = make_isolated_function("f1")
    name2, iso2 = make_isolated_function("f2")

    module = {name1: iso1, name2: iso2}
    with pytest.raises(Exception) as exc:
        _find_target(module)
    assert "Multiple fal.functions found in the module" in str(exc.value)


def test_find_target_single_function_without_name():
    name, iso = make_isolated_function("single")
    module = {name: iso}

    target, function_name, auth, class_name = _find_target(module)

    assert target is iso
    assert function_name == name
    assert auth is None
    assert class_name == name


def test_merge_basic_config_preserves_existing_values():
    target = {"min_concurrency": 2, "regions": ["us-east"]}
    incoming = {"min_concurrency": 10, "max_concurrency": 20}

    merge_basic_config(target, incoming)

    assert target["min_concurrency"] == 2
    assert target["regions"] == ["us-east"]
    assert target["max_concurrency"] == 20


def _extract_wrapped_app_class(loaded):
    freevars = loaded.function.raw_func.__code__.co_freevars
    cls_index = freevars.index("cls")
    return loaded.function.raw_func.__closure__[cls_index].cell_contents


def test_load_function_from_applies_toml_app_files_for_fal_app(tmp_path):
    app_file = tmp_path / "app.py"
    app_file.write_text(
        "import fal\n"
        "\n"
        "class MyApp(fal.App):\n"
        "    @fal.endpoint('/')\n"
        "    def run(self):\n"
        "        return {'ok': True}\n"
    )

    client = SyncServerlessClient(host="api.alpha.fal.ai")
    host = client._create_host(local_file_path=str(app_file))
    options = Options(
        host={
            "app_files": ["assets", "config.yaml"],
            "app_files_ignore": [r"\\.venv/"],
            "app_files_context_dir": ".",
        }
    )

    loaded = load_function_from(host, str(app_file), "MyApp", options=options)
    wrapped_cls = _extract_wrapped_app_class(loaded)

    assert loaded.function.options.host["app_files"] == ["assets", "config.yaml"]
    assert loaded.function.options.host["app_files_ignore"] == [r"\\.venv/"]
    assert loaded.function.options.host["app_files_context_dir"] == "."
    assert wrapped_cls.app_files == ["assets", "config.yaml"]
    assert wrapped_cls.app_files_ignore == [r"\\.venv/"]
    assert wrapped_cls.app_files_context_dir == "."


def test_load_function_from_preserves_app_defined_app_files_over_toml(tmp_path):
    app_file = tmp_path / "app.py"
    app_file.write_text(
        "import fal\n"
        "\n"
        "class MyApp(fal.App):\n"
        "    app_files = ['class-files']\n"
        "    app_files_ignore = ['class-ignore']\n"
        "    app_files_context_dir = 'class-context'\n"
        "\n"
        "    @fal.endpoint('/')\n"
        "    def run(self):\n"
        "        return {'ok': True}\n"
    )

    client = SyncServerlessClient(host="api.alpha.fal.ai")
    host = client._create_host(local_file_path=str(app_file))
    options = Options(
        host={
            "app_files": ["toml-files"],
            "app_files_ignore": ["toml-ignore"],
            "app_files_context_dir": "toml-context",
        }
    )

    loaded = load_function_from(host, str(app_file), "MyApp", options=options)
    wrapped_cls = _extract_wrapped_app_class(loaded)

    assert loaded.function.options.host["app_files"] == ["class-files"]
    assert loaded.function.options.host["app_files_ignore"] == ["class-ignore"]
    assert loaded.function.options.host["app_files_context_dir"] == "class-context"
    assert wrapped_cls.app_files == ["class-files"]
    assert wrapped_cls.app_files_ignore == ["class-ignore"]
    assert wrapped_cls.app_files_context_dir == "class-context"
