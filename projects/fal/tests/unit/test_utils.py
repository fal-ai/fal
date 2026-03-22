from unittest.mock import patch

import pytest

import fal
from fal.api import IsolatedFunction, Options
from fal.api.api import merge_basic_config
from fal.api.client import SyncServerlessClient
from fal.utils import (
    _build_local_wheel_install_target,
    _build_runtime_pip_install_command,
    _find_target,
    _materialize_runtime_wheel_path,
    _parse_python_entry_point,
    load_function_from,
)


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


@patch("fal.utils._build_project_wheel")
def test_load_from_python_entry_point_strips_local_requirements(
    mock_build_wheel, tmp_path
):
    from fal.utils import _load_from_python_entry_point

    wheel_dir = tmp_path / "dist"
    wheel_dir.mkdir()
    wheel_path = wheel_dir / "simple-0.1.0-py3-none-any.whl"
    wheel_path.write_text("wheel")
    mock_build_wheel.return_value = wheel_path

    host = SyncServerlessClient(host="api.alpha.fal.ai")._create_host()
    options = Options(environment={"requirements": [".[func]", "torch"]})
    loaded = _load_from_python_entry_point(
        host,
        "simple.func:simple_func",
        project_root=tmp_path,
        options=options,
    )

    assert loaded.function.options.environment["requirements"] == ["torch", "fal"]
    assert loaded.function.options.host["app_files_context_dir"] == str(
        tmp_path.resolve()
    )
    assert loaded.function.options.host["app_files"] == [
        str(wheel_path.relative_to(tmp_path.resolve()))
    ]


@patch("fal.utils._build_project_wheel")
def test_load_from_python_entry_point_does_not_duplicate_fal_requirement(
    mock_build_wheel, tmp_path
):
    from fal.utils import _load_from_python_entry_point

    wheel_dir = tmp_path / "dist"
    wheel_dir.mkdir()
    wheel_path = wheel_dir / "simple-0.1.0-py3-none-any.whl"
    wheel_path.write_text("wheel")
    mock_build_wheel.return_value = wheel_path

    host = SyncServerlessClient(host="api.alpha.fal.ai")._create_host()
    options = Options(environment={"requirements": [".[func]", "fal>=1.0.0", "torch"]})
    loaded = _load_from_python_entry_point(
        host,
        "simple.func:simple_func",
        project_root=tmp_path,
        options=options,
    )

    assert loaded.function.options.environment["requirements"] == [
        "fal>=1.0.0",
        "torch",
    ]


@patch("fal.utils._build_project_wheel")
def test_load_from_python_entry_point_stages_wheel_inside_context_dir(
    mock_build_wheel, tmp_path
):
    from fal.utils import _load_from_python_entry_point

    external_wheel_dir = tmp_path.parent / "external-no-pickle-wheel"
    external_wheel_dir.mkdir(exist_ok=True)
    external_wheel_path = external_wheel_dir / "simple-0.1.0-py3-none-any.whl"
    external_wheel_path.write_text("wheel")
    mock_build_wheel.return_value = external_wheel_path

    host = SyncServerlessClient(host="api.alpha.fal.ai")._create_host()
    options = Options(
        host={
            "app_files_context_dir": str(tmp_path.resolve()),
            "app_files": ["assets/config.json"],
        }
    )
    loaded = _load_from_python_entry_point(
        host,
        "simple.func:simple_func",
        project_root=tmp_path,
        options=options,
    )

    staged_relative_path = "fal-no-pickle-wheels/simple-0.1.0-py3-none-any.whl"
    assert loaded.function.options.host["app_files"] == [
        "assets/config.json",
        staged_relative_path,
    ]
    assert (tmp_path / staged_relative_path).read_text() == "wheel"


def test_build_local_wheel_install_target_without_extras(tmp_path, monkeypatch):
    wheel_path = tmp_path / "fal-no-pickle-wheels" / "simple-0.1.0-py3-none-any.whl"
    expected = str(wheel_path.resolve())

    assert (
        _build_local_wheel_install_target(
            wheel_local_path=wheel_path,
            wheel_file_name="simple-0.1.0-py3-none-any.whl",
            extras=[],
        )
        == expected
    )


def test_build_local_wheel_install_target_with_extras(tmp_path):
    wheel_local_path = (
        tmp_path / "fal-no-pickle-wheels" / "simple-0.1.0-py3-none-any.whl"
    ).resolve()
    expected = f"simple[func,image] @ {wheel_local_path.as_uri()}"

    assert (
        _build_local_wheel_install_target(
            wheel_local_path=wheel_local_path,
            wheel_file_name="simple-0.1.0-py3-none-any.whl",
            extras=["image", "func", "func"],
        )
        == expected
    )


def test_materialize_runtime_wheel_path_keeps_named_whl(tmp_path):
    wheel_path = tmp_path / "simple-0.1.0-py3-none-any.whl"
    wheel_path.write_text("wheel")

    materialized = _materialize_runtime_wheel_path(
        wheel_local_path=wheel_path,
        wheel_file_name="simple-0.1.0-py3-none-any.whl",
    )

    assert materialized == wheel_path.resolve()


def test_materialize_runtime_wheel_path_copies_blob_path(tmp_path):
    blob_path = tmp_path / "app-blobs" / "fce09c55484593342d0eb63df89b6cae"
    blob_path.parent.mkdir(parents=True)
    blob_path.write_text("wheel")

    materialized = _materialize_runtime_wheel_path(
        wheel_local_path=blob_path,
        wheel_file_name="simple-0.1.0-py3-none-any.whl",
    )

    assert materialized.name == "simple-0.1.0-py3-none-any.whl"
    assert materialized.suffix == ".whl"
    assert materialized.read_text() == "wheel"


def test_build_runtime_pip_install_command(tmp_path):
    command = _build_runtime_pip_install_command(
        install_target="simple[func] @ file:///tmp/simple-0.1.0-py3-none-any.whl",
        target_dir=tmp_path / "site",
    )

    assert command[1:6] == ["-m", "pip", "install", "--no-cache-dir", "--target"]
    assert command[6] == str(tmp_path / "site")
    assert command[7] == "simple[func] @ file:///tmp/simple-0.1.0-py3-none-any.whl"


def test_parse_python_entry_point():
    module_name, symbol_name = _parse_python_entry_point("pkg.mod:MyApp")
    assert module_name == "pkg.mod"
    assert symbol_name == "MyApp"


def test_parse_python_entry_point_invalid_format():
    with pytest.raises(Exception) as exc:
        _parse_python_entry_point("pkg.mod")
    assert "python_entry_point must be in '<module>:<symbol>' format." in str(exc.value)
