from unittest.mock import MagicMock

import pytest

import fal
from fal.api import IsolatedFunction, Options
from fal.api.api import merge_basic_config
from fal.api.client import SyncServerlessClient
from fal.utils import (
    _find_target,
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


def test_parse_python_entry_point():
    module_name, symbol_name = _parse_python_entry_point("pkg.mod:MyApp")
    assert module_name == "pkg.mod"
    assert symbol_name == "MyApp"


def test_parse_python_entry_point_invalid_format():
    with pytest.raises(Exception) as exc:
        _parse_python_entry_point("pkg.mod")
    assert "python_entry_point must be in '<module>:<symbol>' format." in str(exc.value)


def test_load_from_python_entry_point_is_passive():
    """``load_function_from(python_entry_point=...)`` must not hit the worker;
    metadata is fetched explicitly later via ``IsolatedFunction.fetch_metadata``.
    """
    from fal.utils import _load_from_python_entry_point

    host = MagicMock(spec=["run"])  # no .run() attribute access expected
    options = Options(environment={"requirements": ["fal", "torch"]})

    loaded = _load_from_python_entry_point(
        host,
        "simple.app:SimpleApp",
        options=options,
    )

    host.run.assert_not_called()
    assert loaded.function.entrypoint == "simple.app:SimpleApp"
    assert loaded.function.run_entrypoint == "simple.app:SimpleApp.run_local"
    assert loaded.function.raw_func is None
    assert loaded.function.options.gateway["exposed_port"] == 8080
    assert loaded.function.options.environment["requirements"] == ["fal", "torch"]
    assert loaded.function.endpoints == ["/"]
    assert loaded.function.build_metadata() == {}


def test_load_from_python_entry_point_keeps_pyproject_exposed_port():
    from fal.utils import _load_from_python_entry_point

    host = MagicMock(spec=["run"])
    options = Options(gateway={"exposed_port": 9000})
    loaded = _load_from_python_entry_point(
        host,
        "simple.app:SimpleApp",
        options=options,
    )

    assert loaded.function.options.gateway["exposed_port"] == 9000


def test_isolated_function_endpoints_from_metadata():
    iso = IsolatedFunction(
        host=DummyHost(),
        entrypoint="pkg:Sym",
    )
    iso.options.host["metadata"] = {"openapi": {"paths": {"/predict": {}, "/info": {}}}}

    assert sorted(iso.endpoints) == ["/info", "/predict"]


def test_isolated_function_endpoints_fallback_to_routes():
    func = lambda: None  # noqa: E731
    func._routes = ["/custom"]  # type: ignore[attr-defined]
    iso = IsolatedFunction(host=DummyHost(), raw_func=func, options=Options())

    assert iso.endpoints == ["/custom"]


def test_isolated_function_endpoints_default():
    iso = IsolatedFunction(host=DummyHost(), entrypoint="pkg:Sym")
    assert iso.endpoints == ["/"]


def test_isolated_function_fetch_metadata_no_op_without_entrypoint():
    iso = IsolatedFunction(
        host=DummyHost(),
        raw_func=lambda: None,
        options=Options(),
    )
    iso.options.host["metadata"] = {"openapi": {"paths": {"/x": {}}}}

    assert iso.fetch_metadata() == {"openapi": {"paths": {"/x": {}}}}


def test_isolated_function_fetch_metadata_probes_worker():
    host = MagicMock()
    host.run.return_value = {"openapi": {"paths": {"/predict": {}}}}

    iso = IsolatedFunction(host=host, entrypoint="pkg.mod:MyApp")
    metadata = iso.fetch_metadata()

    host.run.assert_called_once()
    _, call_kwargs = host.run.call_args
    assert call_kwargs["entrypoint"] == "pkg.mod:MyApp.build_metadata"
    assert metadata == {"openapi": {"paths": {"/predict": {}}}}
    assert iso.options.host["metadata"] == metadata
    assert iso.endpoints == ["/predict"]


def test_isolated_function_fetch_metadata_is_idempotent():
    host = MagicMock()
    host.run.return_value = {"openapi": {"paths": {"/predict": {}}}}

    iso = IsolatedFunction(host=host, entrypoint="pkg.mod:MyApp")
    first = iso.fetch_metadata()
    second = iso.fetch_metadata()

    host.run.assert_called_once()
    assert first == second == {"openapi": {"paths": {"/predict": {}}}}


def test_isolated_function_fetch_metadata_rejects_non_dict_payload():
    from fal.api import FalServerlessError

    host = MagicMock()
    host.run.return_value = "not a dict"

    iso = IsolatedFunction(host=host, entrypoint="pkg.mod:MyApp")
    with pytest.raises(FalServerlessError, match="non-dict payload"):
        iso.fetch_metadata()


def test_isolated_function_requires_func_or_entrypoint():
    with pytest.raises(Exception, match="raw_func or entrypoint"):
        IsolatedFunction(host=DummyHost())
