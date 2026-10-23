import ssl
from unittest.mock import patch

import certifi
import pytest
from isolate.backends.common import Requirements

import fal
from fal._tls import CERTIFI_REQUIREMENT, create_ssl_context
from fal.api.api import FalServerlessHost, Options, function
from fal.app import wrap_app


def test_ssl_context_uses_only_certifi_roots(monkeypatch):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    expected = ssl.create_default_context(cafile=certifi.where())

    def reject_system_roots(*args, **kwargs):
        raise AssertionError("System roots must not be loaded")

    monkeypatch.setattr(ssl.SSLContext, "load_default_certs", reject_system_roots)

    context = create_ssl_context()

    assert context.get_ca_certs(binary_form=True)
    assert context.get_ca_certs(binary_form=True) == expected.get_ca_certs(
        binary_form=True
    )
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname


def test_ssl_context_honors_ca_file_and_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("SSL_CERT_FILE", certifi.where())
    monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path))

    with patch(
        "ssl.create_default_context", wraps=ssl.create_default_context
    ) as create_context:
        context = create_ssl_context()

    # OpenSSL loads directory certificates lazily; check the configured path.
    create_context.assert_called_once_with(cafile=certifi.where(), capath=str(tmp_path))
    assert context.get_ca_certs(binary_form=True)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname


def test_function_runner_receives_certifi_without_mutating_requirements():
    requirements = [["numpy"], ["pillow"]]

    @function(host=FalServerlessHost(), requirements=requirements)
    def target():
        return "ok"

    prepared = target.host.prepare_options(target.options, func=target.raw_func)

    assert prepared.environment["requirements"] == [
        ["numpy"],
        ["pillow"],
        ["certifi>=2026.10.19"],
    ]
    assert requirements == [["numpy"], ["pillow"]]
    assert target.options.environment["requirements"] == requirements


@pytest.mark.parametrize(
    "kind,serve,layered,pin",
    [
        ("virtualenv", False, False, "certifi==2024.8.30"),
        ("virtualenv", True, True, "Certifi<2025"),
        ("container", True, False, "certifi==2024.8.30"),
        ("conda", False, True, "certifi==2024.8.30"),
    ],
)
def test_explicit_certifi_requirement_opts_out_of_default(kind, serve, layered, pin):
    requirements = [[pin], ["numpy"]] if layered else [pin, "numpy"]
    requirement_key = "pip" if kind == "conda" else "requirements"

    @function(
        host=FalServerlessHost(),
        kind=kind,
        serve=serve,
        **{requirement_key: requirements},
    )
    def target():
        return "ok"

    prepared = target.host.prepare_options(target.options, func=target.raw_func)
    layers = Requirements.from_raw(prepared.environment[requirement_key]).layers
    flattened = [req for layer in layers for req in layer]

    assert pin in flattened
    assert CERTIFI_REQUIREMENT not in flattened
    assert "numpy" in flattened
    if serve:
        assert any(req.startswith("fastapi==") for req in flattened)


def test_app_preserves_explicit_certifi_requirement():
    class PinnedApp(fal.App):
        requirements = ["certifi==2024.8.30"]

        @fal.endpoint("/")
        def run(self) -> str:
            return "ok"

    target = wrap_app(PinnedApp)
    prepared = target.host.prepare_options(target.options, func=target.raw_func)

    assert "certifi==2024.8.30" in prepared.environment["requirements"]
    assert CERTIFI_REQUIREMENT not in prepared.environment["requirements"]
    assert PinnedApp.requirements == ["certifi==2024.8.30"]


@pytest.mark.parametrize(
    "source,definition",
    [
        ("packages", ["certifi=2024.8.30"]),
        ("env_dict", {"dependencies": [{"pip": ["Certifi==2024.8.30"]}]}),
        ("env_yml_str", "dependencies:\n  - conda-forge::certifi<2025\n"),
    ],
)
def test_conda_certifi_declaration_opts_out_of_default(source, definition):
    options = Options(environment={"kind": "conda", source: definition})

    prepared = FalServerlessHost().prepare_options(options)

    assert prepared.environment[source] == definition
    assert prepared.environment["pip"] == []
    assert "pip" not in options.environment


def test_unrelated_conda_pip_dependencies_still_receive_certifi():
    options = Options(environment={"kind": "conda", "packages": [{"pip": ["numpy"]}]})

    prepared = FalServerlessHost().prepare_options(options)

    assert prepared.environment["pip"] == [CERTIFI_REQUIREMENT]
