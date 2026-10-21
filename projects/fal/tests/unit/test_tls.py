import ssl
from unittest.mock import patch

import certifi

from fal._tls import create_ssl_context
from fal.api.api import FalServerlessHost, function


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
