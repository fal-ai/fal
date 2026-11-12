import ssl
from pathlib import Path
from unittest.mock import patch

import certifi

from fal._tls import create_ssl_context
from fal.sdk import RemoteCredentials


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


def test_remote_grpc_credentials_use_certifi_roots(monkeypatch):
    monkeypatch.delenv("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH", raising=False)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)

    with patch("fal.sdk.grpc.ssl_channel_credentials") as ssl_channel_credentials:
        RemoteCredentials().to_grpc()

    ssl_channel_credentials.assert_called_once_with(
        root_certificates=Path(certifi.where()).read_bytes()
    )


def test_remote_grpc_credentials_honor_ssl_cert_file(monkeypatch, tmp_path):
    private_ca = tmp_path / "private-ca.pem"
    private_ca.write_bytes(b"private CA bundle")
    monkeypatch.delenv("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH", raising=False)
    monkeypatch.setenv("SSL_CERT_FILE", str(private_ca))

    with patch("fal.sdk.grpc.ssl_channel_credentials") as ssl_channel_credentials:
        RemoteCredentials().to_grpc()

    ssl_channel_credentials.assert_called_once_with(
        root_certificates=b"private CA bundle"
    )
