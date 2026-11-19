import os
import ssl
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import certifi
import cloudpickle

import fal
from fal._tls import create_ssl_context, grpc_root_certificates
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


def test_pickled_tls_helpers_load_without_certifi():
    # fal is pickled by value into runner envs that may not install certifi.
    registered = "fal" in cloudpickle.list_registry_pickle_by_value()
    cloudpickle.register_pickle_by_value(fal)
    try:
        payload = cloudpickle.dumps((create_ssl_context, grpc_root_certificates))
    finally:
        if not registered:
            cloudpickle.unregister_pickle_by_value(fal)
    code = """
import pickle, ssl, sys
sys.modules["certifi"] = None
create_ssl_context, grpc_root_certificates = pickle.loads(sys.stdin.buffer.read())
assert create_ssl_context().verify_mode == ssl.CERT_REQUIRED
assert grpc_root_certificates() is None
"""
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in ("SSL_CERT_FILE", "SSL_CERT_DIR", "GRPC_DEFAULT_SSL_ROOTS_FILE_PATH")
    }

    subprocess.run([sys.executable, "-c", code], input=payload, check=True, env=env)
