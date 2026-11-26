from __future__ import annotations

import os
import ssl
from pathlib import Path


def _certifi_file() -> str | None:
    # Imported here, not at module level: cloudpickle ships fal by value, so a
    # module-level certifi would be re-imported when unpickling in runner envs
    # that do not install it. Those keep the default roots, as before.
    try:
        import certifi  # noqa: PLC0415
    except ImportError:
        return None
    return certifi.where()


def create_ssl_context() -> ssl.SSLContext:
    ca_file = os.environ.get("SSL_CERT_FILE") or None
    ca_dir = os.environ.get("SSL_CERT_DIR") or None
    if ca_file or ca_dir:
        return ssl.create_default_context(cafile=ca_file, capath=ca_dir)
    return ssl.create_default_context(cafile=_certifi_file())


def grpc_root_certificates() -> bytes | None:
    """PEM roots for gRPC, or None for grpcio's bundled roots."""
    root_file = (
        os.environ.get("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH")
        or os.environ.get("SSL_CERT_FILE")
        or _certifi_file()
    )
    return Path(root_file).read_bytes() if root_file else None
