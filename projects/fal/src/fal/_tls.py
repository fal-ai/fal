from __future__ import annotations

import os
import ssl
from pathlib import Path

try:
    import certifi
except ImportError:
    # fal code is pickled into runner envs that may not install certifi.
    # Those keep the default roots, as before.
    certifi = None  # type: ignore[assignment]


def create_ssl_context() -> ssl.SSLContext:
    ca_file = os.environ.get("SSL_CERT_FILE") or None
    ca_dir = os.environ.get("SSL_CERT_DIR") or None
    if ca_file or ca_dir:
        return ssl.create_default_context(cafile=ca_file, capath=ca_dir)
    if certifi is None:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def grpc_root_certificates() -> bytes | None:
    """PEM roots for gRPC, or None for grpcio's bundled roots."""
    root_file = os.environ.get("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH") or os.environ.get(
        "SSL_CERT_FILE"
    )
    if not root_file and certifi is not None:
        root_file = certifi.where()
    return Path(root_file).read_bytes() if root_file else None
