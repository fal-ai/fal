import os
import ssl

import certifi

CERTIFI_REQUIREMENT = "certifi>=2026.10.19"


def create_ssl_context() -> ssl.SSLContext:
    ca_file = os.environ.get("SSL_CERT_FILE") or None
    ca_dir = os.environ.get("SSL_CERT_DIR") or None
    if ca_file or ca_dir:
        return ssl.create_default_context(cafile=ca_file, capath=ca_dir)
    return ssl.create_default_context(cafile=certifi.where())
