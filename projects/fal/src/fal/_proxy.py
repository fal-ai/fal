"""HTTPS proxy settings from the environment, shared by REST and gRPC calls."""

from __future__ import annotations

import os
import urllib.parse


def https_proxy_for(url_or_host: str) -> str | None:
    """The proxy URL that HTTPS calls to this host go through, if any.

    Reads HTTPS_PROXY / https_proxy and honors NO_PROXY / no_proxy, the same
    way httpx and urllib do.
    """
    # Loaded on first use: importing fal.sdk should not pull in urllib.request.
    import urllib.request  # noqa: PLC0415

    host = urllib.parse.urlsplit(url_or_host).hostname or url_or_host.partition(":")[0]
    proxy = urllib.request.getproxies_environment().get("https")
    if not proxy or urllib.request.proxy_bypass_environment(host):
        return None
    return proxy


def grpc_proxy_options(target: str) -> list[tuple[str, str | int]]:
    """Channel options that send a gRPC channel to `target` through the proxy.

    gRPC only reads the lowercase https_proxy (and grpc_proxy) itself, so the
    proxy is passed as the `grpc.http_proxy` option, which also covers an
    uppercase HTTPS_PROXY. An explicit grpc_proxy variable keeps precedence.
    """
    import urllib.request  # noqa: PLC0415

    host = urllib.parse.urlsplit(target).hostname or target.partition(":")[0]
    if urllib.request.getproxies_environment().get(
        "https"
    ) and urllib.request.proxy_bypass_environment(host):
        # gRPC itself ignores NO_PROXY and would still use a lowercase https_proxy.
        return [("grpc.enable_http_proxy", 0)]
    if os.environ.get("grpc_proxy"):
        return []
    proxy = https_proxy_for(target)
    return [("grpc.http_proxy", proxy)] if proxy else []


def redact(proxy: str) -> str:
    """The proxy URL without credentials, safe to show."""
    parts = urllib.parse.urlsplit(proxy)
    if parts.username is None and parts.password is None:
        return proxy
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urllib.parse.urlunsplit(parts._replace(netloc=f"***@{host}"))


PROXY_HINT = (
    "check that the proxy is up and allows CONNECT to the host; a custom CA "
    "needs SSL_CERT_FILE (and GRPC_DEFAULT_SSL_ROOTS_FILE_PATH for deploys)"
)


def unreachable_message(host: str, reason: object) -> str:
    """Why a connect to `host` failed, naming the proxy in use when one is set."""
    text = str(reason) or type(reason).__name__
    proxy = https_proxy_for(host)
    if not proxy:
        return f"could not reach {host}: {text}"
    return (
        f"could not reach {host} through proxy {redact(proxy)}: {text}\n"
        f"hint: {PROXY_HINT}"
    )
