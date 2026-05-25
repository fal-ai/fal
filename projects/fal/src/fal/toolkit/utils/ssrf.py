"""SSRF-safe HTTP(S) download helpers.

This client connects directly to validated target IPs and does not honor
HTTP_PROXY or HTTPS_PROXY environment variables.
"""

from __future__ import annotations

import contextlib
import http.client
import ipaddress
import os
import socket
import ssl
import tempfile
import urllib.parse
import warnings
from dataclasses import dataclass
from typing import Any, Callable

DEFAULT_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})
DEFAULT_MAX_REDIRECT_HOPS = 5

_BODY_CONTENT = "content"
_BODY_HEADERS = "headers"
_BODY_FILE = "file"
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_CROSS_ORIGIN_STRIPPED_HEADERS: frozenset[str] = frozenset(
    {"authorization", "cookie", "proxy-authorization"}
)
_DEFAULT_PORTS = {"http": 80, "https": 443}
_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")
_NAT64_WELL_KNOWN_PREFIX = ipaddress.ip_network("64:ff9b::/96")
_PROXY_ENV_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")


class SSRFError(Exception):
    pass


class SSRFHTTPStatusError(SSRFError):
    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(f"HTTP request failed with status code {status_code}")


class SSRFConnectionError(SSRFError):
    pass


@dataclass
class SafeResponse:
    status_code: int
    headers: dict[str, str]
    content: bytes = b""


def _socket_getaddrinfo(
    host: str | None,
    port: int | str | None,
    family: int = 0,
    type: int = 0,
    proto: int = 0,
    flags: int = 0,
) -> list[Any]:
    return socket.getaddrinfo(host, port, family, type, proto, flags)


def _warn_if_proxy_configured() -> None:
    configured_proxy_vars = [var for var in _PROXY_ENV_VARS if os.environ.get(var)]
    if not configured_proxy_vars:
        return

    warnings.warn(
        "HTTP_PROXY/HTTPS_PROXY settings are not honored by the SSRF-safe "
        "download client.",
        RuntimeWarning,
        stacklevel=3,
    )


def _embedded_nat64_ipv4(ip: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    if ip not in _NAT64_WELL_KNOWN_PREFIX:
        return None

    return ipaddress.IPv4Address(ip.packed[-4:])


def is_globally_routable_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False

    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    elif isinstance(ip, ipaddress.IPv6Address):
        nat64_ipv4 = _embedded_nat64_ipv4(ip)
        if nat64_ipv4 is not None:
            return is_globally_routable_ip(str(nat64_ipv4))

        transition_addresses = []
        if ip.sixtofour is not None:
            transition_addresses.append(ip.sixtofour)
        if ip.teredo is not None:
            transition_addresses.extend(ip.teredo)

        if any(
            not is_globally_routable_ip(str(embedded_ip))
            for embedded_ip in transition_addresses
        ):
            return False

    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return False

    if isinstance(ip, ipaddress.IPv4Address) and ip in _CGNAT_NETWORK:
        return False

    return True


def _parse_url(url: str):
    try:
        parsed = urllib.parse.urlsplit(url)
    except ValueError as exc:
        raise SSRFError(f"Invalid URL: {exc}") from exc

    if not parsed.scheme:
        raise SSRFError("URL has no scheme")

    return parsed


def _validate_scheme(parsed, allowed: frozenset[str]) -> None:
    if parsed.scheme not in allowed:
        raise SSRFError(f"URL scheme {parsed.scheme!r} is not allowed")


def _ips_from_addrinfo(addrinfo: list[Any]) -> list[str]:
    return [str(info[4][0]) for info in addrinfo]


def _validate_resolved_ips(ips: list[str]) -> list[str]:
    if not ips:
        raise SSRFError("Hostname did not resolve")

    for ip in ips:
        if not is_globally_routable_ip(ip):
            raise SSRFError("Hostname resolves to a non-routable address")

    return ips


def resolve_and_validate_host(hostname: str, port: int | None) -> list[str]:
    try:
        addrinfo = _socket_getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise SSRFError("DNS resolution failed") from exc

    return _validate_resolved_ips(_ips_from_addrinfo(addrinfo))


def _origin_tuple(parsed) -> tuple[str, str | None, int]:
    norm_scheme = parsed.scheme.lower()
    norm_port = parsed.port if parsed.port is not None else _DEFAULT_PORTS[norm_scheme]
    return (norm_scheme, parsed.hostname, norm_port)


def _build_host_authority(parsed) -> str:
    if not parsed.hostname:
        raise SSRFError("URL has no hostname")

    if ":" in parsed.hostname and not parsed.hostname.startswith("["):
        host_authority = f"[{parsed.hostname}]"
    else:
        host_authority = parsed.hostname

    default_port = _DEFAULT_PORTS[parsed.scheme]
    if parsed.port is not None and parsed.port != default_port:
        host_authority = f"{host_authority}:{parsed.port}"

    return host_authority


def merge_headers_for_pinned_request(
    extra_headers: dict[str, str] | None,
    pinned_headers: dict[str, str],
) -> dict[str, str]:
    if not extra_headers:
        return dict(pinned_headers)

    pinned_keys_lower = {key.lower() for key in pinned_headers}
    cleaned_extra = {
        key: value
        for key, value in extra_headers.items()
        if key.lower() not in pinned_keys_lower
    }
    return {**cleaned_extra, **pinned_headers}


def _path_and_query(parsed) -> str:
    path = parsed.path or "/"
    if parsed.query:
        return f"{path}?{parsed.query}"
    return path


def _headers_from_response(response: http.client.HTTPResponse) -> dict[str, str]:
    return {key.lower(): value for key, value in response.getheaders()}


def _content_length_from_headers(headers: dict[str, str]) -> int | None:
    try:
        content_length = int(headers.get("content-length", -1))
    except ValueError:
        return None

    return content_length if content_length >= 0 else None


def _raise_if_declared_size_exceeds_limit(
    headers: dict[str, str],
    max_size: int | None,
) -> None:
    if max_size is None:
        return

    content_length = _content_length_from_headers(headers)
    if content_length is not None and content_length > max_size:
        raise SSRFError(f"File body exceeded {max_size} bytes before download")


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(
        self,
        target_ip: str,
        port: int,
        *,
        server_hostname: str,
        timeout: float,
    ):
        ssl_context = ssl.create_default_context()
        super().__init__(
            target_ip,
            port=port,
            timeout=timeout,
            context=ssl_context,
        )
        self._server_hostname = server_hostname
        self._ssl_context = ssl_context

    def connect(self) -> None:
        source_address = getattr(self, "source_address", None)
        sock = socket.create_connection(
            (self.host, self.port), self.timeout, source_address
        )
        self.sock = self._ssl_context.wrap_socket(
            sock, server_hostname=self._server_hostname
        )


def _open_connection(
    parsed,
    target_ip: str | None,
    *,
    timeout: float,
) -> http.client.HTTPConnection:
    if not parsed.hostname:
        raise SSRFError("URL has no hostname")

    port = parsed.port if parsed.port is not None else _DEFAULT_PORTS[parsed.scheme]
    connect_host = target_ip or parsed.hostname

    if parsed.scheme == "https":
        if target_ip is None:
            return http.client.HTTPSConnection(parsed.hostname, port, timeout=timeout)
        return _PinnedHTTPSConnection(
            target_ip,
            port,
            server_hostname=parsed.hostname,
            timeout=timeout,
        )

    return http.client.HTTPConnection(connect_host, port, timeout=timeout)


def _strip_sensitive_headers_for_cross_origin(
    headers: dict[str, str],
    initial_origin: tuple[str, str | None, int],
    current_url,
) -> dict[str, str]:
    if _origin_tuple(current_url) == initial_origin:
        return headers

    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in _CROSS_ORIGIN_STRIPPED_HEADERS
    }


def _read_response_content(
    response: http.client.HTTPResponse,
    max_size: int | None,
    *,
    expected_size: int | None,
    chunk_size: int = 64 * 1024,
) -> bytes:
    chunks: list[bytes] = []
    bytes_read = 0

    try:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break

            bytes_read += len(chunk)
            if max_size is not None and bytes_read > max_size:
                raise SSRFError(f"File body exceeded {max_size} bytes during download")
            chunks.append(chunk)
    except http.client.IncompleteRead as exc:
        raise SSRFConnectionError(
            "Received less data than expected from the server."
        ) from exc

    if expected_size is not None and bytes_read < expected_size:
        raise SSRFConnectionError("Received less data than expected from the server.")

    return b"".join(chunks)


def _stream_response_to_file(
    response: http.client.HTTPResponse,
    target_path: str,
    max_size: int | None,
    *,
    expected_size: int | None,
    chunk_size: int,
) -> None:
    bytes_written = 0
    temp_file_path = ""
    target_dir = os.path.dirname(target_path) or "."
    target_basename = os.path.basename(target_path)

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            dir=target_dir,
            prefix=f".{target_basename}.tmp.",
        ) as file:
            try:
                temp_file_path = file.name
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break

                    file.write(chunk)
                    bytes_written += len(chunk)
                    if max_size is not None and bytes_written > max_size:
                        raise SSRFError(
                            f"File body exceeded {max_size} bytes during download"
                        )
            except http.client.IncompleteRead as exc:
                raise SSRFConnectionError(
                    "Received less data than expected from the server."
                ) from exc

        if expected_size is not None and bytes_written < expected_size:
            raise SSRFConnectionError(
                "Received less data than expected from the server."
            )

        os.replace(temp_file_path, target_path)
    finally:
        if temp_file_path:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temp_file_path)


def _request_one_hop(
    parsed,
    *,
    target_ip: str | None,
    timeout: float,
    headers: dict[str, str],
    max_size: int | None,
    body_mode: str,
    target_path: str | None = None,
    chunk_size: int = 64 * 1024,
    on_response_headers: Callable[[dict[str, str]], None] | None = None,
) -> SafeResponse:
    connection = _open_connection(parsed, target_ip, timeout=timeout)
    try:
        connection.request("GET", _path_and_query(parsed), headers=headers)
        response = connection.getresponse()
        response_headers = _headers_from_response(response)

        if response.status in _REDIRECT_STATUSES:
            return SafeResponse(response.status, response_headers)

        if response.status < 200 or response.status >= 300:
            raise SSRFHTTPStatusError(response.status)

        if body_mode == _BODY_HEADERS:
            return SafeResponse(response.status, response_headers)

        if on_response_headers is not None:
            on_response_headers(response_headers)

        _raise_if_declared_size_exceeds_limit(response_headers, max_size)
        expected_size = _content_length_from_headers(response_headers)

        if body_mode == _BODY_FILE:
            if target_path is None:
                raise SSRFError("File download target path is required")
            _stream_response_to_file(
                response,
                target_path,
                max_size,
                expected_size=expected_size,
                chunk_size=chunk_size,
            )
            return SafeResponse(response.status, response_headers)

        return SafeResponse(
            response.status,
            response_headers,
            _read_response_content(
                response,
                max_size,
                expected_size=expected_size,
                chunk_size=chunk_size,
            ),
        )
    finally:
        connection.close()


def _request_resolved_url(
    parsed,
    *,
    timeout: float,
    headers: dict[str, str],
    max_size: int | None,
    body_mode: str,
    target_path: str | None,
    chunk_size: int,
    on_response_headers: Callable[[dict[str, str]], None] | None,
) -> SafeResponse:
    hostname = parsed.hostname
    if not hostname:
        raise SSRFError("URL has no hostname")

    last_error: Exception | None = None
    for ip in resolve_and_validate_host(hostname, parsed.port):
        try:
            return _request_one_hop(
                parsed,
                target_ip=ip,
                timeout=timeout,
                headers=headers,
                max_size=max_size,
                body_mode=body_mode,
                target_path=target_path,
                chunk_size=chunk_size,
                on_response_headers=on_response_headers,
            )
        except (OSError, SSRFConnectionError, http.client.IncompleteRead) as exc:
            last_error = exc

    raise SSRFConnectionError("All validated addresses failed") from last_error


def _redirect_target(current_url: str, location: str) -> str:
    return urllib.parse.urljoin(current_url, location)


def _safe_request(
    url: str,
    *,
    timeout: float,
    max_size: int | None,
    max_hops: int,
    headers: dict[str, str] | None,
    allowed_schemes: frozenset[str],
    body_mode: str,
    target_path: str | None = None,
    chunk_size: int = 64 * 1024,
    on_response_headers: Callable[[dict[str, str]], None] | None = None,
) -> SafeResponse:
    _warn_if_proxy_configured()

    current_url = url
    initial_parsed = _parse_url(current_url)
    _validate_scheme(initial_parsed, allowed_schemes)
    initial_origin = _origin_tuple(initial_parsed)
    safe_headers = dict(headers) if headers else {}

    for _ in range(max_hops + 1):
        current_parsed = _parse_url(current_url)
        _validate_scheme(current_parsed, allowed_schemes)

        hop_headers = _strip_sensitive_headers_for_cross_origin(
            safe_headers, initial_origin, current_parsed
        )
        request_headers = merge_headers_for_pinned_request(
            hop_headers, {"Host": _build_host_authority(current_parsed)}
        )

        response = _request_resolved_url(
            current_parsed,
            timeout=timeout,
            headers=request_headers,
            max_size=max_size,
            body_mode=body_mode,
            target_path=target_path,
            chunk_size=chunk_size,
            on_response_headers=on_response_headers,
        )

        if response.status_code not in _REDIRECT_STATUSES:
            if response.status_code < 200 or response.status_code >= 300:
                raise SSRFHTTPStatusError(response.status_code)
            return response

        location = response.headers.get("location")
        if not location:
            raise SSRFError("Redirect response missing Location header")
        current_url = _redirect_target(current_url, location)

    raise SSRFError("Too many redirects")


def ssrf_safe_get(
    url: str,
    *,
    timeout: float = 30.0,
    max_size: int | None = None,
    max_hops: int = DEFAULT_MAX_REDIRECT_HOPS,
    headers: dict[str, str] | None = None,
    allowed_schemes: frozenset[str] = DEFAULT_ALLOWED_SCHEMES,
) -> SafeResponse:
    return _safe_request(
        url,
        timeout=timeout,
        max_size=max_size,
        max_hops=max_hops,
        headers=headers,
        allowed_schemes=allowed_schemes,
        body_mode=_BODY_CONTENT,
        on_response_headers=None,
    )


def ssrf_safe_get_headers(
    url: str,
    *,
    timeout: float = 30.0,
    max_hops: int = DEFAULT_MAX_REDIRECT_HOPS,
    headers: dict[str, str] | None = None,
    allowed_schemes: frozenset[str] = DEFAULT_ALLOWED_SCHEMES,
) -> SafeResponse:
    return _safe_request(
        url,
        timeout=timeout,
        max_size=None,
        max_hops=max_hops,
        headers=headers,
        allowed_schemes=allowed_schemes,
        body_mode=_BODY_HEADERS,
        on_response_headers=None,
    )


def ssrf_safe_get_to_file(
    url: str,
    target_path: os.PathLike[str] | str,
    *,
    timeout: float = 30.0,
    max_size: int | None = None,
    max_hops: int = DEFAULT_MAX_REDIRECT_HOPS,
    headers: dict[str, str] | None = None,
    allowed_schemes: frozenset[str] = DEFAULT_ALLOWED_SCHEMES,
    chunk_size: int = 64 * 1024,
) -> SafeResponse:
    return _ssrf_safe_get_to_file(
        url,
        target_path,
        timeout=timeout,
        max_size=max_size,
        max_hops=max_hops,
        headers=headers,
        allowed_schemes=allowed_schemes,
        chunk_size=chunk_size,
    )


def _ssrf_safe_get_to_file(
    url: str,
    target_path: os.PathLike[str] | str,
    *,
    timeout: float = 30.0,
    max_size: int | None = None,
    max_hops: int = DEFAULT_MAX_REDIRECT_HOPS,
    headers: dict[str, str] | None = None,
    allowed_schemes: frozenset[str] = DEFAULT_ALLOWED_SCHEMES,
    chunk_size: int = 64 * 1024,
    on_response_headers: Callable[[dict[str, str]], None] | None = None,
) -> SafeResponse:
    return _safe_request(
        url,
        timeout=timeout,
        max_size=max_size,
        max_hops=max_hops,
        headers=headers,
        allowed_schemes=allowed_schemes,
        body_mode=_BODY_FILE,
        target_path=os.fspath(target_path),
        chunk_size=chunk_size,
        on_response_headers=on_response_headers,
    )
