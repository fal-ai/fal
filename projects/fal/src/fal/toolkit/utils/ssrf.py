from __future__ import annotations

import http.client
import ipaddress
import os
import socket
import ssl
import urllib.parse
from dataclasses import dataclass
from typing import Any

DEFAULT_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})
DEFAULT_MAX_REDIRECT_HOPS = 5

_CROSS_ORIGIN_STRIPPED_HEADERS: frozenset[str] = frozenset(
    {"authorization", "cookie", "proxy-authorization"}
)
_DEFAULT_PORTS = {"http": 80, "https": 443}
_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")


class SSRFError(Exception):
    pass


class SSRFHTTPStatusError(SSRFError):
    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(f"HTTP request failed with status code {status_code}")


class SSRFSizeExceededError(SSRFError):
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


def is_globally_routable_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False

    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped

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
    chunk_size: int = 64 * 1024,
) -> bytes:
    chunks: list[bytes] = []
    bytes_read = 0

    while True:
        chunk = response.read(chunk_size)
        if not chunk:
            break

        bytes_read += len(chunk)
        if max_size is not None and bytes_read > max_size:
            raise SSRFSizeExceededError(
                f"File body exceeded {max_size} bytes during download"
            )
        chunks.append(chunk)

    return b"".join(chunks)


def _request_one_hop(
    parsed,
    *,
    target_ip: str | None,
    timeout: float,
    headers: dict[str, str],
    max_size: int | None,
) -> SafeResponse:
    connection = _open_connection(parsed, target_ip, timeout=timeout)
    try:
        connection.request("GET", _path_and_query(parsed), headers=headers)
        response = connection.getresponse()
        response_headers = _headers_from_response(response)
        content = _read_response_content(response, max_size)
        return SafeResponse(response.status, response_headers, content)
    finally:
        connection.close()


def _request_one_hop_to_file(
    parsed,
    target_path: str,
    *,
    target_ip: str | None,
    timeout: float,
    headers: dict[str, str],
    max_size: int | None,
    chunk_size: int,
) -> SafeResponse:
    connection = _open_connection(parsed, target_ip, timeout=timeout)
    try:
        connection.request("GET", _path_and_query(parsed), headers=headers)
        response = connection.getresponse()
        response_headers = _headers_from_response(response)

        if response.status in (301, 302, 303, 307, 308):
            return SafeResponse(response.status, response_headers)

        if response.status >= 400:
            raise SSRFHTTPStatusError(response.status)

        bytes_written = 0
        with open(target_path, "wb") as file:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break

                file.write(chunk)
                bytes_written += len(chunk)
                if max_size is not None and bytes_written > max_size:
                    raise SSRFSizeExceededError(
                        f"File body exceeded {max_size} bytes during download"
                    )

        return SafeResponse(response.status, response_headers)
    finally:
        connection.close()


def _redirect_target(current_url: str, location: str) -> str:
    return urllib.parse.urljoin(current_url, location)


def _url_to_string(parsed) -> str:
    return urllib.parse.urlunsplit(parsed)


def ssrf_safe_get(
    url: str,
    *,
    timeout: float = 30.0,
    max_size: int | None = None,
    max_hops: int = DEFAULT_MAX_REDIRECT_HOPS,
    headers: dict[str, str] | None = None,
    allow_internal_hosts: bool = False,
    allowed_schemes: frozenset[str] = DEFAULT_ALLOWED_SCHEMES,
) -> SafeResponse:
    current_url = url
    current_parsed = _parse_url(current_url)
    initial_origin = _origin_tuple(current_parsed)
    safe_headers = dict(headers) if headers else {}

    for _ in range(max_hops + 1):
        current_parsed = _parse_url(current_url)
        _validate_scheme(current_parsed, allowed_schemes)

        hostname = current_parsed.hostname
        if not hostname:
            raise SSRFError("URL has no hostname")

        safe_headers = _strip_sensitive_headers_for_cross_origin(
            safe_headers, initial_origin, current_parsed
        )
        request_headers = merge_headers_for_pinned_request(
            safe_headers, {"Host": _build_host_authority(current_parsed)}
        )

        if allow_internal_hosts:
            response = _request_one_hop(
                current_parsed,
                target_ip=None,
                timeout=timeout,
                headers=request_headers,
                max_size=max_size,
            )
        else:
            validated_ips = resolve_and_validate_host(hostname, current_parsed.port)
            response = _request_one_hop(
                current_parsed,
                target_ip=validated_ips[0],
                timeout=timeout,
                headers=request_headers,
                max_size=max_size,
            )

        if response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get("location")
            if not location:
                raise SSRFError("Redirect response missing Location header")
            current_url = _redirect_target(current_url, location)
            continue

        if response.status_code >= 400:
            raise SSRFHTTPStatusError(response.status_code)

        return response

    raise SSRFError("Too many redirects")


def ssrf_safe_get_to_file(
    url: str,
    target_path: os.PathLike[str] | str,
    *,
    timeout: float = 30.0,
    max_size: int | None = None,
    max_hops: int = DEFAULT_MAX_REDIRECT_HOPS,
    headers: dict[str, str] | None = None,
    allow_internal_hosts: bool = False,
    allowed_schemes: frozenset[str] = DEFAULT_ALLOWED_SCHEMES,
    chunk_size: int = 64 * 1024,
) -> SafeResponse:
    current_url = url
    current_parsed = _parse_url(current_url)
    initial_origin = _origin_tuple(current_parsed)
    safe_headers = dict(headers) if headers else {}
    target_path_str = os.fspath(target_path)

    for _ in range(max_hops + 1):
        current_parsed = _parse_url(current_url)
        _validate_scheme(current_parsed, allowed_schemes)

        hostname = current_parsed.hostname
        if not hostname:
            raise SSRFError("URL has no hostname")

        safe_headers = _strip_sensitive_headers_for_cross_origin(
            safe_headers, initial_origin, current_parsed
        )
        request_headers = merge_headers_for_pinned_request(
            safe_headers, {"Host": _build_host_authority(current_parsed)}
        )

        if allow_internal_hosts:
            response = _request_one_hop_to_file(
                current_parsed,
                target_path_str,
                target_ip=None,
                timeout=timeout,
                headers=request_headers,
                max_size=max_size,
                chunk_size=chunk_size,
            )
        else:
            validated_ips = resolve_and_validate_host(hostname, current_parsed.port)
            response = _request_one_hop_to_file(
                current_parsed,
                target_path_str,
                target_ip=validated_ips[0],
                timeout=timeout,
                headers=request_headers,
                max_size=max_size,
                chunk_size=chunk_size,
            )

        if response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get("location")
            if not location:
                raise SSRFError("Redirect response missing Location header")
            current_url = _redirect_target(current_url, location)
            continue

        return response

    raise SSRFError("Too many redirects")
