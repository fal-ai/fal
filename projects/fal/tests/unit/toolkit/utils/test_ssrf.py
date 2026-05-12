from __future__ import annotations

import socket
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import HTTPException

import fal.toolkit.image as image_toolkit
import fal.toolkit.utils.download_utils as download_utils
from fal.toolkit.image import read_image_from_url
from fal.toolkit.utils import ssrf
from fal.toolkit.utils.download_utils import (
    DownloadError,
    _download_file_python,
    download_file,
)

_request_calls: list[dict[str, Any]] = []
_request_responses: list[ssrf.SafeResponse] = []


def _addrinfo(*ips: str, family: int = socket.AF_INET) -> list[Any]:
    return [(family, socket.SOCK_STREAM, 6, "", (ip, 443)) for ip in ips]


@pytest.fixture(autouse=True)
def reset_client() -> None:
    read_image_from_url.cache_clear()
    _request_calls.clear()
    _request_responses.clear()


def _fake_request_one_hop(
    parsed,
    *,
    target_ip: str | None,
    timeout: float,
    headers: dict[str, str],
    max_size: int | None,
    body_mode: str,
    target_path: str | None = None,
    chunk_size: int = 64 * 1024,
) -> ssrf.SafeResponse:
    _request_calls.append(
        {
            "parsed": parsed,
            "target_ip": target_ip,
            "timeout": timeout,
            "headers": headers,
            "max_size": max_size,
            "body_mode": body_mode,
            "target_path": target_path,
            "chunk_size": chunk_size,
        }
    )
    response = _request_responses.pop(0)
    if (
        body_mode == ssrf._BODY_FILE
        and target_path is not None
        and response.status_code not in {301, 302, 303, 307, 308}
        and response.status_code < 400
    ):
        with open(target_path, "wb") as file:
            file.write(b"hello")
    return response


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.0.1",
        "169.254.169.254",
        "0.0.0.0",
        "224.0.0.1",
        "100.64.0.1",
        "::1",
        "fe80::1",
        "::ffff:169.254.169.254",
        "2002:7f00:0001::",
        "2002:0a00:0001::",
        "2001:0000:4136:e378:8000:63bf:f5ff:fffe",
        "2001:0000:0a00:0001:8000:63bf:c000:02d2",
        "64:ff9b::10.0.0.1",
        "64:ff9b::100.64.0.1",
        "64:ff9b::169.254.169.254",
    ],
)
def test_non_routable_ips_are_rejected(ip: str) -> None:
    assert not ssrf.is_globally_routable_ip(ip)


def test_public_nat64_addresses_are_allowed() -> None:
    assert ssrf.is_globally_routable_ip("64:ff9b::8.8.8.8")


def test_download_file_uses_validated_pinned_ip(tmp_path) -> None:
    _request_responses.append(
        ssrf.SafeResponse(
            200,
            headers={
                "content-disposition": 'attachment; filename="safe.txt"',
                "content-length": "5",
            },
        )
    )

    with patch.object(ssrf, "_socket_getaddrinfo", return_value=_addrinfo("8.8.8.8")):
        with patch.object(ssrf, "_request_one_hop", _fake_request_one_hop):
            path = download_file("https://example.com/file", tmp_path)

    assert path == tmp_path / "safe.txt"
    assert path.read_bytes() == b"hello"

    assert len(_request_calls) == 1
    assert _request_calls[0]["body_mode"] == ssrf._BODY_FILE
    assert _request_calls[0]["target_ip"] == "8.8.8.8"
    assert _request_calls[0]["headers"]["Host"] == "example.com"


def test_download_file_keeps_matching_cached_file_after_single_request(
    tmp_path,
) -> None:
    cached_path = tmp_path / "safe.txt"
    cached_path.write_bytes(b"hello")
    _request_responses.append(
        ssrf.SafeResponse(
            200,
            headers={
                "content-disposition": 'attachment; filename="safe.txt"',
                "content-length": "5",
            },
        )
    )

    with patch.object(ssrf, "_socket_getaddrinfo", return_value=_addrinfo("8.8.8.8")):
        with patch.object(ssrf, "_request_one_hop", _fake_request_one_hop):
            path = download_file("https://example.com/file", tmp_path)

    assert path == cached_path
    assert len(_request_calls) == 1
    assert _request_calls[0]["body_mode"] == ssrf._BODY_FILE


def test_ssrf_safe_get_headers_tries_later_validated_ips() -> None:
    _request_responses.append(ssrf.SafeResponse(200, headers={"content-length": "0"}))

    def fake_request(
        parsed,
        *,
        target_ip: str | None,
        timeout: float,
        headers: dict[str, str],
        max_size: int | None,
        body_mode: str,
        target_path: str | None = None,
        chunk_size: int = 64 * 1024,
    ) -> ssrf.SafeResponse:
        if target_ip == "2001:4860:4860::8888":
            raise OSError("network unreachable")
        return _fake_request_one_hop(
            parsed,
            target_ip=target_ip,
            timeout=timeout,
            headers=headers,
            max_size=max_size,
            body_mode=body_mode,
            target_path=target_path,
            chunk_size=chunk_size,
        )

    with patch.object(
        ssrf,
        "_socket_getaddrinfo",
        return_value=[
            *_addrinfo("2001:4860:4860::8888", family=socket.AF_INET6),
            *_addrinfo("8.8.8.8"),
        ],
    ):
        with patch.object(ssrf, "_request_one_hop", fake_request):
            response = ssrf.ssrf_safe_get_headers("https://example.com/file")

    assert response.status_code == 200
    assert _request_calls[0]["target_ip"] == "8.8.8.8"


def test_ssrf_safe_get_to_file_retries_later_ips_after_short_body(tmp_path) -> None:
    target_path = tmp_path / "safe.txt"
    _request_responses.append(ssrf.SafeResponse(200, headers={"content-length": "5"}))

    def fake_request(
        parsed,
        *,
        target_ip: str | None,
        timeout: float,
        headers: dict[str, str],
        max_size: int | None,
        body_mode: str,
        target_path: str | None = None,
        chunk_size: int = 64 * 1024,
    ) -> ssrf.SafeResponse:
        if target_ip == "8.8.8.8":
            raise ssrf.SSRFConnectionError("short body")
        return _fake_request_one_hop(
            parsed,
            target_ip=target_ip,
            timeout=timeout,
            headers=headers,
            max_size=max_size,
            body_mode=body_mode,
            target_path=target_path,
            chunk_size=chunk_size,
        )

    with patch.object(
        ssrf,
        "_socket_getaddrinfo",
        return_value=_addrinfo("8.8.8.8", "1.1.1.1"),
    ):
        with patch.object(ssrf, "_request_one_hop", fake_request):
            response = ssrf.ssrf_safe_get_to_file(
                "https://example.com/file",
                target_path,
            )

    assert response.status_code == 200
    assert _request_calls[0]["target_ip"] == "1.1.1.1"
    assert target_path.read_bytes() == b"hello"


def test_ssrf_safe_get_headers_warns_when_proxy_env_is_configured(monkeypatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    _request_responses.append(ssrf.SafeResponse(200, headers={"content-length": "0"}))

    with patch.object(ssrf, "_socket_getaddrinfo", return_value=_addrinfo("8.8.8.8")):
        with patch.object(ssrf, "_request_one_hop", _fake_request_one_hop):
            with pytest.warns(RuntimeWarning, match="not honored"):
                response = ssrf.ssrf_safe_get_headers("https://example.com/file")

    assert response.status_code == 200


def test_ssrf_safe_get_headers_blocks_disallowed_scheme() -> None:
    with pytest.raises(ssrf.SSRFError, match="scheme"):
        ssrf.ssrf_safe_get_headers("file:///etc/passwd")


def test_ssrf_safe_get_headers_rejects_non_success_status() -> None:
    _request_responses.append(ssrf.SafeResponse(304, headers={}))

    with patch.object(ssrf, "_socket_getaddrinfo", return_value=_addrinfo("8.8.8.8")):
        with patch.object(ssrf, "_request_one_hop", _fake_request_one_hop):
            with pytest.raises(ssrf.SSRFHTTPStatusError) as exc_info:
                ssrf.ssrf_safe_get_headers("https://example.com/file")

    assert exc_info.value.status_code == 304


def test_ssrf_safe_get_headers_limits_redirect_hops() -> None:
    _request_responses.extend(
        [
            ssrf.SafeResponse(302, headers={"location": "/next"}),
            ssrf.SafeResponse(302, headers={"location": "/again"}),
        ]
    )

    with patch.object(ssrf, "_socket_getaddrinfo", return_value=_addrinfo("8.8.8.8")):
        with patch.object(ssrf, "_request_one_hop", _fake_request_one_hop):
            with pytest.raises(ssrf.SSRFError, match="Too many redirects"):
                ssrf.ssrf_safe_get_headers("https://example.com/file", max_hops=1)

    assert len(_request_calls) == 2


def test_ssrf_safe_get_headers_strips_sensitive_cross_origin_headers() -> None:
    _request_responses.extend(
        [
            ssrf.SafeResponse(302, headers={"location": "https://other.example/file"}),
            ssrf.SafeResponse(200, headers={"content-length": "0"}),
        ]
    )
    resolutions = {
        "first.example": _addrinfo("8.8.8.8"),
        "other.example": _addrinfo("8.8.4.4"),
    }

    def fake_getaddrinfo(host: str, *_args: Any, **_kwargs: Any) -> list[Any]:
        return resolutions[host]

    with patch.object(ssrf, "_socket_getaddrinfo", side_effect=fake_getaddrinfo):
        with patch.object(ssrf, "_request_one_hop", _fake_request_one_hop):
            response = ssrf.ssrf_safe_get_headers(
                "https://first.example/file",
                headers={
                    "Authorization": "token",
                    "Cookie": "session=value",
                    "X-Trace": "keep",
                },
            )

    assert response.status_code == 200
    assert _request_calls[0]["headers"]["Authorization"] == "token"
    assert _request_calls[0]["headers"]["Cookie"] == "session=value"
    assert "Authorization" not in _request_calls[1]["headers"]
    assert "Cookie" not in _request_calls[1]["headers"]
    assert _request_calls[1]["headers"]["X-Trace"] == "keep"
    assert _request_calls[1]["headers"]["Host"] == "other.example"


def test_ssrf_safe_get_headers_restores_sensitive_headers_after_return_redirect() -> (
    None
):
    _request_responses.extend(
        [
            ssrf.SafeResponse(302, headers={"location": "https://other.example/file"}),
            ssrf.SafeResponse(302, headers={"location": "https://first.example/final"}),
            ssrf.SafeResponse(200, headers={"content-length": "0"}),
        ]
    )
    resolutions = {
        "first.example": _addrinfo("8.8.8.8"),
        "other.example": _addrinfo("8.8.4.4"),
    }

    def fake_getaddrinfo(host: str, *_args: Any, **_kwargs: Any) -> list[Any]:
        return resolutions[host]

    with patch.object(ssrf, "_socket_getaddrinfo", side_effect=fake_getaddrinfo):
        with patch.object(ssrf, "_request_one_hop", _fake_request_one_hop):
            response = ssrf.ssrf_safe_get_headers(
                "https://first.example/file",
                headers={
                    "Authorization": "token",
                    "Cookie": "session=value",
                    "X-Trace": "keep",
                },
            )

    assert response.status_code == 200
    assert _request_calls[0]["headers"]["Authorization"] == "token"
    assert _request_calls[1]["headers"].get("Authorization") is None
    assert _request_calls[2]["headers"]["Authorization"] == "token"
    assert _request_calls[2]["headers"]["Cookie"] == "session=value"


def test_download_file_python_preserves_existing_file_on_http_failure(tmp_path) -> None:
    target_path = tmp_path / "safe.txt"
    target_path.write_bytes(b"existing")

    with patch.object(
        download_utils,
        "ssrf_safe_get_to_file",
        side_effect=ssrf.SSRFSizeExceededError("too large"),
    ):
        with pytest.raises(DownloadError, match="too large"):
            _download_file_python("https://example.com/file", target_path)

    assert target_path.read_bytes() == b"existing"


def test_ssrf_safe_get_to_file_preserves_existing_file_on_short_http_body(
    tmp_path,
) -> None:
    target_path = tmp_path / "safe.txt"
    target_path.write_bytes(b"existing")

    class FakeResponse:
        status = 200

        def __init__(self) -> None:
            self._chunks = [b"short", b""]

        def getheaders(self) -> list[tuple[str, str]]:
            return [("content-length", "10")]

        def read(self, _chunk_size: int) -> bytes:
            return self._chunks.pop(0)

    class FakeConnection:
        def request(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def getresponse(self) -> FakeResponse:
            return FakeResponse()

        def close(self) -> None:
            pass

    with patch.object(ssrf, "_socket_getaddrinfo", return_value=_addrinfo("8.8.8.8")):
        with patch.object(ssrf, "_open_connection", return_value=FakeConnection()):
            with pytest.raises(ssrf.SSRFConnectionError):
                ssrf.ssrf_safe_get_to_file("https://example.com/file", target_path)

    assert target_path.read_bytes() == b"existing"


def test_ssrf_safe_get_to_file_preserves_existing_file_on_http_failure(
    tmp_path,
) -> None:
    target_path = tmp_path / "safe.txt"
    target_path.write_bytes(b"existing")

    class FakeResponse:
        status = 200

        def __init__(self) -> None:
            self._chunks = [b"abc", b"def", b""]

        def getheaders(self) -> list[tuple[str, str]]:
            return [("content-length", "6")]

        def read(self, _chunk_size: int) -> bytes:
            return self._chunks.pop(0)

    class FakeConnection:
        def request(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def getresponse(self) -> FakeResponse:
            return FakeResponse()

        def close(self) -> None:
            pass

    with patch.object(ssrf, "_socket_getaddrinfo", return_value=_addrinfo("8.8.8.8")):
        with patch.object(ssrf, "_open_connection", return_value=FakeConnection()):
            with pytest.raises(ssrf.SSRFSizeExceededError):
                ssrf.ssrf_safe_get_to_file(
                    "https://example.com/file",
                    target_path,
                    max_size=4,
                )

    assert target_path.read_bytes() == b"existing"


def test_read_image_from_url_applies_download_limit() -> None:
    def fake_get(url: str, **kwargs: Any) -> ssrf.SafeResponse:
        assert kwargs["max_size"] == image_toolkit.MAX_IMAGE_DOWNLOAD_SIZE
        raise ssrf.SSRFSizeExceededError("too large")

    with patch.object(image_toolkit, "ssrf_safe_get", fake_get):
        with pytest.raises(HTTPException) as exc_info:
            read_image_from_url("https://attacker.example/image.png")

    assert exc_info.value.status_code == 413


def test_read_image_from_url_preserves_data_uri() -> None:
    image = read_image_from_url(
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
        "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        convert_to_rgb=False,
    )

    assert image.size == (1, 1)


def test_download_file_redownloads_truncated_data_uri_cache(tmp_path) -> None:
    url = "data:text/plain;base64,aGVsbG8="
    target_path = tmp_path / download_utils._hash_url(url)
    target_path.write_bytes(b"bad")

    path = download_file(url, tmp_path)

    assert path == target_path
    assert path.read_bytes() == b"hello"


def test_filename_from_response_parses_rfc5987_filename() -> None:
    response = ssrf.SafeResponse(
        200,
        headers={"content-disposition": "attachment; filename*=UTF-8''safe%20name.txt"},
    )

    assert (
        download_utils._filename_from_response("https://example.com/file", response)
        == "safe name.txt"
    )


def test_download_file_blocks_redirect_to_private_ip(tmp_path) -> None:
    _request_responses.append(
        ssrf.SafeResponse(302, headers={"location": "http://internal.example/secrets"})
    )
    resolutions = {
        "evil.example": _addrinfo("8.8.8.8"),
        "internal.example": _addrinfo("169.254.169.254"),
    }

    def fake_getaddrinfo(host: str, *_args: Any, **_kwargs: Any) -> list[Any]:
        return resolutions[host]

    with patch.object(ssrf, "_socket_getaddrinfo", side_effect=fake_getaddrinfo):
        with patch.object(ssrf, "_request_one_hop", _fake_request_one_hop):
            with pytest.raises(DownloadError, match="non-routable"):
                download_file("https://evil.example/file", tmp_path)

    assert len(_request_calls) == 1
    assert _request_calls[0]["body_mode"] == ssrf._BODY_FILE


def test_read_image_from_url_blocks_private_resolution() -> None:
    with patch.object(
        ssrf, "_socket_getaddrinfo", return_value=_addrinfo("169.254.169.254")
    ):
        with pytest.raises(HTTPException):
            read_image_from_url("https://attacker.example/private.png")
