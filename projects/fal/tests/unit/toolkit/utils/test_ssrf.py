from __future__ import annotations

import socket
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from fal.toolkit.image import read_image_from_url
from fal.toolkit.utils import ssrf
from fal.toolkit.utils.download_utils import DownloadError, download_file


_stream_calls: list[dict[str, Any]] = []
_stream_responses: list[ssrf.SafeResponse] = []


def _addrinfo(*ips: str, family: int = socket.AF_INET) -> list[Any]:
    return [(family, socket.SOCK_STREAM, 6, "", (ip, 443)) for ip in ips]


@pytest.fixture(autouse=True)
def reset_client() -> None:
    _stream_calls.clear()
    _stream_responses.clear()


def _fake_stream_one_hop_to_file(
    parsed,
    target_path: str,
    *,
    target_ip: str | None,
    timeout: float,
    headers: dict[str, str],
    max_size: int | None,
    chunk_size: int,
) -> ssrf.SafeResponse:
    _stream_calls.append(
        {
            "parsed": parsed,
            "target_ip": target_ip,
            "timeout": timeout,
            "headers": headers,
            "max_size": max_size,
            "chunk_size": chunk_size,
        }
    )
    response = _stream_responses.pop(0)
    if response.status_code < 300 or response.status_code >= 400:
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
    ],
)
def test_non_routable_ips_are_rejected(ip: str) -> None:
    assert not ssrf.is_globally_routable_ip(ip)


def test_download_file_uses_validated_pinned_ip(tmp_path) -> None:
    _stream_responses.append(
        ssrf.SafeResponse(
            200,
            headers={"content-disposition": 'attachment; filename="safe.txt"'},
        )
    )

    with patch.object(ssrf, "_socket_getaddrinfo", return_value=_addrinfo("8.8.8.8")):
        with patch.object(
            ssrf, "_request_one_hop_to_file", _fake_stream_one_hop_to_file
        ):
            path = download_file("https://example.com/file", tmp_path)

    assert path == tmp_path / "safe.txt"
    assert path.read_bytes() == b"hello"

    assert _stream_calls[0]["target_ip"] == "8.8.8.8"
    assert _stream_calls[0]["headers"]["Host"] == "example.com"


def test_download_file_blocks_redirect_to_private_ip(tmp_path) -> None:
    _stream_responses.append(
        ssrf.SafeResponse(302, headers={"location": "http://internal.example/secrets"})
    )
    resolutions = {
        "evil.example": _addrinfo("8.8.8.8"),
        "internal.example": _addrinfo("169.254.169.254"),
    }

    def fake_getaddrinfo(host: str, *_args: Any, **_kwargs: Any) -> list[Any]:
        return resolutions[host]

    with patch.object(ssrf, "_socket_getaddrinfo", side_effect=fake_getaddrinfo):
        with patch.object(
            ssrf, "_request_one_hop_to_file", _fake_stream_one_hop_to_file
        ):
            with pytest.raises(DownloadError, match="non-routable"):
                download_file("https://evil.example/file", tmp_path)

    assert len(_stream_calls) == 1


def test_read_image_from_url_blocks_private_resolution() -> None:
    with patch.object(
        ssrf, "_socket_getaddrinfo", return_value=_addrinfo("169.254.169.254")
    ):
        with pytest.raises(HTTPException):
            read_image_from_url("https://attacker.example/image.png")
