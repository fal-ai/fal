from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from fal.toolkit.file.providers import fal as providers
from fal.toolkit.file.types import FileData


@contextmanager
def _fake_retry_request(_request, **_kwargs):
    yield MagicMock()


def _assert_lifecycle_headers(
    headers_list: list[dict[str, str]], preference: dict[str, str]
) -> None:
    expected = json.dumps(preference)
    for headers in headers_list:
        if (
            headers.get("X-Fal-Object-Lifecycle") == expected
            and headers.get("X-Fal-Object-Lifecycle-Preference") == expected
        ):
            return
    raise AssertionError("Expected lifecycle headers not found in request headers")


def _assert_multipart_preference_call(mock, preference: dict[str, str]) -> None:
    _, kwargs = mock.call_args
    assert kwargs["object_lifecycle_preference"] == preference


@pytest.mark.parametrize(
    ("multipart_cls", "token_manager", "token", "json_payload"),
    [
        (
            providers.MultipartUploadGCS,
            None,
            None,
            {"file_url": "https://files", "upload_url": "https://upload"},
        ),
        (
            providers.MultipartUpload,
            providers.fal_v2_token_manager,
            providers.FalV2Token(
                token="token",
                token_type="Bearer",
                base_upload_url="https://upload",
                expires_at=datetime.now(timezone.utc),
            ),
            {"file_url": "https://files", "upload_url": "https://upload"},
        ),
        (
            providers.MultipartUploadV3,
            None,
            None,
            {"file_url": "https://files", "upload_url": "https://upload"},
        ),
        (
            providers.InternalMultipartUploadV3,
            providers.fal_v3_token_manager,
            providers.FalV3Token(
                token="token",
                token_type="Bearer",
                base_upload_url="https://upload",
                expires_at=datetime.now(timezone.utc),
            ),
            {"access_url": "https://files", "uploadId": "upload-id"},
        ),
    ],
)
def test_multipart_create_includes_lifecycle_headers(
    multipart_cls, token_manager, token, json_payload
):
    captured_headers: list[dict[str, str]] = []

    def _fake_request(_url, headers=None, method=None, data=None):
        nonlocal captured_headers
        if headers is not None:
            captured_headers.append(headers)
        return MagicMock()

    patches = [
        patch.object(providers, "Request", side_effect=_fake_request),
        patch.object(providers, "_maybe_retry_request", _fake_retry_request),
        patch.object(providers.json, "load", return_value=json_payload),
    ]
    if token_manager:
        patches.append(patch.object(token_manager, "get_token", return_value=token))
    else:
        patches.append(
            patch.object(
                providers, "key_credentials", return_value=("key_id", "key_secret")
            )
        )

    with patches[0], patches[1], patches[2], patches[3]:
        multipart = multipart_cls("file.txt")
        multipart.create(object_lifecycle_preference={"ttl": "1d"})

    _assert_lifecycle_headers(captured_headers, {"ttl": "1d"})


@pytest.mark.parametrize(
    ("repo_cls", "multipart_cls", "method_name"),
    [
        (providers.FalFileRepository, providers.MultipartUploadGCS, "save"),
        (providers.FalFileRepository, providers.MultipartUploadGCS, "save_file"),
        (providers.FalFileRepositoryV2, providers.MultipartUpload, "save"),
        (providers.FalFileRepositoryV2, providers.MultipartUpload, "save_file"),
        (providers.FalFileRepositoryV3, providers.MultipartUploadV3, "save"),
        (providers.FalFileRepositoryV3, providers.MultipartUploadV3, "save_file"),
        (
            providers.InternalFalFileRepositoryV3,
            providers.InternalMultipartUploadV3,
            "save",
        ),
        (
            providers.InternalFalFileRepositoryV3,
            providers.InternalMultipartUploadV3,
            "save_file",
        ),
    ],
)
def test_repository_multipart_passes_lifecycle_preference(
    repo_cls, multipart_cls, method_name
):
    preference = {"ttl": "1d"}
    repo = repo_cls()

    with patch.object(multipart_cls, method_name, return_value="url") as mock:
        if method_name == "save":
            file = FileData(b"data", content_type="text/plain", file_name="file.txt")
            repo.save(file, multipart=True, object_lifecycle_preference=preference)
        else:
            repo.save_file(
                "path/to/file.txt",
                content_type="text/plain",
                multipart=True,
                object_lifecycle_preference=preference,
            )

    _assert_multipart_preference_call(mock, preference)
