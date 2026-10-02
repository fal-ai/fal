from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from starlette.requests import Request

from fal.auth import AuthCredentials
from fal.exceptions.auth import UnauthenticatedException
from fal.toolkit.file import _local_uploader
from fal.toolkit.file import file as files
from fal.toolkit.file._local_uploader import LocalUploader, LocalUploadError
from fal.toolkit.file._upload_policy import UPLOAD_POLICY_KEY
from fal.toolkit.file.providers import fal as remote
from fal.toolkit.file.providers import local
from fal.toolkit.file.types import FileData
from fal.toolkit.image import Image

ACCEPTED = {
    "upload_id": "accepted-id",
    "file_url": "https://fal.media/file.txt",
    "state": "accepted_local",
}
STARTED = {**ACCEPTED, "upload_id": "session-id", "state": "receiving"}


@pytest.fixture
def transport(monkeypatch):
    requests = []

    def install(handler):
        def respond(request):
            requests.append(request)
            return handler(request)

        monkeypatch.setattr(
            _local_uploader,
            "_new_client",
            lambda: httpx.Client(
                transport=httpx.MockTransport(respond), base_url="http://localhost"
            ),
        )
        return requests

    return install


@pytest.fixture
def local_upload(monkeypatch, transport):
    monkeypatch.setenv("FAL_USE_LOCAL_UPLOADER", "1")
    monkeypatch.setattr(
        local, "fetch_auth_credentials", lambda: AuthCredentials("Key", "test:key")
    )
    return transport(lambda request: httpx.Response(202, json=ACCEPTED))


@pytest.mark.parametrize("repository", ["fal_v3", "fal_v2", "cdn"])
@pytest.mark.parametrize("file_type", [files.File, Image])
def test_existing_calls_select_local(local_upload, repository, file_type):
    kwargs = {"format": "png"} if file_type is Image else {}
    result = file_type.from_bytes(
        b"hello", file_name="hello.txt", repository=repository, **kwargs
    )
    assert result.url == ACCEPTED["file_url"]
    assert result.as_bytes() == b"hello"
    assert result.file_size == 5
    (request,) = local_upload
    assert request.url.path == "/uploads"
    assert request.content == b"hello"
    assert request.headers["content-length"] == "5"
    assert request.headers["x-fal-file-name"] == "hello.txt"
    assert request.headers["content-type"] == (
        "image/png" if file_type is Image else "text/plain"
    )
    assert request.headers["authorization"] == "Key test:key"


@pytest.mark.parametrize(
    "file_name,header",
    [
        ("plain (1) 100%.txt", "plain (1) 100%.txt"),
        ("vidéo.mp4", "vid%C3%A9o.mp4"),
        ("日本語.png", "%E6%97%A5%E6%9C%AC%E8%AA%9E.png"),
        ("new\nline.txt", "new%0Aline.txt"),
    ],
)
def test_file_names_travel_in_the_header(local_upload, file_name, header):
    files.File.from_bytes(b"hi", file_name=file_name)
    assert local_upload[0].headers["x-fal-file-name"] == header


@pytest.mark.parametrize("flag", [None, "0", "false", ""])
def test_disabled_uses_unchanged_repository(monkeypatch, local_upload, flag):
    if flag is None:
        monkeypatch.delenv("FAL_USE_LOCAL_UPLOADER")
    else:
        monkeypatch.setenv("FAL_USE_LOCAL_UPLOADER", flag)
    save = Mock(return_value="https://remote.example/file")
    monkeypatch.setattr(remote.FalFileRepositoryV3, "save", save)
    assert files.File.from_bytes(b"hi").url == "https://remote.example/file"
    save.assert_called_once()
    assert not local_upload


def test_explicit_repository_objects_and_destinations_are_preserved(
    monkeypatch, local_upload
):
    assert files.File.from_bytes(b"hi", repository="in_memory").url.startswith("data:")
    save = Mock(return_value="https://remote.example/file")
    monkeypatch.setattr(remote.FalFileRepositoryV3, "save", save)
    files.File.from_bytes(b"hi", repository=remote.FalFileRepositoryV3())
    save.assert_called_once()
    assert not local_upload


@pytest.mark.parametrize("from_path", [False, True])
def test_upload_policy_keeps_precedence(monkeypatch, local_upload, tmp_path, from_path):
    policy = json.dumps(
        {
            "url": "https://bucket.s3.amazonaws.com/",
            "fields": {"key": "outputs/${filename}"},
        }
    )
    request = Request(
        {"type": "http", "headers": [(UPLOAD_POLICY_KEY.encode(), policy.encode())]}
    )
    upload = Mock(return_value="https://bucket.s3.amazonaws.com/outputs/file.txt")
    monkeypatch.setattr(files, "upload_bytes_with_policy", upload)
    if from_path:
        path = tmp_path / "file.txt"
        path.write_bytes(b"hello")
        result = files.File.from_path(path, request=request)
    else:
        result = files.File.from_bytes(b"hello", file_name="file.txt", request=request)
    assert result.url == upload.return_value
    assert not local_upload


@pytest.mark.parametrize("multipart", [False, True, None])
@pytest.mark.parametrize("size", [0, 1024 * 1024 + 3])
def test_files_stream_and_preserve_file_data(local_upload, tmp_path, multipart, size):
    path = tmp_path / "file.bin"
    data = b"a" * size
    path.write_bytes(data)
    result = files.File.from_path(
        path, multipart=multipart, save_kwargs={"multipart_threshold": 1024}
    )
    assert result.file_size == size
    assert result.file_name == path.name
    assert result.file_data == (
        None if multipart or (multipart is None and size > 1024) else data
    )
    (request,) = local_upload
    assert request.content == data
    assert request.headers["content-length"] == str(size)
    path.unlink()  # caller may remove the source immediately after acceptance


@pytest.mark.parametrize(
    "settings",
    [
        None,
        {},
        {"expiration_duration_seconds": 3600},
        {
            "initial_acl": {
                "default": "forbid",
                "rules": [{"action": "allow", "user": "recipient"}],
            }
        },
    ],
)
@pytest.mark.parametrize("scheme", ["Key", "Bearer"])
def test_caller_metadata_and_effective_settings(
    monkeypatch, local_upload, settings, scheme
):
    request = SimpleNamespace(
        headers={"x-fal-cdn-token": "caller-token"}, request_id="request-id"
    )
    monkeypatch.setattr(
        remote, "get_current_app", lambda: SimpleNamespace(current_request=request)
    )
    monkeypatch.setattr(
        local, "fetch_auth_credentials", lambda: AuthCredentials(scheme, "credential")
    )
    local.LocalFileRepository().save(
        FileData(b"hi"), object_lifecycle_preference=settings
    )
    headers = local_upload[0].headers
    assert headers["authorization"] == f"{scheme} credential"
    assert headers["x-fal-cdn-token"] == "caller-token"
    assert headers["x-fal-request-id"] == "request-id"
    if settings:
        assert json.loads(headers["x-fal-object-lifecycle"]) == settings
        assert json.loads(headers["x-fal-object-lifecycle-preference"]) == settings
    else:
        assert "x-fal-object-lifecycle" not in headers
        assert "x-fal-object-lifecycle-preference" not in headers


def test_token_only_does_not_fabricate_settings(monkeypatch, local_upload):
    monkeypatch.setattr(
        local, "fetch_auth_credentials", Mock(side_effect=UnauthenticatedException())
    )
    request = SimpleNamespace(headers={"x-fal-cdn-token": "token"}, request_id=None)
    monkeypatch.setattr(
        remote, "get_current_app", lambda: SimpleNamespace(current_request=request)
    )
    local.LocalFileRepository().save(FileData(b"hi"))
    assert "authorization" not in local_upload[0].headers
    assert "x-fal-object-lifecycle" not in local_upload[0].headers


@pytest.mark.parametrize("status", [401, 413, 503, 307])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("from_path", [False, True])
def test_errors_do_not_retry_or_fall_back(
    monkeypatch, local_upload, transport, tmp_path, status, explicit, from_path
):
    requests = transport(
        lambda request: httpx.Response(
            status, headers={"location": "http://elsewhere"}, text="secret"
        )
    )
    method = "save_file" if from_path else "save"
    fallback = Mock()
    monkeypatch.setattr(remote.FalFileRepository, method, fallback)
    kwargs = {}
    if explicit:
        monkeypatch.delenv("FAL_USE_LOCAL_UPLOADER")
        kwargs["repository"] = local.LocalFileRepository()
    with pytest.raises(LocalUploadError, match=f"HTTP {status}") as caught:
        if from_path:
            path = tmp_path / "file.txt"
            path.write_bytes(b"hi")
            files.File.from_path(path, **kwargs)
        else:
            files.File.from_bytes(b"hi", **kwargs)
    assert "secret" not in str(caught.value)
    assert len(requests) == 1
    fallback.assert_not_called()


def test_local_in_fallback_list_does_not_chain(monkeypatch, local_upload, transport):
    requests = transport(lambda request: httpx.Response(503))
    monkeypatch.delenv("FAL_USE_LOCAL_UPLOADER")
    primary = Mock(side_effect=RuntimeError("Remote upload failed"))
    monkeypatch.setattr(remote.FalFileRepositoryV3, "save", primary)
    fallback = Mock()
    monkeypatch.setattr(remote.FalFileRepository, "save", fallback)
    with pytest.raises(LocalUploadError, match="HTTP 503"):
        files.File.from_bytes(
            b"hi",
            repository="fal_v3",
            fallback_repository=[local.LocalFileRepository(), "fal"],
        )
    primary.assert_called_once()
    assert len(requests) == 1
    fallback.assert_not_called()


def test_explicit_local_auth_failure_does_not_fall_back(monkeypatch, local_upload):
    monkeypatch.delenv("FAL_USE_LOCAL_UPLOADER")
    monkeypatch.setattr(remote, "get_current_app", lambda: None)
    monkeypatch.setattr(
        local, "fetch_auth_credentials", Mock(side_effect=UnauthenticatedException())
    )
    fallback = Mock()
    monkeypatch.setattr(remote.FalFileRepository, "save", fallback)
    with pytest.raises(local.FileUploadException, match="requires fal credentials"):
        files.File.from_bytes(b"hi", repository=local.LocalFileRepository())
    fallback.assert_not_called()
    assert not local_upload


@pytest.mark.parametrize(
    "failure,uncertain",
    [
        (httpx.ConnectError, False),
        (httpx.ReadError, True),
        (httpx.WriteError, True),
        (httpx.ReadTimeout, True),
    ],
)
def test_connection_failure_outcomes(transport, failure, uncertain):
    def fail(request):
        raise failure("secret")

    requests = transport(fail)
    with LocalUploader() as client, pytest.raises(LocalUploadError) as caught:
        client.upload("file", b"hi", 2, {})
    assert caught.value.acceptance_uncertain is uncertain
    assert failure.__name__ in str(caught.value)
    assert "secret" not in str(caught.value)
    assert len(requests) == 1


@pytest.mark.parametrize(
    "body",
    [
        b"broken json",
        b"[]",
        b"{}",
        json.dumps(STARTED).encode(),
    ],
)
def test_invalid_acceptance_is_not_success(transport, body):
    transport(lambda request: httpx.Response(202, content=body))
    with LocalUploader() as client, pytest.raises(LocalUploadError) as caught:
        client.upload("file", b"", 0, {})
    assert caught.value.acceptance_uncertain


def session_response(request):
    if request.method == "PUT":
        return httpx.Response(204)
    if request.method == "DELETE":
        return httpx.Response(202)
    if request.url.path.endswith("/finish"):
        return httpx.Response(202, json=ACCEPTED)
    return httpx.Response(201, json=STARTED)


def test_stream_requires_body_then_explicit_finish(transport):
    requests = transport(session_response)
    with LocalUploader() as client, client.begin_stream("file", {}) as session:
        with pytest.raises(ValueError, match="successfully received"):
            session.finish(5)
        session.send_body(iter([b"he", b"llo"]))
        assert len(requests) == 2  # EOF never finishes automatically
        accepted = session.finish(5)
        assert accepted.upload_id == "accepted-id"  # different from session ID
    assert [r.method for r in requests] == ["POST", "PUT", "POST"]
    assert requests[1].headers["transfer-encoding"] == "chunked"
    assert requests[1].content == b"hello"
    assert json.loads(requests[2].content) == {"size_bytes": 5}


def test_repository_stream_counts_bytes_and_carries_metadata(
    monkeypatch, local_upload, transport
):
    requests = transport(session_response)
    request = SimpleNamespace(headers={"x-fal-cdn-token": "token"}, request_id="rid")
    monkeypatch.setattr(
        remote, "get_current_app", lambda: SimpleNamespace(current_request=request)
    )
    url = local.LocalFileRepository().save_stream(
        iter([b"he", b"llo"]), "generated.txt", "text/plain"
    )
    assert url == ACCEPTED["file_url"]
    assert [r.method for r in requests] == ["POST", "PUT", "POST"]
    assert requests[0].headers["authorization"] == "Key test:key"
    assert requests[0].headers["x-fal-request-id"] == "rid"
    assert requests[0].headers["content-type"] == "text/plain"
    assert requests[1].content == b"hello"
    assert json.loads(requests[2].content) == {"size_bytes": 5}


def test_repository_stream_producer_failure_aborts(local_upload, transport):
    requests = transport(session_response)

    def generate():
        yield b"partial"
        raise RuntimeError("encoder died")

    with pytest.raises(RuntimeError, match="encoder died"):
        local.LocalFileRepository().save_stream(generate(), "clip.mp4", "video/mp4")
    assert [r.method for r in requests] == ["POST", "DELETE"]


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
def test_producer_failure_aborts_without_finish(transport, failure):
    requests = transport(session_response)

    def generate():
        yield b"partial"
        raise failure()

    with LocalUploader() as client, client.begin_stream("file", {}) as session:
        with pytest.raises(failure):
            session.send_body(generate())
    assert [r.method for r in requests] == ["POST", "DELETE"]


def test_session_exit_without_finish_aborts(transport):
    requests = transport(session_response)
    with LocalUploader() as client, client.begin_stream("file", {}) as session:
        session.send_body(b"hi")
    assert [r.method for r in requests] == ["POST", "PUT", "DELETE"]


def test_lost_finish_response_does_not_retry_or_mask_error(transport):
    def respond(request):
        if request.url.path.endswith("/finish"):
            raise httpx.ReadError("lost reply")
        if request.method == "DELETE":
            return httpx.Response(404)
        return session_response(request)

    requests = transport(respond)
    with LocalUploader() as client, pytest.raises(LocalUploadError) as caught:
        with client.begin_stream("file", {}) as session:
            session.send_body(b"hi")
            session.finish(2)
    assert caught.value.acceptance_uncertain
    assert [r.method for r in requests] == ["POST", "PUT", "POST", "DELETE"]


@pytest.mark.asyncio
async def test_async_wrapper_carries_request_context(monkeypatch, local_upload):
    request = SimpleNamespace(
        headers={"x-fal-cdn-token": "caller-token"}, request_id="async-id"
    )
    monkeypatch.setattr(
        remote, "get_current_app", lambda: SimpleNamespace(current_request=request)
    )
    result = await files.File.from_bytes_async(b"hi")
    assert result.url == ACCEPTED["file_url"]
    assert local_upload[0].headers["x-fal-request-id"] == "async-id"
