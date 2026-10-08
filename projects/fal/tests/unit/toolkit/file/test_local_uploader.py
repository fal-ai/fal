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
from fal.toolkit.file._local_uploader import LocalUploadError
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
SLEEP = "fal.toolkit.utils.retry.time.sleep"


@pytest.mark.parametrize("socket_path", [None, "/tmp/custom-uploader.sock"])
def test_client_socket_path(monkeypatch, socket_path):
    monkeypatch.delenv("CDN_UPLOADER_SOCKET_PATH", raising=False)
    if socket_path is not None:
        monkeypatch.setenv("CDN_UPLOADER_SOCKET_PATH", socket_path)
    transport = Mock(wraps=httpx.HTTPTransport)
    monkeypatch.setattr(httpx, "HTTPTransport", transport)

    with _local_uploader._new_client():
        pass

    assert transport.call_args.kwargs["uds"] == (
        socket_path if socket_path is not None else "/run/fal-upload/upload.sock"
    )


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
def caller(monkeypatch):
    def install(headers, request_id):
        request = SimpleNamespace(headers=headers, request_id=request_id)
        monkeypatch.setattr(
            remote, "get_current_app", lambda: SimpleNamespace(current_request=request)
        )

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
    monkeypatch, local_upload, caller, settings, scheme
):
    caller({"x-fal-cdn-token": "caller-token"}, "request-id")
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


def test_token_only_does_not_fabricate_settings(monkeypatch, local_upload, caller):
    monkeypatch.setattr(
        local, "fetch_auth_credentials", Mock(side_effect=UnauthenticatedException())
    )
    caller({"x-fal-cdn-token": "token"}, None)
    local.LocalFileRepository().save(FileData(b"hi"))
    assert "authorization" not in local_upload[0].headers
    assert "x-fal-object-lifecycle" not in local_upload[0].headers


def fail_with_secret(response):
    def respond(request):
        if response is None:
            raise httpx.ReadError("secret")
        return response

    return respond


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(401, text="secret"),
        httpx.Response(413, text="secret"),
        httpx.Response(503, text="secret"),
        httpx.Response(307, headers={"location": "http://secret"}),
        httpx.Response(202, text="secret"),
        None,  # lost response
    ],
)
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("from_path", [False, True])
def test_uploader_failures_fall_back_to_direct_cdn(
    monkeypatch,
    capsys,
    local_upload,
    transport,
    tmp_path,
    response,
    explicit,
    from_path,
):
    requests = transport(fail_with_secret(response))
    method = "save_file" if from_path else "save"
    direct = Mock(return_value="https://direct/file")
    if from_path:
        direct.return_value = (direct.return_value, FileData(b"hi"))
    monkeypatch.setattr(remote.FalFileRepositoryV3, method, direct)
    kwargs = {}
    if explicit:
        monkeypatch.delenv("FAL_USE_LOCAL_UPLOADER")
        kwargs["repository"] = local.LocalFileRepository()
    if from_path:
        path = tmp_path / "file.txt"
        path.write_bytes(b"hi")
        result = files.File.from_path(path, **kwargs)
    else:
        result = files.File.from_bytes(b"hi", **kwargs)
    assert result.url == "https://direct/file"
    assert len(requests) == 1
    direct.assert_called_once()
    out = capsys.readouterr().out
    assert "Uploading directly to CDN" in out
    assert "secret" not in out


def test_missing_credentials_are_not_sent(monkeypatch, local_upload):
    monkeypatch.setattr(remote, "get_current_app", lambda: None)
    monkeypatch.setattr(
        local, "fetch_auth_credentials", Mock(side_effect=UnauthenticatedException())
    )
    with pytest.raises(local.FileUploadException, match="requires fal credentials"):
        local.LocalFileRepository().save(FileData(b"hi"))
    assert not local_upload


@pytest.mark.parametrize("source", ["bytes", "small_file", "large_file"])
@pytest.mark.parametrize("exhausted", [False, True])
def test_definite_rejection_replays_then_uses_direct_cdn(
    monkeypatch, local_upload, transport, tmp_path, source, exhausted
):
    def respond(request):
        if exhausted or len(requests) < 3:
            return httpx.Response(429)
        return httpx.Response(202, json=ACCEPTED)

    requests = transport(respond)
    sleep = Mock()
    monkeypatch.setattr(SLEEP, sleep)
    body = b"x" * (1024 * 1024 + 1)
    path = tmp_path / "clip.mp4"
    path.write_bytes(body)
    kwargs = {
        "multipart": source == "large_file",
        "multipart_threshold": 1,
        "multipart_chunk_size": 12345,
        "multipart_max_concurrency": 2,
        "object_lifecycle_preference": {"expiration_duration_seconds": 60},
    }
    if source == "bytes":
        direct = Mock(return_value="https://direct/file")
        monkeypatch.setattr(remote.FalFileRepositoryV3, "save", direct)
        result = files.File.from_bytes(body, "video/mp4", save_kwargs=dict(kwargs))
    else:
        file_data = None if source == "large_file" else FileData(body)
        direct = Mock(return_value=("https://direct/file", file_data))
        monkeypatch.setattr(remote.FalFileRepositoryV3, "save_file", direct)
        result = files.File.from_path(path, "video/mp4", save_kwargs=dict(kwargs))
    assert result.url == ("https://direct/file" if exhausted else ACCEPTED["file_url"])
    assert result.file_data == (None if source == "large_file" else body)
    assert [request.content for request in requests] == [body] * 3
    assert [call.args[0] for call in sleep.call_args_list] == [0.1, 0.2]
    if not exhausted:
        direct.assert_not_called()
    elif source != "bytes":
        direct.assert_called_once_with(path, content_type="video/mp4", **kwargs)
    else:
        direct.assert_called_once()
        (data,) = direct.call_args.args
        assert data.data == body
        assert direct.call_args.kwargs == kwargs


@pytest.mark.parametrize(
    "failure",
    [httpx.ConnectError, httpx.ReadError, httpx.WriteError, httpx.ReadTimeout],
)
def test_connection_failures_name_the_class_only(transport, failure):
    def fail(request):
        raise failure("secret")

    requests = transport(fail)
    with pytest.raises(LocalUploadError) as caught:
        _local_uploader.upload("file", b"hi", 2, {})
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
    with pytest.raises(LocalUploadError):
        _local_uploader.upload("file", b"", 0, {})


def session_response(request):
    if request.method == "PUT":
        return httpx.Response(204)
    if request.method == "DELETE":
        return httpx.Response(202)
    if request.url.path.endswith("/finish"):
        return httpx.Response(202, json=ACCEPTED)
    return httpx.Response(201, json=STARTED)


def test_repository_stream_counts_bytes_and_carries_metadata(
    local_upload, transport, caller
):
    requests = transport(session_response)
    caller({"x-fal-cdn-token": "token"}, "rid")
    url = local.LocalFileRepository().save_stream(
        iter([b"he", b"llo"]), "generated.txt", "text/plain"
    )
    assert url == ACCEPTED["file_url"]
    assert [r.method for r in requests] == ["POST", "PUT", "POST"]
    assert requests[0].headers["authorization"] == "Key test:key"
    assert requests[0].headers["x-fal-request-id"] == "rid"
    assert requests[0].headers["content-type"] == "text/plain"
    assert requests[1].headers["transfer-encoding"] == "chunked"
    assert requests[1].content == b"hello"
    assert json.loads(requests[2].content) == {"size_bytes": 5}


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
def test_producer_failure_aborts_without_finish(local_upload, transport, failure):
    requests = transport(session_response)

    def generate():
        yield b"partial"
        raise failure()

    with pytest.raises(failure):
        local.LocalFileRepository().save_stream(generate(), "clip.mp4", "video/mp4")
    assert [r.method for r in requests] == ["POST", "DELETE"]


def test_lost_finish_response_aborts_and_raises(transport):
    def respond(request):
        if request.url.path.endswith("/finish"):
            raise httpx.ReadError("lost reply")
        if request.method == "DELETE":
            return httpx.Response(404)
        return session_response(request)

    requests = transport(respond)
    with pytest.raises(LocalUploadError, match="ReadError"):
        _local_uploader.upload_stream("file", iter([b"hi"]), {})
    assert [r.method for r in requests] == ["POST", "PUT", "POST", "DELETE"]


@pytest.mark.asyncio
async def test_async_wrapper_carries_request_context(local_upload, caller):
    caller({"x-fal-cdn-token": "caller-token"}, "async-id")
    result = await files.File.from_bytes_async(b"hi")
    assert result.url == ACCEPTED["file_url"]
    assert local_upload[0].headers["x-fal-request-id"] == "async-id"
