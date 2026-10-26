from __future__ import annotations

import io
import json
import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.parse import quote, unquote
from uuid import UUID

import httpx
import pytest
from starlette.requests import Request

from fal.auth import AuthCredentials
from fal.toolkit.exceptions import FileUploadException
from fal.toolkit.file import _local_uploader
from fal.toolkit.file import _upload_policy as upload_policy
from fal.toolkit.file import file as files
from fal.toolkit.file._local_uploader import LocalUploadError, LocalUploadRefused
from fal.toolkit.file._upload_policy import UPLOAD_POLICY_KEY
from fal.toolkit.file.providers import fal as remote
from fal.toolkit.file.providers import local
from fal.toolkit.file.types import FileData

ACCEPTED = {
    "upload_id": "accepted-id",
    "file_url": "https://fal.media/file.txt",
    "state": "accepted_local",
}


@pytest.mark.parametrize("socket_path", [None, "/tmp/custom-api.sock"])
def test_client_socket_path(monkeypatch, socket_path):
    monkeypatch.delenv("FAL_API_SOCKET", raising=False)
    # Only FAL_API_SOCKET selects the socket; the old variable is ignored.
    monkeypatch.setenv("CDN_UPLOADER_SOCKET_PATH", "/run/fal-upload/upload.sock")
    if socket_path is not None:
        monkeypatch.setenv("FAL_API_SOCKET", socket_path)
    transport = Mock(wraps=httpx.HTTPTransport)
    monkeypatch.setattr(httpx, "HTTPTransport", transport)

    with _local_uploader._new_client():
        pass

    assert transport.call_args.kwargs["uds"] == (
        socket_path if socket_path is not None else "/run/fal/api.sock"
    )


@pytest.mark.parametrize("app_name", [None, "github_110602490-test"])
def test_client_names_the_app(monkeypatch, app_name):
    monkeypatch.delenv("FAL_APP_NAME", raising=False)
    if app_name is not None:
        monkeypatch.setenv("FAL_APP_NAME", app_name)

    with _local_uploader._new_client() as http:
        assert http.headers.get("x-fal-application") == app_name


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
        app = SimpleNamespace(current_request=request)
        monkeypatch.setattr(remote, "get_current_app", lambda: app)
        monkeypatch.setattr(_local_uploader, "get_current_app", lambda: app)

    return install


@pytest.fixture
def local_upload(monkeypatch, transport):
    monkeypatch.setenv("FAL_USE_LOCAL_UPLOADER", "1")
    monkeypatch.setattr(
        remote, "fetch_auth_credentials", lambda: AuthCredentials("Key", "test:key")
    )
    return transport(lambda request: httpx.Response(202, json=ACCEPTED))


@pytest.mark.parametrize("repository", ["fal_v3", "fal_v2", "cdn"])
def test_existing_calls_select_local(local_upload, repository):
    result = files.File.from_bytes(
        b"hello", file_name="hello.txt", repository=repository
    )
    assert result.url == ACCEPTED["file_url"]
    assert result.as_bytes() == b"hello"
    assert result.file_size == 5
    (request,) = local_upload
    assert request.url.path == "/v1/uploads"
    assert request.content == b"hello"
    assert request.headers["content-length"] == "5"
    assert request.headers["x-fal-file-name"] == "hello.txt"
    assert request.headers["content-type"] == "text/plain"
    assert request.headers["authorization"] == "Key test:key"


@pytest.mark.parametrize(
    "file_name,header",
    [
        ("plain (1) 100%.txt", "plain (1) 100%.txt"),
        ("vidéo.mp4", "vid?o.mp4"),
        (" new\nline.txt ", "new?line.txt"),
    ],
)
def test_file_names_are_sanitized_like_the_rest_api(local_upload, file_name, header):
    files.File.from_bytes(b"hi", file_name=file_name)
    assert local_upload[0].headers["x-fal-file-name"] == header


@pytest.mark.parametrize("flag", [None, "0"])
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


POLICY = {
    "url": "https://bucket.s3.amazonaws.com/",
    "fields": {"key": "outputs/${filename}", "policy": "signed", "x-amz-flag": True},
}


def _policy_request():
    header = (UPLOAD_POLICY_KEY.encode(), json.dumps(POLICY).encode())
    return Request({"type": "http", "headers": [header]})


@pytest.fixture
def policy_upload(monkeypatch, transport):
    """Opt in; record in-process uploads and fail on any fal CDN upload."""
    monkeypatch.setenv("FAL_USE_LOCAL_UPLOADER", "1")
    monkeypatch.setattr("fal.toolkit.file._local_uploader.time.sleep", Mock())
    in_process = Mock(return_value="https://bucket.s3.amazonaws.com/outputs/direct")
    monkeypatch.setattr(local, "upload_bytes_with_policy", in_process)
    monkeypatch.setattr(local, "upload_path_with_policy", in_process)
    for repository in (remote.FalFileRepositoryV3, remote.FalFileRepository):
        for method in ("save", "save_file"):
            monkeypatch.setattr(repository, method, Mock(side_effect=AssertionError))

    def install(response):
        respond = isinstance(response, (httpx.Response, type))
        handler = _respond(response) if respond else response
        return transport(handler), in_process

    return install


def _accept_like_the_uploader(request):
    """Substitute ${filename} and build file_url as isolate-cloud's s3_post.rs does."""
    policy = json.loads(request.headers[UPLOAD_POLICY_KEY])
    key = policy["fields"]["key"].replace(
        "${filename}", unquote(request.headers["x-fal-file-name"])
    )
    file_url = policy["url"].rstrip("/") + "/" + quote(key, safe="/~")
    return httpx.Response(202, json={**ACCEPTED, "file_url": file_url})


@pytest.mark.parametrize(
    "file_name,header",
    [
        ("cat.txt", "cat.txt"),
        ("vidéo (1).mp4", "vid%C3%A9o%20%281%29.mp4"),
    ],
)
@pytest.mark.parametrize("explicit", [False, True])
def test_upload_policy_goes_through_the_local_uploader(
    monkeypatch, policy_upload, caller, explicit, file_name, header
):
    requests, in_process = policy_upload(_accept_like_the_uploader)
    caller(
        {"x-fal-cdn-token": "caller-token", "x-fal-endpoint": "fal-ai/test"},
        "request-id",
    )
    monkeypatch.setattr(upload_policy, "get_current_app", remote.get_current_app)
    monkeypatch.setattr(upload_policy, "uuid4", lambda: UUID(int=7))
    kwargs = {"repository": local.LocalFileRepository()} if explicit else {}
    result = files.File.from_bytes(
        b"hello", "text/plain", file_name, request=_policy_request(), **kwargs
    )
    in_process_url, fields = upload_policy._prepare_upload(
        upload_policy.UploadPolicy(**POLICY), file_name, "text/plain"
    )
    assert result.url == in_process_url
    assert result.as_bytes() == b"hello"
    (request,) = requests
    assert request.url.path == "/v1/uploads"
    assert request.content == b"hello"
    assert json.loads(request.headers[UPLOAD_POLICY_KEY]) == POLICY
    assert request.headers["x-fal-file-name"] == f"{UUID(int=7).hex}-{header}"
    assert fields["key"] == f"outputs/{UUID(int=7).hex}-{file_name}"
    assert request.headers["content-type"] == "text/plain"
    assert request.headers["x-fal-request-id"] == "request-id"
    assert request.headers["x-fal-endpoint"] == "fal-ai/test"
    # Without credentials an uploader that predates policies cannot use the CDN.
    assert "authorization" not in request.headers
    assert "x-fal-cdn-token" not in request.headers
    in_process.assert_not_called()


@pytest.mark.parametrize("stage_to_disk", [False, True])
def test_upload_policy_files_stream_from_disk_when_large(
    policy_upload, tmp_path, stage_to_disk
):
    requests, _ = policy_upload(httpx.Response(202, json=ACCEPTED))
    path = tmp_path / "out.bin"
    path.write_bytes(b"data")
    result = files.File.from_path(
        path, multipart=stage_to_disk, request=_policy_request()
    )
    assert result.url == ACCEPTED["file_url"]
    assert result.file_data == (None if stage_to_disk else b"data")
    assert requests[0].content == b"data"


@pytest.mark.parametrize(
    "response,attempts",
    [
        (httpx.Response(429), 3),
        (httpx.Response(401), 1),  # an uploader that predates policies
        (httpx.ConnectError, 1),
    ],
)
def test_upload_policy_refusals_upload_in_process(policy_upload, response, attempts):
    requests, in_process = policy_upload(response)
    result = files.File.from_bytes(
        b"hi", "text/plain", "cat.txt", request=_policy_request()
    )
    assert result.url == in_process.return_value
    assert len(requests) == attempts
    # The in-process upload keeps the key name the uploader was offered.
    in_process.assert_called_once_with(
        upload_policy.UploadPolicy(**POLICY),
        "cat.txt",
        b"hi",
        "text/plain",
        name=unquote(requests[0].headers["x-fal-file-name"]),
        request=None,
    )


@pytest.mark.parametrize("status", [413, 422])
def test_upload_policy_rejections_are_raised(policy_upload, status):
    _, in_process = policy_upload(httpx.Response(status, text="secret"))
    with pytest.raises(upload_policy.UploadPolicyError) as caught:
        files.File.from_bytes(b"hi", request=_policy_request())
    assert caught.value.status_code == 424
    assert "secret" not in str(caught.value)
    in_process.assert_not_called()


@pytest.mark.parametrize("response", [httpx.Response(503), httpx.ReadError])
def test_upload_policy_possibly_accepted_failures_are_raised(policy_upload, response):
    _, in_process = policy_upload(response)
    with pytest.raises(LocalUploadError) as caught:
        files.File.from_bytes(b"hi", request=_policy_request())
    assert not isinstance(caught.value, LocalUploadRefused)
    in_process.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"file_name": "a/b.txt"},
        {"content_type": "image/png", "file_name": "b.txt"},
    ],
)
def test_upload_policy_input_is_checked_before_handoff(
    monkeypatch, policy_upload, kwargs
):
    requests, in_process = policy_upload(httpx.Response(202, json=ACCEPTED))
    signed = {**POLICY, "fields": {**POLICY["fields"], "Content-Type": "text/plain"}}
    header = (UPLOAD_POLICY_KEY.encode(), json.dumps(signed).encode())
    request = Request({"type": "http", "headers": [header]})
    with pytest.raises(upload_policy.UploadPolicyInputError):
        files.File.from_bytes(b"hi", request=request, **kwargs)
    assert not requests
    in_process.assert_not_called()


def test_upload_policy_without_opt_in_uploads_in_process(monkeypatch, policy_upload):
    monkeypatch.delenv("FAL_USE_LOCAL_UPLOADER")
    requests, in_process = policy_upload(httpx.Response(202, json=ACCEPTED))
    result = files.File.from_bytes(b"hi", request=_policy_request())
    assert result.url == in_process.return_value
    assert not requests


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


def test_file_chunks_are_bounded(tmp_path):
    path = tmp_path / "file.bin"
    data = b"a\n" * (128 * 1024) + b"end"
    path.write_bytes(data)
    chunks = list(_local_uploader._chunks(path))
    assert all(0 < len(chunk) <= 64 * 1024 for chunk in chunks)
    assert b"".join(chunks) == data


@pytest.mark.parametrize("settings", [None, {"expiration_duration_seconds": 3600}])
def test_caller_metadata_and_effective_settings(local_upload, caller, settings):
    caller(
        {
            "x-fal-cdn-token": "caller-token",
            "x-fal-request-id": "request-id",
            "x-fal-endpoint": "fal-ai/test",
            "x-fal-object-lifecycle-preference": '{"expiration_duration_seconds": 1}',
            "x-fal-caller-billing-country-code": "Türkiye",
            "x-fal-application": "github_110602490-spoofed",
            "content-type": "application/json",
            "x-app-custom": "request-only",
        },
        "request-id",
    )
    local.LocalFileRepository().save(
        FileData(b"hi"), object_lifecycle_preference=settings
    )
    headers = local_upload[0].headers
    assert headers["authorization"] == "Key test:key"
    # Context is forwarded; uploader inputs and non-ASCII values are not.
    assert headers.get_list("x-fal-cdn-token") == ["caller-token"]
    assert headers.get_list("x-fal-request-id") == ["request-id"]
    assert headers.get_list("content-type") == ["application/octet-stream"]
    assert headers["x-fal-endpoint"] == "fal-ai/test"
    assert "x-fal-caller-billing-country-code" not in headers
    assert "x-fal-application" not in headers
    assert "x-app-custom" not in headers
    if settings:
        assert json.loads(headers["x-fal-object-lifecycle"]) == settings
        assert json.loads(headers["x-fal-object-lifecycle-preference"]) == settings
    else:
        assert "x-fal-object-lifecycle" not in headers
        assert "x-fal-object-lifecycle-preference" not in headers


def _respond(response):
    def respond(request):
        if isinstance(response, type):
            raise response("secret")
        return response

    return respond


@pytest.mark.parametrize(
    "response,refused",
    [
        (httpx.Response(401, text="secret"), True),
        (httpx.Response(404, text="secret"), True),  # no /v1 routes
        (httpx.Response(413, text="secret"), True),
        (httpx.Response(307, headers={"location": "http://secret"}), True),
        (httpx.ConnectError, True),
        (httpx.WriteError, True),
        (httpx.Response(503, text="secret"), False),
        (httpx.Response(202, text="secret"), False),
        (httpx.Response(202, json={**ACCEPTED, "state": "receiving"}), False),
        (httpx.ReadError, False),
        (httpx.ReadTimeout, False),
    ],
)
@pytest.mark.parametrize("source", ["bytes", "small_file", "large_file"])
def test_only_refusals_fall_back_to_direct_cdn(
    monkeypatch, capsys, local_upload, transport, tmp_path, response, refused, source
):
    requests = transport(_respond(response))
    direct = Mock(return_value="https://direct/file")
    direct_file = Mock(return_value=("https://direct/file", None))
    legacy = Mock()
    monkeypatch.setattr(remote.FalFileRepositoryV3, "save", direct)
    monkeypatch.setattr(remote.FalFileRepositoryV3, "save_file", direct_file)
    monkeypatch.setattr(remote.FalFileRepository, "save", legacy)
    monkeypatch.setattr(remote.FalFileRepository, "save_file", legacy)
    path = tmp_path / "file.txt"
    path.write_bytes(b"hi")

    def save():
        if source == "bytes":
            return files.File.from_bytes(b"hi")
        return files.File.from_path(path, multipart=source == "large_file")

    if refused:
        assert save().url == "https://direct/file"
        (direct if source == "bytes" else direct_file).assert_called_once()
        out = capsys.readouterr().out
        assert "Uploading directly to CDN" in out
        assert "secret" not in out
    else:
        with pytest.raises(LocalUploadError) as caught:
            save()
        assert not isinstance(caught.value, LocalUploadRefused)
        assert "secret" not in str(caught.value)
        direct.assert_not_called()
        direct_file.assert_not_called()
    assert len(requests) == 1
    legacy.assert_not_called()


@pytest.mark.skipif(os.name != "posix", reason="Unix sockets require a POSIX host")
@pytest.mark.parametrize("old_wiring", [False, True])
def test_missing_socket_falls_back_to_direct_cdn(monkeypatch, tmp_path, old_wiring):
    monkeypatch.setenv("FAL_USE_LOCAL_UPLOADER", "1")
    monkeypatch.delenv("FAL_API_SOCKET", raising=False)
    monkeypatch.setattr(_local_uploader, "DEFAULT_SOCKET_PATH", str(tmp_path / "x"))
    if old_wiring:
        monkeypatch.setenv("CDN_UPLOADER_SOCKET_PATH", "/run/fal-upload/upload.sock")
    monkeypatch.setattr(
        remote, "fetch_auth_credentials", lambda: AuthCredentials("Key", "test:key")
    )
    direct = Mock(return_value="https://direct/file")
    monkeypatch.setattr(remote.FalFileRepositoryV3, "save", direct)
    assert files.File.from_bytes(b"hi").url == "https://direct/file"
    direct.assert_called_once()


def test_missing_httpx_falls_back_to_v3(monkeypatch):
    monkeypatch.setenv("FAL_USE_LOCAL_UPLOADER", "1")
    monkeypatch.setattr(
        remote, "fetch_auth_credentials", lambda: AuthCredentials("Key", "test:key")
    )
    monkeypatch.setitem(sys.modules, "httpx", None)
    legacy = Mock()
    monkeypatch.setattr(remote.FalFileRepository, "save", legacy)
    direct = Mock(return_value="https://direct/file")
    monkeypatch.setattr(remote.FalFileRepositoryV3, "save", direct)
    assert files.File.from_bytes(b"hi").url == "https://direct/file"
    direct.assert_called_once()
    legacy.assert_not_called()


@pytest.mark.parametrize("source", ["bytes", "large_file"])
@pytest.mark.parametrize("exhausted", [False, True])
def test_rejection_replays_then_uses_direct_cdn(
    monkeypatch, local_upload, transport, tmp_path, source, exhausted
):
    def respond(request):
        if exhausted or len(requests) < 3:
            return httpx.Response(429)
        return httpx.Response(202, json=ACCEPTED)

    requests = transport(respond)
    sleep = Mock()
    monkeypatch.setattr("fal.toolkit.file._local_uploader.time.sleep", sleep)
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
        direct = Mock(return_value=("https://direct/file", None))
        monkeypatch.setattr(remote.FalFileRepositoryV3, "save_file", direct)
        result = files.File.from_path(path, "video/mp4", save_kwargs=dict(kwargs))
    assert result.url == ("https://direct/file" if exhausted else ACCEPTED["file_url"])
    assert result.file_data == (None if source == "large_file" else body)
    assert [request.content for request in requests] == [body] * 3
    assert [call.args[0] for call in sleep.call_args_list] == [0.1, 0.2]
    if not exhausted:
        direct.assert_not_called()
    elif source == "bytes":
        direct.assert_called_once()
        (data,) = direct.call_args.args
        assert data.data == body
        assert direct.call_args.kwargs == kwargs
    else:
        direct.assert_called_once_with(path, content_type="video/mp4", **kwargs)


@pytest.mark.asyncio
async def test_async_wrapper_carries_request_context(local_upload, caller):
    caller({"x-fal-cdn-token": "caller-token"}, "async-id")
    result = await files.File.from_bytes_async(b"hi")
    assert result.url == ACCEPTED["file_url"]
    assert local_upload[0].headers["x-fal-request-id"] == "async-id"


def _completion(*statuses):
    """Accept the upload, then answer status requests with `statuses` in order."""
    remaining = list(statuses)

    def respond(request):
        if request.method == "POST":
            return httpx.Response(202, json=ACCEPTED)
        return _respond(remaining.pop(0))(request)

    return respond


PENDING = httpx.Response(200, json={"state": "pending"})
COMPLETED = httpx.Response(200, json={"state": "completed"})


@pytest.mark.parametrize("from_path", [False, True])
def test_wait_returns_after_cdn_completion(
    local_upload, transport, tmp_path, from_path
):
    requests = transport(_completion(PENDING, COMPLETED))
    save_kwargs = {"wait_for_completion": True}
    if from_path:
        path = tmp_path / "file.txt"
        path.write_bytes(b"hi")
        result = files.File.from_path(path, save_kwargs=save_kwargs)
    else:
        result = files.File.from_bytes(b"hi", save_kwargs=save_kwargs)
    assert result.url == ACCEPTED["file_url"]
    assert [(r.method, str(r.url)) for r in requests[1:]] == [
        ("GET", "http://localhost/v1/uploads/accepted-id?wait=60")
    ] * 2


@pytest.mark.parametrize(
    "status",
    [
        httpx.Response(200, json={"state": "failed", "failure": {"rejected": 403}}),
        httpx.Response(404, text="secret"),  # completed before a restart, or expired
        httpx.Response(200, text="secret"),
        httpx.ReadError,
    ],
)
def test_failed_or_unknown_completion_is_raised(
    monkeypatch, local_upload, transport, status
):
    transport(_completion(PENDING, status))
    direct = Mock()
    legacy = Mock()
    monkeypatch.setattr(remote.FalFileRepositoryV3, "save", direct)
    monkeypatch.setattr(remote.FalFileRepository, "save", legacy)
    with pytest.raises(LocalUploadError) as caught:
        files.File.from_bytes(b"hi", save_kwargs={"wait_for_completion": True})
    assert not isinstance(caught.value, LocalUploadRefused)
    assert "secret" not in str(caught.value)
    direct.assert_not_called()
    legacy.assert_not_called()


def _session(request):
    if request.url.path == "/v1/upload-sessions":
        started = {**ACCEPTED, "upload_id": "session-id", "state": "receiving"}
        return httpx.Response(201, json=started)
    if request.method == "PUT":
        return httpx.Response(204)
    return httpx.Response(202, json=ACCEPTED)


def test_stream_finishes_with_the_byte_count(local_upload, transport):
    requests = transport(_session)
    url = local.LocalFileRepository().save_stream(
        iter([b"he", b"llo"]), "out.txt", "text/plain"
    )
    assert url == ACCEPTED["file_url"]
    assert [(r.method, r.url.path) for r in requests] == [
        ("POST", "/v1/upload-sessions"),
        ("PUT", "/v1/upload-sessions/session-id/body"),
        ("POST", "/v1/upload-sessions/session-id/finish"),
    ]
    assert requests[0].headers["x-fal-file-name"] == "out.txt"
    assert requests[1].content == b"hello"
    assert json.loads(requests[2].content) == {"size_bytes": 5}


def test_failed_producer_aborts_the_stream(local_upload, transport):
    requests = transport(_session)

    def generate():
        yield b"partial"
        raise RuntimeError("encoder died")

    with pytest.raises(RuntimeError, match="encoder died"):
        local.LocalFileRepository().save_stream(generate(), "out.mp4", "video/mp4")
    assert [r.method for r in requests] == ["POST", "DELETE"]


@pytest.mark.parametrize("source", ["bytes", "path"])
def test_reserve_hands_out_the_url_before_the_body(
    local_upload, transport, caller, tmp_path, source
):
    requests = transport(_session)
    caller({"x-fal-cdn-token": "caller-token"}, "request-id")
    pending = files.File.reserve("text/plain", "out.txt")
    assert pending.url == ACCEPTED["file_url"]
    (opened,) = requests
    assert opened.url.path == "/v1/upload-sessions"
    assert opened.headers["x-fal-file-name"] == "out.txt"
    assert opened.headers["x-fal-request-id"] == "request-id"

    path = tmp_path / "out.txt"
    path.write_bytes(b"hello")
    result = pending.complete(b"hello" if source == "bytes" else path)
    assert (result.url, result.file_name, result.file_size) == (
        ACCEPTED["file_url"],
        "out.txt",
        5,
    )
    assert [(r.method, r.url.path) for r in requests[1:]] == [
        ("PUT", "/v1/upload-sessions/session-id/body"),
        ("POST", "/v1/upload-sessions/session-id/finish"),
    ]
    assert requests[1].content == b"hello"
    assert json.loads(requests[2].content) == {"size_bytes": 5}
    # The body carries no request context; the session already has it.
    assert "x-fal-request-id" not in requests[1].headers


def test_reserve_abort_and_failed_complete_delete_the_session(local_upload, transport):
    requests = transport(_session)
    files.File.reserve("text/plain").abort()
    assert [r.method for r in requests] == ["POST", "DELETE"]

    requests.clear()
    pending = files.File.reserve("text/plain")
    with pytest.raises(FileNotFoundError):
        pending.complete("/missing/out.txt")
    assert [r.method for r in requests] == ["POST", "DELETE"]


def _fake_cdn(monkeypatch, complete_statuses=()):
    """Answer the REST multipart API and record each request; each /complete
    answers the next of `complete_statuses` with an error, then succeeds."""
    calls = []
    failures = list(complete_statuses)

    @contextmanager
    def respond(request, **_kwargs):
        calls.append(request)
        if failures and request.full_url.endswith("/complete"):
            raise HTTPError(request.full_url, failures.pop(0), "", {}, None)
        body = {"file_url": "https://v3.fal.media/out", "upload_url": "https://up/x"}
        response = io.BytesIO(json.dumps(body).encode())
        response.headers = {"ETag": "etag"}
        yield response

    monkeypatch.setattr(remote, "_maybe_retry_request", respond)
    return calls


@pytest.mark.parametrize(
    "refusal,body,parts",
    [
        (httpx.Response(404), b"hello", [b"hel", b"lo"]),
        (httpx.ConnectError, b"", [b""]),
    ],
)
def test_reserve_falls_back_to_a_direct_multipart_upload(
    monkeypatch, local_upload, transport, tmp_path, refusal, body, parts
):
    transport(_respond(refusal))
    monkeypatch.setattr(remote.MultipartUploadV3, "MULTIPART_CHUNK_SIZE", 3)
    calls = _fake_cdn(monkeypatch)
    pending = files.File.reserve("text/plain", "out.txt")
    assert pending.url == "https://v3.fal.media/out"
    assert [(c.get_method(), c.full_url) for c in calls] == [
        (
            "POST",
            f"https://{remote.REST_HOST}/storage/upload/initiate-multipart"
            "?storage_type=fal-cdn-v3",
        )
    ]
    path = tmp_path / "out.txt"
    path.write_bytes(body)
    assert pending.complete(path).url == pending.url
    assert sorted((c.get_method(), c.full_url, c.data) for c in calls[1:]) == [
        ("POST", "https://up/x/complete", calls[-1].data),
        *(("PUT", f"https://up/x/{n}", part) for n, part in enumerate(parts, 1)),
    ]


def test_reserve_failed_complete_raises_and_a_retry_sends_fresh_parts(
    monkeypatch, local_upload, transport
):
    transport(_respond(httpx.Response(404)))
    # 409 also means a deleted or expired upload, so it is never success.
    calls = _fake_cdn(monkeypatch, complete_statuses=[409])
    pending = files.File.reserve("text/plain", "out.txt")
    with pytest.raises(FileUploadException):
        pending.complete(b"hi")
    assert pending.complete(b"hi").url == pending.url
    completes = [c for c in calls if c.full_url == "https://up/x/complete"]
    assert [json.loads(c.data)["parts"] for c in completes] == [
        [{"partNumber": 1, "etag": "etag"}]
    ] * 2


def _caller_request(**headers):
    return Request(
        {
            "type": "http",
            "headers": [(k.encode(), v.encode()) for k, v in headers.items()],
        }
    )


@pytest.mark.parametrize("via_uploader", [True, False])
def test_reserve_names_the_caller_from_the_given_request(
    monkeypatch, local_upload, transport, caller, via_uploader
):
    requests = transport(_session if via_uploader else _respond(httpx.Response(404)))
    calls = _fake_cdn(monkeypatch)
    # The request in flight, if any, is not the caller.
    caller({"x-fal-cdn-token": "other-token"}, "other-id")
    request = _caller_request(**{"x-fal-cdn-token": "token", "x-fal-request-id": "rid"})
    files.File.reserve("text/plain", "out.txt", request=request)
    if via_uploader:
        (opened,) = requests
        headers = dict(opened.headers)
    else:
        (initiated,) = calls
        headers = {k.lower(): v for k, v in initiated.header_items()}
    assert (headers["x-fal-cdn-token"], headers["x-fal-request-id"]) == ("token", "rid")


@pytest.mark.parametrize("via_uploader", [True, False])
@pytest.mark.parametrize("acknowledged", [True, False])
def test_reserve_reports_cdn_signing_for_selected_destination(
    monkeypatch, local_upload, transport, via_uploader, acknowledged
):
    def session(request):
        response = _session(request)
        if request.url.path == "/v1/upload-sessions":
            body = response.json()
            body["cdn_signing_requested"] = acknowledged
            return httpx.Response(201, json=body)
        return response

    requests = transport(session if via_uploader else _respond(httpx.Response(404)))
    calls = _fake_cdn(monkeypatch)
    request = _caller_request(**{"x-fal-cdn-token": "token", "x-fal-request-id": "rid"})
    pending = files.File.reserve(
        "text/plain",
        "out.txt",
        request=request,
        extra_headers={"x-fal-provenance": "true"},
    )

    assert pending._cdn_signing_requested is (acknowledged if via_uploader else False)
    initiated = requests[0] if via_uploader else calls[0]
    headers = (
        {k.lower(): v for k, v in initiated.header_items()}
        if not via_uploader
        else initiated.headers
    )
    if via_uploader:
        assert headers["x-fal-provenance"] == "true"
    else:
        assert "x-fal-provenance" not in headers
    assert headers["x-fal-request-id"] == "rid"


def test_ambient_provenance_does_not_sign_an_unselected_output(
    local_upload, transport, caller
):
    requests = transport(_session)
    caller({"x-fal-provenance": "true", "x-fal-cdn-token": "token"}, "rid")
    pending = files.File.reserve("text/plain", "out.txt")
    assert pending._cdn_signing_requested is False
    assert "x-fal-provenance" not in requests[0].headers


def test_reserve_explicit_headers_override_forwarded_context(
    local_upload, transport, caller
):
    requests = transport(_session)
    caller({"x-fal-endpoint": "incoming"}, "rid")
    files.File.reserve(extra_headers={"X-Fal-Endpoint": "explicit"})
    assert requests[0].headers.get_list("x-fal-endpoint") == ["explicit"]


@pytest.mark.parametrize("given", [True, False])
def test_reserve_policy_url_matches_the_upload(
    monkeypatch, policy_upload, tmp_path, given
):
    requests, in_process = policy_upload(_accept_like_the_uploader)
    request = _policy_request()
    request.scope["headers"] += [
        (b"x-fal-request-id", b"rid"),
        (b"x-fal-endpoint", b"fal-ai/test"),
    ]
    # Given, or in flight at reserve time; either way gone before complete.
    app = SimpleNamespace(current_request=None if given else request)
    for module in (files, upload_policy, remote, _local_uploader):
        monkeypatch.setattr(module, "get_current_app", lambda: app)
    pending = files.File.reserve(
        "text/plain", "cat.txt", request=request if given else None
    )
    app.current_request = None
    assert pending.url.startswith(POLICY["url"].rstrip("/") + "/outputs/")
    assert requests == []
    path = tmp_path / "cat.txt"
    path.write_bytes(b"hi")
    pending.complete(path)
    (request,) = requests
    # The uploader builds the same URL from the reserved key name.
    assert _accept_like_the_uploader(request).json()["file_url"] == pending.url
    assert request.headers["x-fal-request-id"] == "rid"
    assert request.headers["x-fal-endpoint"] == "fal-ai/test"
    in_process.assert_not_called()


def test_reserve_policy_complete_queues_the_post(monkeypatch, tmp_path):
    """Without the uploader, complete does not wait for the caller's bucket."""
    monkeypatch.delenv("FAL_USE_LOCAL_UPLOADER", raising=False)
    submit = Mock()
    monkeypatch.setattr(upload_policy, "_submit", submit)
    request = _policy_request()
    request.scope["headers"] += [(b"x-fal-request-id", b"rid")]
    pending = files.File.reserve("text/plain", "cat.txt", request=request)
    path = tmp_path / "cat.txt"
    path.write_bytes(b"hi")
    assert pending.complete(path).url == pending.url
    # Queued under the reserving request, with nothing in flight now.
    submit.assert_called_once()
    assert submit.call_args.args[1] == "rid"


def test_in_memory_cannot_reserve():
    with pytest.raises(NotImplementedError):
        files.File.reserve("text/plain", repository="in_memory")
