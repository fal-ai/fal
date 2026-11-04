import json
from http import HTTPStatus
from types import SimpleNamespace

import httpx
import pytest

from fal.api.api import FalServerlessError
from fal.api.client import SyncServerlessClient
from fal.exceptions import (
    VolumeAlreadyExistsError,
    VolumeControlPlaneUnavailableError,
    VolumeInvalidMountPathError,
    VolumeInvalidNameError,
)
from fal.sdk import FalServerlessKeyCredentials
from fal.volumes import Volume, normalize_volumes


def test_volume_name_is_canonical_lowercase():
    assert Volume("My.Vol_1-x").name == "my.vol_1-x"


@pytest.mark.parametrize("name", ["", "-lead", "has space", "a" * 33, "é", 5])
def test_volume_rejects_invalid_name(name):
    with pytest.raises(VolumeInvalidNameError) as exc_info:
        Volume(name)
    assert exc_info.value.reason == "VOLUME_INVALID_NAME"
    assert exc_info.value.http_status is None
    assert isinstance(exc_info.value, ValueError)


def test_normalize_volumes_accepts_volume_and_str():
    assert normalize_volumes({"/a": Volume("A"), "/b": "B"}) == {"/a": "a", "/b": "b"}


@pytest.mark.parametrize("path", ["/", "/a/b", "a", "/.", "/..", "/a/", "/a b"])
def test_normalize_volumes_rejects_mount_path(path):
    with pytest.raises(VolumeInvalidMountPathError) as exc_info:
        normalize_volumes({path: "vol"})
    assert exc_info.value.mount_path == path


def test_normalize_volumes_rejects_more_than_max():
    volumes = {f"/v{i}": "vol" for i in range(6)}
    with pytest.raises(VolumeInvalidMountPathError, match="At most 5 volumes"):
        normalize_volumes(volumes)
    assert len(normalize_volumes(dict(list(volumes.items())[:5]))) == 5


@pytest.mark.parametrize("volumes", [["/a"], {"/a": 1}, {"/a": None}])
def test_normalize_volumes_rejects_shape(volumes):
    with pytest.raises(ValueError):
        normalize_volumes(volumes)


@pytest.fixture
def rest(monkeypatch):
    """Serve the REST volumes endpoint through an httpx mock transport."""
    state = SimpleNamespace(
        requests=[],
        response=httpx.Response(HTTPStatus.CREATED, json={"name": "my-volume"}),
    )

    def handler(request):
        state.requests.append(request)
        return state.response

    real_client = httpx.Client

    def mock_client(**kwargs):
        state.client_kwargs = kwargs
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", mock_client)
    monkeypatch.setattr(
        "fal.sdk.get_credentials",
        lambda **kwargs: FalServerlessKeyCredentials("k", "s"),
    )
    return state


def _error_response(status, content):
    return httpx.Response(status, content=content)


def test_create_volume_returns_canonical_name(rest):
    name = SyncServerlessClient(host="api.example.com").volumes.create("My-Volume")

    assert name == "my-volume"
    (request,) = rest.requests
    assert request.method == "POST"
    assert str(request.url) == "https://rest.example.com/volumes/"
    assert json.loads(request.content) == {"name": "my-volume"}
    assert request.headers["authorization"] == "Key k:s"
    assert rest.client_kwargs["follow_redirects"] is True


def test_create_volume_validates_name_before_request(rest):
    with pytest.raises(VolumeInvalidNameError):
        SyncServerlessClient().volumes.create("bad name")
    assert rest.requests == []


@pytest.mark.parametrize(
    ("status", "reason", "expected"),
    [
        (409, "VOLUME_ALREADY_EXISTS", VolumeAlreadyExistsError),
        (503, "VOLUME_CONTROL_PLANE_UNAVAILABLE", VolumeControlPlaneUnavailableError),
    ],
)
def test_create_volume_maps_known_reason(rest, status, reason, expected):
    rest.response = _error_response(
        status, json.dumps({"detail": "nope", "reason": reason}).encode()
    )
    with pytest.raises(expected) as exc_info:
        SyncServerlessClient().volumes.create("my-volume")
    exc = exc_info.value
    assert exc.message == "nope"
    assert exc.reason == reason
    assert exc.http_status == status
    assert exc.grpc_status is None
    assert exc.volume_name == "my-volume"


def test_create_volume_unknown_reason_is_generic(rest):
    rest.response = _error_response(
        418, json.dumps({"detail": "teapot", "reason": "VOLUME_BREWING"}).encode()
    )
    with pytest.raises(FalServerlessError, match="418 teapot"):
        SyncServerlessClient().volumes.create("my-volume")


def test_create_volume_non_json_error_is_generic(rest):
    rest.response = _error_response(502, b"bad gateway")
    with pytest.raises(FalServerlessError, match="502 bad gateway"):
        SyncServerlessClient().volumes.create("my-volume")
