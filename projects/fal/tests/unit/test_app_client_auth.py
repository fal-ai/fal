from unittest.mock import Mock

import httpx

from fal import flags
from fal.api import SpawnInfo
from fal.app import AppClient, AppSpawnInfo
from fal.sdk import AuthenticatedCredentials


def test_connect_uses_spawn_login_for_health_and_endpoint(monkeypatch):
    user = Mock(bearer_token="Bearer first")
    user.get_account.return_value = {"user_id": "team-id"}
    info = SpawnInfo(credentials=AuthenticatedCredentials(user=user, team="team"))
    info.url = f"https://{flags.FAL_RUN_HOST}/owner/app"
    info.stream = Mock()
    spawn = AppSpawnInfo(info)

    class Example:
        def echo(self, data):
            pass

    Example.echo.route_signature = Mock(path="/echo")
    monkeypatch.setattr(Example, "spawn", lambda: spawn, raising=False)
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={})

    client_class = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: client_class(transport=httpx.MockTransport(respond), **kwargs),
    )
    with AppClient.connect(Example) as client:
        user.bearer_token = "Bearer refreshed"
        client.echo({})

    assert [
        (r.method, r.url.path, r.headers["Authorization"], r.headers["X-Fal-User-Id"])
        for r in requests
    ] == [
        ("GET", "/owner/app/health", "Bearer first", "team-id"),
        ("POST", "/owner/app/echo", "Bearer refreshed", "team-id"),
    ]

    info.url = "https://external.example/owner/app"
    spawn.wait()
    assert "Authorization" not in requests[-1].headers
    assert "X-Fal-User-Id" not in requests[-1].headers
