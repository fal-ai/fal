import httpx
import pytest

from fal import apps
from fal.apps import Completed, RequestHandle
from fal.sdk import FalServerlessKeyCredentials


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(apps, "_RETRY_BASE_DELAY", 0.0)
    monkeypatch.setattr(apps, "_RETRY_MAX_DELAY", 0.0)


def _handle(responses: list) -> tuple[list[int], RequestHandle]:
    """Serve each entry of `responses` in order: an exception is raised, a
    Response is returned, and the last entry repeats."""
    calls = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        calls[0] += 1
        reply = responses[min(calls[0], len(responses)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply

    handle = RequestHandle(
        app_id="user/app",
        request_id="req",
        _client=httpx.Client(transport=httpx.MockTransport(handler)),
        _creds=FalServerlessKeyCredentials(key_id="id", key_secret="secret"),
    )
    return calls, handle


def _completed() -> httpx.Response:
    return httpx.Response(200, json={"logs": [], "status": "COMPLETED"})


def test_status_retries_read_timeout():
    calls, handle = _handle([httpx.ReadTimeout("stalled"), _completed()])

    assert isinstance(handle.status(), Completed)
    assert calls[0] == 2


def test_status_retries_rate_limit():
    calls, handle = _handle([httpx.Response(429), _completed()])

    assert isinstance(handle.status(), Completed)
    assert calls[0] == 2


def test_status_retries_ingress_error_but_not_platform_error():
    calls, handle = _handle(
        [
            httpx.Response(503, text="<html>nginx</html>"),
            httpx.Response(
                503, text="<html>nginx</html>", headers={"x-fal-request-id": "r"}
            ),
        ]
    )

    with pytest.raises(httpx.HTTPStatusError) as exc:
        handle.status()

    assert exc.value.response.status_code == 503
    assert calls[0] == 2


def test_status_does_not_retry_user_timeout():
    calls, handle = _handle(
        [
            httpx.Response(
                504,
                text="<html>nginx</html>",
                headers={"x-fal-request-timeout-type": "user"},
            )
        ]
    )

    with pytest.raises(httpx.HTTPStatusError):
        handle.status()

    assert calls[0] == 1


def test_status_gives_up_after_max_attempts():
    calls, handle = _handle([httpx.ReadTimeout("stalled")])

    with pytest.raises(httpx.ReadTimeout):
        handle.status()

    assert calls[0] == apps._RETRY_MAX_ATTEMPTS


def test_fetch_result_retries_read_timeout():
    calls, handle = _handle(
        [httpx.ReadTimeout("stalled"), httpx.Response(200, json={"result": 3})]
    )

    assert handle.fetch_result() == {"result": 3}
    assert calls[0] == 2


def test_fetch_result_keeps_error_text_without_retrying():
    calls, handle = _handle([httpx.Response(400, json={"detail": "not done"})])

    with pytest.raises(httpx.HTTPStatusError, match="400: ") as exc:
        handle.fetch_result()

    assert "not done" in str(exc.value)
    assert calls[0] == 1
