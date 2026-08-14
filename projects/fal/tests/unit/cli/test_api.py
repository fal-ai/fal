import json
from unittest.mock import patch

import httpx
import pytest
import rich

from fal.cli.api import _api, _format_log, _response_detail
from fal.cli.main import parse_args
from fal.sdk import Credentials

REQUEST_ID = "00000000-0000-4000-8000-000000000001"
TRACEBACK = (
    "ModuleNotFoundError: No module named 'torch'\n"
    '  File "/app/app.py", line 18, in run\n'
    "    import torch\n"
)


def _log(message):
    return {"timestamp": "2020-01-01T00:00:00Z", "message": message, "labels": {}}


class FakeQueue:
    """Serves the queue endpoints `fal api` polls, for a single request.

    The gateway answers each status poll with the log entries inside a moving
    time window rather than everything so far, so `window` bounds how many
    entries a poll may return -- the oldest fall out as new ones arrive, which
    is what a request outliving the window looks like from the client.
    """

    def __init__(
        self,
        *,
        response,
        status_logs,
        error=None,
        error_type=None,
        in_progress_polls=1,
        window=None,
        poll_failures=(),
    ):
        self.response = response
        self.status_logs = status_logs
        self.error = error
        self.error_type = error_type
        self.in_progress_polls = in_progress_polls
        self.window = window or len(status_logs)
        self.poll_failures = dict(poll_failures)
        self.polls = 0

    def _visible(self, upto):
        return self.status_logs[max(0, upto - self.window) : upto]

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"request_id": REQUEST_ID})

        if not request.url.path.endswith("/status/"):
            return self.response

        self.polls += 1
        if self.polls in self.poll_failures:
            return self.poll_failures[self.polls]

        if self.polls <= self.in_progress_polls:
            # One more entry becomes visible per poll, so a later poll can hold
            # entries an earlier one did not.
            return httpx.Response(
                202, json={"status": "IN_PROGRESS", "logs": self._visible(self.polls)}
            )

        body = {"status": "COMPLETED", "logs": self._visible(len(self.status_logs))}
        if self.error is not None:
            body["error"] = self.error
        if self.error_type is not None:
            body["error_type"] = self.error_type
        return httpx.Response(200, json=body)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self._handle))


@pytest.fixture(autouse=True)
def wide_console():
    # Keep panel borders from wrapping the strings the assertions look for.
    console = rich.console.Console(width=200)
    with patch.object(rich, "get_console", return_value=console):
        yield


@pytest.fixture
def run_api(capsys):
    def _run(queue: FakeQueue):
        args = parse_args(["api", "owner/app", "prompt=hi"])
        patch_client = patch("fal.apps._get_http_client", return_value=queue.client())
        patch_creds = patch("fal.apps.get_credentials", return_value=Credentials())
        with patch_client, patch_creds:
            code = _api(args)
        return code, capsys.readouterr().out

    return _run


def _failing_queue(**kwargs):
    kwargs.setdefault(
        "status_logs", [_log("starting up"), _log(json.dumps({"traceback": TRACEBACK}))]
    )
    kwargs.setdefault(
        "response", httpx.Response(500, json={"detail": "Internal Server Error"})
    )
    return FakeQueue(**kwargs)


def test_failed_request_reports_detail_and_exits_nonzero(run_api):
    code, out = run_api(_failing_queue())

    assert code == 1
    assert f"Request {REQUEST_ID} failed with HTTP 500" in out
    assert "Internal Server Error" in out


def test_failed_request_shows_app_traceback(run_api):
    _, out = run_api(_failing_queue())

    # The traceback only arrives with the terminal status, wrapped in a JSON
    # envelope that has to be unpacked to be readable.
    assert "ModuleNotFoundError: No module named 'torch'" in out
    assert '{"traceback"' not in out


def test_failed_request_prefers_gateway_error(run_api):
    queue = _failing_queue(
        status_logs=[_log("starting up")],
        error="Request timed out",
        error_type="request_timeout",
    )

    code, out = run_api(queue)

    assert code == 1
    assert "Request timed out" in out


def test_successful_request_returns_result(run_api):
    queue = FakeQueue(
        response=httpx.Response(200, json={"text": "hi"}),
        status_logs=[_log("starting up"), _log("done")],
    )

    code, out = run_api(queue)

    assert code is None
    assert "'text'" in out


def test_logs_are_not_duplicated_across_polls(run_api):
    queue = _failing_queue(
        status_logs=[_log("starting up"), _log("still going"), _log("nearly there")],
        in_progress_polls=3,
    )

    _, out = run_api(queue)

    assert queue.polls > 3
    assert out.count("starting up") == 1
    assert out.count("still going") == 1


def test_traceback_survives_a_log_window_that_slides(run_api):
    # A request outliving the log window: by the time it fails, the early
    # entries have fallen out and the batch is no longer a growing prefix.
    queue = _failing_queue(
        status_logs=[
            _log("progress 1"),
            _log("progress 2"),
            _log("progress 3"),
            _log("progress 4"),
            _log(json.dumps({"traceback": TRACEBACK})),
        ],
        in_progress_polls=4,
        window=2,
    )

    code, out = run_api(queue)

    assert code == 1
    assert "ModuleNotFoundError: No module named 'torch'" in out


def test_already_seen_logs_are_not_reprinted_on_failure(run_api):
    # The terminal status adds nothing new; the failure panel must stay away
    # rather than repeat the whole log.
    queue = _failing_queue(
        status_logs=[_log("starting up"), _log("still going")],
        in_progress_polls=2,
    )

    _, out = run_api(queue)

    assert out.count("starting up") == 1
    assert out.count("still going") == 1


def test_transient_poll_failure_does_not_claim_the_request_failed(run_api):
    # A 503 from one status poll says nothing about the request itself.
    queue = _failing_queue(
        status_logs=[_log("starting up"), _log("still going")],
        in_progress_polls=4,
        poll_failures=[(2, httpx.Response(503, text="upstream unavailable"))],
    )

    code, out = run_api(queue)

    assert code == 1
    assert "may still be running" in out
    assert "failed with HTTP" not in out


def test_square_brackets_in_logs_and_detail_survive(run_api):
    # Rich parses markup in a plain string: "[/]" raises MarkupError and
    # "[gw0]" is swallowed as a style tag.
    queue = _failing_queue(
        status_logs=[_log("worker [gw0] died"), _log("expected [/] token")],
        response=httpx.Response(500, json={"detail": "bad [/] input"}),
    )

    code, out = run_api(queue)

    assert code == 1
    assert "[gw0]" in out
    assert "expected [/] token" in out
    assert "bad [/] input" in out


def test_format_log_unwraps_traceback_envelope():
    assert _format_log(_log(json.dumps({"traceback": TRACEBACK}))) == TRACEBACK.rstrip()


def test_format_log_passes_through_plain_messages():
    assert _format_log(_log("plain message")) == "plain message"
    assert _format_log(_log('{"not": "a traceback"}')) == '{"not": "a traceback"}'


def test_response_detail_falls_back_to_body_text():
    assert _response_detail(httpx.Response(502, text="bad gateway")) == "bad gateway"
    assert (
        _response_detail(httpx.Response(422, json={"detail": [{"loc": ["prompt"]}]}))
        == '[{"loc": ["prompt"]}]'
    )
