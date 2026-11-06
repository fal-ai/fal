import json

import pytest

from fal.dev.remote import proxy

pytestmark = pytest.mark.usefixtures("dev_account")


def test_forward_refuses_child_control_paths():
    for path in ("/__fal_dev_child/reload", "/%5F_fal_dev_child/reload", "x"):
        with pytest.raises(ValueError):
            proxy.forward({"path": path}, 1)


def _fake_child(monkeypatch, statuses, times):
    calls: list[dict] = []

    def forward(request, timeout):
        index = len(calls)
        calls.append({**request, "timeout": timeout})
        return {
            "status_code": statuses[index],
            "content_type": "application/json",
            "elapsed_seconds": times[index],
            "body": json.dumps({"run": index}),
        }

    monkeypatch.setattr(proxy, "forward", forward)
    return calls


def test_forward_runs_times_warmups_and_runs_separately(monkeypatch):
    calls = _fake_child(monkeypatch, [200] * 5, [9.0, 1.0, 2.0, 3.0, 4.0])
    result = proxy.forward_runs({"path": "/", "warmups": 1, "repeat": 4})

    assert len(calls) == 5
    assert [run["elapsed_seconds"] for run in result["warmup_runs"]] == [9.0]
    assert [run["elapsed_seconds"] for run in result["runs"]] == [1, 2, 3, 4]
    # The last response comes back, not all of them.
    assert json.loads(result["body"]) == {"run": 4}


def test_forward_runs_stops_at_the_first_failure(monkeypatch):
    calls = _fake_child(monkeypatch, [200, 422, 200], [1.0, 1.0, 1.0])
    result = proxy.forward_runs({"path": "/", "repeat": 3})

    assert len(calls) == 2
    assert result["status_code"] == 422
    assert [run["status_code"] for run in result["runs"]] == [200, 422]


@pytest.mark.parametrize(
    "payload",
    [{"repeat": 0}, {"repeat": 101}, {"warmups": -1}, {"warmups": "2"}],
)
def test_forward_runs_checks_counts(payload):
    with pytest.raises(ValueError):
        proxy.forward_runs({"path": "/", **payload})


def test_forward_runs_stops_at_the_time_budget(monkeypatch):
    calls = _fake_child(monkeypatch, [200] * 3, [1.0] * 3)
    clock = iter([0.0, 10.0, 4000.0])
    monkeypatch.setattr(proxy.time, "monotonic", lambda: next(clock))
    result = proxy.forward_runs({"path": "/", "repeat": 3})
    assert "time budget" in result["stopped"]
    assert len(result["runs"]) == 2
    # The second run only gets what is left of the budget.
    assert calls[1]["timeout"] == proxy.RUNS_BUDGET_SECONDS - 10


def test_forward_runs_keeps_runs_when_a_later_run_fails_to_connect(monkeypatch):
    calls = _fake_child(monkeypatch, [200], [1.0])
    forward = proxy.forward

    def flaky(request, timeout):
        if calls:
            raise TimeoutError("timed out")
        return forward(request, timeout)

    monkeypatch.setattr(proxy, "forward", flaky)
    result = proxy.forward_runs({"path": "/", "repeat": 3})
    assert result["stopped"] == "run 2 did not complete: timed out"
    assert [run["elapsed_seconds"] for run in result["runs"]] == [1.0]
