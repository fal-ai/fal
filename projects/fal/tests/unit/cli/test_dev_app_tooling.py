import json

import pytest

from fal.cli.main import parse_args
from fal.cli.parser import FalParserExit
from fal.dev.account import Account
from fal.dev.app import bench as app_bench
from fal.dev.app import client
from fal.dev.app import session as app_session
from fal.dev.app.session import Session
from fal.dev.errors import DevError
from fal.dev.remote import proxy

pytestmark = pytest.mark.usefixtures("dev_account")


def _session(**kwargs) -> Session:
    values = {
        "name": "dev-demo-me",
        "file_path": "/src/demo_app.py",
        "root": "/src",
        "module": "demo_app",
        "class_name": "Demo",
        "run_url": "https://fal.run/me/dev-demo-me",
        "token": "t",
        "account": Account("me", True),
        "machine_type": "GPU-H100",
        "manifest": {"demo_app.py": "abc"},
    }
    return Session(**{**values, **kwargs})


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(app_bench, "BENCHMARKS_DIR", tmp_path / "benchmarks")
    monkeypatch.setattr(app_session, "SESSIONS_DIR", tmp_path / "sessions")
    return tmp_path


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


def test_stats_use_nearest_rank_p90():
    values = [float(n) for n in range(1, 11)]
    result = app_bench.stats(values)
    assert result["median"] == 5.5
    assert result["p90"] == 9.0
    assert (result["min"], result["max"]) == (1.0, 10.0)
    assert app_bench.stats([3.0])["p90"] == 3.0
    assert app_bench.percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.9) == 5.0


def test_save_load_and_compare(home):
    session = _session()

    def run(name, runs, manifest):
        result = {
            "runs": [{"elapsed_seconds": value} for value in runs],
            "warmup_runs": [{"elapsed_seconds": 5.0}],
        }
        record = app_bench.record(
            name=name,
            session=_session(manifest=manifest),
            path="/",
            body={"prompt": "a cat"},
            result=result,
            warmups=1,
        )
        return app_bench.save(record)

    run("before", [2.0, 2.2, 2.1], {"a.py": "1"})
    run("after", [1.0, 1.1, 1.2], {"a.py": "2"})

    before = app_bench.load("before", session.name)
    after = app_bench.load(
        str(app_bench.BENCHMARKS_DIR / session.name / "after.json"), None
    )
    assert before["stats"]["median"] == 2.1
    assert before["source"] != after["source"]
    assert app_bench.mismatches(before, after) == []

    lines = app_bench.compare(before, after)
    median = next(line for line in lines if line.startswith("median"))
    assert "2.100s" in median
    assert "-1.000s (-47.6%)" in median
    assert [run["name"] for run in app_bench.saved(session.name)] == [
        "before",
        "after",
    ]


def test_compare_flags_different_requests(home):
    base = {
        "app": "a.py::A",
        "path": "/",
        "method": "POST",
        "payload_sha256": "a",
        "machine_type": "x",
    }
    other = {**base, "payload_sha256": "b"}
    assert app_bench.mismatches(base, other) == ["payload_sha256: a vs b"]


def test_benchmark_names_are_checked(home, tmp_path):
    for bad in ("../escape", "a\n", ""):
        with pytest.raises(DevError):
            app_bench.check_name(bad)
    with pytest.raises(DevError):
        app_bench.load("missing", "dev-demo-me")
    foreign = tmp_path / "foreign.json"
    foreign.write_text('{"hello": 1}')
    with pytest.raises(DevError, match="not a saved benchmark"):
        app_bench.load(str(foreign), None)
    partial = tmp_path / "partial.json"
    partial.write_text(
        json.dumps(
            {"name": "x", "path": "/", "repeat": 1, "created_at": 0, "stats": {}}
        )
    )
    with pytest.raises(DevError, match="not a saved benchmark"):
        app_bench.load(str(partial), None)
    # Stray files in the session folder do not break listing.
    folder = app_bench.BENCHMARKS_DIR / "dev-demo-me"
    folder.mkdir(parents=True)
    (folder / "stray.json").write_text("[1, 2]")
    assert app_bench.saved("dev-demo-me") == []


def _fake_box(monkeypatch, statuses, truncated=False):
    sent = []

    def call(_session, action, payload=None, **_):
        if action == "status":
            return {"phase": "ready", "module": "demo_app"}
        sent.append(payload)
        runs = [
            {"status_code": s, "elapsed_seconds": 1.0 + i}
            for i, s in enumerate(statuses)
        ]
        return {
            "status_code": statuses[-1],
            "content_type": "application/json",
            "elapsed_seconds": 1.5,
            "body": "{}",
            "runs": runs,
            "warmup_runs": [{"status_code": 200, "elapsed_seconds": 9}],
            "stopped": "run 3 did not complete: boom" if truncated else "",
        }

    monkeypatch.setattr(client, "call", call)
    return sent


def _request(*extra):
    args = parse_args(
        ["dev", "app", "request", "/", "--data", '{"prompt": "a"}', *extra]
    )
    args.func(args)


def test_request_saves_a_named_benchmark(home, monkeypatch, capsys):
    session = _session()
    session.save()
    sent = _fake_box(monkeypatch, [200, 200, 200])
    _request("--warmups", "1", "--repeat", "3", "--save", "base")

    assert sent[0]["warmups"] == 1
    assert sent[0]["repeat"] == 3
    saved = app_bench.load("base", session.name)
    assert saved["runs"] == [1.0, 2.0, 3.0]
    assert saved["warmup_runs"] == [9]
    out = capsys.readouterr().out
    assert "median 2.000s" in out
    assert "Saved benchmark" in out

    _request("--repeat", "3", "--save", "base")
    assert "Overwrote benchmark" in capsys.readouterr().out
    assert [run["name"] for run in app_bench.saved(session.name)] == ["base"]


@pytest.mark.parametrize(
    "statuses,truncated", [([200, 500], False), ([200, 200], True)]
)
def test_failed_or_truncated_benchmarks_are_not_saved(
    home, monkeypatch, capsys, statuses, truncated
):
    _session().save()
    _fake_box(monkeypatch, statuses, truncated=truncated)
    with pytest.raises(SystemExit):
        _request("--repeat", "3", "--save", "base")
    out = capsys.readouterr().out
    assert "Not saved" in out
    assert app_bench.saved("dev-demo-me") == []


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


def test_request_flags_are_bounded(capsys):
    with pytest.raises(FalParserExit):
        parse_args(["dev", "app", "request", "/", "--repeat", "0"])
    with pytest.raises(FalParserExit):
        parse_args(["dev", "app", "request", "/", "--repeat", "abc"])
    assert "invalid integer 'abc'" in capsys.readouterr().err
    args = parse_args(["dev", "app", "benchmark", "compare", "a", "b"])
    assert (args.baseline, args.candidate) == ("a", "b")


def test_empty_benchmark_name_is_refused(home):
    _session().save()
    with pytest.raises(DevError):
        _request("--save", "")


def test_compare_columns_fit_source_digests():
    run = {
        "name": "a",
        "repeat": 5,
        "source": "0123456789ab",
        "stats": dict.fromkeys(app_bench.STAT_KEYS, 1.0),
    }
    lines = app_bench.compare(run, {**run, "name": "b", "source": "ba9876543210"})
    source = next(line for line in lines if line.startswith("source"))
    assert source.split() == ["source", "0123456789ab", "ba9876543210"]
