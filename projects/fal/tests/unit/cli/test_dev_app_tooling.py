import pytest
from dev.app import fakes
from dev.app.fakes import make_request, make_session

from fal.cli.main import parse_args
from fal.cli.parser import FalParserExit
from fal.dev.app import bench as app_bench
from fal.dev.app import client

home = fakes.home

pytestmark = pytest.mark.usefixtures("dev_account")


def _fake_box(monkeypatch, statuses, truncated=False):
    sent = []

    def call(make_session, action, payload=None, **_):
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


def test_request_saves_a_named_benchmark(home, monkeypatch, capsys):
    session = make_session()
    session.save()
    sent = _fake_box(monkeypatch, [200, 200, 200])
    make_request("--warmups", "1", "--repeat", "3", "--save", "base")

    assert sent[0]["warmups"] == 1
    assert sent[0]["repeat"] == 3
    saved = app_bench.load("base", session.name)
    assert saved["runs"] == [1.0, 2.0, 3.0]
    assert saved["warmup_runs"] == [9]
    out = capsys.readouterr().out
    assert "median 2.000s" in out
    assert "Saved benchmark" in out

    make_request("--repeat", "3", "--save", "base")
    assert "Overwrote benchmark" in capsys.readouterr().out
    assert [run["name"] for run in app_bench.saved(session.name)] == ["base"]


@pytest.mark.parametrize(
    "statuses,truncated", [([200, 500], False), ([200, 200], True)]
)
def test_failed_or_truncated_benchmarks_are_not_saved(
    home, monkeypatch, capsys, statuses, truncated
):
    make_session().save()
    _fake_box(monkeypatch, statuses, truncated=truncated)
    with pytest.raises(SystemExit):
        make_request("--repeat", "3", "--save", "base")
    out = capsys.readouterr().out
    assert "Not saved" in out
    assert app_bench.saved("dev-demo-me") == []


def test_request_flags_are_bounded(capsys):
    with pytest.raises(FalParserExit):
        parse_args(["dev", "app", "request", "/", "--repeat", "0"])
    with pytest.raises(FalParserExit):
        parse_args(["dev", "app", "request", "/", "--repeat", "abc"])
    assert "invalid integer 'abc'" in capsys.readouterr().err
    args = parse_args(["dev", "app", "benchmark", "compare", "a", "b"])
    assert (args.baseline, args.candidate) == ("a", "b")
