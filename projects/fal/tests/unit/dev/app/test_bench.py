import json

import pytest

from dev.app import fakes
from dev.app.fakes import make_request, make_session
from fal.dev.app import bench as app_bench
from fal.dev.errors import DevError

home = fakes.home

pytestmark = pytest.mark.usefixtures("dev_account")


def test_stats_use_nearest_rank_p90():
    values = [float(n) for n in range(1, 11)]
    result = app_bench.stats(values)
    assert result["median"] == 5.5
    assert result["p90"] == 9.0
    assert (result["min"], result["max"]) == (1.0, 10.0)
    assert app_bench.stats([3.0])["p90"] == 3.0
    assert app_bench.percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.9) == 5.0


def test_save_load_and_compare(home):
    session = make_session()

    def run(name, runs, manifest):
        result = {
            "runs": [{"elapsed_seconds": value} for value in runs],
            "warmup_runs": [{"elapsed_seconds": 5.0}],
        }
        record = app_bench.record(
            name=name,
            session=make_session(manifest=manifest),
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


def test_empty_benchmark_name_is_refused(home, capsys):
    make_session().save()
    # The command reports the DevError (see fal.dev.cli.error_panel) and exits 1.
    assert make_request("--save", "") == 1
    assert "Error:" in capsys.readouterr().err


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
