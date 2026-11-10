import json
import time
from unittest.mock import MagicMock

import pytest

from fal.dev import config, inventory
from fal.dev.account import Account
from fal.sdk import RunnerState
from tests.unit.dev.fakes import make_box, make_runner

ACCOUNT = Account("my-team", False)


def _app(alias, min_concurrency=0, active_runners=0, machine="GPU-H100"):
    app = MagicMock(
        alias=alias, min_concurrency=min_concurrency, active_runners=active_runners
    )
    app.machine_types = [machine]
    return app


def _client(apps, runners=None):
    client = MagicMock()
    client.apps.list.return_value = apps
    runners = runners or {}
    client.apps.runners.side_effect = lambda alias, **_: runners.get(alias, [])
    return client


def test_lists_only_devbox_apps_sorted():
    client = _client([_app("zzz"), _app("devbox-b"), _app("dev-x"), _app("devbox-a")])

    rows = inventory.list_boxes(client, ACCOUNT, None)

    assert [r.name for r in rows] == ["a", "b"]
    assert {r.state for r in rows} == {"STOPPED"}
    client.apps.runners.assert_not_called()


def test_states_and_uptime():
    live = make_runner("r1", uptime=125, machine="GPU-B200")
    client = _client(
        [
            _app("devbox-up", min_concurrency=1, active_runners=1),
            _app("devbox-wait", min_concurrency=1),
            _app("devbox-down", active_runners=1),
        ],
        {
            "devbox-up": [live],
            "devbox-wait": [make_runner("r2", RunnerState.PENDING)],
            "devbox-down": [make_runner("r3")],
        },
    )

    rows = {r.name: r for r in inventory.list_boxes(client, ACCOUNT, None, holds=False)}

    assert rows["up"].state == "RUNNING"
    assert rows["up"].machine_type == "GPU-B200"
    assert rows["up"].uptime_seconds == 125
    assert rows["wait"].state == "STARTING"
    assert rows["wait"].uptime_seconds is None
    assert rows["down"].state == "STOPPING"


def test_volume_name_follows_the_box_name():
    long = "a" * 30
    client = _client([_app(f"devbox-{long}")])

    (row,) = inventory.list_boxes(client, ACCOUNT, None)

    assert row.volume == config.volume_name(long)
    assert len(row.volume) <= config.VOLUME_NAME_MAX


def test_holds_are_read_for_running_boxes_only(monkeypatch):
    asked = []

    def hold_until(box):
        asked.append(box.name)
        return time.time() + 600, True

    monkeypatch.setattr(inventory, "_hold", hold_until)
    client = _client(
        [
            _app("devbox-up", min_concurrency=1),
            _app("devbox-off"),
        ],
        {"devbox-up": [make_runner("r1")]},
    )

    rows = {r.name: r for r in inventory.list_boxes(client, ACCOUNT, None)}

    assert asked == ["up"]
    assert inventory.hold_left(rows["up"]) > 0
    assert inventory.hold_left(rows["off"]) is None


def test_a_failed_hold_read_is_marked_unknown():
    box = MagicMock()
    box.hold_until.side_effect = RuntimeError("boom")

    assert inventory._hold(box) == (None, False)


def test_a_failed_runner_lookup_is_unknown_not_stopped():
    client = _client([_app("devbox-a", min_concurrency=1)])
    client.apps.runners.side_effect = RuntimeError("blip")

    (row,) = inventory.list_boxes(client, ACCOUNT, None)

    assert row.state == "UNKNOWN"


def test_expired_hold_is_not_a_hold():
    row = inventory.BoxRow(
        "a", "devbox-a", "RUNNING", None, None, 100.0, True, "devbox-a"
    )

    assert inventory.hold_left(row, now=200.0) is None
    assert inventory.hold_left(row, now=40.0) == 60.0


def test_rows_serialize_to_json():
    client = _client([_app("devbox-a")])

    (row,) = inventory.list_boxes(client, ACCOUNT, None)

    assert json.loads(json.dumps(row.as_dict()))["name"] == "a"


def test_a_crash_looping_box_is_failed_and_dead_runners_are_ignored():
    client = _client(
        [
            _app("devbox-bad", min_concurrency=1),
            _app("devbox-old", min_concurrency=1),
        ],
        {
            "devbox-bad": [make_runner("r1", RunnerState.CRASH_BACKOFF)],
            "devbox-old": [make_runner("r2", RunnerState.DEAD)],
        },
    )

    rows = {r.name: r.state for r in inventory.list_boxes(client, ACCOUNT, None)}

    assert rows == {"bad": "FAILED", "old": "STOPPED"}


def test_a_hold_is_known_only_where_it_was_read_or_cannot_exist():
    client = _client([_app("devbox-a", min_concurrency=1), _app("devbox-b")])
    client.apps.runners.side_effect = RuntimeError("blip")

    rows = {r.name: r for r in inventory.list_boxes(client, ACCOUNT, None)}

    assert rows["a"].state == "UNKNOWN"
    assert not rows["a"].hold_known
    assert rows["b"].state == "STOPPED"
    assert rows["b"].hold_known


CASES = {
    "up": (1, [("r1", RunnerState.RUNNING)]),
    "idle-up": (1, [("r1", RunnerState.IDLE)]),
    "armed-no-runners": (1, []),
    "armed-only-dead": (1, [("r1", RunnerState.DEAD)]),
    "starting": (1, [("r1", RunnerState.PENDING)]),
    "setup": (0, [("r1", RunnerState.SETUP)]),
    "crash": (1, [("r1", RunnerState.CRASH_BACKOFF)]),
    "draining": (0, [("r1", RunnerState.DRAINING)]),
    "unarmed-live": (0, [("r1", RunnerState.RUNNING)]),
    "stopped": (0, []),
}


@pytest.mark.parametrize("case", CASES)
def test_list_and_status_agree_on_every_state(case):
    armed, specs = CASES[case]
    runners = [make_runner(i, st) for i, st in specs]
    alias = f"devbox-{case}"
    client = _client(
        [_app(alias, min_concurrency=armed, active_runners=len(runners))],
        {alias: runners},
    )
    (row,) = inventory.list_boxes(client, ACCOUNT, None, holds=False)

    box = make_box(runners, min_concurrency=armed)
    box.client.apps.list.return_value = client.apps.list.return_value[:1]
    box.app = alias

    assert row.state.lower() == box.state().code
