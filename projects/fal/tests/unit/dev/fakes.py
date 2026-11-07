from datetime import timedelta
from unittest.mock import MagicMock

from fal.dev.account import Account
from fal.dev.box import Box
from fal.sdk import RunnerState


def make_runner(runner_id, state=RunnerState.RUNNING, uptime=60, machine="GPU-H100"):
    runner = MagicMock()
    runner.runner_id = runner_id
    runner.state = state
    runner.uptime = timedelta(seconds=uptime)
    runner.machine_type = machine
    return runner


def make_box(runners=(), min_concurrency=1, deployed=True):
    box = Box("demo", account=Account("my-team", False))
    box.client = MagicMock()
    box.client.apps.runners.return_value = list(runners)
    app = MagicMock(alias="devbox-demo", min_concurrency=min_concurrency)
    app.machine_types = ["GPU-H100"]
    box.client.apps.list.return_value = [app] if deployed else []
    # Owner lookups go over REST; tests of the guard set these themselves.
    box.owner_identity = lambda: "me"
    box.stored_owner = lambda attempts=3: None
    return box
