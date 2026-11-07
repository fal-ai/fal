"""The dev boxes of an account: what `fal dev list` shows."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from fal.api.client import SyncServerlessClient
from fal.dev import config
from fal.dev.account import Account
from fal.dev.box import LIVE_STATES, Box, derive_state

# Boxes whose holds are read at once; each read is a command on the box.
HOLD_WORKERS = 8


@dataclass(frozen=True)
class BoxRow:
    name: str
    app: str
    # RUNNING, STARTING, STOPPING, FAILED, STOPPED, or UNKNOWN (runners unreadable)
    state: str
    machine_type: str | None
    uptime_seconds: float | None
    hold_until: float | None
    # False when the hold could not be read, so "no hold" is not certain.
    hold_known: bool
    # The box's own volume; boxes deployed with --team-data do not use it.
    volume: str

    def as_dict(self) -> dict:
        return asdict(self)


def _row(app, runners: list | None) -> BoxRow:
    name = app.alias[len(config.APP_PREFIX) :]
    if runners is None:
        state, runner = "UNKNOWN", None
    else:
        code, runner = derive_state(app, runners)
        state = code.upper()
    machine = runner.machine_type if runner else None
    if machine is None and app.machine_types:
        machine = app.machine_types[0]
    up = runner is not None and runner.state in LIVE_STATES
    return BoxRow(
        name=name,
        app=app.alias,
        state=state,
        machine_type=machine,
        uptime_seconds=runner.uptime.total_seconds() if up else None,
        hold_until=None,
        # Only a running box's hold is read; for the others, "no hold" is a guess.
        hold_known=state not in {"UNKNOWN", "RUNNING"},
        volume=config.volume_name(name),
    )


def _runners(client: SyncServerlessClient, app) -> list | None:
    """The app's runners; None when they could not be read."""
    # Only boxes that are armed or still have a runner can have one up.
    if not (app.min_concurrency or app.active_runners):
        return []
    try:
        return client.apps.runners(app.alias)
    except Exception:
        return None


def _hold(box: Box) -> tuple[float | None, bool]:
    """The hold's end and whether it could be read; best effort."""
    try:
        return box.hold_until(), True
    except Exception:
        return None, False


def list_boxes(
    client: SyncServerlessClient,
    account: Account,
    host: str | None,
    *,
    holds: bool = True,
) -> list[BoxRow]:
    """Every `devbox-*` app of the account, by name.

    One app listing for all boxes, then a runner lookup only for boxes that are
    up or starting, and, for running ones, a read of their hold.
    """
    apps = sorted(
        (a for a in client.apps.list() if a.alias.startswith(config.APP_PREFIX)),
        key=lambda a: a.alias,
    )
    rows = [_row(app, _runners(client, app)) for app in apps]
    running = [i for i, row in enumerate(rows) if row.state == "RUNNING"]
    if holds and running:
        boxes = [Box(rows[i].name, host=host, account=account) for i in running]
        with ThreadPoolExecutor(max_workers=HOLD_WORKERS) as pool:
            for i, (until, known) in zip(running, pool.map(_hold, boxes)):
                rows[i] = BoxRow(
                    **{**rows[i].as_dict(), "hold_until": until, "hold_known": known}
                )
    return rows


def hold_left(row: BoxRow, now: float | None = None) -> float | None:
    """Seconds of hold left, None when there is no hold."""
    if row.hold_until is None:
        return None
    left = row.hold_until - (time.time() if now is None else now)
    return left if left > 0 else None
