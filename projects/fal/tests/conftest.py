from __future__ import annotations

import os
import uuid
from functools import partial
from pathlib import Path
from typing import Callable

import pytest


def pytest_configure(config):
    unit_directory = Path(__file__).parent / "unit"
    paths = [Path(arg.split("::", 1)[0]).resolve() for arg in config.args]
    if paths and all(
        path == unit_directory or unit_directory in path.parents for path in paths
    ):
        # Unit collection imports fal before any autouse fixtures can run.
        patch = pytest.MonkeyPatch()
        patch.setenv("FAL_CONFIG_PATH", os.devnull)
        patch.delenv("FAL_PROFILE", raising=False)
        config.add_cleanup(patch.undo)


def pytest_report_header():
    from fal.flags import GRPC_HOST
    from fal.sdk import get_credentials

    return [f"TARGET: {GRPC_HOST}", f"AUTH: {get_credentials()}"]


@pytest.fixture(scope="function")
def isolated_client():
    from fal import function

    return partial(function, machine_type="XS", keep_alive=0)


@pytest.fixture(scope="session")
def make_tmp_app_name() -> Callable[[str], str]:
    def _make_tmp_app_name(prefix: str = "test") -> str:
        short_id = uuid.uuid4().hex[:8]
        return f"{prefix or 'test'}-{short_id}"

    return _make_tmp_app_name
