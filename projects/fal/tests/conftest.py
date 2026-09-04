from __future__ import annotations

import sys
import uuid
from functools import partial
from typing import Callable

import pytest

from fal import function
from fal.flags import GRPC_HOST
from fal.sdk import get_credentials

print("TARGET:", GRPC_HOST, file=sys.stderr)
print("AUTH:", get_credentials(), file=sys.stderr)

# Suites that deploy to the platform and wait on cold starts; the package
# default timeout is sized for unit tests.
REMOTE_SUITES = {"e2e", "integration"}
REMOTE_SUITE_TIMEOUT = 300


def pytest_collection_modifyitems(items):
    for item in items:
        if item.get_closest_marker("timeout"):
            continue
        if REMOTE_SUITES & set(item.path.parts):
            item.add_marker(pytest.mark.timeout(REMOTE_SUITE_TIMEOUT))


@pytest.fixture(scope="function")
def isolated_client():
    return partial(function, machine_type="XS", keep_alive=0)


@pytest.fixture(scope="function")
def make_tmp_app_name() -> Callable[[str], str]:
    def _make_tmp_app_name(prefix: str = "test") -> str:
        short_id = uuid.uuid4().hex[:8]
        return f"{prefix or 'test'}-{short_id}"

    return _make_tmp_app_name
