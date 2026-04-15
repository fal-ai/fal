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


@pytest.fixture(scope="function")
def isolated_client():
    return partial(function, machine_type="XS", keep_alive=0)


@pytest.fixture(scope="function")
def make_tmp_app_name() -> Callable[[str], str]:
    def _make_tmp_app_name(prefix: str = "test") -> str:
        short_id = uuid.uuid4().hex[:8]
        return f"{prefix or 'test'}-{short_id}"

    return _make_tmp_app_name
