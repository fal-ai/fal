from __future__ import annotations

import sys
from functools import partial

import pytest

from fal import function
from fal.flags import GRPC_HOST
from fal.sdk import get_default_credentials

print("TARGET:", GRPC_HOST, file=sys.stderr)
print("AUTH:", get_default_credentials(), file=sys.stderr)


@pytest.fixture(scope="function")
def isolated_client():
    return partial(function, machine_type="XS", keep_alive=0)
