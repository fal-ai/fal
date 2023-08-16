from __future__ import annotations

from functools import partial

import pytest
from fal import isolated


@pytest.fixture(scope="function")
def isolated_client():
    return partial(isolated, machine_type="M", keep_alive=0)
