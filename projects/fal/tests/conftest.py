from __future__ import annotations

from functools import partial

import pytest
from fal import function


@pytest.fixture(scope="function")
def isolated_client():
    return partial(function, machine_type="XS", keep_alive=0)
