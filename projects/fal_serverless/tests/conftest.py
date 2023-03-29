from __future__ import annotations

from functools import partial

import pytest
from fal_serverless import isolated


@pytest.fixture(scope="function")
def isolated_client():
    return partial(isolated, keep_alive=0)
