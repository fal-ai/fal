from pathlib import Path

import pytest

# Every test deploys a real app, and cold starts have been observed to exceed the
# 60s default from pyproject.toml on their own. Tests with their own
# @pytest.mark.timeout keep it untouched.
DEFAULT_REMOTE_TIMEOUT = 180

_E2E_DIR = Path(__file__).parent


def pytest_collection_modifyitems(items):
    for item in items:
        # The hook receives every collected item, not just this directory's.
        if _E2E_DIR not in Path(str(item.fspath)).parents:
            continue
        if item.get_closest_marker("timeout") is None:
            item.add_marker(pytest.mark.timeout(DEFAULT_REMOTE_TIMEOUT))
