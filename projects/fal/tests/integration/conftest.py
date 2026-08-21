from pathlib import Path

import pytest

# Every test here provisions a real remote environment, so the 60s default from
# pyproject.toml has to cover provisioning, execution and assertions at once and
# routinely runs out during provisioning alone. Give the whole directory a
# safety net instead; tests needing a different budget set their own
# @pytest.mark.timeout, which is left untouched.
DEFAULT_REMOTE_TIMEOUT = 180

_INTEGRATION_DIR = Path(__file__).parent


def pytest_collection_modifyitems(items):
    for _item in items:
        # The hook receives every collected item, not just this directory's.
        if _INTEGRATION_DIR not in Path(str(_item.fspath)).parents:
            continue
        if _item.get_closest_marker("timeout") is None:
            _item.add_marker(pytest.mark.timeout(DEFAULT_REMOTE_TIMEOUT))
