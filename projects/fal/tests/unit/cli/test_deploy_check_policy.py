"""The admin deploy-check policy lookup fails closed.

Before this fix, a non-200 status or an empty payload from the users
endpoint returned False ("no check required"), so a team that requires
deploy confirmation could deploy without it whenever that call failed.
Only a raised exception failed closed.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from fal.cli.deploy_check import _admin_requires_deploy_check

LOOKUP = "openapi_fal_rest.api.users.get_current_user.sync_detailed"


def lookup(response=None, error=None):
    client = MagicMock()
    with patch(LOOKUP) as sync_detailed:
        if error is not None:
            sync_detailed.side_effect = error
        else:
            sync_detailed.return_value = response
        return _admin_requires_deploy_check(client)


def test_policy_on_requires_check():
    response = SimpleNamespace(
        status_code=200, parsed=SimpleNamespace(deploy_check=True)
    )
    assert lookup(response) is True


def test_policy_off_does_not_require_check():
    response = SimpleNamespace(
        status_code=200, parsed=SimpleNamespace(deploy_check=False)
    )
    assert lookup(response) is False


@pytest.mark.parametrize("status", [404, 500])
def test_non_200_requires_check(status):
    assert lookup(SimpleNamespace(status_code=status, parsed=None)) is True


def test_empty_payload_requires_check():
    assert lookup(SimpleNamespace(status_code=200, parsed=None)) is True


def test_exception_requires_check():
    assert lookup(error=httpx.ConnectError("boom")) is True
