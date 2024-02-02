from __future__ import annotations

from openapi_fal_rest.client import Client

import fal.flags as flags
from fal.sdk import get_default_credentials


class CredentialsClient(Client):
    def get_headers(self) -> dict[str, str]:
        creds = get_default_credentials()
        return {
            **creds.to_headers(),
            **self.headers,
        }


# TODO: accept more auth methods
REST_CLIENT = CredentialsClient(
    flags.REST_URL,
    timeout=30,
    verify_ssl=not flags.TEST_MODE,
    raise_on_unexpected_status=False,
    follow_redirects=True,
)
