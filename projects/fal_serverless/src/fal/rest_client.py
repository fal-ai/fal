from __future__ import annotations

import fal.flags as flags
from fal.sdk import get_default_credentials
from openapi_fal_rest.client import AuthenticatedClient


class CredentialsClient(AuthenticatedClient):
    def __init__(
        self,
        base_url: str,
        *,
        # defaults come from openapi_fal_rest.Client
        timeout: float = 5,
        verify_ssl: bool = True,
        raise_on_unexpected_status: bool = False,
        follow_redirects: bool = False,
    ):
        super().__init__(
            base_url,
            token="",  # token will be ignored, but required by the constructor
            timeout=timeout,
            verify_ssl=verify_ssl,
            raise_on_unexpected_status=raise_on_unexpected_status,
            follow_redirects=follow_redirects,
        )

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
