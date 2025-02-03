from __future__ import annotations

import functools
import time
import warnings

import click
import httpx

from fal.auth.local import ActiveAuthConfig, UserInfo, open_browser
from fal.console import console
from fal.console.icons import CHECK_ICON

WEBSITE_URL = "https://fal.ai"

AUTH0_DOMAIN = "auth.fal.ai"
AUTH0_JWKS_URL = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"
AUTH0_ALGORITHMS = ["RS256"]
AUTH0_ISSUER = f"https://{AUTH0_DOMAIN}/"
AUTH0_FAL_API_AUDIENCE_ID = "fal-cloud"
AUTH0_CLIENT_ID = "TwXR51Vz8JbY8GUUMy6EyuVR0fTO7N4N"
AUTH0_SCOPE = "openid profile email offline_access"


def logout_url(return_url: str):
    return f"https://{AUTH0_DOMAIN}/v2/logout?client_id={AUTH0_CLIENT_ID}&returnTo={return_url}"


def login() -> ActiveAuthConfig:
    """
    Runs the device authorization flow and returns the auth config
    """
    device_code_payload = {
        "audience": AUTH0_FAL_API_AUDIENCE_ID,
        "client_id": AUTH0_CLIENT_ID,
        "scope": AUTH0_SCOPE,
    }
    device_code_response = httpx.post(
        f"https://{AUTH0_DOMAIN}/oauth/device/code", data=device_code_payload
    )

    if device_code_response.status_code != 200:
        raise click.ClickException("Error generating the device code")

    device_code_data = device_code_response.json()
    device_user_code = device_code_data["user_code"]
    device_confirmation_url = device_code_data["verification_uri_complete"]

    url = logout_url(device_confirmation_url)

    open_browser(url, device_user_code)

    # This is needed to suppress the ResourceWarning emitted
    # when the process is waiting for user confirmation
    warnings.filterwarnings("ignore", category=ResourceWarning)

    token_payload = {
        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        "device_code": device_code_data["device_code"],
        "client_id": AUTH0_CLIENT_ID,
    }

    with console.status("Waiting for confirmation...") as status:
        while True:
            token_response = httpx.post(
                f"https://{AUTH0_DOMAIN}/oauth/token", data=token_payload
            )

            token_data = token_response.json()
            if token_response.status_code == 200:
                status.update(spinner=None)
                console.print(f"{CHECK_ICON} Authenticated successfully, welcome!")

                access_token = token_data["id_token"]

                validate_id_token(access_token)

                return ActiveAuthConfig(
                    provider="auth0",
                    refresh_token=token_data["refresh_token"],
                    access_token=access_token,
                )

            elif token_data["error"] not in ("authorization_pending", "slow_down"):
                status.update(spinner=None)
                raise click.ClickException(token_data["error_description"])

            else:
                time.sleep(device_code_data["interval"])


def refresh(token: str) -> ActiveAuthConfig:
    token_payload = {
        "grant_type": "refresh_token",
        "client_id": AUTH0_CLIENT_ID,
        "refresh_token": token,
    }

    token_response = httpx.post(
        f"https://{AUTH0_DOMAIN}/oauth/token", data=token_payload
    )

    token_data = token_response.json()
    if token_response.status_code == 200:
        validate_id_token(token_data["id_token"])

        return ActiveAuthConfig(
            provider="auth0",
            refresh_token=token_data["refresh_token"],
            access_token=token_data["id_token"],
        )
    else:
        raise click.ClickException(token_data["error_description"])


def revoke(token: str) -> None:
    token_payload = {
        "client_id": AUTH0_CLIENT_ID,
        "token": token,
    }

    token_response = httpx.post(
        f"https://{AUTH0_DOMAIN}/oauth/revoke", data=token_payload
    )

    if token_response.status_code != 200:
        token_data = token_response.json()
        raise click.ClickException(token_data["error_description"])

    open_browser(logout_url(WEBSITE_URL), None)


def get_user_info(bearer_token: str) -> UserInfo:
    userinfo_response = httpx.post(
        f"https://{AUTH0_DOMAIN}/userinfo",
        headers={"Authorization": bearer_token},
    )

    if userinfo_response.status_code != 200:
        raise click.ClickException(userinfo_response.content.decode("utf-8"))

    data = userinfo_response.json()

    return UserInfo(
        name=data["name"],
        id=data["sub"],
    )


@functools.lru_cache
def build_jwk_client():
    from jwt import PyJWKClient

    return PyJWKClient(AUTH0_JWKS_URL, cache_keys=True)


def validate_id_token(token: str):
    """
    id_token is intended for the client (this sdk) only.
    Never send one to another service.
    """
    from jwt import decode

    jwk_client = build_jwk_client()

    decode(
        token,
        key=jwk_client.get_signing_key_from_jwt(token).key,
        algorithms=AUTH0_ALGORITHMS,
        issuer=AUTH0_ISSUER,
        audience=AUTH0_CLIENT_ID,
        leeway=60,  # 1 minute, to account for clock skew
        options={
            "verify_signature": True,
            "verify_exp": True,
            "verify_iat": True,
            "verify_aud": True,
            "verify_iss": True,
        },
    )
