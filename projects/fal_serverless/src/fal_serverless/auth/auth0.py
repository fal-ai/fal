from __future__ import annotations

import time
import warnings

import click
import requests
from auth0.v3.authentication.token_verifier import (
    AsymmetricSignatureVerifier,
    TokenVerifier,
)
from fal_serverless.console import console
from fal_serverless.console.icons import CHECK_ICON
from fal_serverless.console.ux import get_browser
from rich.prompt import Confirm

AUTH0_DOMAIN = "auth.fal.ai"
AUTH0_JWKS_URL = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"
AUTH0_ALGORITHMS = ["RS256"]
AUTH0_ISSUER = f"https://{AUTH0_DOMAIN}/"
AUTH0_FAL_API_AUDIENCE_ID = "fal-cloud"
AUTH0_CLIENT_ID = "TwXR51Vz8JbY8GUUMy6EyuVR0fTO7N4N"
AUTH0_SCOPE = "openid profile email offline_access"


def login() -> dict:
    """
    Runs the device authorization flow and stores the user object in memory
    """
    device_code_payload = {
        "audience": AUTH0_FAL_API_AUDIENCE_ID,
        "client_id": AUTH0_CLIENT_ID,
        "scope": AUTH0_SCOPE,
    }
    device_code_response = requests.post(
        f"https://{AUTH0_DOMAIN}/oauth/device/code", data=device_code_payload
    )

    if device_code_response.status_code != 200:
        raise click.ClickException("Error generating the device code")

    device_code_data = device_code_response.json()
    device_user_code = device_code_data["user_code"]
    device_confirmation_url = device_code_data["verification_uri_complete"]

    browser = get_browser()
    console.print()
    if browser is None:
        console.print(
            f"1. On your computer or mobile device navigate to: {device_confirmation_url}"
        )
        console.print(
            f"2. Confirm it shows the following code: [markdown.code]{device_user_code}[/]\n"
        )
    else:
        console.print(
            f"Once the page loads, confirm it shows the following code: [markdown.code]{device_user_code}[/]"
        )
        Confirm.get_input(
            console=console,
            prompt="A browser will open with the login page when you [bold]Press Enter...[/]",
            password=True,
        )
        # This is needed to suppress the ResourceWarning emitted
        # when the process is waiting for user confirmation
        warnings.filterwarnings("ignore", category=ResourceWarning)
        browser.open_new_tab(device_confirmation_url)

    token_payload = {
        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        "device_code": device_code_data["device_code"],
        "client_id": AUTH0_CLIENT_ID,
    }

    with console.status("Waiting for confirmation...") as status:
        while True:
            token_response = requests.post(
                f"https://{AUTH0_DOMAIN}/oauth/token", data=token_payload
            )

            token_data = token_response.json()
            if token_response.status_code == 200:
                status.update(spinner=None)
                console.print(f"{CHECK_ICON} Authenticated successfully, welcome!")

                validate_id_token(token_data["id_token"])

                return token_data

            elif token_data["error"] not in ("authorization_pending", "slow_down"):
                status.update(spinner=None)
                raise click.ClickException(token_data["error_description"])

            else:
                time.sleep(device_code_data["interval"])


def refresh(token: str) -> dict:
    token_payload = {
        "grant_type": "refresh_token",
        "client_id": AUTH0_CLIENT_ID,
        "refresh_token": token,
    }

    token_response = requests.post(
        f"https://{AUTH0_DOMAIN}/oauth/token", data=token_payload
    )

    token_data = token_response.json()
    if token_response.status_code == 200:
        # DEBUG: print("Authenticated!")

        validate_id_token(token_data["id_token"])

        return token_data
    else:
        raise click.ClickException(token_data["error_description"])


def revoke(token: str):
    token_payload = {
        "client_id": AUTH0_CLIENT_ID,
        "token": token,
    }

    token_response = requests.post(
        f"https://{AUTH0_DOMAIN}/oauth/revoke", data=token_payload
    )

    if token_response.status_code != 200:
        token_data = token_response.json()
        raise click.ClickException(token_data["error_description"])


def get_user_info(bearer_token: str) -> dict:
    userinfo_response = requests.post(
        f"https://{AUTH0_DOMAIN}/userinfo",
        headers={"Authorization": bearer_token},
    )

    if userinfo_response.status_code != 200:
        raise click.ClickException(userinfo_response.content.decode("utf-8"))

    return userinfo_response.json()


def validate_id_token(token: str):
    """
    Verify the token and its precedence.
    `id_token`s are intended for the client (this sdk) only.
    Never send one to another service.

    :param id_token:
    """
    sv = AsymmetricSignatureVerifier(AUTH0_JWKS_URL)
    tv = TokenVerifier(
        signature_verifier=sv,
        issuer=AUTH0_ISSUER,
        audience=AUTH0_CLIENT_ID,
    )
    tv.verify(token)


def validate_access_token(token: str):
    from datetime import timedelta

    from jwt import decode

    decode(
        token,
        leeway=timedelta(minutes=-30),  # Mark as expired some time before it expires
        options={"verify_exp": True, "verify_signature": False},
    )
