import functools
import hashlib
import os
from base64 import urlsafe_b64encode
from http.server import HTTPServer, SimpleHTTPRequestHandler
from threading import Event, Thread
from typing import Tuple, Union
from urllib.parse import parse_qs, urlencode, urlparse

import click
import httpx

from fal.auth.local import ActiveAuthConfig, UserInfo, open_browser
from fal.console import console
from fal.console.icons import CHECK_ICON

# We need a port that's unlikely to be used by the user
PORT = 43781

WORKOS_CLIENT_ID = "client_01JCPB2MYS0GR84MZ8NJE7A3XW"
WORKOS_DOMAIN = "api.workos.com"
WORKOS_JWKS_URL = f"https://{WORKOS_DOMAIN}/sso/jwks/{WORKOS_CLIENT_ID}"
WORKOS_ALGORITHMS = ["RS256"]
WORKOS_ISSUER = "https://session.fal.ai"

REDIRECT_PATH = "/api/auth/v2/callback"
REDIRECT_URI = f"http://127.0.0.1:3000{REDIRECT_PATH}"

# Code verifier needs to be a string compatible with base64 (without padding),
# which is then hashed to the challenge
# Using the underlying bytes and hashing them for the challenge doesn't work
code_verifier_bytes = os.urandom(96)
code_verifier = urlsafe_b64encode(code_verifier_bytes).decode("utf8").rstrip("=")

code_challenge_bytes = hashlib.sha256(code_verifier.encode("utf8")).digest()
code_challenge = urlsafe_b64encode(code_challenge_bytes).decode("utf8").rstrip("=")

params = {
    "client_id": WORKOS_CLIENT_ID,
    "redirect_uri": REDIRECT_URI,
    "response_type": "code",
    "code_challenge": code_challenge,
    "code_challenge_method": "S256",
    "provider": "GitHubOAuth",
}
authorization_url = (
    "https://api.workos.com/user_management/authorize" + "?" + urlencode(params)
)

TOKEN_FILE = "workos_token"


class WorkOSHandler(SimpleHTTPRequestHandler):
    auth_event: Union[Event, None] = None
    refresh_token: Union[str, None] = None
    access_token: Union[str, None] = None
    error_message: Union[str, None] = None

    def do_GET(self):
        try:
            self.process_get()
        except Exception as e:
            self.return_bad_request_error(f"Error authenticating: {e}")

    def process_get(self):
        if not self.path.startswith(REDIRECT_PATH):
            # Ignore as browser may be asking for favicon and other stuff
            self.send_response(404)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        # Parse the URL and get the 'code' parameter
        parsed_url = urlparse(self.path)
        params = parse_qs(parsed_url.query)
        code_params = params.get("code")
        code = code_params[0] if code_params else None

        if not code:
            self.return_bad_request_error("No authorization code received")
            return

        with httpx.Client() as client:
            token_response = client.post(
                "https://api.workos.com/user_management/authenticate",
                data={
                    "client_id": WORKOS_CLIENT_ID,
                    "code": code,
                    "grant_type": "authorization_code",
                    "code_verifier": code_verifier,
                },
            )

            if token_response.status_code != 200:
                self.return_bad_request_error(
                    "Error getting authentication token: "
                    + f"{token_response.status_code} {token_response.text}"
                )
                return

            auth_response = token_response.json()

            access_token = auth_response["access_token"]
            refresh_token = auth_response["refresh_token"]

            try:
                check_accesss_token_and_get_session(access_token, verify=True)
            except Exception as e:
                self.return_bad_request_error(
                    f"Error validating the obtained credentials: {e}"
                )
                return

            WorkOSHandler.refresh_token = refresh_token
            WorkOSHandler.access_token = access_token

            if WorkOSHandler.auth_event:
                WorkOSHandler.auth_event.set()

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"Success!")

    def return_bad_request_error(self, message: str):
        self.send_response(400)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(message.encode())

        WorkOSHandler.error_message = message
        if WorkOSHandler.auth_event:
            WorkOSHandler.auth_event.set()

    def log_message(self, format, *args):
        # Don't emit logs
        pass


def run_server(auth_event: Event):
    server_address = ("127.0.0.1", PORT)
    WorkOSHandler.auth_event = auth_event

    try:
        httpd = HTTPServer(server_address, WorkOSHandler)
        open_browser(authorization_url, None)

        # Keep server up until auth is complete
        while not auth_event.is_set():
            httpd.handle_request()

        httpd.server_close()
    except OSError:
        WorkOSHandler.error_message = (
            f"Failed to start authentication server on port {PORT} as it's already in "
            "use. Please ensure no other applications are using this port."
        )
        auth_event.set()


def login() -> ActiveAuthConfig:
    auth_event = Event()

    server_thread = Thread(target=run_server, args=(auth_event,), daemon=True)
    server_thread.start()

    auth_event.wait()

    if WorkOSHandler.error_message:
        raise click.ClickException(WorkOSHandler.error_message)

    if not WorkOSHandler.refresh_token:
        raise click.ClickException(
            "Missing refresh token. This is a bug, please let the Fal team know"
        )

    if not WorkOSHandler.access_token:
        raise click.ClickException(
            "Missing access token. This is a bug, please let the Fal team know"
        )

    console.print(f"{CHECK_ICON} Authenticated successfully, welcome!")

    return ActiveAuthConfig(
        provider="workos",
        refresh_token=WorkOSHandler.refresh_token,
        access_token=WorkOSHandler.access_token,
    )


def refresh(refresh_token: str) -> Tuple[ActiveAuthConfig, UserInfo]:
    """
    WorkOS doesn't have an API for user detils without a client secret, we can only
    get the data when refreshing the token, so we send back that info
    """
    token_payload = {
        "client_id": WORKOS_CLIENT_ID,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }

    token_response = httpx.post(
        f"https://{WORKOS_DOMAIN}/user_management/authenticate", data=token_payload
    )

    token_data = token_response.json()
    if token_response.status_code == 200:
        access_token = token_data["access_token"]
        check_accesss_token_and_get_session(access_token, verify=True)

        user = token_data.get("user", {})
        user_name = (
            user.get("first_name", "") + " " + user.get("last_name", "")
        ).strip()
        user_id = user.get("id", "")

        return ActiveAuthConfig(
            provider="workos",
            refresh_token=token_data["refresh_token"],
            access_token=access_token,
        ), UserInfo(
            name=user_name,
            id=user_id,
        )
    else:
        raise click.ClickException(token_data["error_description"])


def revoke(access_token: str):
    session_id = check_accesss_token_and_get_session(access_token, verify=False)

    open_browser(
        f"https://{WORKOS_DOMAIN}/user_management/sessions/logout?session_id={session_id}",
        None,
    )


@functools.lru_cache
def build_jwk_client():
    from jwt import PyJWKClient

    return PyJWKClient(WORKOS_JWKS_URL, cache_keys=True)


def check_accesss_token_and_get_session(token: str, verify: bool) -> str:
    """
    id_token is intended for the client (this sdk) only.
    Never send one to another service.
    """
    from jwt import decode

    jwk_client = build_jwk_client()

    res = decode(
        token,
        key=jwk_client.get_signing_key_from_jwt(token).key,
        algorithms=WORKOS_ALGORITHMS,
        issuer=WORKOS_ISSUER,
        audience=WORKOS_CLIENT_ID,
        leeway=60,  # 1 minute, to account for clock skew
        options={
            "verify_signature": verify,
            "verify_exp": verify,
            "verify_iat": verify,
            # Doesn't work for WorkOS
            "verify_aud": False,
            "verify_iss": verify,
        },
    )

    return res["sid"]
