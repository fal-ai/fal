"""Metered.ca TURN credential minting for WMA runners (SSRF-safe, cached).

fal WMA media flows *peer-to-peer* between the browser and the aiortc runner
(see ``fal/wma/_raw.py``). On a permissive network a STUN server is enough:
each side gathers a server-reflexive candidate and the media path is direct. On
a restrictive network (UDP blocked, symmetric NAT, corporate firewall) a direct
path never forms and a **TURN relay** is required. The WMA bridge now owns the
credentials: it gives the browser an ICE list and forwards the runner's
short-lived list with ``/start-session``. This module validates that forwarded
list against Metered's host boundary. Provider-neutral runner selection and
aiortc conversion live in :mod:`fal.wma.ice`.

Getting the relay *hostname* right is load-bearing. Metered serves TURN on
region-specific hosts under ``*.relay.metered.ca`` (e.g.
``global.relay.metered.ca``) — **not** on the project's ``<app>.metered.live``
API host. So this module does not guess: after minting it fetches Metered's
canonical *ICE Servers Array* (``GET /api/v1/turn/credentials?apiKey=...``),
which carries the authoritative relay URLs, and strictly validates it
(:func:`parse_metered_ice_array`). Only if that fetch fails does it fall back to
synthesising against the account-independent global relay host
(:data:`FALLBACK_RELAY_HOST`) — still a real relay, never the API host.

Security model:

* Only the **secret key** creates credentials, server-side, via
  ``POST https://<domain>/api/v1/turn/credential?secretKey=...``; the returned
  ``apiKey`` then fetches the ICE array. The secret is never logged, never
  returned to a caller, never placed in an exception message, and never
  synthesised into a URL shown to anyone — only the *derived* ephemeral
  ``username``/``credential`` pair ever reaches a peer. Every relay URL that
  reaches a peer is validated to a Metered host (:func:`_validate_ice_url`).
* ``METERED_DOMAIN`` is validated to an expected Metered HTTPS host before any
  request (:func:`sanitize_metered_domain`): a bare hostname on a short
  allow-list of Metered apex domains, no scheme/port/path/userinfo, no IP
  literal — so a mistyped or injected value can't turn this into an SSRF probe.
  The fetch itself uses a redirect-refusing opener, a strict timeout, and a
  bounded response read, so even a compromised-but-allow-listed host can't
  redirect the runner elsewhere or stream an unbounded body at it.
* New raw-path runners do not receive the account secret at all. The WMA bridge
  forwards only a short-lived ICE list. The runner validates that additive
  request field again before giving it to aiortc, because callers can invoke a
  fal endpoint directly and request bodies are never trusted merely because the
  bridge normally produced them.

Legacy runner-minted credentials are cached with a refresh margin so a busy
runner does not mint one credential per session; a fetch failure is never cached
(the prior still-valid credential is kept, and a cold cache simply raises), so a
transient Metered blip degrades to STUN-only rather than being pinned as "no
TURN".

Pure-Python + stdlib only, so it imports and unit-tests on a CPU host without
``aiortc``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Mapping, Sequence

from fal.compat import run_in_thread
from fal.wma.ice import (
    DEFAULT_STUN_URL,
    MAX_CREDENTIAL_LENGTH,
    MAX_ICE_ENTRIES,
    MAX_ICE_URL_LENGTH,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Metered apex domains a configured ``METERED_DOMAIN`` may live under. The
#: operator sets a per-project subdomain such as ``my-app.metered.live``; we only
#: ever talk to a host under one of these, which bounds the SSRF surface to
#: Metered's own infrastructure regardless of what the secret contains.
ALLOWED_METERED_APEXES: tuple[str, ...] = ("metered.live", "metered.ca")

#: Credential lifetime requested from Metered. Comfortably above a session's
#: 10-minute hard cap so a credential minted at session start stays valid for the
#: whole session; the cache refreshes well before it expires (see the margin).
DEFAULT_TURN_EXPIRY_SECONDS = 3600
#: Refresh a cached credential this long before it actually expires, so a session
#: never starts with a credential about to lapse mid-stream.
DEFAULT_REFRESH_MARGIN_SECONDS = 600
#: Absolute floor on a cache entry's validity so a tiny/misconfigured expiry can't
#: make the cache thrash (one fetch per call).
MIN_CACHE_TTL_SECONDS = 30

#: Network budget and response ceiling for the credential request. The response
#: is a tiny JSON object; anything larger is treated as hostile/broken.
HTTP_TIMEOUT_SECONDS = 10.0
MAX_RESPONSE_BYTES = 64 * 1024

#: The path we POST to (with the secret key) to *mint* a fresh ephemeral
#: credential. Its response carries ``username`` / ``password`` / ``apiKey`` —
#: but crucially **not** the relay hostnames.
_CREDENTIAL_PATH = "/api/v1/turn/credential"
#: The canonical "ICE Servers Array" path (GET, authenticated with the *apiKey*
#: returned by the mint above). This is the authoritative source of the relay
#: URLs: Metered serves TURN on region-specific hosts under ``*.relay.metered.ca``
#: (e.g. ``global.relay.metered.ca``) — **not** on the project's
#: ``<app>.metered.live`` API host. Synthesising ``turn:<app>.metered.live:…``
#: therefore points a peer at a host that runs no TURN service, so no relay
#: candidate ever gathers. We fetch this array instead of guessing hostnames.
_ICE_ARRAY_PATH = "/api/v1/turn/credentials"

#: Account-independent relay hosts used *only* as a last-resort fallback when the
#: canonical array fetch fails after a credential was successfully minted. This is
#: Metered's global relay (correct for the free/global tier and a strictly
#: better-than-STUN-only best effort otherwise) — never the ``.metered.live`` API
#: host, which is not a relay.
FALLBACK_RELAY_HOST = "global.relay.metered.ca"
FALLBACK_STUN_HOST = "stun.relay.metered.ca"
METERED_RELAY_APEX = "relay.metered.ca"

#: A conservative RFC-1123 hostname (labels 1-63 chars, no leading/trailing '-').
_HOSTNAME_RE = re.compile(
    r"^(?=.{1,253}$)([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)"
    r"(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+$"
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MeteredError(Exception):
    """Base class for Metered ICE errors (safe to surface; never carries the key)."""


class MeteredConfigError(MeteredError):
    """Static misconfiguration (bad domain, missing key); never transient."""


class MeteredFetchError(MeteredError):
    """A credential request failed (network / HTTP / bad response). Redacted."""


# ---------------------------------------------------------------------------
# Domain validation (SSRF gate)
# ---------------------------------------------------------------------------


def sanitize_metered_domain(domain: str) -> str:
    """Validate a configured ``METERED_DOMAIN`` and return a bare Metered host.

    Accepts either a bare host (``my-app.metered.live``) or an ``https://`` URL
    whose host is a Metered host; rejects everything else. The returned value is
    a lowercase hostname with no scheme, port, path, query, or userinfo, and it
    is guaranteed to sit under one of :data:`ALLOWED_METERED_APEXES`. Raises
    :class:`MeteredConfigError` (never echoing anything secret) otherwise.

    This is the SSRF gate: the credential URL is built only from the returned
    host, so an operator can never point the runner at ``169.254.169.254`` or an
    internal service even by mistake.
    """
    if not isinstance(domain, str):
        raise MeteredConfigError("METERED_DOMAIN must be a string")
    raw = domain.strip()
    if not raw:
        raise MeteredConfigError("METERED_DOMAIN is empty")

    host = raw
    if "://" in host:
        parsed = urllib.parse.urlsplit(host)
        if parsed.scheme != "https":
            raise MeteredConfigError("METERED_DOMAIN must use https")
        if parsed.username or parsed.password:
            raise MeteredConfigError("METERED_DOMAIN must not contain credentials")
        if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
            raise MeteredConfigError("METERED_DOMAIN must not contain a path or query")
        if parsed.port is not None and parsed.port != 443:
            raise MeteredConfigError("METERED_DOMAIN must not specify a non-443 port")
        host = parsed.hostname or ""
    else:
        # Bare host: still reject an embedded port/path/userinfo/whitespace.
        if any(c in host for c in "/@?# \t") or ":" in host:
            raise MeteredConfigError("METERED_DOMAIN must be a bare hostname")

    host = host.strip(".").lower()
    if not host or not _HOSTNAME_RE.match(host):
        raise MeteredConfigError("METERED_DOMAIN is not a valid hostname")
    if not any(
        host == apex or host.endswith("." + apex) for apex in ALLOWED_METERED_APEXES
    ):
        raise MeteredConfigError(
            "METERED_DOMAIN must be a Metered host "
            f"(one of {', '.join(ALLOWED_METERED_APEXES)})"
        )
    # A bare apex has no project subdomain — reject so we never hit the apex API.
    if host in ALLOWED_METERED_APEXES:
        raise MeteredConfigError("METERED_DOMAIN must include the project subdomain")
    return host


def synthesize_ice_servers(
    host: str, username: str, credential: str, *, stun_host: str | None = None
) -> list[dict[str, str]]:
    """Build a browser-shaped ICE list for a **relay** host + ephemeral creds.

    Returns entries in the standard Metered shape (``urls`` plus ``username`` /
    ``credential`` on the TURN entries). Order is browser-oriented (browsers use
    every entry): a reliable public STUN first, then the relay's STUN, then TURN
    over UDP/TCP on 80/443, and finally ``turns:`` (TLS) on 443 for the
    strict-firewall case. :func:`fal.wma.ice.ice_servers_for_aiortc`
    re-orders/prunes this for the runner, which only honours the first STUN and
    first TURN.

    ``host`` must be an actual **relay** host (e.g. ``global.relay.metered.ca``),
    *not* the ``<app>.metered.live`` API host — the API host runs no TURN service.
    This is now used only as the last-resort fallback path; the primary path
    fetches Metered's canonical ICE array (:func:`parse_metered_ice_array`), which
    carries the correct region-specific relay hostnames directly.
    """
    if not username or not credential:
        raise MeteredFetchError("Metered credential response missing username/password")
    turn = {"username": username, "credential": credential}
    stun_host = stun_host or host
    return [
        {"urls": DEFAULT_STUN_URL},
        {"urls": f"stun:{stun_host}:80"},
        {"urls": f"turn:{host}:80", **turn},
        {"urls": f"turn:{host}:80?transport=tcp", **turn},
        {"urls": f"turn:{host}:443", **turn},
        {"urls": f"turn:{host}:443?transport=tcp", **turn},
        {"urls": f"turns:{host}:443?transport=tcp", **turn},
    ]


def _validate_ice_url(url: Any) -> str:
    """Validate a single ``urls`` value from a Metered ICE array; return it as-is.

    Strict (Q7 in the task): rejects anything that is not a bounded
    ``stun|stuns|turn|turns`` URI whose host sits under an allow-listed Metered
    apex (which covers ``*.relay.metered.ca`` — it ends in ``.metered.ca``). This
    is the SSRF/only-Metered gate for the *response*: even a compromised
    allow-listed endpoint cannot get an arbitrary ``turn:`` host into the peer's
    RTCPeerConnection. Raises :class:`MeteredFetchError` (never echoing anything
    secret) on anything invalid.
    """
    if not isinstance(url, str) or not url or len(url) > MAX_ICE_URL_LENGTH:
        raise MeteredFetchError("Metered ICE array entry has an invalid url")
    scheme, _, rest = url.partition(":")
    if scheme not in ("stun", "stuns", "turn", "turns") or not rest:
        raise MeteredFetchError("Metered ICE array entry has an unsupported scheme")
    # No userinfo/path is ever valid in a STUN/TURN URI. Parse and validate the
    # optional port here so malformed relay alternatives cannot reach aiortc's
    # TURN selection and raise while otherwise valid entries are present.
    if "@" in rest or "/" in rest:
        raise MeteredFetchError("Metered ICE array url has an invalid host")
    authority, _, query = rest.partition("?")
    if query and query not in ("transport=udp", "transport=tcp"):
        raise MeteredFetchError("Metered ICE array url has an invalid transport")
    if query and scheme in ("stun", "stuns"):
        # aiortc's STUN/TURN URI parser raises on ``stun:...?transport=...``,
        # so an accepted entry would crash negotiation instead of gathering.
        raise MeteredFetchError("Metered STUN url must not specify a transport")
    if scheme == "turns" and query == "transport=udp":
        # aiortc silently discards TURNS-over-UDP, which would misreport the
        # session as TURN-capable while no relay candidate can ever gather.
        raise MeteredFetchError("Metered TURNS url must use TCP")
    if authority.count(":") > 1:
        raise MeteredFetchError("Metered ICE array url has an invalid host")
    host, separator, port = authority.rpartition(":")
    if not separator:
        host = authority
    # str.isdecimal, not isdigit: non-decimal Unicode digits pass isdigit()
    # but make int() raise, escaping the module's error contract.
    elif not port.isdecimal() or not 1 <= int(port) <= 65535:
        raise MeteredFetchError("Metered ICE array url has an invalid port")
    host = host.strip(".").lower()
    if not host or not _HOSTNAME_RE.match(host):
        raise MeteredFetchError("Metered ICE array url has an invalid host")
    if not any(
        host == apex or host.endswith("." + apex) for apex in ALLOWED_METERED_APEXES
    ):
        raise MeteredFetchError("Metered ICE array url host is not a Metered host")
    return url


def _validate_forwarded_ice_url(url: str) -> str:
    """Validate a bridge-forwarded URL against Metered's relay boundary."""
    valid_url = _validate_ice_url(url)
    rest = valid_url.partition(":")[2]
    host = re.split(r"[:?]", rest, maxsplit=1)[0].strip(".").lower()
    if not (host == METERED_RELAY_APEX or host.endswith("." + METERED_RELAY_APEX)):
        raise MeteredFetchError("Forwarded ICE url host is not a Metered relay host")
    return valid_url


def parse_metered_ice_array(
    data: Any, *, username: str, credential: str
) -> list[dict[str, str]]:
    """Strictly validate Metered's canonical ICE Servers Array into browser shape.

    ``data`` is the JSON returned by ``GET /api/v1/turn/credentials?apiKey=…`` —
    an array of ``{"urls": …, "username": …, "credential": …}`` objects carrying
    the *authoritative relay hostnames*. Every entry is validated
    (:func:`_validate_ice_url`, bounded count/sizes, Metered-only hosts); TURN/
    TURNS entries carry the freshly-minted ``username``/``credential`` (nonempty,
    bounded) rather than whatever the response echoed, so a malformed/hostile
    response can never inject empty or oversized credentials. A public STUN is
    guaranteed first so gathering always has a reflexive path.

    Raises :class:`MeteredFetchError` when the array yields no usable TURN entry —
    the caller then falls back to the corrected synthesized list.
    """
    if not username or not credential:
        raise MeteredFetchError("Metered credential response missing username/password")
    if len(username) > MAX_CREDENTIAL_LENGTH or len(credential) > MAX_CREDENTIAL_LENGTH:
        raise MeteredFetchError("Metered credential is implausibly large")
    if not isinstance(data, list):
        raise MeteredFetchError("Metered ICE array was not a JSON array")
    if len(data) > MAX_ICE_ENTRIES:
        raise MeteredFetchError("Metered ICE array has too many entries")

    turn = {"username": username, "credential": credential}
    entries: list[dict[str, str]] = [{"urls": DEFAULT_STUN_URL}]
    seen: set[str] = {DEFAULT_STUN_URL}
    have_turn = False
    for raw in data:
        if not isinstance(raw, dict):
            continue
        url_field = raw.get("urls")
        urls = url_field if isinstance(url_field, list) else [url_field]
        for url in urls:
            if len(entries) >= MAX_ICE_ENTRIES:
                # Bound the total URL count, not just the object count: an
                # entry may carry a list of urls, and the runner-side
                # validators reject lists over the cap wholesale.
                break
            try:
                valid = _validate_ice_url(url)
            except MeteredFetchError:
                continue  # skip a single bad entry, don't fail the whole array
            if valid in seen:
                continue
            seen.add(valid)
            if valid.startswith(("turn:", "turns:")):
                entries.append({"urls": valid, **turn})
                have_turn = True
            else:
                entries.append({"urls": valid})
    if not have_turn:
        raise MeteredFetchError("Metered ICE array carried no usable TURN entry")
    return entries


def validate_forwarded_ice_servers(
    data: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Validate the bridge-provisioned runner ICE list.

    ``/start-session`` remains directly callable, so this is an SSRF boundary:
    only the exact historical public STUN URL and bounded URLs beneath Metered's
    relay domain are accepted. Invalid input fails the list as a whole; callers
    fall back to the known public STUN server rather than using a partial list.
    """
    if len(data) > MAX_ICE_ENTRIES:
        raise MeteredFetchError("Forwarded ICE array has too many entries")

    entries: list[dict[str, str]] = []
    for raw in data:
        if not isinstance(raw, Mapping):
            raise MeteredFetchError("Forwarded ICE array entry is not an object")
        url = raw.get("urls")
        if not isinstance(url, str):
            raise MeteredFetchError("Forwarded ICE array entry has an invalid url")
        if url == DEFAULT_STUN_URL:
            valid_url = DEFAULT_STUN_URL
        else:
            valid_url = _validate_forwarded_ice_url(url)

        is_turn = valid_url.startswith(("turn:", "turns:"))
        username = raw.get("username")
        credential = raw.get("credential")
        if is_turn:
            if not isinstance(username, str) or not isinstance(credential, str):
                raise MeteredFetchError(
                    "Forwarded TURN entry is missing username/credential"
                )
            if (
                not username
                or not credential
                or len(username) > MAX_CREDENTIAL_LENGTH
                or len(credential) > MAX_CREDENTIAL_LENGTH
            ):
                raise MeteredFetchError("Forwarded TURN credential is invalid")
            entries.append(
                {
                    "urls": valid_url,
                    "username": username,
                    "credential": credential,
                }
            )
        else:
            if username is not None or credential is not None:
                raise MeteredFetchError("Forwarded STUN entry carried credentials")
            entries.append({"urls": valid_url})

    if not entries:
        raise MeteredFetchError("Forwarded ICE array is empty")
    return entries


# ---------------------------------------------------------------------------
# Credential fetch (stdlib urllib; no third-party HTTP dep on the CPU apps)
# ---------------------------------------------------------------------------


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect so an allow-listed host can't bounce us elsewhere."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        raise MeteredFetchError("Metered endpoint returned an unexpected redirect")


_OPENER = urllib.request.build_opener(_NoRedirect())


def _http_json(url: str, *, method: str, body: bytes | None, kind: str) -> Any:
    """Issue one redirect-refusing, time- and size-bounded request; return JSON.

    ``url`` is built by the caller from a pre-validated host (never from a
    response), so this is SSRF-safe. Every failure raises
    :class:`MeteredFetchError` with a **redacted** message: neither the URL (which
    can carry the secret key / apiKey) nor the raw exception (``URLError.reason`` /
    ``OSError`` can echo the target) is ever interpolated. ``kind`` labels the
    request in the message (e.g. ``"credential"``) without revealing anything.
    """
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with _OPENER.open(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except MeteredFetchError:
        raise
    except urllib.error.HTTPError as exc:
        # str(exc) is "HTTP Error <code>: <reason>" — no URL, so no key leak.
        raise MeteredFetchError(
            f"Metered {kind} request rejected (HTTP {exc.code})"
        ) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise MeteredFetchError(
            f"Metered {kind} request failed (network error)"
        ) from None

    if len(raw) > MAX_RESPONSE_BYTES:
        raise MeteredFetchError(f"Metered {kind} response too large")
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise MeteredFetchError(f"Metered {kind} response was not valid JSON") from None


def fetch_metered_credential(
    host: str, secret_key: str, expiry_seconds: int
) -> tuple[str, str, str | None]:
    """POST to mint a credential; return ``(username, password, api_key)``.

    SSRF-safe by construction: ``host`` is a pre-validated Metered host, the URL
    is built here (never from the response), redirects are refused, the timeout is
    strict, and the body read is bounded. The secret key is sent only in the query
    string (the Metered contract) and is **never** logged or placed in an
    exception. ``api_key`` (the per-credential key Metered returns) is used to
    fetch the canonical ICE array (:func:`fetch_metered_ice_array`); it may be
    ``None`` if the response omits it, in which case the caller falls back to the
    corrected synthesized relay list.
    """
    query = urllib.parse.urlencode({"secretKey": secret_key})
    url = urllib.parse.urlunsplit(("https", host, _CREDENTIAL_PATH, query, ""))
    body = json.dumps({"expiryInSeconds": int(expiry_seconds)}).encode("utf-8")
    data = _http_json(url, method="POST", body=body, kind="credential")
    if not isinstance(data, dict):
        raise MeteredFetchError("Metered credential response was not an object")

    username = data.get("username")
    # Metered returns the secret under "password"; accept "credential" defensively.
    credential = data.get("password") or data.get("credential")
    api_key = data.get("apiKey")
    if not isinstance(username, str) or not username:
        raise MeteredFetchError("Metered credential response missing username")
    if not isinstance(credential, str) or not credential:
        raise MeteredFetchError("Metered credential response missing password")
    return (
        username,
        credential,
        (api_key if isinstance(api_key, str) and api_key else None),
    )


def fetch_metered_ice_array(host: str, api_key: str) -> Any:
    """GET Metered's canonical ICE Servers Array (authoritative relay hostnames).

    ``host`` is the pre-validated project API host; ``api_key`` is the
    per-credential key from :func:`fetch_metered_credential`. Returns the raw JSON
    (a list) for :func:`parse_metered_ice_array` to validate. Redacted/bounded
    exactly like the mint request — the apiKey rides only the query string and is
    never logged or echoed.
    """
    query = urllib.parse.urlencode({"apiKey": api_key})
    url = urllib.parse.urlunsplit(("https", host, _ICE_ARRAY_PATH, query, ""))
    return _http_json(url, method="GET", body=None, kind="ICE array")


def mint_ice_servers(
    host: str, secret_key: str, expiry_seconds: int
) -> list[dict[str, str]]:
    """Mint a credential and resolve the browser-shaped ICE list for ``host``.

    Primary path: mint (secret key) → fetch the canonical ICE array (apiKey) →
    strictly validate it (:func:`parse_metered_ice_array`). The array carries the
    correct region-specific ``*.relay.metered.ca`` hostnames, so no hostname is
    ever guessed. If the array fetch/parse fails (or the response omitted the
    apiKey), fall back to synthesising against the account-independent global
    relay host — still a real relay, never the ``.metered.live`` API host. The
    fallback is logged (redacted) so an operator can see the degrade.
    """
    username, password, api_key = fetch_metered_credential(
        host, secret_key, expiry_seconds
    )
    if api_key is not None:
        try:
            raw = fetch_metered_ice_array(host, api_key)
            return parse_metered_ice_array(raw, username=username, credential=password)
        except MeteredError as exc:
            logger.warning(
                "wma: Metered canonical ICE array unavailable (%s); falling back to "
                "the global relay host",
                type(exc).__name__,
            )
    else:
        logger.warning(
            "wma: Metered mint response omitted apiKey; falling back to the global "
            "relay host"
        )
    return synthesize_ice_servers(
        FALLBACK_RELAY_HOST, username, password, stun_host=FALLBACK_STUN_HOST
    )


# ---------------------------------------------------------------------------
# Provider (cache + refresh + thread/async safety)
# ---------------------------------------------------------------------------

#: A provider fetch: given ``(host, secret_key, expiry_seconds)`` return the
#: browser-shaped ICE entries (STUN + TURN). The default is
#: :func:`mint_ice_servers`; tests inject a fake.
# Runtime type alias: ``typing`` generics for the 3.8 floor.
FetchFn = Callable[[str, str, int], List[Dict[str, str]]]


class MeteredIceProvider:
    """Mints and caches ephemeral Metered TURN ICE servers for one runner.

    Thread-safe and async-safe. :meth:`get_ice_servers` returns cached entries
    while they are fresh (valid past the refresh margin) and otherwise mints a new
    credential; concurrent callers are serialised so only one fetch happens. A
    fetch failure is never cached — a still-valid cache is preserved and a cold
    cache re-raises — so a transient outage degrades to STUN-only rather than
    being pinned as permanently broken.
    """

    def __init__(
        self,
        domain: str,
        secret_key: str,
        *,
        expiry_seconds: int = DEFAULT_TURN_EXPIRY_SECONDS,
        refresh_margin_seconds: int = DEFAULT_REFRESH_MARGIN_SECONDS,
        fetch_fn: FetchFn | None = None,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._host = sanitize_metered_domain(domain)
        if not isinstance(secret_key, str) or not secret_key.strip():
            raise MeteredConfigError("METERED_SECRET_KEY is empty")
        self._secret_key = secret_key.strip()
        self._expiry = max(int(expiry_seconds), MIN_CACHE_TTL_SECONDS)
        margin = max(int(refresh_margin_seconds), 0)
        self._ttl = max(self._expiry - margin, MIN_CACHE_TTL_SECONDS)
        self._fetch_fn = fetch_fn or mint_ice_servers
        self._time = time_fn
        self._state_lock = threading.Lock()
        self._fetch_lock = threading.Lock()
        self._cache: tuple[list[dict[str, str]], float] | None = None

    @property
    def host(self) -> str:
        return self._host

    def _cached(self, now: float) -> list[dict[str, str]] | None:
        with self._state_lock:
            if self._cache is not None and now < self._cache[1]:
                return [dict(e) for e in self._cache[0]]
        return None

    def get_ice_servers(self, *, force_refresh: bool = False) -> list[dict[str, str]]:
        """Return browser-shaped ICE entries, minting/refreshing as needed.

        Raises :class:`MeteredFetchError` only when a fetch is required and fails
        with no still-valid cache to fall back to.
        """
        if not force_refresh:
            cached = self._cached(self._time())
            if cached is not None:
                return cached
        # Serialise fetches so a burst of callers doesn't stampede Metered.
        with self._fetch_lock:
            if not force_refresh:
                cached = self._cached(self._time())
                if cached is not None:
                    return cached
            entries = self._fetch_fn(self._host, self._secret_key, self._expiry)
            if not entries or not any(
                str(e.get("urls", "")).startswith(("turn:", "turns:")) for e in entries
            ):
                raise MeteredFetchError("Metered fetch produced no usable TURN entry")
            expires_at = self._time() + self._ttl
            with self._state_lock:
                self._cache = (entries, expires_at)
            return [dict(e) for e in entries]

    async def get_ice_servers_async(
        self, *, force_refresh: bool = False
    ) -> list[dict[str, str]]:
        """Async wrapper: runs the (blocking) fetch off the event loop."""
        return await run_in_thread(self.get_ice_servers, force_refresh=force_refresh)

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None, **kwargs: Any
    ) -> MeteredIceProvider | None:
        """Build a provider from ``METERED_DOMAIN`` / ``METERED_SECRET_KEY``.

        Returns ``None`` when *both* are absent (STUN-only; local/dev/tests).
        Raises :class:`MeteredConfigError` when exactly one is set or the domain
        is invalid — a partial/broken configuration must fail clearly, not
        silently pretend TURN is available.
        """
        env = os.environ if environ is None else environ
        domain = (env.get("METERED_DOMAIN") or "").strip()
        secret = (env.get("METERED_SECRET_KEY") or "").strip()
        if not domain and not secret:
            return None
        if not domain or not secret:
            raise MeteredConfigError(
                "METERED_DOMAIN and METERED_SECRET_KEY must both be set"
            )
        return cls(domain, secret, **kwargs)
