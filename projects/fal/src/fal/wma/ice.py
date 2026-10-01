"""Provider-neutral ICE configuration for WMA runners."""

from __future__ import annotations

import asyncio
import inspect
import ipaddress
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence, Union

from fal.compat import run_in_thread

logger = logging.getLogger(__name__)

# Public STUN used whenever TURN is unavailable.
DEFAULT_STUN_URL = "stun:stun.l.google.com:19302"
DEFAULT_SERVER_PROVIDER_TIMEOUT_SECONDS = 10.0
MAX_ICE_ENTRIES = 24
MAX_ICE_URL_LENGTH = 256
MAX_CREDENTIAL_LENGTH = 1024

_SERVER_HOSTNAME_RE = re.compile(
    r"^(?=.{1,253}$)[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
    r"(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)*$"
)

# TURN is configured, reachable, and a credential was available most recently.
ICE_STATUS_TURN = "turn"
# No TURN provider is configured, or the selected provider returned STUN only.
ICE_STATUS_STUN_ONLY = "stun_only"
# A configured provider or forwarded list returned invalid data.
ICE_STATUS_MISCONFIGURED = "misconfigured"
# A configured provider could not supply credentials.
ICE_STATUS_UNREACHABLE = "unreachable"
# Credentials are supplied per session by the WMA bridge.
ICE_STATUS_BRIDGE_MANAGED = "bridge_managed"
# Credentials are supplied per session by app-owned provider code.
ICE_STATUS_SERVER_MANAGED = "server_managed"

IceServerProviderResult = Sequence[Mapping[str, Any]]
# Runtime type aliases are evaluated at import, so they must use
# ``typing.Union`` for the 3.8/3.9 floor.
IceServerProvider = Callable[
    [], Union[IceServerProviderResult, Awaitable[IceServerProviderResult]]
]


class _MeteredProvider(Protocol):
    @property
    def host(self) -> str: ...

    def get_ice_servers(self) -> list[dict[str, str]]: ...

    async def get_ice_servers_async(self) -> list[dict[str, str]]: ...


class IceServerConfigError(Exception):
    """An app-owned ICE provider returned an invalid server configuration."""


def stun_only_ice_servers() -> list[dict[str, str]]:
    """Return the fallback ICE list used when TURN is unavailable."""
    return [{"urls": DEFAULT_STUN_URL}]


def _validate_server_ice_url(url: Any) -> str:
    """Validate a STUN/TURN URI supplied by trusted app-owned provider code."""
    if not isinstance(url, str) or not url or len(url) > MAX_ICE_URL_LENGTH:
        raise IceServerConfigError("ICE server entry has an invalid url")

    scheme, separator, rest = url.partition(":")
    if separator != ":" or scheme not in ("stun", "turn", "turns"):
        raise IceServerConfigError("ICE server entry has an unsupported scheme")
    if not rest or any(character in rest for character in "/@# \t\r\n"):
        raise IceServerConfigError("ICE server entry has an invalid host")

    authority, query_separator, query = rest.partition("?")
    if query_separator and query not in ("transport=udp", "transport=tcp"):
        raise IceServerConfigError("ICE server entry has an invalid transport")
    if scheme == "stun" and query_separator:
        raise IceServerConfigError("STUN server entry must not specify a transport")
    if scheme == "turns" and query == "transport=udp":
        raise IceServerConfigError("TURNS server entry must use TCP")

    port: str | None = None
    if authority.startswith("["):
        # aiortc 1.15's STUN/TURN URI parser cannot consume IPv6 literals.
        raise IceServerConfigError("ICE server entry uses an unsupported IPv6 host")
    if authority.count(":") > 1:
        raise IceServerConfigError("ICE server entry has an invalid host")
    if ":" in authority:
        host, port = authority.rsplit(":", 1)
    else:
        host = authority
    normalized_host = host.rstrip(".").lower()
    if not normalized_host:
        raise IceServerConfigError("ICE server entry has an invalid host")
    try:
        ipaddress.IPv4Address(normalized_host)
    except ValueError:
        if not _SERVER_HOSTNAME_RE.match(normalized_host):
            raise IceServerConfigError("ICE server entry has an invalid host")

    # isdecimal, not isdigit: non-decimal Unicode digits pass isdigit() but
    # make int() raise, escaping the IceServerConfigError contract.
    if port is not None and (not port.isdecimal() or not 1 <= int(port) <= 65535):
        raise IceServerConfigError("ICE server entry has an invalid port")
    return url


def validate_server_ice_servers(data: Any) -> list[dict[str, str]]:
    """Validate an ICE list returned by trusted app-owned provider code.

    Unlike bridge forwarding, this path deliberately permits non-Metered hosts:
    the provider is application code chosen by the deployer, not request input.
    Bounds and STUN/TURN structure are still checked before aiortc uses the list.
    """
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
        raise IceServerConfigError("ICE server configuration must be a sequence")
    if len(data) > MAX_ICE_ENTRIES:
        raise IceServerConfigError("ICE server configuration has too many entries")

    entries: list[dict[str, str]] = []
    for raw in data:
        if not isinstance(raw, Mapping):
            raise IceServerConfigError("ICE server entry is not an object")
        url_field = raw.get("urls")
        if isinstance(url_field, str):
            urls: Sequence[Any] = (url_field,)
        elif isinstance(url_field, Sequence) and not isinstance(url_field, bytes):
            if not url_field:
                raise IceServerConfigError("ICE server entry has no urls")
            urls = url_field
        else:
            raise IceServerConfigError("ICE server entry has an invalid url")

        if len(entries) + len(urls) > MAX_ICE_ENTRIES:
            raise IceServerConfigError("ICE server configuration has too many entries")
        valid_urls = [_validate_server_ice_url(url) for url in urls]
        has_turn = any(url.startswith(("turn:", "turns:")) for url in valid_urls)
        username = raw.get("username")
        credential = raw.get("credential")
        if has_turn:
            if not isinstance(username, str) or not isinstance(credential, str):
                raise IceServerConfigError(
                    "TURN server entry is missing username/credential"
                )
            if (
                not username
                or not credential
                or len(username) > MAX_CREDENTIAL_LENGTH
                or len(credential) > MAX_CREDENTIAL_LENGTH
            ):
                raise IceServerConfigError("TURN server credential is invalid")
        elif username is not None or credential is not None:
            raise IceServerConfigError("STUN server entry carried credentials")

        for valid_url in valid_urls:
            if valid_url.startswith(("turn:", "turns:")):
                assert isinstance(username, str)
                assert isinstance(credential, str)
                entries.append(
                    {
                        "urls": valid_url,
                        "username": username,
                        "credential": credential,
                    }
                )
            else:
                entries.append({"urls": valid_url})

    if not entries:
        raise IceServerConfigError("ICE server configuration is empty")
    return entries


def _turn_sort_key(entry: dict[str, str]) -> tuple[int, int, int, str]:
    url = str(entry.get("urls", ""))
    is_tls = url.startswith("turns:")
    is_tcp = url.endswith("?transport=tcp")
    scheme, _, remainder = url.partition(":")
    authority = remainder.partition("?")[0]
    _, port_separator, port_text = authority.rpartition(":")
    if port_separator and port_text.isdecimal():
        port = int(port_text)
    else:
        port = 5349 if scheme == "turns" else 3478
    is_443 = port == 443
    return (1 if is_tls else 0, 0 if is_443 else 1, 1 if is_tcp else 0, url)


def ice_servers_for_aiortc(
    entries: list[dict[str, str]], *, host: str | None = None
) -> list[dict[str, str]]:
    """Reduce a browser ICE list to the first STUN and preferred TURN for aiortc.

    ``host`` is accepted for backwards compatibility and ignored.
    """
    stun = next(
        (entry for entry in entries if entry.get("urls", "").startswith("stun:")),
        None,
    )
    result = [dict(stun)] if stun is not None else []
    turns = [
        entry
        for entry in entries
        if entry.get("urls", "").startswith(("turn:", "turns:"))
    ]
    if turns:
        result.append(dict(sorted(turns, key=_turn_sort_key)[0]))
    return result


def ice_candidate_type_counts(sdp: str) -> dict[str, int]:
    """Count gathered ICE candidate types without retaining addresses or creds."""
    counts: dict[str, int] = {}
    for raw_line in sdp.splitlines():
        line = raw_line.strip()
        if not (line.startswith("a=candidate:") or line.startswith("candidate:")):
            continue
        tokens = line.split()
        for index, token in enumerate(tokens):
            if token == "typ" and index + 1 < len(tokens):
                candidate_type = tokens[index + 1]
                if candidate_type in ("host", "srflx", "prflx", "relay"):
                    counts[candidate_type] = counts.get(candidate_type, 0) + 1
                break
    return counts


def build_rtc_ice_servers(
    entries: list[dict[str, str]], *, host: str | None = None
) -> list[Any]:
    """Build the reduced aiortc ``RTCIceServer`` list lazily."""
    from aiortc import RTCIceServer

    servers: list[Any] = []
    for entry in ice_servers_for_aiortc(entries):
        kwargs: dict[str, Any] = {"urls": entry["urls"]}
        if entry.get("username") is not None:
            kwargs["username"] = entry["username"]
        if entry.get("credential") is not None:
            kwargs["credential"] = entry["credential"]
        servers.append(RTCIceServer(**kwargs))
    return servers


@dataclass
class RunnerIceConfig:
    """Resolve runner ICE from the bridge, environment, or app-owned code."""

    provider: _MeteredProvider | None
    status: str
    bridge_managed: bool = False
    _server_provider: IceServerProvider | None = None
    _server_provider_timeout_seconds: float = DEFAULT_SERVER_PROVIDER_TIMEOUT_SECONDS

    @property
    def turn_configured(self) -> bool:
        """Whether TURN is runner-managed or expected from the bridge."""
        return (
            self.provider is not None
            or self._server_provider is not None
            or self.bridge_managed
        )

    @property
    def turn_available(self) -> bool:
        """Whether TURN was available for the most recently built list."""
        return self.status == ICE_STATUS_TURN

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        log: logging.Logger | None = None,
        warm: bool = True,
        **kwargs: Any,
    ) -> RunnerIceConfig:
        """Resolve app-owned Metered credentials from the environment."""
        log = log or logger
        try:
            provider = MeteredIceProvider.from_env(environ, **kwargs)
        except MeteredConfigError as exc:
            log.error(
                "wma: METERED_* misconfigured (%s: %s); TURN disabled, STUN-only",
                type(exc).__name__,
                exc,
            )
            return cls(None, ICE_STATUS_MISCONFIGURED)
        if provider is None:
            log.info("wma: no Metered secrets configured; using STUN-only ICE")
            return cls(None, ICE_STATUS_STUN_ONLY)
        if warm:
            try:
                provider.get_ice_servers()
            except MeteredError as exc:
                log.error(
                    "wma: Metered TURN warmup failed (%s: %s); TURN configured but "
                    "unreachable, serving STUN-only until it recovers",
                    type(exc).__name__,
                    exc,
                )
                return cls(provider, ICE_STATUS_UNREACHABLE)
        log.info("wma: Metered TURN configured for host %s", provider.host)
        return cls(provider, ICE_STATUS_TURN)

    @classmethod
    def from_bridge(cls) -> RunnerIceConfig:
        """Build a config whose per-session ICE list comes from WMA."""
        return cls(None, ICE_STATUS_BRIDGE_MANAGED, bridge_managed=True)

    @classmethod
    def from_server(
        cls,
        provider: IceServerProvider,
        *,
        timeout_seconds: float = DEFAULT_SERVER_PROVIDER_TIMEOUT_SECONDS,
    ) -> RunnerIceConfig:
        """Build a config backed by app-owned sync or async provider code."""
        if not callable(provider):
            raise TypeError("ICE server provider must be callable")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("ICE server provider timeout must be positive and finite")
        return cls(
            None,
            ICE_STATUS_SERVER_MANAGED,
            _server_provider=provider,
            _server_provider_timeout_seconds=float(timeout_seconds),
        )

    async def build_ice_servers_async(
        self,
        forwarded: Sequence[Mapping[str, Any]] | None = None,
        *,
        forwarded_status: str | None = None,
    ) -> tuple[list[Any], bool]:
        """Build one session's aiortc ICE list and TURN availability."""
        if self.bridge_managed and forwarded:
            try:
                entries = validate_forwarded_ice_servers(forwarded)
            except MeteredFetchError as exc:
                self.status = ICE_STATUS_MISCONFIGURED
                logger.error(
                    "wma: rejected bridge-provisioned ICE (%s); STUN-only",
                    type(exc).__name__,
                )
                return build_rtc_ice_servers(stun_only_ice_servers()), False
            turn_available = any(
                entry["urls"].startswith(("turn:", "turns:")) for entry in entries
            )
            if turn_available:
                self.status = ICE_STATUS_TURN
            elif forwarded_status in {
                ICE_STATUS_STUN_ONLY,
                ICE_STATUS_MISCONFIGURED,
                ICE_STATUS_UNREACHABLE,
            }:
                self.status = forwarded_status
            else:
                self.status = ICE_STATUS_STUN_ONLY
            return build_rtc_ice_servers(entries), turn_available

        if self.bridge_managed:
            self.status = ICE_STATUS_STUN_ONLY
            return build_rtc_ice_servers(stun_only_ice_servers()), False

        if self._server_provider is not None:
            server_provider = self._server_provider

            async def _resolve_provider() -> Any:
                # ``run_in_thread`` (the SDK's 3.8-compatible ``asyncio.to_thread``)
                # runs a sync provider off the loop; an async provider returns a
                # coroutine from the thread, awaited here on the loop.
                provided = await run_in_thread(server_provider)
                if inspect.isawaitable(provided):
                    provided = await provided
                return provided

            try:
                provided = await asyncio.wait_for(
                    _resolve_provider(),
                    timeout=self._server_provider_timeout_seconds,
                )
            except Exception as exc:
                self.status = ICE_STATUS_UNREACHABLE
                logger.error(
                    "wma: app-owned ICE provider failed (%s); STUN-only",
                    type(exc).__name__,
                )
                return build_rtc_ice_servers(stun_only_ice_servers()), False
            try:
                entries = validate_server_ice_servers(provided)
            except IceServerConfigError as exc:
                self.status = ICE_STATUS_MISCONFIGURED
                logger.error(
                    "wma: app-owned ICE provider returned invalid configuration "
                    "(%s: %s); STUN-only",
                    type(exc).__name__,
                    exc,
                )
                return build_rtc_ice_servers(stun_only_ice_servers()), False
            turn_available = any(
                entry["urls"].startswith(("turn:", "turns:")) for entry in entries
            )
            self.status = ICE_STATUS_TURN if turn_available else ICE_STATUS_STUN_ONLY
            return build_rtc_ice_servers(entries), turn_available

        if self.provider is None:
            self.status = ICE_STATUS_STUN_ONLY
            return build_rtc_ice_servers(stun_only_ice_servers()), False
        try:
            entries = await self.provider.get_ice_servers_async()
        except MeteredError as exc:
            self.status = ICE_STATUS_UNREACHABLE
            logger.error(
                "wma: Metered TURN fetch failed at session start (%s); STUN-only",
                type(exc).__name__,
            )
            return build_rtc_ice_servers(stun_only_ice_servers()), False
        self.status = ICE_STATUS_TURN
        return build_rtc_ice_servers(entries), True


# Resolve these before fal serializes an app. Registry source modules are not
# importable by package name inside a runner, so session methods must not use a
# dynamic import. This intentionally follows the class definition
# because ``metered`` imports provider-neutral constants from this module.
from fal.wma.metered import (  # noqa: E402
    MeteredConfigError,
    MeteredError,
    MeteredFetchError,
    MeteredIceProvider,
    validate_forwarded_ice_servers,
)
