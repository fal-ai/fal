"""Unit tests for the Metered TURN ICE helper (``fal/wma/metered.py``).

Covers the security-critical surface: domain validation / SSRF gate, the
credential request (method/body/query + secret redaction), response schema and
URL allow-listing, the cache/refresh/concurrency behaviour, missing/invalid
secret handling, and the aiortc ICE-server mapping. All network is faked — no
real Metered call is made, and no secret value is ever asserted into an error or
a URL that a caller could see.
"""

from __future__ import annotations

import builtins
import json
import sys
import threading
import types
import urllib.error

import pytest

from fal.wma import ice as i
from fal.wma import metered as m

# ---------------------------------------------------------------------------
# Domain validation (SSRF gate)
# ---------------------------------------------------------------------------


class TestSanitizeMeteredDomain:
    @pytest.mark.parametrize(
        "given,expected",
        [
            ("my-app.metered.live", "my-app.metered.live"),
            ("My-App.Metered.LIVE", "my-app.metered.live"),
            ("https://my-app.metered.live", "my-app.metered.live"),
            ("https://my-app.metered.live/", "my-app.metered.live"),
            ("https://my-app.metered.live:443", "my-app.metered.live"),
            ("proj.metered.ca", "proj.metered.ca"),
            ("  proj.metered.ca  ", "proj.metered.ca"),
        ],
    )
    def test_accepts_valid_metered_hosts(self, given, expected):
        assert m.sanitize_metered_domain(given) == expected

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "   ",
            "metered.live",  # bare apex, no project subdomain
            "metered.ca",
            "evil.com",
            "app.metered.live.evil.com",  # apex must be a suffix boundary
            "app.evil-metered.live",  # not under the real apex
            "http://app.metered.live",  # not https
            "https://app.metered.live/api/v1/turn/credential",  # path
            "https://app.metered.live?x=1",  # query
            "https://user:pw@app.metered.live",  # userinfo
            "https://app.metered.live:8080",  # non-443 port
            "app.metered.live:22",  # embedded port
            "app.metered.live/path",  # embedded path
            "a b.metered.live",  # whitespace
            "169.254.169.254",  # metadata IP
            "http://169.254.169.254/latest/meta-data",
            "app.metered.live@evil.com",
        ],
    )
    def test_rejects_everything_else(self, bad):
        with pytest.raises(m.MeteredConfigError):
            m.sanitize_metered_domain(bad)

    def test_rejects_non_string(self):
        with pytest.raises(m.MeteredConfigError):
            m.sanitize_metered_domain(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ICE-server synthesis + allow-listing
# ---------------------------------------------------------------------------


class TestSynthesizeIceServers:
    def test_shape_and_credentials(self):
        entries = m.synthesize_ice_servers("h.metered.live", "user1", "pass1")
        assert entries[0] == {"urls": i.DEFAULT_STUN_URL}
        urls = [e["urls"] for e in entries]
        assert "turn:h.metered.live:443?transport=tcp" in urls
        assert "turns:h.metered.live:443?transport=tcp" in urls
        # Every TURN/TURNS entry carries the ephemeral pair; STUN entries do not.
        for e in entries:
            if e["urls"].startswith(("turn:", "turns:")):
                assert e["username"] == "user1"
                assert e["credential"] == "pass1"
            else:
                assert "username" not in e

    def test_missing_credentials_raises(self):
        with pytest.raises(m.MeteredFetchError):
            m.synthesize_ice_servers("h.metered.live", "", "pass")
        with pytest.raises(m.MeteredFetchError):
            m.synthesize_ice_servers("h.metered.live", "user", "")

    def test_only_metered_and_stun_schemes(self):
        entries = m.synthesize_ice_servers("h.metered.live", "u", "c")
        for e in entries:
            assert e["urls"].split(":", 1)[0] in {"stun", "turn", "turns"}


class TestStunOnly:
    def test_stun_only_list(self):
        assert i.stun_only_ice_servers() == [{"urls": i.DEFAULT_STUN_URL}]


# ---------------------------------------------------------------------------
# aiortc mapping (first-STUN / first-TURN, turns is tcp-only)
# ---------------------------------------------------------------------------


class TestIceServersForAiortc:
    def test_reduces_to_one_stun_and_one_turn(self):
        entries = m.synthesize_ice_servers("h.metered.live", "u", "c")
        reduced = i.ice_servers_for_aiortc(entries)
        assert len(reduced) == 2
        assert reduced[0]["urls"].startswith("stun:")
        assert reduced[1]["urls"].startswith("turn")
        # The runner prefers UDP :443; restrictive clients independently choose
        # from their full multi-transport ICE server list.
        assert reduced[1]["urls"] == "turn:h.metered.live:443"
        assert reduced[1]["username"] == "u"

    def test_stun_only_maps_to_single_stun(self):
        reduced = i.ice_servers_for_aiortc(i.stun_only_ice_servers())
        assert reduced == [{"urls": i.DEFAULT_STUN_URL}]

    def test_prefers_udp_443_regardless_of_host(self):
        # Selection is host-independent, so it works on the canonical relay hosts.
        entries = m.synthesize_ice_servers("global.relay.metered.ca", "u", "c")
        reduced = i.ice_servers_for_aiortc(entries, host="ignored.example")
        assert reduced[1]["urls"] == "turn:global.relay.metered.ca:443"

    def test_prefers_plain_turn_over_turns_tls(self):
        # A list with ONLY a TLS turns: entry still selects it, but a plain TURN
        # is always preferred when present.
        only_tls = [
            {"urls": i.DEFAULT_STUN_URL},
            {
                "urls": "turns:h.metered.ca:443?transport=tcp",
                "username": "u",
                "credential": "c",
            },
        ]
        assert i.ice_servers_for_aiortc(only_tls)[1]["urls"].startswith("turns:")

    def test_turn_preference_parses_port_instead_of_hostname_text(self):
        entries = [
            {
                "urls": "turn:443.example.com:80",
                "username": "port-80",
                "credential": "c",
            },
            {
                "urls": "turn:relay.example.com:443",
                "username": "port-443",
                "credential": "c",
            },
        ]

        assert i.ice_servers_for_aiortc(entries)[0]["username"] == "port-443"

    def test_build_rtc_ice_servers_maps_fields(self, monkeypatch):
        # Inject a fake aiortc so the runner-only import resolves off-runner.
        captured: list[dict] = []

        class FakeRTCIceServer:
            def __init__(self, **kwargs):
                captured.append(kwargs)

        fake = types.ModuleType("aiortc")
        fake.RTCIceServer = FakeRTCIceServer  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "aiortc", fake)

        entries = m.synthesize_ice_servers("h.metered.live", "u", "c")
        servers = i.build_rtc_ice_servers(entries)
        assert len(servers) == 2
        # STUN carries no credentials; TURN carries username+credential.
        assert "username" not in captured[0]
        assert captured[1]["username"] == "u"
        assert captured[1]["credential"] == "c"
        assert captured[1]["urls"] == "turn:h.metered.live:443"


# ---------------------------------------------------------------------------
# Credential fetch (method/body/query + redaction + limits)
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self, n: int = -1) -> bytes:
        return self._body if n < 0 else self._body[:n]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeOpener:
    def __init__(self, handler):
        self._handler = handler
        self.last_request = None
        self.last_timeout = None

    def open(self, request, timeout=None):
        self.last_request = request
        self.last_timeout = timeout
        return self._handler(request)


SECRET = "super-secret-key-value-1234567890"


class TestFetchMeteredCredential:
    def _install(self, monkeypatch, handler) -> _FakeOpener:
        opener = _FakeOpener(handler)
        monkeypatch.setattr(m, "_OPENER", opener)
        return opener

    def test_request_shape_and_success(self, monkeypatch):
        body = json.dumps(
            {"username": "eph-user", "password": "eph-pass", "apiKey": "ak-123"}
        ).encode()
        opener = self._install(monkeypatch, lambda req: _FakeResponse(body))

        user, cred, api_key = m.fetch_metered_credential("h.metered.live", SECRET, 3600)

        assert (user, cred, api_key) == ("eph-user", "eph-pass", "ak-123")
        req = opener.last_request
        assert req.method == "POST"
        assert req.full_url.startswith("https://h.metered.live/api/v1/turn/credential?")
        assert f"secretKey={SECRET}" in req.full_url
        assert json.loads(req.data) == {"expiryInSeconds": 3600}
        assert req.headers["Content-type"] == "application/json"
        assert opener.last_timeout == m.HTTP_TIMEOUT_SECONDS

    def test_accepts_credential_field_alias_and_missing_apikey(self, monkeypatch):
        # No apiKey in the response -> api_key is None (caller falls back).
        body = json.dumps({"username": "u", "credential": "c"}).encode()
        self._install(monkeypatch, lambda req: _FakeResponse(body))
        assert m.fetch_metered_credential("h.metered.live", SECRET, 60) == (
            "u",
            "c",
            None,
        )

    def test_oversized_response_rejected(self, monkeypatch):
        big = (
            b'{"username":"u","password":"' + b"x" * (m.MAX_RESPONSE_BYTES + 1) + b'"}'
        )
        self._install(monkeypatch, lambda req: _FakeResponse(big))
        with pytest.raises(m.MeteredFetchError, match="too large"):
            m.fetch_metered_credential("h.metered.live", SECRET, 60)

    def test_non_json_response_rejected(self, monkeypatch):
        self._install(monkeypatch, lambda req: _FakeResponse(b"<html>nope"))
        with pytest.raises(m.MeteredFetchError, match="not valid JSON"):
            m.fetch_metered_credential("h.metered.live", SECRET, 60)

    @pytest.mark.parametrize("payload", [{"password": "c"}, {"username": "u"}, {}])
    def test_missing_fields_rejected(self, monkeypatch, payload):
        body = json.dumps(payload).encode()
        self._install(monkeypatch, lambda req: _FakeResponse(body))
        with pytest.raises(m.MeteredFetchError):
            m.fetch_metered_credential("h.metered.live", SECRET, 60)

    def test_http_error_is_redacted(self, monkeypatch):
        def handler(req):
            raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)

        self._install(monkeypatch, handler)
        with pytest.raises(m.MeteredFetchError) as excinfo:
            m.fetch_metered_credential("h.metered.live", SECRET, 60)
        msg = str(excinfo.value)
        assert "401" in msg
        assert SECRET not in msg  # the URL (which carries the key) is never echoed

    def test_network_error_is_redacted(self, monkeypatch):
        def handler(req):
            raise urllib.error.URLError(f"cannot reach {req.full_url}")

        self._install(monkeypatch, handler)
        with pytest.raises(m.MeteredFetchError) as excinfo:
            m.fetch_metered_credential("h.metered.live", SECRET, 60)
        assert SECRET not in str(excinfo.value)

    def test_redirect_handler_refuses(self):
        with pytest.raises(m.MeteredFetchError, match="redirect"):
            m._NoRedirect().redirect_request(None, None, 302, "Found", {}, "http://x")


# ---------------------------------------------------------------------------
# Provider: cache / refresh / concurrency / no-cache-on-failure
# ---------------------------------------------------------------------------


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class TestMeteredIceProvider:
    def test_caches_within_ttl_and_refreshes_after(self):
        clock = _Clock()
        calls = {"n": 0}

        def fetch(host, secret, expiry):
            calls["n"] += 1
            return m.synthesize_ice_servers(host, f"user{calls['n']}", "pass")

        p = m.MeteredIceProvider(
            "h.metered.live",
            SECRET,
            expiry_seconds=100,
            refresh_margin_seconds=10,
            fetch_fn=fetch,
            time_fn=clock,
        )
        p.get_ice_servers()
        assert calls["n"] == 1
        clock.t = 80.0
        p.get_ice_servers()
        assert calls["n"] == 1  # still within ttl (90s)
        clock.t = 95.0
        p.get_ice_servers()
        assert calls["n"] == 2  # refreshed past ttl

    def test_force_refresh(self):
        calls = {"n": 0}

        def fetch(host, secret, expiry):
            calls["n"] += 1
            return m.synthesize_ice_servers(host, "u", "c")

        p = m.MeteredIceProvider("h.metered.live", SECRET, fetch_fn=fetch)
        p.get_ice_servers()
        p.get_ice_servers(force_refresh=True)
        assert calls["n"] == 2

    def test_concurrent_callers_share_one_fetch(self):
        calls = {"n": 0}
        started = threading.Event()

        def slow_fetch(host, secret, expiry):
            calls["n"] += 1
            started.set()
            # Simulate a slow network call so threads overlap.
            import time as _t

            _t.sleep(0.05)
            return m.synthesize_ice_servers(host, "u", "c")

        p = m.MeteredIceProvider("h.metered.live", SECRET, fetch_fn=slow_fetch)
        results: list = []

        def worker():
            results.append(p.get_ice_servers())

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert calls["n"] == 1  # serialised: only one fetch across 8 callers
        assert all(r == results[0] for r in results)

    def test_failure_is_not_cached(self):
        state = {"fail": True}

        def fetch(host, secret, expiry):
            if state["fail"]:
                raise m.MeteredFetchError("down")
            return m.synthesize_ice_servers(host, "u", "c")

        p = m.MeteredIceProvider("h.metered.live", SECRET, fetch_fn=fetch)
        with pytest.raises(m.MeteredFetchError):
            p.get_ice_servers()
        assert p._cache is None
        state["fail"] = False
        # Recovers on the next call rather than being pinned as broken.
        assert p.get_ice_servers()[0]["urls"] == i.DEFAULT_STUN_URL

    def test_empty_secret_rejected(self):
        with pytest.raises(m.MeteredConfigError):
            m.MeteredIceProvider("h.metered.live", "  ")


# ---------------------------------------------------------------------------
# from_env + RunnerIceConfig status
# ---------------------------------------------------------------------------


class TestFromEnv:
    def test_absent_secrets_returns_none(self):
        assert m.MeteredIceProvider.from_env({}) is None

    def test_partial_config_raises(self):
        with pytest.raises(m.MeteredConfigError):
            m.MeteredIceProvider.from_env({"METERED_DOMAIN": "a.metered.live"})
        with pytest.raises(m.MeteredConfigError):
            m.MeteredIceProvider.from_env({"METERED_SECRET_KEY": "x"})

    def test_valid_config_builds_provider(self):
        p = m.MeteredIceProvider.from_env(
            {"METERED_DOMAIN": "a.metered.live", "METERED_SECRET_KEY": "x"},
            fetch_fn=lambda h, s, e: m.synthesize_ice_servers(h, "u", "c"),
        )
        assert p is not None and p.host == "a.metered.live"


class TestRunnerIceConfig:
    def test_bridge_managed_has_no_provider_or_secret(self):
        rc = i.RunnerIceConfig.from_bridge()
        assert rc.provider is None
        assert rc.bridge_managed
        assert rc.status == i.ICE_STATUS_BRIDGE_MANAGED
        assert rc.turn_configured
        assert not rc.turn_available

    def test_server_managed_has_only_app_provider(self):
        def provider():
            return [{"urls": "stun:stun.example.com:3478"}]

        rc = i.RunnerIceConfig.from_server(provider)
        assert rc.provider is None
        assert rc.status == i.ICE_STATUS_SERVER_MANAGED
        assert rc.turn_configured
        assert not rc.turn_available

    def test_server_provider_must_be_callable(self):
        with pytest.raises(TypeError, match="must be callable"):
            i.RunnerIceConfig.from_server(None)  # type: ignore[arg-type]

    @pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
    def test_server_provider_timeout_must_be_positive_and_finite(self, timeout):
        with pytest.raises(ValueError, match="positive and finite"):
            i.RunnerIceConfig.from_server(lambda: [], timeout_seconds=timeout)

    def test_stun_only_when_absent(self):
        rc = i.RunnerIceConfig.from_env({}, warm=False)
        assert rc.status == i.ICE_STATUS_STUN_ONLY
        assert not rc.turn_available and not rc.turn_configured

    def test_misconfigured_never_raises(self):
        rc = i.RunnerIceConfig.from_env(
            {"METERED_DOMAIN": "not-a-metered-host", "METERED_SECRET_KEY": "x"},
            warm=False,
        )
        assert rc.status == i.ICE_STATUS_MISCONFIGURED
        assert not rc.turn_available
        assert rc.provider is None

    def test_unreachable_downgrades_but_keeps_provider(self):
        def boom(*a):
            raise m.MeteredFetchError("down")

        rc = i.RunnerIceConfig.from_env(
            {"METERED_DOMAIN": "a.metered.live", "METERED_SECRET_KEY": "x"},
            warm=True,
            fetch_fn=boom,
        )
        assert rc.status == i.ICE_STATUS_UNREACHABLE
        assert not rc.turn_available
        assert rc.provider is not None  # retried per-session

    def test_turn_status_when_warm_succeeds(self):
        rc = i.RunnerIceConfig.from_env(
            {"METERED_DOMAIN": "a.metered.live", "METERED_SECRET_KEY": "x"},
            warm=True,
            fetch_fn=lambda h, s, e: m.synthesize_ice_servers(h, "u", "c"),
        )
        assert rc.status == i.ICE_STATUS_TURN
        assert rc.turn_available and rc.turn_configured

    def test_from_env_does_not_import_registry_on_runner(self, monkeypatch):
        real_import = builtins.__import__

        def runner_import(name, *args, **kwargs):
            if name == "registry" or name.startswith("registry."):
                raise ModuleNotFoundError("No module named 'registry'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", runner_import)
        rc = i.RunnerIceConfig.from_env(
            {
                "METERED_DOMAIN": "a.metered.live",
                "METERED_SECRET_KEY": "x",
            },
            warm=False,
        )

        assert rc.provider is not None
        assert rc.status == i.ICE_STATUS_TURN

    @pytest.mark.asyncio
    async def test_build_ice_servers_async_stun_only(self, monkeypatch):
        fake = types.ModuleType("aiortc")
        fake.RTCIceServer = lambda **kw: kw  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "aiortc", fake)
        rc = i.RunnerIceConfig(None, i.ICE_STATUS_STUN_ONLY)
        servers, turn = await rc.build_ice_servers_async()
        assert turn is False
        assert servers == [{"urls": i.DEFAULT_STUN_URL}]

    @pytest.mark.asyncio
    async def test_build_ice_servers_async_turn(self, monkeypatch):
        fake = types.ModuleType("aiortc")
        fake.RTCIceServer = lambda **kw: kw  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "aiortc", fake)
        provider = m.MeteredIceProvider(
            "a.metered.live",
            "x",
            fetch_fn=lambda h, s, e: m.synthesize_ice_servers(h, "u", "c"),
        )
        rc = i.RunnerIceConfig(provider, i.ICE_STATUS_UNREACHABLE)
        servers, turn = await rc.build_ice_servers_async()
        assert turn is True
        assert rc.status == i.ICE_STATUS_TURN
        assert rc.turn_available
        assert servers[0]["urls"].startswith("stun:")
        assert servers[1]["urls"].startswith("turn")
        assert servers[1]["username"] == "u"

    @pytest.mark.asyncio
    async def test_build_ice_servers_async_fetch_failure_falls_back(self, monkeypatch):
        fake = types.ModuleType("aiortc")
        fake.RTCIceServer = lambda **kw: kw  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "aiortc", fake)

        def boom(*a):
            raise m.MeteredFetchError("down")

        provider = m.MeteredIceProvider("a.metered.live", "x", fetch_fn=boom)
        rc = i.RunnerIceConfig(provider, i.ICE_STATUS_TURN)
        servers, turn = await rc.build_ice_servers_async()
        assert turn is False
        assert rc.status == i.ICE_STATUS_UNREACHABLE
        assert not rc.turn_available
        assert servers == [{"urls": i.DEFAULT_STUN_URL}]

    @pytest.mark.asyncio
    async def test_bridge_forwarded_turn_is_validated_and_used(self, monkeypatch):
        fake = types.ModuleType("aiortc")
        fake.RTCIceServer = lambda **kw: kw  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "aiortc", fake)
        rc = i.RunnerIceConfig.from_bridge()
        servers, turn = await rc.build_ice_servers_async(
            [
                {"urls": i.DEFAULT_STUN_URL},
                {
                    "urls": "turn:global.relay.metered.ca:443",
                    "username": "ephemeral-user",
                    "credential": "ephemeral-password",
                },
            ],
            forwarded_status=i.ICE_STATUS_TURN,
        )
        assert turn is True
        assert rc.status == i.ICE_STATUS_TURN
        assert rc.turn_configured
        assert servers[1] == {
            "urls": "turn:global.relay.metered.ca:443",
            "username": "ephemeral-user",
            "credential": "ephemeral-password",
        }

    @pytest.mark.asyncio
    async def test_bridge_malformed_turn_port_falls_back(self, monkeypatch):
        fake = types.ModuleType("aiortc")
        fake.RTCIceServer = lambda **kw: kw  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "aiortc", fake)
        rc = i.RunnerIceConfig.from_bridge()

        servers, turn = await rc.build_ice_servers_async(
            [
                {
                    "urls": "turn:global.relay.metered.ca:443",
                    "username": "u",
                    "credential": "p",
                },
                {
                    "urls": "turn:global.relay.metered.ca:443extra",
                    "username": "u",
                    "credential": "p",
                },
            ],
            forwarded_status=i.ICE_STATUS_TURN,
        )

        assert turn is False
        assert rc.status == i.ICE_STATUS_MISCONFIGURED
        assert servers == [{"urls": i.DEFAULT_STUN_URL}]

    @pytest.mark.asyncio
    async def test_bridge_build_does_not_import_registry_on_runner(self, monkeypatch):
        fake = types.ModuleType("aiortc")
        fake.RTCIceServer = lambda **kw: kw  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "aiortc", fake)
        real_import = builtins.__import__

        def runner_import(name, *args, **kwargs):
            if name == "registry" or name.startswith("registry."):
                raise ModuleNotFoundError("No module named 'registry'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", runner_import)
        rc = i.RunnerIceConfig.from_bridge()

        servers, turn = await rc.build_ice_servers_async(
            [{"urls": i.DEFAULT_STUN_URL}],
            forwarded_status=i.ICE_STATUS_STUN_ONLY,
        )

        assert turn is False
        assert servers == [{"urls": i.DEFAULT_STUN_URL}]

    @pytest.mark.asyncio
    async def test_hostile_forwarded_ice_falls_back_without_contacting_it(
        self, monkeypatch
    ):
        fake = types.ModuleType("aiortc")
        fake.RTCIceServer = lambda **kw: kw  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "aiortc", fake)
        rc = i.RunnerIceConfig.from_bridge()
        servers, turn = await rc.build_ice_servers_async(
            [
                {
                    "urls": "turn:169.254.169.254:80",
                    "username": "u",
                    "credential": "p",
                }
            ],
            forwarded_status=i.ICE_STATUS_TURN,
        )
        assert turn is False
        assert rc.status == i.ICE_STATUS_MISCONFIGURED
        assert rc.turn_configured
        assert servers == [{"urls": i.DEFAULT_STUN_URL}]

    @pytest.mark.parametrize(
        "url",
        [
            "turn:customer-project.metered.live:443",
            "turn:api.metered.ca:443",
        ],
    )
    def test_forwarded_ice_rejects_non_relay_metered_hosts(self, url):
        with pytest.raises(m.MeteredFetchError, match="not a Metered relay host"):
            m.validate_forwarded_ice_servers(
                [{"urls": url, "username": "u", "credential": "p"}]
            )

    @pytest.mark.asyncio
    async def test_bridge_managed_stays_configured_without_forwarded_ice(
        self, monkeypatch
    ):
        fake = types.ModuleType("aiortc")
        fake.RTCIceServer = lambda **kw: kw  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "aiortc", fake)
        rc = i.RunnerIceConfig.from_bridge()

        servers, turn = await rc.build_ice_servers_async()

        assert turn is False
        assert rc.status == i.ICE_STATUS_STUN_ONLY
        assert rc.turn_configured
        assert servers == [{"urls": i.DEFAULT_STUN_URL}]

    @pytest.mark.asyncio
    async def test_sync_server_provider_supports_non_metered_turn_off_loop(
        self, monkeypatch
    ):
        fake = types.ModuleType("aiortc")
        fake.RTCIceServer = lambda **kw: kw  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "aiortc", fake)
        event_loop_thread = threading.get_ident()
        provider_threads = []

        def provider():
            provider_threads.append(threading.get_ident())
            return [
                {"urls": "stun:stun.example.com:3478"},
                {
                    "urls": "turn:turn.example.com:443?transport=tcp",
                    "username": "app-user",
                    "credential": "app-password",
                },
            ]

        rc = i.RunnerIceConfig.from_server(provider)
        servers, turn = await rc.build_ice_servers_async()

        assert len(provider_threads) == 1
        assert provider_threads[0] != event_loop_thread
        assert turn is True
        assert rc.status == i.ICE_STATUS_TURN
        assert servers == [
            {"urls": "stun:stun.example.com:3478"},
            {
                "urls": "turn:turn.example.com:443?transport=tcp",
                "username": "app-user",
                "credential": "app-password",
            },
        ]

    @pytest.mark.asyncio
    async def test_async_server_provider_is_awaited(self, monkeypatch):
        fake = types.ModuleType("aiortc")
        fake.RTCIceServer = lambda **kw: kw  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "aiortc", fake)

        async def provider():
            return [
                {
                    "urls": "turns:turn.example.com:5349?transport=tcp",
                    "username": "u",
                    "credential": "p",
                }
            ]

        rc = i.RunnerIceConfig.from_server(provider)
        servers, turn = await rc.build_ice_servers_async()

        assert turn is True
        assert servers == [
            {
                "urls": "turns:turn.example.com:5349?transport=tcp",
                "username": "u",
                "credential": "p",
            }
        ]

    @pytest.mark.asyncio
    async def test_server_provider_ignores_request_forwarded_ice(self, monkeypatch):
        fake = types.ModuleType("aiortc")
        fake.RTCIceServer = lambda **kw: kw  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "aiortc", fake)
        rc = i.RunnerIceConfig.from_server(
            lambda: [{"urls": "stun:app-owned.example.com:3478"}]
        )

        servers, turn = await rc.build_ice_servers_async(
            [
                {
                    "urls": "turn:global.relay.metered.ca:443",
                    "username": "forwarded-user",
                    "credential": "forwarded-password",
                }
            ]
        )

        assert turn is False
        assert servers == [{"urls": "stun:app-owned.example.com:3478"}]

    @pytest.mark.asyncio
    async def test_env_provider_ignores_request_forwarded_ice(self, monkeypatch):
        fake = types.ModuleType("aiortc")
        fake.RTCIceServer = lambda **kw: kw  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "aiortc", fake)
        provider = m.MeteredIceProvider(
            "app.metered.live",
            "app-secret",
            fetch_fn=lambda h, s, e: m.synthesize_ice_servers(h, "owner", "owned"),
        )
        rc = i.RunnerIceConfig(provider, i.ICE_STATUS_TURN)

        servers, turn = await rc.build_ice_servers_async(
            [
                {
                    "urls": "turn:global.relay.metered.ca:443",
                    "username": "forwarded-user",
                    "credential": "forwarded-password",
                }
            ]
        )

        assert turn is True
        assert servers[1]["username"] == "owner"
        assert servers[1]["credential"] == "owned"

    @pytest.mark.asyncio
    async def test_invalid_server_provider_result_falls_back(self, monkeypatch):
        fake = types.ModuleType("aiortc")
        fake.RTCIceServer = lambda **kw: kw  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "aiortc", fake)
        rc = i.RunnerIceConfig.from_server(
            lambda: [
                {
                    "urls": "turn:https://not-a-turn-uri.example.com",
                    "username": "u",
                    "credential": "p",
                }
            ]
        )

        servers, turn = await rc.build_ice_servers_async()

        assert turn is False
        assert rc.status == i.ICE_STATUS_MISCONFIGURED
        assert servers == [{"urls": i.DEFAULT_STUN_URL}]

    @pytest.mark.asyncio
    async def test_server_provider_failure_falls_back_without_leaking_error(
        self, monkeypatch, caplog
    ):
        fake = types.ModuleType("aiortc")
        fake.RTCIceServer = lambda **kw: kw  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "aiortc", fake)
        secret = "provider-secret-that-must-not-be-logged"

        def provider():
            raise RuntimeError(secret)

        rc = i.RunnerIceConfig.from_server(provider)
        servers, turn = await rc.build_ice_servers_async()

        assert turn is False
        assert rc.status == i.ICE_STATUS_UNREACHABLE
        assert servers == [{"urls": i.DEFAULT_STUN_URL}]
        assert secret not in caplog.text

    @pytest.mark.asyncio
    async def test_server_provider_timeout_falls_back(self, monkeypatch):
        fake = types.ModuleType("aiortc")
        fake.RTCIceServer = lambda **kw: kw  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "aiortc", fake)
        release = threading.Event()

        def provider():
            release.wait()
            return [{"urls": "stun:too-late.example.com"}]

        try:
            rc = i.RunnerIceConfig.from_server(provider, timeout_seconds=0.01)
            servers, turn = await rc.build_ice_servers_async()
        finally:
            release.set()

        assert turn is False
        assert rc.status == i.ICE_STATUS_UNREACHABLE
        assert servers == [{"urls": i.DEFAULT_STUN_URL}]


class TestValidateServerIceServers:
    def test_accepts_hostname_ipv4_and_supported_tls_servers(self):
        entries = i.validate_server_ice_servers(
            [
                {"urls": "stun:stun.internal:3478"},
                {
                    "urls": "turn:192.0.2.10:3478?transport=udp",
                    "username": "u",
                    "credential": "p",
                },
                {
                    "urls": "turns:turn.example.com:5349?transport=tcp",
                    "username": "u",
                    "credential": "p",
                },
            ]
        )
        assert len(entries) == 3

    def test_flattens_standard_url_arrays(self):
        entries = i.validate_server_ice_servers(
            [
                {"urls": ["stun:stun.example.com:3478"]},
                {
                    "urls": [
                        "turn:turn.example.com:3478?transport=udp",
                        "turn:turn.example.com:443?transport=tcp",
                    ],
                    "username": "u",
                    "credential": "p",
                },
            ]
        )

        assert entries == [
            {"urls": "stun:stun.example.com:3478"},
            {
                "urls": "turn:turn.example.com:3478?transport=udp",
                "username": "u",
                "credential": "p",
            },
            {
                "urls": "turn:turn.example.com:443?transport=tcp",
                "username": "u",
                "credential": "p",
            },
        ]

    def test_flattens_mixed_url_array_with_object_level_turn_credentials(self):
        entries = i.validate_server_ice_servers(
            [
                {
                    "urls": [
                        "stun:stun.example.com:3478",
                        "turn:turn.example.com:443",
                    ],
                    "username": "u",
                    "credential": "p",
                }
            ]
        )

        assert entries == [
            {"urls": "stun:stun.example.com:3478"},
            {
                "urls": "turn:turn.example.com:443",
                "username": "u",
                "credential": "p",
            },
        ]

    def test_accepted_servers_are_consumed_by_aiortc(self):
        pytest.importorskip("aiortc")
        from aiortc.rtcicetransport import connection_kwargs

        entries = i.validate_server_ice_servers(
            [
                {"urls": "stun:stun.example.com:3478"},
                {
                    "urls": "turns:turn.example.com:5349?transport=tcp",
                    "username": "u",
                    "credential": "p",
                },
            ]
        )
        kwargs = connection_kwargs(i.build_rtc_ice_servers(entries))

        assert kwargs["stun_server"] == ("stun.example.com", 3478)
        assert kwargs["turn_server"] == ("turn.example.com", 5349)
        assert kwargs["turn_ssl"] is True

    def test_bounds_expanded_url_arrays(self):
        with pytest.raises(i.IceServerConfigError):
            i.validate_server_ice_servers(
                [{"urls": [f"stun:stun-{i}.example.com" for i in range(25)]}]
            )

    @pytest.mark.parametrize(
        "entries",
        [
            [],
            "stun:stun.example.com",
            [{"urls": "https://example.com"}],
            [{"urls": "turn:turn.example.com:0", "username": "u", "credential": "p"}],
            [{"urls": "turn:turn.example.com:3478"}],
            [{"urls": "stun:stun.example.com:3478", "username": "u"}],
            [{"urls": []}],
            [{"urls": "stuns:stun.example.com:5349"}],
            [{"urls": "stun:stun.example.com:3478?transport=udp"}],
            [
                {
                    "urls": "turns:turn.example.com:5349?transport=udp",
                    "username": "u",
                    "credential": "p",
                }
            ],
            [
                {
                    "urls": "turns:[2001:db8::1]:5349?transport=tcp",
                    "username": "u",
                    "credential": "p",
                }
            ],
            [
                {
                    "urls": "turn:user@turn.example.com:3478",
                    "username": "u",
                    "credential": "p",
                }
            ],
            [{"urls": "turn:[not-ipv6]:3478", "username": "u", "credential": "p"}],
        ],
    )
    def test_rejects_malformed_server_configuration(self, entries):
        with pytest.raises(i.IceServerConfigError):
            i.validate_server_ice_servers(entries)


# ---------------------------------------------------------------------------
# App wiring: secrets declared, /info reports TURN honestly
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Canonical ICE Servers Array: strict parsing + correct relay hostnames
# ---------------------------------------------------------------------------


# A realistic Metered ICE array — the relay lives on *.relay.metered.ca, NOT on
# the <app>.metered.live API host. This is exactly the hostname the old code got
# wrong (it synthesised turn:<app>.metered.live:... which runs no TURN service).
_CANONICAL_ARRAY = [
    {"urls": "stun:stun.relay.metered.ca:80"},
    {
        "urls": "turn:global.relay.metered.ca:80",
        "username": "echoed",
        "credential": "echoed",
    },
    {
        "urls": "turn:global.relay.metered.ca:80?transport=tcp",
        "username": "e",
        "credential": "e",
    },
    {"urls": "turn:global.relay.metered.ca:443", "username": "e", "credential": "e"},
    {
        "urls": "turn:global.relay.metered.ca:443?transport=tcp",
        "username": "e",
        "credential": "e",
    },
    {
        "urls": "turns:global.relay.metered.ca:443?transport=tcp",
        "username": "e",
        "credential": "e",
    },
]


class TestValidateIceUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "stun:stun.relay.metered.ca:80",
            "turn:global.relay.metered.ca:443?transport=tcp",
            "turns:global.relay.metered.ca:443?transport=tcp",
            "turn:my-app.metered.live:80",
        ],
    )
    def test_accepts_metered_relay_urls(self, url):
        assert m._validate_ice_url(url) == url

    @pytest.mark.parametrize(
        "url",
        [
            "https://evil.example/steal",
            "turn:evil.com:443",
            "turn:global.relay.metered.ca.evil.com:443",  # apex not a suffix boundary
            "turn:169.254.169.254:443",
            "turn:user@evil.com:443",  # userinfo
            "turn:global.relay.metered.ca:443/path",  # path
            "turn:global.relay.metered.ca:",
            "turn:global.relay.metered.ca:443extra",
            "turn:global.relay.metered.ca:0",
            "turn:global.relay.metered.ca:65536",
            "",
            "turn:" + "x" * (m.MAX_ICE_URL_LENGTH + 1) + ".metered.ca:443",
        ],
    )
    def test_rejects_non_metered_or_malformed(self, url):
        with pytest.raises(m.MeteredFetchError):
            m._validate_ice_url(url)


class TestParseMeteredIceArray:
    def test_uses_relay_hosts_and_minted_creds(self):
        entries = m.parse_metered_ice_array(
            _CANONICAL_ARRAY, username="minted-user", credential="minted-pass"
        )
        # A public STUN is always guaranteed first.
        assert entries[0] == {"urls": i.DEFAULT_STUN_URL}
        urls = [e["urls"] for e in entries]
        # The correct relay host is carried through verbatim; the broken
        # <app>.metered.live host is nowhere in the result.
        assert "turn:global.relay.metered.ca:443?transport=tcp" in urls
        assert not any(".metered.live" in u for u in urls)
        # Every TURN entry carries the *minted* pair (never the response's echoes).
        for e in entries:
            if e["urls"].startswith(("turn:", "turns:")):
                assert e["username"] == "minted-user"
                assert e["credential"] == "minted-pass"

    def test_drops_bad_entries_but_keeps_valid_turn(self):
        mixed = [
            {"urls": "https://evil.example"},
            {"urls": "turn:evil.com:443", "username": "x", "credential": "y"},
            {"urls": "turn:global.relay.metered.ca:443?transport=tcp"},
            "not-a-dict",
        ]
        entries = m.parse_metered_ice_array(mixed, username="u", credential="c")
        urls = [e["urls"] for e in entries]
        assert urls == [
            i.DEFAULT_STUN_URL,
            "turn:global.relay.metered.ca:443?transport=tcp",
        ]

    def test_no_turn_raises(self):
        with pytest.raises(m.MeteredFetchError, match="no usable TURN"):
            m.parse_metered_ice_array(
                [{"urls": "stun:stun.relay.metered.ca:80"}],
                username="u",
                credential="c",
            )

    def test_non_array_raises(self):
        with pytest.raises(m.MeteredFetchError, match="not a JSON array"):
            m.parse_metered_ice_array({"urls": "x"}, username="u", credential="c")

    def test_too_many_entries_raises(self):
        big = [
            {
                "urls": "turn:global.relay.metered.ca:443",
                "username": "u",
                "credential": "c",
            }
        ] * (m.MAX_ICE_ENTRIES + 1)
        with pytest.raises(m.MeteredFetchError, match="too many"):
            m.parse_metered_ice_array(big, username="u", credential="c")

    def test_missing_creds_raise(self):
        with pytest.raises(m.MeteredFetchError):
            m.parse_metered_ice_array(_CANONICAL_ARRAY, username="", credential="c")


class TestFetchMeteredIceArray:
    def test_request_shape(self, monkeypatch):
        body = json.dumps(_CANONICAL_ARRAY).encode()
        opener = _FakeOpener(lambda req: _FakeResponse(body))
        monkeypatch.setattr(m, "_OPENER", opener)
        data = m.fetch_metered_ice_array("h.metered.live", "ak-123")
        assert isinstance(data, list) and len(data) == len(_CANONICAL_ARRAY)
        req = opener.last_request
        assert req.method == "GET"
        assert req.full_url.startswith(
            "https://h.metered.live/api/v1/turn/credentials?"
        )
        assert "apiKey=ak-123" in req.full_url

    def test_http_error_redacted(self, monkeypatch):
        def handler(req):
            raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, None)

        monkeypatch.setattr(m, "_OPENER", _FakeOpener(handler))
        with pytest.raises(m.MeteredFetchError) as excinfo:
            m.fetch_metered_ice_array("h.metered.live", "ak-super-secret")
        assert "403" in str(excinfo.value)
        assert "ak-super-secret" not in str(excinfo.value)


class TestMintIceServers:
    def test_primary_path_uses_canonical_array(self, monkeypatch):
        monkeypatch.setattr(
            m, "fetch_metered_credential", lambda h, s, e: ("mu", "mp", "ak-1")
        )
        monkeypatch.setattr(
            m, "fetch_metered_ice_array", lambda h, ak: _CANONICAL_ARRAY
        )
        entries = m.mint_ice_servers("h.metered.live", SECRET, 3600)
        urls = [e["urls"] for e in entries]
        assert "turn:global.relay.metered.ca:443?transport=tcp" in urls
        assert not any(".metered.live" in u for u in urls)
        # minted creds attached
        turn = next(e for e in entries if e["urls"].startswith("turn:"))
        assert turn["username"] == "mu" and turn["credential"] == "mp"

    def test_falls_back_to_global_relay_when_no_apikey(self, monkeypatch):
        monkeypatch.setattr(
            m, "fetch_metered_credential", lambda h, s, e: ("mu", "mp", None)
        )

        # Must NOT call the array endpoint when there's no apiKey.
        def boom(*a):
            raise AssertionError("should not fetch array without apiKey")

        monkeypatch.setattr(m, "fetch_metered_ice_array", boom)
        entries = m.mint_ice_servers("h.metered.live", SECRET, 3600)
        urls = [e["urls"] for e in entries]
        # Fallback synthesises against the GLOBAL RELAY host, never the API host.
        assert any(m.FALLBACK_RELAY_HOST in u for u in urls)
        assert not any(".metered.live" in u for u in urls)

    def test_falls_back_when_array_fetch_fails(self, monkeypatch):
        monkeypatch.setattr(
            m, "fetch_metered_credential", lambda h, s, e: ("mu", "mp", "ak-1")
        )

        def boom(h, ak):
            raise m.MeteredFetchError("array down")

        monkeypatch.setattr(m, "fetch_metered_ice_array", boom)
        entries = m.mint_ice_servers("h.metered.live", SECRET, 3600)
        assert any(m.FALLBACK_RELAY_HOST in e["urls"] for e in entries)


class TestIceCandidateTypeCounts:
    def test_counts_types_without_addresses(self):
        sdp = (
            "v=0\r\n"
            "a=candidate:1 1 udp 2122260223 192.168.1.5 54321 typ host generation 0\r\n"
            "a=candidate:2 1 udp 1686052607 203.0.113.9 40000 typ srflx"
            " raddr 0.0.0.0\r\n"
            "a=candidate:3 1 tcp 25108479 198.51.100.2 443 typ relay raddr 0.0.0.0\r\n"
            "a=candidate:4 1 udp 1 10.0.0.1 5 typ host\r\n"
        )
        counts = i.ice_candidate_type_counts(sdp)
        assert counts == {"host": 2, "srflx": 1, "relay": 1}

    def test_empty_when_no_candidates(self):
        assert (
            i.ice_candidate_type_counts("v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\n")
            == {}
        )
