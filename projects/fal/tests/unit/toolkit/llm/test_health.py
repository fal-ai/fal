from __future__ import annotations

import time
import types

import httpx
import pytest

from fal.toolkit.llm import LLMApp, _app


def test_health_and_engine_stats(served):
    app, url, _, _ = served
    h = httpx.get(f"{url}/health").json()
    assert h["status"] == "ok"
    assert h["engine"]["vllm:num_requests_running"] == 2.0
    s = httpx.get(f"{url}/engine-stats").json()
    assert s["stats"]["vllm:kv_cache_usage_perc"] == 0.25
    assert "preread_s" in s["boot"] and "engine_ready_s" in s["boot"]
    # The fake engine prints no vLLM milestones, so none are recorded.
    assert "weights_loaded_s" not in s["boot"]
    assert "other_metric" not in s["stats"]
    assert s["stats"]["spec_decode_acceptance_rate"] == 0.75
    # The per-position counter is not summed into the accepted total.
    assert s["stats"]["vllm:spec_decode_num_accepted_tokens_total"] == 30.0
    assert h["engine"]["spec_decode_acceptance_rate"] == 0.75

    app._proc.kill()
    app._proc.wait()
    assert httpx.get(f"{url}/health").status_code == 503


def test_health_check_is_the_engine_probe(served):
    app, url, _, _ = served
    config = type(app).get_health_check_config()
    assert config is not None
    assert (config.path, config.method) == ("/engine-health", "POST")
    assert config.timeout_seconds == 10
    h = httpx.post(f"{url}/engine-health").json()
    assert h["status"] == "ok"
    assert h["engine"]["vllm:generation_tokens_total"] == 50.0

    app._proc.kill()
    app._proc.wait()
    r = httpx.post(f"{url}/engine-health")
    assert r.status_code == 503
    assert "exited" in r.json()["detail"]


@pytest.mark.parametrize("stream", [False, True])
def test_engine_5xx_asks_gateway_for_health_check(served, stream):
    # One failed request must not replace the runner: on a 5xx the gateway
    # probes the health check (still passing here) instead.
    _, url, _, _ = served
    for status in (500, 503):
        r = httpx.post(
            f"{url}/v1/chat/completions",
            json={"fail": True, "fail_status": status, "stream": stream},
        )
        assert r.status_code == status
        assert r.json() == {"error": "boom"}
        assert r.headers["x-fal-runner-health-check"] == "true"
    r = httpx.post(f"{url}/v1/chat/completions", json={"fail": True, "stream": stream})
    assert r.status_code == 400
    assert "x-fal-runner-health-check" not in r.headers
    r = httpx.post(f"{url}/v1/chat/completions", json={"stream": stream})
    assert r.status_code == 200
    assert "x-fal-runner-health-check" not in r.headers
    assert httpx.post(f"{url}/engine-health").status_code == 200


def test_consecutive_engine_errors_fail_health(served, monkeypatch):
    app, url, _, _ = served
    monkeypatch.setattr(type(app), "engine_error_threshold", 3)
    fail = {"fail": True, "fail_status": 500}
    for _ in range(2):
        assert httpx.post(f"{url}/v1/chat/completions", json=fail).status_code == 500
    assert httpx.post(f"{url}/engine-health").status_code == 200
    # Any engine reply below 500 (a 4xx is the caller's fault) resets the count.
    httpx.post(f"{url}/v1/chat/completions", json={"fail": True})
    for _ in range(2):
        httpx.post(f"{url}/v1/chat/completions", json={**fail, "stream": True})
    assert httpx.post(f"{url}/engine-health").status_code == 200
    httpx.post(f"{url}/v1/chat/completions", json=fail)
    r = httpx.post(f"{url}/engine-health")
    assert r.status_code == 503
    assert "3 requests in a row" in r.json()["detail"]
    # A successful stream resets it.
    assert httpx.post(f"{url}/v1/chat/completions", json={"stream": True}).is_success
    assert httpx.post(f"{url}/engine-health").status_code == 200
    # /v1/models is not inference: it neither counts nor resets.
    for _ in range(2):
        httpx.post(f"{url}/v1/chat/completions", json=fail)
    assert httpx.get(f"{url}/v1/models").status_code == 200
    httpx.post(f"{url}/v1/chat/completions", json=fail)
    assert httpx.post(f"{url}/engine-health").status_code == 503


def test_engine_errors_off_by_default(served):
    app, url, _, _ = served
    assert type(app).engine_error_threshold is None
    for _ in range(12):
        httpx.post(
            f"{url}/v1/chat/completions", json={"fail": True, "fail_status": 500}
        )
    assert httpx.post(f"{url}/engine-health").status_code == 200


def test_engine_errors_decay(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(_app.time, "monotonic", lambda: clock[0])
    probe = types.SimpleNamespace(_engine_errors=0, _engine_error_at=0.0)
    count = types.MethodType(LLMApp._count_engine_reply, probe)
    probe._recent_engine_errors = types.MethodType(LLMApp._recent_engine_errors, probe)
    for _ in range(3):
        count(500)
        clock[0] += 30.0
    assert probe._recent_engine_errors(clock[0]) == 3
    # A gap longer than the window starts over, for counting and for health.
    clock[0] += 31.0
    assert probe._recent_engine_errors(clock[0]) == 0
    count(503)
    assert probe._recent_engine_errors(clock[0]) == 1
    count(404)
    assert probe._recent_engine_errors(clock[0]) == 0


def test_no_probe_header_while_draining(served):
    app, url, _, _ = served
    assert app._engine_reply(b"{}", 500).headers["x-fal-runner-health-check"]
    app._draining = True
    try:
        assert "x-fal-runner-health-check" not in app._engine_reply(b"{}", 500).headers
    finally:
        app._draining = False


def _stall_probe(timeout):
    return types.SimpleNamespace(engine_stall_timeout_s=timeout, _progress=None)


def _stats(tokens, running=1.0, waiting=0.0):
    return {
        "vllm:prompt_tokens_total": 100.0,
        "vllm:generation_tokens_total": tokens,
        "vllm:num_requests_running": running,
        "vllm:num_requests_waiting": waiting,
    }


def test_stall_needs_pending_requests_and_frozen_counters():
    probe = _stall_probe(180.0)
    stalled = LLMApp._stalled_for
    assert stalled(probe, _stats(10.0), 0.0) is None
    assert stalled(probe, _stats(10.0), 100.0) is None
    assert stalled(probe, _stats(10.0), 181.0) == 181.0
    # Progress restarts the clock.
    assert stalled(probe, _stats(11.0), 200.0) is None
    assert stalled(probe, _stats(11.0), 300.0) is None
    # An idle engine is not stalled, however long its counters sit still.
    assert stalled(probe, _stats(11.0, running=0.0), 1000.0) is None
    assert stalled(probe, _stats(11.0, running=0.0), 5000.0) is None
    # The clock starts at the first busy sample, not at an old idle one.
    assert stalled(probe, _stats(11.0, running=0.0, waiting=3.0), 5100.0) is None
    assert stalled(probe, _stats(11.0, running=0.0, waiting=3.0), 5200.0) is None
    # Queued but nothing running counts as pending.
    assert stalled(probe, _stats(11.0, running=0.0, waiting=3.0), 5281.0) == 181.0


def test_stall_clock_ignores_idle_gap_before_failed_reads():
    probe = _stall_probe(180.0)
    stalled = LLMApp._stalled_for
    assert stalled(probe, _stats(10.0, running=0.0), 0.0) is None
    assert stalled(probe, {}, 5.0) is None
    assert stalled(probe, _stats(10.0), 200.0) is None
    assert stalled(probe, _stats(10.0), 381.0) == 181.0


def test_stall_check_skipped_when_off_or_metrics_missing():
    off = _stall_probe(None)
    assert LLMApp._stalled_for(off, _stats(1.0), 0.0) is None
    assert LLMApp._stalled_for(off, _stats(1.0), 1e6) is None
    probe = _stall_probe(1.0)
    no_counters = {"vllm:num_requests_running": 4.0}
    assert LLMApp._stalled_for(probe, no_counters, 0.0) is None
    assert LLMApp._stalled_for(probe, no_counters, 1e6) is None
    # SGLang counts tokens per finished request, so it is never judged.
    sglang = {
        "sglang:num_running_reqs": 1.0,
        "sglang:generation_tokens_total": 5.0,
        "sglang:prompt_tokens_total": 5.0,
    }
    assert LLMApp._stalled_for(probe, sglang, 0.0) is None
    assert LLMApp._stalled_for(probe, sglang, 1e6) is None
    # A failed /metrics read (empty stats) neither stalls nor resets the clock.
    assert LLMApp._stalled_for(probe, _stats(1.0), 0.0) is None
    assert LLMApp._stalled_for(probe, {}, 5.0) is None
    assert LLMApp._stalled_for(probe, _stats(1.0), 10.0) == 10.0


def test_health_fails_on_stalled_engine(served, monkeypatch):
    app, url, _, _ = served
    # The fake engine reports 2 running requests and fixed token counters.
    monkeypatch.setattr(type(app), "engine_stall_timeout_s", 0.2)
    assert httpx.post(f"{url}/engine-health").status_code == 200
    time.sleep(0.4)
    r = httpx.post(f"{url}/engine-health")
    assert r.status_code == 503
    assert "no progress" in r.json()["detail"]
    assert httpx.get(f"{url}/health").status_code == 503
