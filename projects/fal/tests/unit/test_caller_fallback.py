"""Unit tests for the universal caller-defined fallback engine (fal._caller_fallback).

Covers the pure pieces that the framework middleware composes: lenient chain
parsing, status -> trigger mapping, output normalization, and the chain walk
(including robustness against bad caller input and the output-shape guard).
"""

import asyncio

import pytest

from fal._caller_fallback import (
    FallbackNode,
    normalize_output,
    parse_fallback_chain,
    run_fallback_chain,
    trigger_for_status,
)

GPT = "fal-ai/gpt-image-2"
SEEDREAM = "fal-ai/bytedance/seedream/v4.5/edit"


# --- parse_fallback_chain -----------------------------------------------------


def test_parse_valid_chain():
    body = {
        "prompt": "x",
        "fal_fallback": [
            {"endpoint": GPT, "input": {"prompt": "x"}, "on": ["error"], "timeout": 30}
        ],
    }
    chain = parse_fallback_chain(body)
    assert len(chain) == 1
    assert chain[0].endpoint == GPT
    assert chain[0].input == {"prompt": "x"}
    assert chain[0].on == ["error"]
    assert chain[0].timeout == 30


@pytest.mark.parametrize("body", [None, "str", 123, {}, {"fal_fallback": "nope"}, {"fal_fallback": {}}])
def test_parse_absent_or_malformed_yields_empty(body):
    assert parse_fallback_chain(body) == []


def test_parse_skips_bad_items_keeps_good():
    body = {"fal_fallback": ["not a dict", {"endpoint": GPT}, 42, {"input": {"a": 1}}]}
    chain = parse_fallback_chain(body)
    # Two dict items survive (one with endpoint, one endpoint-less but parseable).
    assert [n.endpoint for n in chain] == [GPT, None]


def test_parse_lenient_defaults():
    chain = parse_fallback_chain({"fal_fallback": [{"endpoint": GPT}]})
    node = chain[0]
    assert node.input == {}
    assert node.on == ["error", "timeout"]
    assert node.timeout is None


# --- trigger_for_status -------------------------------------------------------


@pytest.mark.parametrize(
    "status,expected",
    [(408, "timeout"), (504, "timeout"), (500, "error"), (502, "error"), (503, "error")],
)
def test_trigger_for_failure_statuses(status, expected):
    assert trigger_for_status(status) == expected


@pytest.mark.parametrize("status", [200, 201, 301, 400, 401, 403, 422, 429])
def test_trigger_none_for_non_failure(status):
    assert trigger_for_status(status) is None


# --- normalize_output ---------------------------------------------------------


def test_normalize_single_to_list():
    assert normalize_output({"image": {"url": "a"}}, "images")["images"] == [{"url": "a"}]


def test_normalize_list_to_single():
    assert normalize_output({"images": [{"url": "a"}, {"url": "b"}]}, "image")["image"] == {"url": "a"}


def test_normalize_passthrough_and_non_dict():
    r = {"images": [{"url": "a"}]}
    assert normalize_output(r, "images") is r
    assert normalize_output("x", "images") == "x"
    assert normalize_output({"images": []}, None) == {"images": []}


# --- run_fallback_chain -------------------------------------------------------


def _forward_returning(mapping):
    async def _forward(endpoint, payload):
        value = mapping[endpoint]
        if isinstance(value, Exception):
            raise value
        return value

    return _forward


def test_chain_first_eligible_node_wins():
    chain = parse_fallback_chain({"fal_fallback": [{"endpoint": GPT, "input": {"prompt": "x"}}]})
    forward = _forward_returning({GPT: {"images": [{"url": "ok"}]}})
    result, served = asyncio.run(run_fallback_chain(chain, trigger="error", forward=forward))
    assert served == GPT
    assert result == {"images": [{"url": "ok"}]}


def test_chain_skips_node_when_trigger_not_listed():
    chain = parse_fallback_chain({"fal_fallback": [{"endpoint": GPT, "on": ["error"]}]})
    forward = _forward_returning({GPT: {"images": []}})
    result, served = asyncio.run(run_fallback_chain(chain, trigger="timeout", forward=forward))
    assert (result, served) == (None, None)


def test_chain_advances_past_failing_node():
    chain = parse_fallback_chain(
        {"fal_fallback": [{"endpoint": GPT}, {"endpoint": SEEDREAM}]}
    )
    forward = _forward_returning(
        {GPT: RuntimeError("down"), SEEDREAM: {"images": [{"url": "ok"}]}}
    )
    result, served = asyncio.run(run_fallback_chain(chain, trigger="error", forward=forward))
    assert served == SEEDREAM
    assert result == {"images": [{"url": "ok"}]}


def test_chain_exhausted_returns_none():
    chain = parse_fallback_chain({"fal_fallback": [{"endpoint": GPT}]})
    forward = _forward_returning({GPT: RuntimeError("down")})
    assert asyncio.run(run_fallback_chain(chain, trigger="error", forward=forward)) == (None, None)


def test_chain_endpointless_node_skipped():
    chain = parse_fallback_chain(
        {"fal_fallback": [{"input": {"a": 1}}, {"endpoint": GPT}]}
    )
    forward = _forward_returning({GPT: {"images": [{"url": "ok"}]}})
    result, served = asyncio.run(run_fallback_chain(chain, trigger="error", forward=forward))
    assert served == GPT


def test_chain_output_guard_skips_incompatible_then_serves():
    chain = parse_fallback_chain(
        {"fal_fallback": [{"endpoint": GPT}, {"endpoint": SEEDREAM}]}
    )
    forward = _forward_returning(
        {GPT: {"video": {"url": "v"}}, SEEDREAM: {"images": [{"url": "ok"}]}}
    )
    result, served = asyncio.run(
        run_fallback_chain(chain, trigger="error", forward=forward, output_field="images")
    )
    # GPT returned a video (no `images`) -> rejected; chain advanced to SEEDREAM.
    assert served == SEEDREAM
    assert result["images"] == [{"url": "ok"}]


def test_chain_normalizes_winner_to_output_field():
    chain = parse_fallback_chain({"fal_fallback": [{"endpoint": GPT}]})
    forward = _forward_returning({GPT: {"image": {"url": "single"}}})
    result, served = asyncio.run(
        run_fallback_chain(chain, trigger="error", forward=forward, output_field="images")
    )
    assert result["images"] == [{"url": "single"}]
