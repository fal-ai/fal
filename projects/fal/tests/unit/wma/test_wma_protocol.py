# ruff: noqa: E402
from __future__ import annotations

import json

import pytest

pytest.importorskip("pydantic", minversion="2")
from fal.wma import (
    KeysMessage,
    KeyState,
    PingMessage,
    PromptMessage,
    ResetMessage,
    control_json,
    parse_control_message,
)

KEY_ORDER = ("W", "A", "S", "D", "I", "J", "K", "L")
CONFLICT_GROUPS = (("W", "S"), ("A", "D"), ("I", "K"), ("J", "L"))


def make_key_state() -> KeyState:
    return KeyState(KEY_ORDER, CONFLICT_GROUPS)


class TestKeyState:
    def test_initial_sample_is_all_false(self):
        assert make_key_state().sample() == {key: False for key in KEY_ORDER}

    def test_pressed_keys_persist_across_samples(self):
        state = make_key_state()
        state.update(pressed=["W"], activated=[])
        assert state.sample()["W"] is True
        assert state.sample()["W"] is True

    def test_activated_keys_are_consumed_by_one_sample(self):
        state = make_key_state()
        state.update(pressed=[], activated=["D"])
        assert state.sample()["D"] is True
        assert state.sample()["D"] is False

    def test_tap_between_samples_is_not_lost(self):
        state = make_key_state()
        state.update(pressed=[], activated=["A"])
        # A later report without the tap must not erase it before it is sampled.
        state.update(pressed=[], activated=[])
        assert state.sample()["A"] is True

    def test_activated_wins_conflict_against_pressed(self):
        state = make_key_state()
        state.update(pressed=["W"], activated=["S"])
        sample = state.sample()
        assert sample["S"] is True
        assert sample["W"] is False
        # The tap is consumed; the held key acts again on the next sample.
        assert state.sample()["W"] is True

    def test_conflicting_pressed_keys_are_both_kept(self):
        state = make_key_state()
        state.update(pressed=["W", "S"], activated=[])
        sample = state.sample()
        assert sample["W"] is True and sample["S"] is True

    def test_unknown_and_lowercase_keys(self):
        state = make_key_state()
        state.update(pressed=["w", "SHIFT", "x"], activated=["escape"])
        sample = state.sample()
        assert sample["W"] is True
        assert sum(sample.values()) == 1


class TestControlMessages:
    def test_parse_keys(self):
        message = parse_control_message(
            '{"type": "keys", "pressed": ["W"], "activated": ["D"]}'
        )
        assert isinstance(message, KeysMessage)
        assert message.pressed == ["W"]

    def test_parse_prompt(self):
        message = parse_control_message('{"type": "prompt", "prompt": "a forest"}')
        assert isinstance(message, PromptMessage)

    def test_parse_reset_and_ping(self):
        reset = parse_control_message('{"type": "reset", "preset": "example"}')
        assert isinstance(reset, ResetMessage)
        assert reset.image_url is None
        ping = parse_control_message('{"type": "ping", "ts": 12.5}')
        assert isinstance(ping, PingMessage)

    @pytest.mark.parametrize(
        "raw",
        [
            "not json",
            '{"type": "unknown"}',
            '{"pressed": ["W"]}',
            '{"type": "prompt", "prompt": ""}',
        ],
    )
    def test_malformed_messages_raise_value_error(self, raw):
        with pytest.raises(ValueError):
            parse_control_message(raw)

    def test_control_json_roundtrip(self):
        payload = json.loads(control_json("stats", generation_fps=15.9))
        assert payload == {"type": "stats", "generation_fps": 15.9}
