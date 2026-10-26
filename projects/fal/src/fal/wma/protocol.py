"""Typed data-channel control protocol for interactive WMA apps.

Client -> server messages ride a WebRTC data channel as JSON; the schema here
is shared between runners and clients so both sides validate against the same
contract. Public contracts expose semantic commands such as ``forward``. The
legacy key report remains accepted as a compatibility input, but physical keys
are not the model API.

Everything in this module is pure Python (no aiortc/av), so it can be imported
and unit-tested locally.
"""

import json
import threading
from typing import Any, Callable, Dict, List, Literal, Set, Tuple, Type, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    create_model,
)
from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated

# ---------------------------------------------------------------------------
# Control-channel protocol (client -> server, JSON over a WebRTC data channel)
# ---------------------------------------------------------------------------


class KeysMessage(BaseModel):
    """Legacy physical-key report retained for existing WMA clients."""

    model_config = ConfigDict(json_schema_extra={"deprecated": True})

    type: Literal["keys"]
    pressed: List[str] = Field(default_factory=list, max_length=32)
    activated: List[str] = Field(default_factory=list, max_length=32)


class CommandsMessage(BaseModel):
    """Semantic command-state report.

    ``active`` contains commands currently held by the user and ``activated``
    contains edge-triggered commands since the previous report. A browser may
    map a keyboard, gamepad, touch control, or agent action to these names.
    """

    type: Literal["commands"]
    active: List[str] = Field(default_factory=list, max_length=32)
    activated: List[str] = Field(default_factory=list, max_length=32)


def commands_message(
    name: str,
    *,
    commands: Tuple[str, ...],
    conflict_groups: Tuple[Tuple[str, str], ...] = (),
) -> Type[CommandsMessage]:
    """A semantic command report narrowed to one app's vocabulary."""

    command = Literal[commands]  # type: ignore[valid-type]
    reported = List[command]

    def field() -> Any:
        return Field(default_factory=list, max_length=len(commands))

    model = create_model(
        name,
        __base__=CommandsMessage,
        __doc__=CommandsMessage.__doc__,
        active=(reported, field()),
        activated=(reported, field()),
    )
    if conflict_groups:
        model.model_config = ConfigDict(
            json_schema_extra={
                "x-fal-ui": {"conflictGroups": [list(pair) for pair in conflict_groups]}
            }
        )
        model.model_rebuild(force=True)
    return model


def _key_normalizer(keys: Tuple[str, ...]) -> Callable[[Any], Any]:
    """Fold a client key report onto an app's vocabulary: upper-case it, drop
    keys the app does not bind, and collapse repeats.

    This is the leniency :class:`KeyState` has always applied on the way in
    (``{k.upper() for k in pressed} & self._known``), moved to the message
    boundary. Moving it is what lets the field be typed as the app's key enum
    without tightening the wire: a browser sending ``"w"`` or a stray
    ``"SHIFT"`` is normalized exactly as before rather than failing validation
    and losing the whole report.
    """

    known = {key.upper() for key in keys}

    def normalize(value: Any) -> Any:
        if not isinstance(value, list):
            return value
        bound: List[str] = []
        for entry in value:
            if not isinstance(entry, str):
                # Not ours to coerce; let the enum report it as the type error.
                return value
            upper = entry.upper()
            if upper in known and upper not in bound:
                bound.append(upper)
        return bound

    return normalize


def keys_message(
    name: str,
    *,
    keys: Tuple[str, ...],
    conflict_groups: Tuple[Tuple[str, str], ...] = (),
) -> Type[KeysMessage]:
    """A :class:`KeysMessage` narrowed to one app's key bindings.

    Typing the fields as ``Literal[*keys]`` publishes the bindings as a JSON
    Schema ``enum`` (order preserved) instead of an opaque ``list[str]``.
    ``conflict_groups`` rides along as a ``ui`` hint because mutual exclusion
    between array items has no JSON Schema spelling. Both arguments are the
    ones handed to :class:`KeyState`, so contract and sampling cannot drift.
    """

    key = Literal[keys]  # type: ignore[valid-type]
    reported = Annotated[List[key], BeforeValidator(_key_normalizer(keys))]

    def field() -> Any:
        # Normalization dedupes against a fixed vocabulary, so the key count is
        # the true bound rather than the base class's arbitrary one.
        return Field(default_factory=list, max_length=len(keys))

    model = create_model(
        name,
        __base__=KeysMessage,
        # create_model does not inherit the base docstring, and the published
        # schema would otherwise lose the pressed/activated semantics -- the
        # one part of this message an enum cannot explain.
        __doc__=KeysMessage.__doc__,
        pressed=(reported, field()),
        activated=(reported, field()),
    )
    schema_extra: Dict[str, Any] = {"deprecated": True}
    if conflict_groups:
        schema_extra["x-fal-ui"] = {
            "conflictGroups": [list(pair) for pair in conflict_groups]
        }
    model.model_config = ConfigDict(json_schema_extra=schema_extra)
    model.model_rebuild(force=True)
    return model


class PromptMessage(BaseModel):
    """Hot-swap the world prompt without restarting the stream."""

    type: Literal["prompt"]
    prompt: str = Field(min_length=1, max_length=2000)


class ResetMessage(BaseModel):
    """Restart the world, optionally from a new prompt, a named seed preset, or a
    seed image URL (``image_url`` wins when both are given)."""

    type: Literal["reset"]
    prompt: Union[str, None] = Field(default=None, max_length=2000)
    preset: Union[str, None] = Field(default=None, max_length=100)
    image_url: Union[str, None] = Field(default=None, max_length=4096)


class PingMessage(BaseModel):
    """Latency probe; the server echoes ``ts`` back in a ``pong`` payload."""

    type: Literal["ping"]
    ts: Union[float, None] = None


ControlMessage = Annotated[
    Union[Union[Union[KeysMessage, PromptMessage], ResetMessage], PingMessage],
    Field(discriminator="type"),
]

_CONTROL_MESSAGE_ADAPTER: TypeAdapter = TypeAdapter(ControlMessage)


def parse_control_message(
    raw: Union[str, bytes], adapter: Union[TypeAdapter, None] = None
) -> ControlMessage:
    """Parse one data-channel message. Raises ``ValueError`` on malformed input.

    An app that narrowed its vocabulary passes the adapter for its own union so
    that the rejection wording, and the point at which a bad message is
    rejected, stay the same across every app.
    """
    try:
        return (adapter or _CONTROL_MESSAGE_ADAPTER).validate_json(raw)
    except ValidationError as exc:
        raise ValueError(f"invalid control message: {exc.errors()[0]['msg']}") from exc


def control_json(type_: str, **fields: Any) -> str:
    """Serialize a server -> client control payload (sent over the data channel)."""
    return json.dumps({"type": type_, **fields})


# Control-channel protocol (server -> client). These three messages are the
# ones every interactive app sends with the same field names; per-app payloads
# (``session_info``, ``stats``) belong to the app that sends them.


class PongMessage(BaseModel):
    """Reply to a ``ping``, echoing the client's timestamp for RTT measurement.

    The echoed timestamp has two wire spellings: apps that handle ``ping``
    themselves echo ``ts``; the fallback handler in
    :class:`fal.wma.sdk.Session` replies with ``client_ts``. Clients read
    whichever is present. ``server_ts`` is the (unsynchronized) runner clock; it
    only shows how the round trip divides once the client has an offset estimate.
    """

    type: Literal["pong"] = "pong"
    ts: Union[float, None] = None
    client_ts: Union[float, None] = None
    server_ts: Union[float, None] = None


class ErrorMessage(BaseModel):
    """A session failure. Depending on the app and failure, the peer may remain
    usable or may close; clients should observe connection state before retrying."""

    type: Literal["error"] = "error"
    error: str


class StreamExhaustedMessage(BaseModel):
    """The model has no more frames to send and the current peer is closing.
    Clients must negotiate a new session to continue."""

    type: Literal["stream_exhausted"] = "stream_exhausted"


ServerMessage = Annotated[
    Union[Union[PongMessage, ErrorMessage], StreamExhaustedMessage],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Key state (pressed/activated sets with conflict resolution)
# ---------------------------------------------------------------------------


class KeyState:
    """Thread-safe key-state accumulator sampled once per generation step.

    Semantics match the interaction model used by keyboard-driven world models:
    clients report ``pressed`` (currently held) and ``activated``
    (tapped since last report). ``sample()`` merges ``pressed | activated`` with
    ``activated`` winning conflicts, then consumes ``activated`` so a tap acts for
    exactly one step while held keys keep acting.
    """

    def __init__(
        self,
        key_order: Tuple[str, ...],
        conflict_groups: Tuple[Tuple[str, str], ...] = (),
    ) -> None:
        self.key_order = key_order
        self.conflict_groups = conflict_groups
        self._known = set(key_order)
        self._pressed: Set[str] = set()
        self._activated: Set[str] = set()
        self._lock = threading.Lock()

    def update(self, pressed: List[str], activated: List[str]) -> None:
        """Apply a client key report. Unknown keys are ignored."""
        new_pressed = {k.upper() for k in pressed} & self._known
        new_activated = {k.upper() for k in activated} & self._known
        with self._lock:
            self._pressed = new_pressed
            # Merge (|=) so a tap reported between samples is never dropped; the
            # freshly activated keys win conflicts against older ones.
            merged = self._activated | new_activated
            self._activated = self._resolve_conflicts(
                merged, high_priority=new_activated
            )

    def sample(self) -> Dict[str, bool]:
        """Snapshot the effective key state and consume ``activated``."""
        with self._lock:
            combined = self._pressed | self._activated
            combined = self._resolve_conflicts(combined, high_priority=self._activated)
            self._activated.clear()
        return {key: key in combined for key in self.key_order}

    def _resolve_conflicts(self, keys: Set[str], high_priority: Set[str]) -> Set[str]:
        """For each conflicting pair with both keys present, keep the high-priority
        one. If both or neither are high-priority, keep both (don't guess)."""
        result = set(keys)
        for a, b in self.conflict_groups:
            if a not in result or b not in result:
                continue
            a_high = a in high_priority
            b_high = b in high_priority
            if a_high and not b_high:
                result.discard(b)
            elif b_high and not a_high:
                result.discard(a)
        return result
