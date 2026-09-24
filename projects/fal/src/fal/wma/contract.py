"""Machine-readable contracts for WMA realtime sessions.

OpenAPI describes how a client creates a session. AsyncAPI describes what the
client sends and receives after WebRTC negotiation. A :class:`RealtimeContract`
is the single declaration from which both documents are rendered, so their
message and media descriptions cannot drift.

The OpenAPI path carries only a small ``x-fal-realtime`` discovery object. The
actual message contract is a valid, standalone AsyncAPI 3.1 document. WebRTC
media tracks are not messages and have no standard AsyncAPI binding, so they
live in the narrowly scoped ``x-fal-media`` extension on that document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Union

#: Key holding realtime discovery on the OpenAPI session path item.
CONTRACT_EXTENSION = "x-fal-realtime"

#: Version of the fal discovery object's shape. This is deliberately separate
#: from the WMA session protocol version and the AsyncAPI specification version.
CONTRACT_VERSION = 1

#: AsyncAPI specification and application-contract versions.
ASYNCAPI_SPEC_VERSION = "3.1.0"
ASYNCAPI_DOCUMENT_VERSION = "1.0.0"

#: The wire transport and fal session protocol are separate facts. WebRTC is
#: the transport; WMA defines how fal negotiates and uses it.
TRANSPORT_PROTOCOL = "webrtc"
SESSION_PROTOCOL = "wma"
SESSION_PROTOCOL_VERSION = 1

#: Relative document locations. A metadata host may rewrite these while
#: preserving the relationship.
ASYNCAPI_URL = "./asyncapi.json"
OPENAPI_URL = "./openapi.json"

DEFAULT_CONTENT_TYPE = "application/json"
CLIENT_OPERATION = "sendControl"
SERVER_OPERATION = "receiveEvents"

TrackKind = Literal["video", "audio"]
Direction = Literal["send", "receive"]
TrackSource = Literal["camera", "microphone", "screen"]

#: A Pydantic model class, or a discriminated-union alias. Unions are not
#: classes, so this cannot be narrowed to ``type[BaseModel]``.
MessageSchema = Any


def _type_adapter(schema: MessageSchema) -> Any:
    """Return a pydantic ``TypeAdapter`` for ``schema``.

    Imported lazily: rendering realtime contracts is the one part of
    ``fal.wma`` that requires pydantic v2 (``TypeAdapter``). Everything else —
    sessions, negotiation, billing — works on the SDK's full pydantic range,
    so the requirement is scoped to contract rendering rather than the
    package import.
    """
    try:
        from pydantic import TypeAdapter
    except ImportError as exc:  # pydantic v1
        raise RuntimeError(
            "declaring a RealtimeContract requires pydantic v2 "
            "(pydantic.TypeAdapter is unavailable)"
        ) from exc
    return TypeAdapter(schema)


@dataclass(frozen=True)
class Constraint:
    """A MediaTrackConstraint range for width, height, or frame rate."""

    min: float | None = None
    max: float | None = None
    ideal: float | None = None
    exact: float | None = None

    def __post_init__(self) -> None:
        if self.to_openapi() == {}:
            raise ValueError("a Constraint must bound something")

    def to_openapi(self) -> dict[str, float]:
        bounds = (
            ("min", self.min),
            ("max", self.max),
            ("ideal", self.ideal),
            ("exact", self.exact),
        )
        return {name: value for name, value in bounds if value is not None}


# Evaluated at runtime (it is a value, not an annotation), so it must use
# ``typing.Union`` for the 3.8/3.9 floor.
Measure = Union[int, float, "Constraint"]


def _measure(value: Measure) -> Any:
    return value.to_openapi() if isinstance(value, Constraint) else value


@dataclass(frozen=True)
class Track:
    """One WebRTC media track, described from the client's perspective.

    Client-to-model tracks carry MediaTrackConstraints, which a browser can
    pass to ``getUserMedia``. Model-to-client tracks carry MediaTrackSettings,
    which describe the stream the model already sends.
    """

    kind: TrackKind
    source: TrackSource | None = None
    required: bool = False
    width: Measure | None = None
    height: Measure | None = None
    frame_rate: Measure | None = None

    @property
    def measures(self) -> dict[str, Measure]:
        declared = (
            ("width", self.width),
            ("height", self.height),
            ("frameRate", self.frame_rate),
        )
        return {name: value for name, value in declared if value is not None}

    def to_openapi(self, direction: Direction = "send") -> dict[str, Any]:
        published: dict[str, Any] = {"kind": self.kind}
        if self.source is not None:
            published["source"] = self.source
        if self.required:
            published["required"] = True
        measures = self.measures
        if measures:
            key = "constraints" if direction == "send" else "settings"
            published[key] = {name: _measure(value) for name, value in measures.items()}
        return published


@dataclass(frozen=True)
class MediaContract:
    """Tracks the client sends and receives during the WebRTC session."""

    send: tuple[Track, ...] = ()
    receive: tuple[Track, ...] = ()

    def __post_init__(self) -> None:
        for track in self.receive:
            if any(isinstance(value, Constraint) for value in track.measures.values()):
                raise ValueError(
                    "an inbound track reports settings, not constraints: "
                    "nothing negotiates what the model already sends"
                )
            if track.source is not None:
                raise ValueError(
                    "source says where the browser captures an outbound track, "
                    "so it cannot describe one it receives"
                )

    def to_openapi(self) -> dict[str, Any]:
        return {
            "send": [track.to_openapi("send") for track in self.send],
            "receive": [track.to_openapi("receive") for track in self.receive],
        }


@dataclass(frozen=True)
class RealtimeContract:
    """What a client may exchange with one live model session.

    ``session_params`` is intentionally absent from v1. Session configuration
    must gain concrete wire semantics before it becomes part of the contract.
    """

    media: MediaContract = field(default_factory=MediaContract)
    client_messages: MessageSchema = None
    server_messages: MessageSchema = None


def message_types(schema: MessageSchema) -> tuple[str, ...]:
    """Return the ``type`` discriminator values accepted by ``schema``."""

    rendered = _type_adapter(schema).json_schema()
    discriminator = rendered.get("discriminator")
    if isinstance(discriminator, dict) and "mapping" in discriminator:
        return tuple(sorted(discriminator["mapping"]))

    literal = rendered.get("properties", {}).get("type", {})
    values = [literal["const"]] if "const" in literal else literal.get("enum", ())
    if any(not isinstance(value, str) for value in values):
        # AsyncAPI message names are strings and WMA dispatch compares
        # string wire types; a numeric tag would publish an invalid name.
        raise ValueError("wire 'type' discriminators must be strings")
    return tuple(sorted(values))


def apply_contract(
    spec: dict[str, Any],
    *,
    path: str,
    asyncapi_url: str = ASYNCAPI_URL,
) -> dict[str, Any]:
    """Attach realtime discovery to one OpenAPI path item, in place."""

    path_item = spec.get("paths", {}).get(path)
    if path_item is None:
        return spec

    path_item[CONTRACT_EXTENSION] = {
        "schemaVersion": CONTRACT_VERSION,
        "transport": {
            "protocol": TRANSPORT_PROTOCOL,
            "sessionProtocol": SESSION_PROTOCOL,
            "version": SESSION_PROTOCOL_VERSION,
        },
        "asyncapi": {"url": asyncapi_url},
    }
    return spec


_COMPONENT_KEY = re.compile(r"^[a-zA-Z0-9.\-_]+$")


def _normalize_discriminators(schemas: dict[str, Any]) -> None:
    """Rewrite OpenAPI discriminator objects to AsyncAPI's string form, in place.

    Union members reached through a discriminator mapping also get the tag
    property marked required: runtime validation cannot select a branch
    without the tag even when the member model defaults it, so publishing it
    as optional would under-describe the wire.
    """

    def require_tag(ref: str, property_name: str) -> None:
        member = schemas.get(ref.rsplit("/", 1)[-1])
        if isinstance(member, dict) and property_name in member.get("properties", {}):
            required = member.setdefault("required", [])
            if property_name not in required:
                required.insert(0, property_name)

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            discriminator = value.get("discriminator")
            if isinstance(discriminator, dict) and "propertyName" in discriminator:
                for ref in discriminator.get("mapping", {}).values():
                    require_tag(ref, discriminator["propertyName"])
                value["discriminator"] = discriminator["propertyName"]
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(schemas)


def _pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _register_role_schemas(
    schemas: dict[str, Any],
    schema: MessageSchema,
    *,
    prefix: str,
) -> dict[str, str]:
    """Register one directional union and return wire type -> payload ref.

    Direction is part of every generated component name. Consequently a
    client and server message may share a wire ``type`` without overwriting one
    another or being forced to share a payload schema.
    """

    rendered = _type_adapter(schema).json_schema(
        ref_template=f"#/components/schemas/{prefix}{{model}}"
    )
    definitions = rendered.pop("$defs", {})
    for definition_name, definition in definitions.items():
        schemas[f"{prefix}{definition_name}"] = definition

    discriminator = rendered.get("discriminator")
    if isinstance(discriminator, dict) and "mapping" in discriminator:
        # OpenAPI/Pydantic discriminator objects are not AsyncAPI Schema
        # Objects (AsyncAPI's discriminator is a string). The channel and
        # operation already enumerate every concrete directional message, so
        # publishing the redundant union envelope would only make the document
        # invalid without adding dispatch information.
        return dict(sorted(discriminator["mapping"].items()))

    envelope_name = f"{prefix}Message"
    if envelope_name in schemas:
        # The nested definition was registered from ``$defs`` above; writing
        # the envelope over it would silently repoint every nested reference.
        raise ValueError(
            f"a nested model named 'Message' collides with the generated "
            f"{envelope_name!r} payload envelope; rename the nested model"
        )
    wire_types = message_types(schema)
    if not wire_types:
        # Without a literal ``type`` the message would silently vanish from
        # the published contract while remaining live on the wire.
        raise ValueError(
            f"{schema!r} declares no literal 'type' discriminator; every WMA "
            "wire message must carry one"
        )
    schemas[envelope_name] = rendered
    envelope_ref = f"#/components/schemas/{envelope_name}"
    return {wire_type: envelope_ref for wire_type in wire_types}


def _publish_role(
    *,
    role: str,
    schema: MessageSchema,
    schema_prefix: str,
    component_schemas: dict[str, Any],
    component_messages: dict[str, Any],
    channel_messages: dict[str, Any],
) -> list[dict[str, str]]:
    role_prefix = f"{schema_prefix}{role.title()}"
    payloads = _register_role_schemas(
        component_schemas,
        schema,
        prefix=role_prefix,
    )

    operation_messages: list[dict[str, str]] = []
    for wire_type, payload_ref in payloads.items():
        message_id = f"{role}.{wire_type}"
        if not _COMPONENT_KEY.fullmatch(message_id):
            raise ValueError(
                f"message type {wire_type!r} cannot be used as an AsyncAPI "
                "component key"
            )
        component_messages[message_id] = {
            "name": wire_type,
            "title": f"{role.title()} {wire_type} message",
            "payload": {"$ref": payload_ref},
        }
        channel_messages[message_id] = {
            "$ref": f"#/components/messages/{_pointer_token(message_id)}"
        }
        operation_messages.append(
            {"$ref": (f"#/channels/control/messages/{_pointer_token(message_id)}")}
        )
    return operation_messages


def render_asyncapi(
    contract: RealtimeContract,
    *,
    title: str,
    schema_prefix: str,
    channel_address: str,
    openapi_operation_id: str,
    openapi_url: str = OPENAPI_URL,
) -> dict[str, Any]:
    """Render a standalone AsyncAPI 3 client contract.

    Operation actions are deliberately from the client perspective: the
    generated client sends control messages and receives model events.
    """

    component_schemas: dict[str, Any] = {}
    component_messages: dict[str, Any] = {}
    channel_messages: dict[str, Any] = {}
    operations: dict[str, Any] = {}

    for role, schema, operation_id, action in (
        ("client", contract.client_messages, CLIENT_OPERATION, "send"),
        ("server", contract.server_messages, SERVER_OPERATION, "receive"),
    ):
        if schema is None:
            continue
        messages = _publish_role(
            role=role,
            schema=schema,
            schema_prefix=schema_prefix,
            component_schemas=component_schemas,
            component_messages=component_messages,
            channel_messages=channel_messages,
        )
        operations[operation_id] = {
            "action": action,
            "channel": {"$ref": "#/channels/control"},
            "messages": messages,
        }

    # Data-channel messages carry their own wire discriminator. Pydantic omits
    # fields with defaults from JSON Schema's required list, but every runtime
    # event includes ``type`` and clients cannot dispatch without it. Only the
    # message payload schemas are rewritten — a nested model with its own
    # optional ``type`` property is not a wire message and keeps its shape.
    payload_names = {
        message["payload"]["$ref"].rsplit("/", 1)[-1]
        for message in component_messages.values()
    }
    for name in payload_names:
        schema = component_schemas[name]
        if "type" not in schema.get("properties", {}):
            # A wire message whose schema publishes no ``type`` property —
            # e.g. ``type: Literal["start"] = Field(alias="kind")`` — would
            # describe payloads the data channel cannot dispatch.
            raise ValueError(
                f"message payload schema {name!r} publishes no 'type' "
                "property; the wire discriminator must be spelled 'type' "
                "(field aliases are not part of the contract)"
            )
        type_property = schema["properties"]["type"]
        type_values = (
            [type_property["const"]]
            if "const" in type_property
            else type_property.get("enum", [])
        )
        if any(not isinstance(value, str) for value in type_values):
            # Union members keep their numeric const even though Pydantic
            # stringifies the discriminator mapping keys, so the check must
            # run on the published payload, not only in message_types().
            raise ValueError("wire 'type' discriminators must be strings")
        required = schema.setdefault("required", [])
        if "type" not in required:
            required.insert(0, "type")

    # Pydantic emits OpenAPI-style ``discriminator: {propertyName, mapping}``
    # objects; AsyncAPI Schema Objects define discriminator as a string. The
    # root-level union discriminator never reaches the document (the channel
    # enumerates concrete messages instead), but a discriminated union NESTED
    # inside a payload keeps its object form and must be normalized.
    _normalize_discriminators(component_schemas)

    document: dict[str, Any] = {
        "asyncapi": ASYNCAPI_SPEC_VERSION,
        "info": {
            "title": title,
            "version": ASYNCAPI_DOCUMENT_VERSION,
            "description": (
                "Client contract for the WebRTC session created by the linked "
                "OpenAPI operation."
            ),
        },
        "defaultContentType": DEFAULT_CONTENT_TYPE,
        "servers": {
            "session": {
                "host": "{sessionHost}",
                "protocol": TRANSPORT_PROTOCOL,
                "description": "Runtime-assigned WebRTC peer negotiated over HTTP.",
                "variables": {
                    "sessionHost": {
                        "default": "runtime-assigned.invalid",
                        "description": (
                            "Logical peer supplied by the OpenAPI session negotiation."
                        ),
                    }
                },
                "x-fal-negotiated-by": openapi_operation_id,
            }
        },
        "x-fal-openapi": {
            "url": openapi_url,
            "operationId": openapi_operation_id,
        },
        "x-fal-media": {
            "perspective": "client",
            **contract.media.to_openapi(),
        },
    }
    if channel_messages:
        document["channels"] = {
            "control": {
                "address": channel_address,
                "servers": [{"$ref": "#/servers/session"}],
                "messages": channel_messages,
            }
        }
        document["operations"] = operations
        document["components"] = {
            "schemas": component_schemas,
            "messages": component_messages,
        }
    return document
