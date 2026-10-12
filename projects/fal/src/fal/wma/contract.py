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

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Set, Tuple, Union

from pydantic import BaseModel

from fal.wma.profile import CommandWorld, ConfiguredSession, publish_profile
from fal.wma.ui import ExampleUI

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
class MessageExample:
    """One named data-channel payload example published through AsyncAPI."""

    name: str
    payload: Dict[str, Any]
    summary: Union[str, None] = None
    ui: Union[ExampleUI, None] = None


@dataclass(frozen=True)
class MessagePresentation:
    """Opt-in native AsyncAPI title/summary for one wire message type."""

    message_type: str
    title: str
    summary: Union[str, None] = None


@dataclass(frozen=True)
class Constraint:
    """A MediaTrackConstraint range for width, height, or frame rate."""

    min: Union[float, None] = None
    max: Union[float, None] = None
    ideal: Union[float, None] = None
    exact: Union[float, None] = None

    def __post_init__(self) -> None:
        if self.to_openapi() == {}:
            raise ValueError("a Constraint must bound something")

    def to_openapi(self) -> Dict[str, float]:
        bounds = (
            ("min", self.min),
            ("max", self.max),
            ("ideal", self.ideal),
            ("exact", self.exact),
        )
        return {name: value for name, value in bounds if value is not None}


Measure = Union[Union[int, float], Constraint]


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
    source: Union[TrackSource, None] = None
    required: bool = False
    width: Union[Measure, None] = None
    height: Union[Measure, None] = None
    frame_rate: Union[Measure, None] = None

    @property
    def measures(self) -> Dict[str, Measure]:
        declared = (
            ("width", self.width),
            ("height", self.height),
            ("frameRate", self.frame_rate),
        )
        return {name: value for name, value in declared if value is not None}

    def to_openapi(self, direction: Direction = "send") -> Dict[str, Any]:
        published: Dict[str, Any] = {"kind": self.kind}
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

    send: Tuple[Track, ...] = ()
    receive: Tuple[Track, ...] = ()

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

    def to_openapi(self) -> Dict[str, Any]:
        return {
            "send": [track.to_openapi("send") for track in self.send],
            "receive": [track.to_openapi("receive") for track in self.receive],
        }


@dataclass(frozen=True)
class RealtimeContract:
    """What a client may exchange with one live model session.

    ``interaction`` optionally describes client behavior on the control channel.
    It does not change server validation or the HTTP session parameters.
    """

    media: MediaContract = field(default_factory=MediaContract)
    client_messages: MessageSchema = None
    server_messages: MessageSchema = None
    client_message_examples: Tuple[MessageExample, ...] = ()
    server_message_examples: Tuple[MessageExample, ...] = ()
    interaction: Union[Union[ConfiguredSession, CommandWorld], None] = None
    client_message_presentation: Tuple[MessagePresentation, ...] = ()
    server_message_presentation: Tuple[MessagePresentation, ...] = ()

    def __post_init__(self) -> None:
        if self.client_message_examples and self.client_messages is None:
            raise ValueError("client message examples require a client message schema")
        if self.server_message_examples and self.server_messages is None:
            raise ValueError("server message examples require a server message schema")
        for role in ("client", "server"):
            if (
                getattr(self, f"{role}_message_presentation")
                and getattr(self, f"{role}_messages") is None
            ):
                raise ValueError(
                    f"{role} message presentation requires a message schema"
                )


def message_types(schema: MessageSchema) -> Tuple[str, ...]:
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
    spec: Dict[str, Any],
    *,
    path: str,
    asyncapi_url: str = ASYNCAPI_URL,
) -> Dict[str, Any]:
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


def _normalize_discriminators(schemas: Dict[str, Any]) -> None:
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
    schemas: Dict[str, Any],
    schema: MessageSchema,
    *,
    prefix: str,
) -> Dict[str, str]:
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


def _schema_matches_value(
    value: Any,
    schema: Dict[str, Any],
    component_schemas: Dict[str, Any],
) -> bool:
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
        component = component_schemas.get(ref.rsplit("/", 1)[1])
        if isinstance(component, dict) and not _schema_matches_value(
            value, component, component_schemas
        ):
            return False
    if any(
        isinstance(variant, dict)
        and not _schema_matches_value(value, variant, component_schemas)
        for variant in schema.get("allOf", ())
    ):
        return False
    for keyword in ("anyOf", "oneOf"):
        variants = schema.get(keyword)
        if isinstance(variants, list) and not any(
            isinstance(variant, dict)
            and _schema_matches_value(value, variant, component_schemas)
            for variant in variants
        ):
            return False
    schema_type = schema.get("type")
    if schema_type == "object" and not isinstance(value, dict):
        return False
    if schema_type == "array" and not isinstance(value, list):
        return False
    if schema_type == "null" and value is not None:
        return False
    if "const" in schema and value != schema["const"]:
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if isinstance(value, dict):
        if not set(schema.get("required", ())).issubset(value):
            return False
        for name, property_schema in schema.get("properties", {}).items():
            if name in value and isinstance(property_schema, dict):
                if not _schema_matches_value(
                    value[name], property_schema, component_schemas
                ):
                    return False
    return True


def _schema_title(
    schema: Dict[str, Any], component_schemas: Dict[str, Any]
) -> Union[str, None]:
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
        component = component_schemas.get(ref.rsplit("/", 1)[1])
        if isinstance(component, dict):
            return _schema_title(component, component_schemas)
    title = schema.get("title")
    return title if isinstance(title, str) else None


def _source_child(source: Any, key: Union[str, int]) -> Any:
    if isinstance(source, BaseModel) and isinstance(key, str):
        for name, field_info in type(source).model_fields.items():
            aliases = {name, field_info.alias, field_info.serialization_alias}
            if key in aliases:
                return getattr(source, name)
        return None
    if isinstance(source, dict):
        return source.get(key)
    if isinstance(source, list) and isinstance(key, int) and key < len(source):
        return source[key]
    return None


def _filter_payload_by_schema(
    value: Any,
    schema: Dict[str, Any],
    component_schemas: Dict[str, Any],
    source: Any = None,
) -> Any:
    """Remove values omitted from closed public schemas at every depth."""
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
        component = component_schemas.get(ref.rsplit("/", 1)[1])
        if isinstance(component, dict):
            value = _filter_payload_by_schema(
                value, component, component_schemas, source
            )

    for keyword in ("allOf", "anyOf", "oneOf"):
        variants = schema.get(keyword)
        if not isinstance(variants, list):
            continue
        if keyword == "allOf":
            matches = [variant for variant in variants if isinstance(variant, dict)]
        else:
            model_title = (
                type(source).model_config.get("title") or type(source).__name__
                if isinstance(source, BaseModel)
                else None
            )
            matches = [
                variant
                for variant in variants
                if isinstance(variant, dict)
                and model_title == _schema_title(variant, component_schemas)
            ]
            if len(matches) != 1:
                matches = [
                    variant
                    for variant in variants
                    if isinstance(variant, dict)
                    and _schema_matches_value(value, variant, component_schemas)
                ]
        if len(matches) == 1 or keyword == "allOf":
            for variant in matches:
                value = _filter_payload_by_schema(
                    value, variant, component_schemas, source
                )

    if isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, dict):
            return [
                _filter_payload_by_schema(
                    item,
                    items,
                    component_schemas,
                    _source_child(source, index),
                )
                for index, item in enumerate(value)
            ]
        return value

    if not isinstance(value, dict):
        return value
    declared_properties = schema.get("properties")
    properties = declared_properties if isinstance(declared_properties, dict) else {}
    additional = schema.get("additionalProperties")
    if isinstance(declared_properties, dict) and not (
        additional is True or isinstance(additional, dict)
    ):
        value = {key: item for key, item in value.items() if key in properties}
    return {
        key: (
            _filter_payload_by_schema(
                item,
                property_schema,
                component_schemas,
                _source_child(source, key),
            )
            if isinstance(
                (property_schema := properties.get(key, additional)),
                dict,
            )
            else item
        )
        for key, item in value.items()
    }


def _publish_role(
    *,
    role: str,
    schema: MessageSchema,
    schema_prefix: str,
    component_schemas: Dict[str, Any],
    component_messages: Dict[str, Any],
    channel_messages: Dict[str, Any],
    examples: Tuple[MessageExample, ...],
    presentation: Tuple[MessagePresentation, ...],
) -> List[Dict[str, str]]:
    role_prefix = f"{schema_prefix}{role.title()}"
    payloads = _register_role_schemas(
        component_schemas,
        schema,
        prefix=role_prefix,
    )
    presentation_by_type: Dict[str, MessagePresentation] = {}
    for item in presentation:
        if item.message_type not in payloads:
            raise ValueError(
                f"unknown {role} message presentation: {item.message_type!r}"
            )
        if item.message_type in presentation_by_type:
            raise ValueError(
                f"duplicate {role} message presentation: {item.message_type!r}"
            )
        if not item.title.strip():
            raise ValueError("message presentation titles cannot be blank")
        presentation_by_type[item.message_type] = item
    examples_by_type: Dict[str, List[Dict[str, Any]]] = {}
    example_names_by_type: Dict[str, Set[str]] = {}
    adapter = _type_adapter(schema)
    for example in examples:
        name = example.name.strip()
        if not name:
            raise ValueError("message example names cannot be blank")
        validated = adapter.validate_python(example.payload)
        payload = adapter.dump_python(
            validated,
            mode="json",
            by_alias=True,
            exclude_none=True,
        )
        if not isinstance(payload, dict):
            raise ValueError("message example payloads must serialize to objects")
        wire_type = payload.get("type")
        if wire_type not in payloads:
            raise ValueError(
                f"message example {name!r} has unknown wire type {wire_type!r}"
            )
        # A model can intentionally omit private runtime fields from its public
        # schema. Do not reintroduce their defaults through named examples.
        public_schema = component_schemas[payloads[wire_type].rsplit("/", 1)[1]]
        payload = _filter_payload_by_schema(
            payload,
            public_schema,
            component_schemas,
            validated,
        )
        names = example_names_by_type.setdefault(wire_type, set())
        if name in names:
            raise ValueError(
                f"message example name {name!r} is duplicated for {wire_type!r}"
            )
        names.add(name)
        published_example: Dict[str, Any] = {"name": name, "payload": payload}
        if example.summary is not None:
            published_example["summary"] = example.summary
        if example.ui is not None:
            published_example.update(example.ui.schema_extra())
        examples_by_type.setdefault(wire_type, []).append(published_example)

    operation_messages: List[Dict[str, str]] = []
    for wire_type, payload_ref in payloads.items():
        message_id = f"{role}.{wire_type}"
        if not _COMPONENT_KEY.fullmatch(message_id):
            raise ValueError(
                f"message type {wire_type!r} cannot be used "
                "as an AsyncAPI component key"
            )
        component_message: Dict[str, Any] = {
            "name": wire_type,
            "title": f"{role.title()} {wire_type} message",
            "payload": {"$ref": payload_ref},
        }
        if display := presentation_by_type.get(wire_type):
            component_message["title"] = display.title
            if display.summary is not None:
                component_message["summary"] = display.summary
        if message_examples := examples_by_type.get(wire_type):
            component_message["examples"] = message_examples
        component_messages[message_id] = component_message
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
) -> Dict[str, Any]:
    """Render a standalone AsyncAPI 3 client contract.

    Operation actions are deliberately from the client perspective: the
    generated client sends control messages and receives model events.
    """

    component_schemas: Dict[str, Any] = {}
    component_messages: Dict[str, Any] = {}
    channel_messages: Dict[str, Any] = {}
    operations: Dict[str, Any] = {}

    for role, schema, examples, operation_id, action in (
        (
            "client",
            contract.client_messages,
            contract.client_message_examples,
            CLIENT_OPERATION,
            "send",
        ),
        (
            "server",
            contract.server_messages,
            contract.server_message_examples,
            SERVER_OPERATION,
            "receive",
        ),
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
            examples=examples,
            presentation=getattr(contract, f"{role}_message_presentation"),
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

    document: Dict[str, Any] = {
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
    if contract.interaction is not None:
        publish_profile(
            document,
            contract.interaction,
            client=contract.client_messages,
            server=contract.server_messages,
        )
    return document
