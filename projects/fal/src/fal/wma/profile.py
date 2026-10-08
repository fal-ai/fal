"""Optional WMA interaction declarations, published as ``x-fal-wma`` 0.1.

These describe an existing protocol; they do not execute it. Version 0.1 only
supports configured/versioned input and automatic command-world interactions.
Messages, payloads, examples and correlation locations remain native AsyncAPI.
"""

import re
import types
from dataclasses import dataclass
from typing import Any, Dict, Literal, Set, Tuple, Type, Union, get_args, get_origin

from pydantic import BaseModel
from typing_extensions import Annotated

PROFILE_EXTENSION = "x-fal-wma"
PROFILE_VERSION = "0.1"


@dataclass(frozen=True)
class Sequence:
    """Client-generated, strictly increasing integers shared by setup and updates.

    Start anew for each session attempt. Gaps are allowed; do not reuse a value
    after uncertain delivery. ``field`` is a JSON Pointer into both payloads.
    """

    field: str
    initial: int = 1
    increment: int = 1


@dataclass(frozen=True)
class Event:
    """A server message; correlation locates its request's sequence value."""

    message: Type[BaseModel]
    correlation: str


@dataclass(frozen=True)
class VersionedInput:
    """One text update and its independent, correlated outcomes.

    ``replace-pending`` means newer updates supersede unfinished preparation;
    the server need not emit a terminal event for every superseded update.
    Applied means accepted for generation, not necessarily visible in playback.
    Clients build updates from the published template plus the text/sequence
    bindings only. Other message options retain their server defaults.
    """

    message: Type[BaseModel]
    input_field: str
    pending: Event
    applied: Event
    rejected: Event
    policy: Literal["replace-pending"]


@dataclass(frozen=True)
class ErrorMapping:
    """Classify finite error codes; uncorrelated/unknown errors are diagnostic.

    An input failure only affects a submission when correlation is present.
    Session failures are terminal even without a correlation value.
    """

    message: Type[BaseModel]
    code_field: str
    description_field: str
    correlation: str
    session_failure: Tuple[str, ...] = ()
    input_failure: Tuple[str, ...] = ()
    diagnostic: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ConfiguredSession:
    """Configure once, await its acknowledgement, then allow text updates.

    This is a client interaction policy, not a claim that the server rejects
    all earlier updates. Clients must not replay setup or automatically resend
    updates: a replay acknowledgement need not prove initialization finished.
    Individual input outcomes never substitute for the setup acknowledgement.
    """

    setup: Type[BaseModel]
    ready: Event
    sequence: Sequence
    updates: VersionedInput
    errors: Union[ErrorMapping, None] = None
    ended: Union[Type[BaseModel], None] = None


@dataclass(frozen=True)
class CommandState:
    """A complete held-state report, plus newly activated semantic commands."""

    message: Type[BaseModel]
    state_field: str
    activated_field: Union[str, None] = None


@dataclass(frozen=True)
class SerialAction:
    """An unversioned mutation with one outstanding acknowledgement at a time."""

    message: Type[BaseModel]
    applied: Type[BaseModel]


@dataclass(frozen=True)
class SerialInput(SerialAction):
    """A text mutation; acceptance affects generation, not current playback."""

    input_field: str


@dataclass(frozen=True)
class CommandErrors:
    """Uncorrelated diagnostics; clients must not invent per-input outcomes."""

    message: Type[BaseModel]
    description_field: str


@dataclass(frozen=True)
class CommandWorld:
    """An automatic world with commands and optional serialized mutations.

    The first video frame establishes readiness. Setup is optional and accepted
    once; setup/reset/text actions share one in-flight slot and are never replayed.
    """

    commands: CommandState
    setup: Union[SerialAction, None] = None
    reset: Union[SerialAction, None] = None
    updates: Union[SerialInput, None] = None
    errors: Union[CommandErrors, None] = None
    stop: Union[Type[BaseModel], None] = None
    ended: Union[Type[BaseModel], None] = None


def _models(schema: Any) -> Tuple[Type[BaseModel], ...]:
    if get_origin(schema) is Annotated:
        return _models(get_args(schema)[0])
    if get_origin(schema) in (Union, getattr(types, "UnionType", Union)):
        return tuple(model for member in get_args(schema) for model in _models(member))
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return (schema,)
    raise ValueError("profile messages must be Pydantic models or model unions")


class _Bindings:
    """Resolve declarations against the exact models published in this contract."""

    def __init__(self, document: Dict[str, Any], client: Any, server: Any) -> None:
        self.document = document
        self.models = {
            role: _models(schema) if schema is not None else ()
            for role, schema in (("client", client), ("server", server))
        }

    def message(self, model: Type[BaseModel], role: str) -> Dict[str, Any]:
        if model not in self.models[role]:
            raise ValueError(f"{model!r} is not a declared {role} message model")
        tag = model.model_json_schema().get("properties", {}).get("type", {})
        wire_type = tag.get("const")
        if not isinstance(wire_type, str):
            raise ValueError("profile messages must have one literal 'type' value")
        return self.document["components"]["messages"][f"{role}.{wire_type}"]

    def ref(self, model: Type[BaseModel], role: str) -> Dict[str, str]:
        message = self.message(model, role)
        return {"$ref": f"#/channels/control/messages/{role}.{message['name']}"}

    def field(
        self,
        model: Type[BaseModel],
        role: str,
        pointer: str,
        kind: str,
        *,
        optional: bool = False,
    ) -> Dict[str, Any]:
        # Deliberately only object properties and local payload refs. An array,
        # union traversal or computed expression has no v0 binding semantics.
        if not isinstance(pointer, str) or not pointer.startswith("/"):
            raise ValueError(f"{pointer!r} must be a payload JSON Pointer")
        if re.search(r"~(?![01])", pointer):
            raise ValueError(f"{pointer!r} contains an invalid JSON Pointer escape")
        schema = self.message(model, role)["payload"]
        tokens = pointer[1:].split("/")
        for index, encoded in enumerate(tokens):
            schema = self.resolve(schema)
            token = encoded.replace("~1", "/").replace("~0", "~")
            if schema.get("type") != "object" or token not in schema.get(
                "properties", {}
            ):
                raise ValueError(
                    f"{model.__name__}: {pointer!r} is not an object field"
                )
            if not (optional and index == len(tokens) - 1):
                if token not in schema.get("required", []):
                    raise ValueError(f"{model.__name__}: {pointer!r} must be required")
            schema = schema["properties"][token]
        schema = self.resolve(schema)
        if optional and "anyOf" in schema:
            alternatives = [
                member for member in schema["anyOf"] if member.get("type") != "null"
            ]
            if len(alternatives) == 1:
                schema = self.resolve(alternatives[0])
        if schema.get("type") != kind:
            raise ValueError(f"{model.__name__}: {pointer!r} must have type {kind}")
        return schema

    def text_template(
        self, updates: VersionedInput, sequence: Sequence
    ) -> Dict[str, str]:
        return self.payload_template(
            updates.message,
            {
                sequence.field: sequence.initial + sequence.increment,
                updates.input_field: "",
            },
        )

    def payload_template(
        self, model: Type[BaseModel], fields: Dict[str, Any]
    ) -> Dict[str, str]:
        """Require a complete action after the consumer supplies bound fields."""
        message = self.message(model, "client")
        template = {"type": message["name"]}
        payload: Dict[str, Any] = dict(template)
        for pointer, value in fields.items():
            if pointer == "/type":
                raise ValueError("an input cannot replace the message discriminator")
            tokens = [
                token.replace("~1", "/").replace("~0", "~")
                for token in pointer[1:].split("/")
            ]
            parent = payload
            for token in tokens[:-1]:
                parent = parent.setdefault(token, {})
            parent[tokens[-1]] = value

        def check_required(schema: Dict[str, Any], value: Any) -> None:
            schema = self.resolve(schema)
            if schema.get("type") != "object":
                return
            if any(key not in value for key in schema.get("required", [])):
                raise ValueError("input has required fields without bindings")
            for key, child in value.items():
                check_required(schema["properties"][key], child)

        check_required(message["payload"], payload)
        return template

    def resolve(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        visited: Set[str] = set()
        while "$ref" in schema:
            ref = schema["$ref"]
            if ref in visited or not ref.startswith("#/components/schemas/"):
                raise ValueError(f"unsupported profile payload reference: {ref}")
            visited.add(ref)
            schema = self.document["components"]["schemas"][ref.rsplit("/", 1)[-1]]
        return schema

    def correlate(
        self, model: Type[BaseModel], role: str, pointer: str, *, optional: bool = False
    ) -> Dict[str, str]:
        self.field(model, role, pointer, "integer", optional=optional)
        message = self.message(model, role)
        correlation = {"location": f"$message.payload#{pointer}"}
        previous = message.get("correlationId")
        if previous is not None and previous != correlation:
            raise ValueError("a message cannot have conflicting correlation locations")
        message["correlationId"] = correlation
        return self.ref(model, role)


def _check_sequence(
    schema: Dict[str, Any], sequence: Sequence, *, update: bool
) -> None:
    for value, label in (
        (sequence.initial, "initial"),
        (sequence.increment, "increment"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            or value > 2**53 - 1
        ):
            raise ValueError(f"sequence {label} must be a positive safe integer")
    value = sequence.initial + sequence.increment if update else sequence.initial
    if (
        value > 2**53 - 1
        or ("minimum" in schema and value < schema["minimum"])
        or ("exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"])
        or ("maximum" in schema and value > schema["maximum"])
        or ("exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"])
        or ("multipleOf" in schema and value % schema["multipleOf"] != 0)
        or ("const" in schema and value != schema["const"])
        or ("enum" in schema and value not in schema["enum"])
    ):
        raise ValueError(
            "sequence initial setup/update value violates its payload field schema"
        )


def publish_profile(
    document: Dict[str, Any],
    interaction: Union[ConfiguredSession, CommandWorld],
    *,
    client: Any,
    server: Any,
) -> None:
    """Validate and enrich a freshly rendered document; never modify models."""

    if isinstance(interaction, CommandWorld):
        _publish_command_world(document, interaction, client=client, server=server)
        return
    if not isinstance(interaction, ConfiguredSession):
        raise ValueError("unsupported WMA interaction profile")
    bindings = _Bindings(document, client, server)
    updates = interaction.updates
    sequence = interaction.sequence
    if updates.policy != "replace-pending":
        raise ValueError("unsupported versioned input policy")
    if interaction.setup is updates.message:
        raise ValueError("setup and updates must be distinct messages")
    server_models = [
        event.message
        for event in (
            interaction.ready,
            updates.pending,
            updates.applied,
            updates.rejected,
        )
    ]
    server_models.extend(
        model
        for model in (
            interaction.errors.message if interaction.errors else None,
            interaction.ended,
        )
        if model is not None
    )
    if len(set(server_models)) != len(server_models):
        raise ValueError("profile server events must have distinct meanings")
    for model, is_update in ((interaction.setup, False), (updates.message, True)):
        schema = bindings.field(model, "client", sequence.field, "integer")
        _check_sequence(schema, sequence, update=is_update)
        bindings.correlate(model, "client", sequence.field)
    # A raw message can also support non-text variants. This action always
    # supplies text, even when that leaf is optional/nullable on the raw model.
    bindings.field(
        updates.message, "client", updates.input_field, "string", optional=True
    )
    template = bindings.text_template(updates, sequence)

    def event_ref(event: Event) -> Dict[str, str]:
        return bindings.correlate(event.message, "server", event.correlation)

    profile: Dict[str, Any] = {
        "profileVersion": PROFILE_VERSION,
        "perspective": "client",
        "requiredFeatures": ["configured-session/1", "versioned-input/1"],
        "sequence": {
            "field": sequence.field,
            "initial": sequence.initial,
            "increment": sequence.increment,
            "scope": "session",
        },
        "configuredSession": {
            "setup": bindings.ref(interaction.setup, "client"),
            "ready": event_ref(interaction.ready),
            "replay": "never",
        },
        "versionedInput": {
            "message": bindings.ref(updates.message, "client"),
            "inputField": updates.input_field,
            "payloadTemplate": template,
            "policy": updates.policy,
            "pending": event_ref(updates.pending),
            "applied": event_ref(updates.applied),
            "rejected": event_ref(updates.rejected),
        },
    }
    if interaction.errors is not None:
        errors = interaction.errors
        code = bindings.field(errors.message, "server", errors.code_field, "string")
        bindings.field(errors.message, "server", errors.description_field, "string")
        codes = (*errors.session_failure, *errors.input_failure, *errors.diagnostic)
        allowed = [code["const"]] if "const" in code else code.get("enum", [])
        if len(set(codes)) != len(codes) or any(item not in allowed for item in codes):
            raise ValueError("error codes must be distinct members of the message enum")
        profile["errors"] = {
            "message": bindings.correlate(
                errors.message, "server", errors.correlation, optional=True
            ),
            "codeField": errors.code_field,
            "descriptionField": errors.description_field,
            "sessionFailure": list(errors.session_failure),
            "inputFailure": list(errors.input_failure),
            "diagnostic": list(errors.diagnostic),
            "uncorrelatedInput": "diagnostic",
            "unknownCode": "diagnostic",
        }
    if interaction.ended is not None:
        profile["ended"] = bindings.ref(interaction.ended, "server")
    document[PROFILE_EXTENSION] = profile


def _publish_command_world(
    document: Dict[str, Any],
    interaction: CommandWorld,
    *,
    client: Any,
    server: Any,
) -> None:
    bindings = _Bindings(document, client, server)
    media = document.get("x-fal-media", {})
    if not any(track.get("kind") == "video" for track in media.get("receive", [])):
        raise ValueError("a first-frame command world requires received video")
    commands = interaction.commands
    fields = [commands.state_field]
    if commands.activated_field is not None:
        fields.append(commands.activated_field)
    if len(set(fields)) != len(fields):
        raise ValueError("command state and activation fields must be distinct")
    vocabularies = []
    for pointer in fields:
        schema = bindings.field(
            commands.message, "client", pointer, "array", optional=True
        )
        items = bindings.resolve(schema.get("items", {}))
        values = [items["const"]] if "const" in items else items.get("enum", [])
        if not values or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise ValueError("command fields must contain a finite string vocabulary")
        if schema.get("minItems", 0) > 0:
            raise ValueError("command fields must permit an empty release report")
        vocabularies.append(set(values))
    if len(vocabularies) > 1 and vocabularies[0] != vocabularies[1]:
        raise ValueError("command state and activation vocabularies must match")
    bindings.payload_template(commands.message, {pointer: [] for pointer in fields})
    state: Dict[str, Any] = {
        "message": bindings.ref(commands.message, "client"),
        "stateField": commands.state_field,
    }
    if commands.activated_field is not None:
        state["activatedField"] = commands.activated_field
    world: Dict[str, Any] = {
        "startup": "automatic",
        "ready": "first-frame",
        "commands": state,
    }
    clients = [commands.message]
    servers = []
    for name, action in (
        ("setup", interaction.setup),
        ("reset", interaction.reset),
        ("updates", interaction.updates),
    ):
        if action is None:
            continue
        clients.append(action.message)
        servers.append(action.applied)
        declaration: Dict[str, Any] = {
            "message": bindings.ref(action.message, "client"),
            "applied": bindings.ref(action.applied, "server"),
            "replay": "never",
        }
        if name == "setup":
            declaration["limit"] = "once-per-session"
        if isinstance(action, SerialInput):
            bindings.field(
                action.message, "client", action.input_field, "string", optional=True
            )
            if action.input_field == "/type":
                raise ValueError(
                    "the text input cannot replace the message discriminator"
                )
            declaration.update(
                inputField=action.input_field,
                payloadTemplate=bindings.payload_template(
                    action.message, {action.input_field: ""}
                ),
                policy="serial",
            )
        world[name] = declaration
    if interaction.errors is not None:
        error = interaction.errors
        bindings.field(error.message, "server", error.description_field, "string")
        world["errors"] = {
            "message": bindings.ref(error.message, "server"),
            "descriptionField": error.description_field,
        }
        servers.append(error.message)
    if interaction.stop is not None:
        bindings.payload_template(interaction.stop, {})
        world["stop"] = bindings.ref(interaction.stop, "client")
        clients.append(interaction.stop)
    profile: Dict[str, Any] = {
        "profileVersion": PROFILE_VERSION,
        "perspective": "client",
        "requiredFeatures": ["command-world/1"],
        "commandWorld": world,
    }
    if interaction.ended is not None:
        profile["ended"] = bindings.ref(interaction.ended, "server")
        servers.append(interaction.ended)
    if len(set(clients)) != len(clients) or len(set(servers)) != len(servers):
        raise ValueError("command-world messages must have distinct meanings")
    document[PROFILE_EXTENSION] = profile
