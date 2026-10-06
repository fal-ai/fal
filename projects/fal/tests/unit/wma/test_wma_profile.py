# ruff: noqa: E402
"""Publishing/consumer-boundary tests for the optional WMA profile."""

import json
from dataclasses import replace
from typing import List, Literal, Union

import pytest
from typing_extensions import Annotated

pytest.importorskip("pydantic", minversion="2")
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from fal.wma import MediaContract, MessageExample, RealtimeContract, Track
from fal.wma import profile as p
from tests.unit.wma.test_wma_contract import (
    assert_internal_refs_resolve,
    render_realtime,
)


# Intentionally unlike Director: nested/escaped pointers, initial value 7,
# and a wire type shared across directions. No app IDs appear in the helpers.
class Ticket(BaseModel):
    serial: int = Field(ge=7, alias="serial/id~")


class Begin(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["begin"]
    ticket: Ticket
    prompt: str = Field(min_length=1)


class Change(Begin):
    type: Literal["change"]  # type: ignore[assignment]


class Ready(BaseModel):
    type: Literal["begin"] = "begin"
    ticket: Ticket


class Waiting(Ready):
    type: Literal["waiting"] = "waiting"  # type: ignore[assignment]


class Applied(Ready):
    type: Literal["active"] = "active"  # type: ignore[assignment]


class Rejected(Ready):
    type: Literal["refused"] = "refused"  # type: ignore[assignment]


class Failure(BaseModel):
    type: Literal["problem"] = "problem"
    code: Literal["fatal", "stale", "invalid"]
    error: str
    serial: Union[int, None] = None


class Ended(BaseModel):
    type: Literal["finished"] = "finished"


CLIENT = Annotated[Union[Begin, Change], Field(discriminator="type")]
SERVER = Annotated[
    Union[
        Union[Union[Union[Union[Ready, Waiting], Applied], Rejected], Failure], Ended
    ],
    Field(discriminator="type"),
]
POINTER = "/ticket/serial~1id~0"


def interaction() -> p.ConfiguredSession:
    return p.ConfiguredSession(
        setup=Begin,
        ready=p.Event(Ready, POINTER),
        sequence=p.Sequence(field=POINTER, initial=7, increment=2),
        updates=p.VersionedInput(
            message=Change,
            input_field="/prompt",
            pending=p.Event(Waiting, POINTER),
            applied=p.Event(Applied, POINTER),
            rejected=p.Event(Rejected, POINTER),
            policy="replace-pending",
        ),
        errors=p.ErrorMapping(
            message=Failure,
            code_field="/code",
            description_field="/error",
            correlation="/serial",
            session_failure=("fatal",),
            input_failure=("stale",),
            diagnostic=("invalid",),
        ),
        ended=Ended,
    )


def contract(**kwargs) -> RealtimeContract:
    return RealtimeContract(client_messages=CLIENT, server_messages=SERVER, **kwargs)


def test_renamed_data_only_profile_uses_native_correlations_and_directional_refs():
    document = render_realtime(contract(interaction=interaction()))
    profile = document["x-fal-wma"]
    assert profile["profileVersion"] == "0.1"
    assert profile["requiredFeatures"] == ["configured-session/1", "versioned-input/1"]
    assert profile["configuredSession"] == {
        "setup": {"$ref": "#/channels/control/messages/client.begin"},
        "ready": {"$ref": "#/channels/control/messages/server.begin"},
        "replay": "never",
    }
    assert profile["sequence"] == {
        "field": POINTER,
        "initial": 7,
        "increment": 2,
        "scope": "session",
    }
    assert profile["versionedInput"]["applied"] == {
        "$ref": "#/channels/control/messages/server.active"
    }
    assert profile["errors"]["uncorrelatedInput"] == "diagnostic"
    assert profile["ended"] == {"$ref": "#/channels/control/messages/server.finished"}
    for key in ("client.begin", "client.change", "server.begin", "server.active"):
        assert document["components"]["messages"][key]["correlationId"] == {
            "location": f"$message.payload#{POINTER}"
        }
    assert document["x-fal-media"] == {
        "perspective": "client",
        "send": [],
        "receive": [],
    }
    assert_internal_refs_resolve(document)
    assert json.loads(json.dumps(document)) == document


def test_profile_does_not_mutate_shared_models_or_other_contracts():
    before = render_realtime(contract())
    original_schema = Ready.model_json_schema()
    render_realtime(contract(interaction=interaction()))
    assert render_realtime(contract()) == before
    assert Ready.model_json_schema() == original_schema
    assert "x-fal-wma" not in before


def test_camera_app_without_profile_has_no_configuration_requirement():
    document = render_realtime(
        RealtimeContract(
            media=MediaContract(
                send=(Track(kind="video", source="camera", required=True),),
                receive=(Track(kind="video"),),
            ),
            client_messages=Change,
        )
    )
    assert "x-fal-wma" not in document
    assert set(document["components"]["messages"]) == {"client.change"}
    assert_internal_refs_resolve(document)


def test_optional_errors_and_ended_are_not_invented():
    document = render_realtime(
        contract(interaction=replace(interaction(), errors=None, ended=None))
    )
    assert "errors" not in document["x-fal-wma"]
    assert "ended" not in document["x-fal-wma"]


@pytest.mark.parametrize(
    "pointer", ["ticket/serial", "/ticket/~2", "/missing", "/ticket"]
)
def test_invalid_sequence_pointer_fails_publishing(pointer):
    profile = replace(interaction(), sequence=p.Sequence(field=pointer, initial=7))
    with pytest.raises(ValueError):
        render_realtime(contract(interaction=profile))


@pytest.mark.parametrize("initial", [True, 0, -1, 1.5, 2**53, 6])
def test_invalid_sequence_initial_fails_publishing(initial):
    profile = replace(
        interaction(), sequence=p.Sequence(field=POINTER, initial=initial)
    )
    with pytest.raises(ValueError, match="sequence initial"):
        render_realtime(contract(interaction=profile))


@pytest.mark.parametrize("increment", [True, 0, -1, 1.5, 2**53])
def test_invalid_sequence_increment_fails_publishing(increment):
    profile = replace(
        interaction(),
        sequence=p.Sequence(field=POINTER, initial=7, increment=increment),
    )
    with pytest.raises(ValueError, match="sequence increment"):
        render_realtime(contract(interaction=profile))


@pytest.mark.parametrize("minimum,maximum,valid", [(9, 20, True), (7, 8, False)])
def test_first_update_is_validated_at_initial_plus_increment(minimum, maximum, valid):
    class NextTicket(BaseModel):
        serial: int = Field(ge=minimum, le=maximum, alias="serial/id~")

    class NextChange(BaseModel):
        type: Literal["next"]
        ticket: NextTicket
        prompt: str

    profile = interaction()
    declaration = replace(
        contract(
            interaction=replace(
                profile, updates=replace(profile.updates, message=NextChange)
            )
        ),
        client_messages=Annotated[
            Union[Begin, NextChange], Field(discriminator="type")
        ],
    )
    if valid:
        render_realtime(declaration)
    else:
        with pytest.raises(ValueError, match="initial setup/update"):
            render_realtime(declaration)


def test_wrong_direction_is_rejected_even_when_wire_type_matches():
    profile = replace(interaction(), setup=Ready)
    with pytest.raises(ValueError, match="not a declared client"):
        render_realtime(contract(interaction=profile))


def test_unregistered_model_with_same_wire_type_is_rejected():
    class OtherBegin(Begin):
        different_required_field: int

    profile = replace(interaction(), setup=OtherBegin)
    with pytest.raises(ValueError, match="not a declared client"):
        render_realtime(contract(interaction=profile))


def test_optional_ready_correlation_is_rejected():
    class OptionalReady(BaseModel):
        type: Literal["optional"]
        serial: Union[int, None] = None

    profile = replace(interaction(), ready=p.Event(OptionalReady, "/serial"))
    with pytest.raises(ValueError, match="must be required"):
        render_realtime(
            replace(
                contract(interaction=profile),
                server_messages=Annotated[
                    Union[
                        Union[
                            Union[
                                Union[Union[Union[Ready, Waiting], Applied], Rejected],
                                Failure,
                            ],
                            Ended,
                        ],
                        OptionalReady,
                    ],
                    Field(discriminator="type"),
                ],
            )
        )


def test_wrong_input_field_type_is_rejected():
    profile = interaction()
    profile = replace(profile, updates=replace(profile.updates, input_field=POINTER))
    with pytest.raises(ValueError, match="must have type string"):
        render_realtime(contract(interaction=profile))


def test_ambiguous_server_event_meanings_are_rejected():
    profile = interaction()
    profile = replace(profile, ready=profile.updates.applied)
    with pytest.raises(ValueError, match="distinct meanings"):
        render_realtime(contract(interaction=profile))


@pytest.mark.parametrize("codes", [("unknown",), ("fatal", "fatal"), ("stale",)])
def test_unknown_duplicate_or_overlapping_error_codes_are_rejected(codes):
    profile = interaction()
    assert profile.errors is not None
    profile = replace(profile, errors=replace(profile.errors, session_failure=codes))
    with pytest.raises(ValueError, match="error codes"):
        render_realtime(contract(interaction=profile))


def test_invalid_error_description_pointer_is_rejected():
    profile = interaction()
    assert profile.errors is not None
    profile = replace(
        profile, errors=replace(profile.errors, description_field="/code2")
    )
    with pytest.raises(ValueError, match="not an object field"):
        render_realtime(contract(interaction=profile))


def build_text_update(document, text, version):
    """A consumer using only the published template and field bindings."""
    profile = document["x-fal-wma"]
    update = profile["versionedInput"]
    payload = json.loads(json.dumps(update["payloadTemplate"]))
    for pointer, value in (
        (profile["sequence"]["field"], version),
        (update["inputField"], text),
    ):
        tokens = [
            token.replace("~1", "/").replace("~0", "~")
            for token in pointer[1:].split("/")
        ]
        parent = payload
        for token in tokens[:-1]:
            parent = parent.setdefault(token, {})
        parent[tokens[-1]] = value
    return payload


def test_template_and_escaped_bindings_construct_a_complete_update():
    document = render_realtime(contract(interaction=interaction()))
    payload = build_text_update(document, "New scene", 9)
    parsed = TypeAdapter(CLIENT).validate_python(payload)
    assert isinstance(parsed, Change)
    assert parsed.ticket.serial == 9
    assert parsed.prompt == "New scene"
    assert document["x-fal-wma"]["versionedInput"]["payloadTemplate"] == {
        "type": "change"
    }


def test_text_action_can_bind_an_optional_field_without_selecting_other_variants():
    class OptionalText(BaseModel):
        type: Literal["edit"]
        ticket: Ticket
        prompt: Union[str, None] = None
        media_url: Union[str, None] = None
        replace_pending: bool = True

    profile = interaction()
    declaration = RealtimeContract(
        client_messages=Annotated[
            Union[Begin, OptionalText], Field(discriminator="type")
        ],
        server_messages=SERVER,
        interaction=replace(
            profile, updates=replace(profile.updates, message=OptionalText)
        ),
    )
    document = render_realtime(declaration)
    payload = build_text_update(document, "Next scene", 9)
    assert set(payload) == {"type", "ticket", "prompt"}
    parsed = OptionalText.model_validate(payload)
    assert parsed.prompt == "Next scene"
    assert parsed.media_url is None
    assert parsed.replace_pending is True


@pytest.mark.parametrize("nested", [False, True])
def test_template_rejects_unbound_required_fields(nested):
    class ExtraTicket(Ticket):
        owner: str

    class ExtraField(Change):
        mode: str

    class NestedField(Change):
        ticket: ExtraTicket

    update_model = NestedField if nested else ExtraField
    profile = interaction()
    declaration = RealtimeContract(
        client_messages=Annotated[
            Union[Begin, update_model], Field(discriminator="type")
        ],
        server_messages=SERVER,
        interaction=replace(
            profile, updates=replace(profile.updates, message=update_model)
        ),
    )
    with pytest.raises(ValueError, match="required fields without bindings"):
        render_realtime(declaration)


def test_input_cannot_overwrite_the_message_type():
    profile = interaction()
    profile = replace(profile, updates=replace(profile.updates, input_field="/type"))
    with pytest.raises(ValueError, match="discriminator"):
        render_realtime(contract(interaction=profile))


def test_existing_native_examples_coexist_with_profile_correlations():
    item = MessageExample(
        name="opening",
        summary="Start the session",
        payload={
            "type": "begin",
            "ticket": {"serial/id~": 7},
            "prompt": "A coastal village",
        },
    )
    declaration = contract(interaction=interaction(), client_message_examples=(item,))
    document = render_realtime(declaration)
    message = document["components"]["messages"]["client.begin"]
    assert message["correlationId"] == {"location": f"$message.payload#{POINTER}"}
    assert message["examples"] == [
        {"name": item.name, "summary": item.summary, "payload": item.payload}
    ]
    message["examples"][0]["payload"]["ticket"]["serial/id~"] = 100
    assert item.payload["ticket"]["serial/id~"] == 7


class Move(BaseModel):
    type: Literal["move"]
    held: List[Literal["forward", "left"]] = Field(default_factory=list)
    pressed: List[Literal["forward", "left"]] = Field(default_factory=list)


class Choose(BaseModel):
    type: Literal["choose"]
    scene: Union[str, None] = None


class Restart(Choose):
    type: Literal["restart"]  # type: ignore[assignment]


class Describe(BaseModel):
    type: Literal["describe"]
    text: str = Field(min_length=1)


class Stop(BaseModel):
    type: Literal["halt"]


class Chosen(BaseModel):
    type: Literal["chosen"] = "chosen"


class Restarted(BaseModel):
    type: Literal["restarted"] = "restarted"


class Described(BaseModel):
    type: Literal["described"] = "described"


COMMAND_CLIENT = Annotated[
    Union[Union[Union[Union[Move, Choose], Restart], Describe], Stop],
    Field(discriminator="type"),
]
COMMAND_SERVER = Annotated[
    Union[Union[Union[Union[Chosen, Restarted], Described], Failure], Ended],
    Field(discriminator="type"),
]


def command_interaction() -> p.CommandWorld:
    return p.CommandWorld(
        commands=p.CommandState(Move, "/held", "/pressed"),
        setup=p.SerialAction(Choose, Chosen),
        reset=p.SerialAction(Restart, Restarted),
        updates=p.SerialInput(Describe, Described, "/text"),
        errors=p.CommandErrors(Failure, "/error"),
        stop=Stop,
        ended=Ended,
    )


def command_contract(**kwargs) -> RealtimeContract:
    return RealtimeContract(
        media=MediaContract(receive=(Track(kind="video"),)),
        client_messages=COMMAND_CLIENT,
        server_messages=COMMAND_SERVER,
        interaction=kwargs.pop("interaction", command_interaction()),
        **kwargs,
    )


def test_automatic_command_world_publishes_exact_refs_without_correlations():
    document = render_realtime(command_contract())
    profile = document["x-fal-wma"]
    assert profile["requiredFeatures"] == ["command-world/1"]
    assert "sequence" not in profile and "configuredSession" not in profile
    world = profile["commandWorld"]
    assert world["startup"] == "automatic"
    assert world["ready"] == "first-frame"
    assert world["commands"] == {
        "message": {"$ref": "#/channels/control/messages/client.move"},
        "stateField": "/held",
        "activatedField": "/pressed",
    }
    assert world["setup"] == {
        "message": {"$ref": "#/channels/control/messages/client.choose"},
        "applied": {"$ref": "#/channels/control/messages/server.chosen"},
        "limit": "once-per-session",
        "replay": "never",
    }
    assert world["updates"]["policy"] == "serial"
    assert world["updates"]["inputField"] == "/text"
    payload = {**world["updates"]["payloadTemplate"], "text": "A blue sky"}
    assert isinstance(TypeAdapter(COMMAND_CLIENT).validate_python(payload), Describe)
    assert world["errors"]["descriptionField"] == "/error"
    assert world["stop"] == {"$ref": "#/channels/control/messages/client.halt"}
    assert profile["ended"] == {"$ref": "#/channels/control/messages/server.finished"}
    assert not any(
        "correlationId" in message
        for message in document["components"]["messages"].values()
    )
    assert_internal_refs_resolve(document)


def test_command_only_world_does_not_invent_setup_or_text_controls():
    interaction = p.CommandWorld(commands=p.CommandState(Move, "/held"))
    document = render_realtime(command_contract(interaction=interaction))
    assert set(document["x-fal-wma"]["commandWorld"]) == {
        "startup",
        "ready",
        "commands",
    }
    assert "activatedField" not in document["x-fal-wma"]["commandWorld"]["commands"]


@pytest.mark.parametrize("field", ["held", "/missing", "/type", "/held/0", "/~2"])
def test_command_world_rejects_invalid_state_binding(field):
    interaction = replace(command_interaction(), commands=p.CommandState(Move, field))
    with pytest.raises(ValueError):
        render_realtime(command_contract(interaction=interaction))


def test_command_world_requires_video_for_first_frame_readiness():
    with pytest.raises(ValueError, match="requires received video"):
        render_realtime(replace(command_contract(), media=MediaContract()))


@pytest.mark.parametrize("invalid", ["unbounded", "nonempty", "different", "unbound"])
def test_command_world_rejects_unusable_control_schemas(invalid):
    class Unbounded(Move):
        # Widen the vocabulary deliberately to test publication rejection.
        held: List[str] = Field(default_factory=list)  # type: ignore[assignment]

    class Nonempty(Move):
        held: List[Literal["forward", "left"]] = Field(min_length=1)

    class Different(Move):
        # Use an incompatible vocabulary deliberately to test publication rejection.
        pressed: List[Literal["right"]] = Field(default_factory=list)  # type: ignore[assignment]

    class Unbound(Move):
        required_mode: str

    model = {
        "unbounded": Unbounded,
        "nonempty": Nonempty,
        "different": Different,
        "unbound": Unbound,
    }[invalid]
    declaration = RealtimeContract(
        media=MediaContract(receive=(Track(kind="video"),)),
        client_messages=model,
        interaction=p.CommandWorld(commands=p.CommandState(model, "/held", "/pressed")),
    )
    with pytest.raises(ValueError):
        render_realtime(declaration)


def test_command_world_rejects_ambiguous_and_wrong_direction_events():
    interaction = command_interaction()
    for action in (p.SerialAction(Restart, Chosen), p.SerialAction(Restart, Choose)):
        with pytest.raises(ValueError):
            render_realtime(
                command_contract(interaction=replace(interaction, reset=action))
            )


def test_serial_input_rejects_unbound_required_fields():
    class ExtraDescribe(Describe):
        mode: str

    declaration = RealtimeContract(
        media=MediaContract(receive=(Track(kind="video"),)),
        client_messages=Annotated[
            Union[Move, ExtraDescribe], Field(discriminator="type")
        ],
        server_messages=Described,
        interaction=p.CommandWorld(
            commands=p.CommandState(Move, "/held"),
            updates=p.SerialInput(ExtraDescribe, Described, "/text"),
        ),
    )
    with pytest.raises(ValueError, match="required fields without bindings"):
        render_realtime(declaration)


def test_reset_can_bind_optional_text_without_changing_the_raw_action():
    interaction = replace(
        command_interaction(), reset=p.SerialInput(Restart, Restarted, "/scene")
    )
    document = render_realtime(command_contract(interaction=interaction))
    reset = document["x-fal-wma"]["commandWorld"]["reset"]
    assert reset["inputField"] == "/scene"
    assert reset["payloadTemplate"] == {"type": "restart"}
    assert reset["policy"] == "serial"
    omitted = TypeAdapter(COMMAND_CLIENT).validate_python(reset["payloadTemplate"])
    described = TypeAdapter(COMMAND_CLIENT).validate_python(
        {**reset["payloadTemplate"], "scene": "A new world"}
    )
    assert isinstance(omitted, Restart) and omitted.scene is None
    assert isinstance(described, Restart) and described.scene == "A new world"
