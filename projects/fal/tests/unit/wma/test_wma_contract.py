"""Tests for linked OpenAPI and AsyncAPI WMA contracts."""

import json
from typing import ClassVar, Dict, List, Literal, Optional, Union

import pydantic
import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing_extensions import Annotated

if not hasattr(pydantic, "TypeAdapter"):
    pytest.skip(
        "realtime contract rendering requires pydantic v2",
        allow_module_level=True,
    )

from fal.wma import (
    ASYNCAPI_SPEC_VERSION,
    ASYNCAPI_URL,
    CONTRACT_EXTENSION,
    CONTRACT_VERSION,
    CommandsMessage,
    Constraint,
    ControlMessage,
    KeysMessage,
    MediaContract,
    MessageExample,
    RealtimeContract,
    Track,
    apply_contract,
    commands_message,
    keys_message,
    message_types,
    parse_control_message,
    render_asyncapi,
)
from fal.wma.sdk import _schema_prefix


def omit_private_fields(schema, model):
    hidden = model.SCHEMA_IGNORES
    schema["properties"] = {
        key: value for key, value in schema["properties"].items() if key not in hidden
    }
    if "required" in schema:
        schema["required"] = [key for key in schema["required"] if key not in hidden]


class OrderedBaseModel(BaseModel):
    SCHEMA_IGNORES: ClassVar[set] = set()
    model_config = ConfigDict(json_schema_extra=omit_private_fields)


SESSION_PATH = "/start-session"


class Configure(BaseModel):
    type: Literal["configure"]
    level: int = 1


class Pong(BaseModel):
    type: Literal["pong"]


SampleMessage = Annotated[Union[Configure, Pong], Field(discriminator="type")]


def empty_spec(path: str = SESSION_PATH) -> dict:
    return {"openapi": "3.1.0", "paths": {path: {"post": {}}}}


def render_openapi(path: str = SESSION_PATH) -> dict:
    return apply_contract(empty_spec(path), path=SESSION_PATH)


def render_realtime(contract: RealtimeContract, *, prefix: str = "Sample") -> dict:
    return render_asyncapi(
        contract,
        title=f"{prefix} realtime client API",
        schema_prefix=prefix,
        channel_address="fal",
        openapi_operation_id="startRealtimeSession",
    )


def resolve_pointer(document: dict, ref: str):
    assert ref.startswith("#/"), ref
    value = document
    for token in ref[2:].split("/"):
        value = value[token.replace("~1", "/").replace("~0", "~")]
    return value


def assert_internal_refs_resolve(document: dict) -> None:
    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "$ref":
                    resolve_pointer(document, child)
                else:
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(document)


class TestConstraint:
    def test_only_the_stated_bounds_are_published(self):
        assert Constraint(min=640, ideal=1280).to_openapi() == {
            "min": 640,
            "ideal": 1280,
        }

    def test_a_constraint_that_bounds_nothing_is_refused(self):
        with pytest.raises(ValueError, match="must bound something"):
            Constraint()


class TestTrack:
    def test_only_kind_is_published_when_nothing_else_is_known(self):
        assert Track(kind="video").to_openapi() == {"kind": "video"}

    def test_an_outbound_track_publishes_capture_constraints(self):
        published = Track(
            kind="video", source="camera", width=640, height=360, frame_rate=24.0
        ).to_openapi("send")
        assert published["constraints"] == {
            "width": 640,
            "height": 360,
            "frameRate": 24.0,
        }

    def test_an_inbound_track_publishes_settings_instead(self):
        published = Track(kind="video", width=1280).to_openapi("receive")
        assert published == {"kind": "video", "settings": {"width": 1280}}

    def test_a_range_survives_into_the_constraint(self):
        published = Track(
            kind="video", source="camera", width=Constraint(min=640, ideal=1280)
        ).to_openapi("send")
        assert published["constraints"]["width"] == {"min": 640, "ideal": 1280}

    def test_a_required_source_track_says_so(self):
        published = Track(kind="video", source="camera", required=True).to_openapi()
        assert published == {"kind": "video", "source": "camera", "required": True}


class TestMediaContract:
    def test_both_directions_are_always_present(self):
        assert MediaContract().to_openapi() == {"send": [], "receive": []}

    def test_two_same_kind_tracks_remain_distinct_and_ordered(self):
        contract = MediaContract(
            receive=(
                Track(kind="video", width=1280),
                Track(kind="video", width=320),
            )
        )
        assert [
            track["settings"]["width"] for track in contract.to_openapi()["receive"]
        ] == [1280, 320]

    def test_a_camera_transform_has_input_constraints_and_output_settings(self):
        contract = MediaContract(
            send=(Track(kind="video", source="camera", required=True, width=640),),
            receive=(Track(kind="video", width=1280),),
        )
        published = contract.to_openapi()
        assert published["send"][0]["constraints"] == {"width": 640}
        assert published["receive"][0]["settings"] == {"width": 1280}

    def test_a_negotiable_inbound_track_is_refused(self):
        with pytest.raises(ValueError, match="settings, not constraints"):
            MediaContract(receive=(Track(kind="video", width=Constraint(min=640)),))

    def test_an_inbound_track_cannot_claim_a_capture_source(self):
        with pytest.raises(ValueError, match="cannot describe one it receives"):
            MediaContract(receive=(Track(kind="video", source="camera"),))


class TestMessageTypes:
    def test_a_union_reports_every_member(self):
        assert message_types(SampleMessage) == ("configure", "pong")

    def test_a_single_model_reports_its_own_type(self):
        assert message_types(Pong) == ("pong",)

    def test_the_compatibility_protocol_remains_key_based(self):
        assert message_types(ControlMessage) == (
            "keys",
            "ping",
            "prompt",
            "reset",
        )


class TestOpenApiDiscovery:
    def test_the_extension_links_to_asyncapi_without_embedding_it(self):
        spec = render_openapi()
        extension = spec["paths"][SESSION_PATH][CONTRACT_EXTENSION]
        assert extension == {
            "schemaVersion": CONTRACT_VERSION,
            "transport": {
                "protocol": "webrtc",
                "sessionProtocol": "wma",
                "version": 1,
            },
            "asyncapi": {"url": ASYNCAPI_URL},
        }
        assert "components" not in spec

    def test_schema_and_transport_versions_are_separate(self):
        extension = render_openapi()["paths"][SESSION_PATH][CONTRACT_EXTENSION]
        assert "schemaVersion" in extension
        assert "version" not in extension
        assert "version" in extension["transport"]

    def test_a_spec_without_the_session_path_is_left_alone(self):
        spec = render_openapi("/other")
        assert spec["paths"]["/other"] == {"post": {}}


class TestAsyncApiDocument:
    def test_document_is_asyncapi_3_and_links_back_to_openapi(self):
        document = render_realtime(RealtimeContract())
        assert document["asyncapi"] == ASYNCAPI_SPEC_VERSION
        assert document["x-fal-openapi"] == {
            "url": "./openapi.json",
            "operationId": "startRealtimeSession",
        }
        assert document["servers"]["session"]["protocol"] == "webrtc"

    def test_media_lives_in_the_asyncapi_extension(self):
        media = MediaContract(
            send=(Track(kind="video", source="camera", required=True),),
            receive=(Track(kind="video", width=1280),),
        )
        document = render_realtime(RealtimeContract(media=media))
        assert document["x-fal-media"] == {
            "perspective": "client",
            "send": [{"kind": "video", "source": "camera", "required": True}],
            "receive": [{"kind": "video", "settings": {"width": 1280}}],
        }

    def test_operations_are_directional_from_the_client_perspective(self):
        document = render_realtime(
            RealtimeContract(
                client_messages=Configure,
                server_messages=Pong,
            )
        )
        assert document["operations"]["sendControl"]["action"] == "send"
        assert document["operations"]["receiveEvents"]["action"] == "receive"

    def test_messages_use_real_asyncapi_references(self):
        document = render_realtime(
            RealtimeContract(
                client_messages=SampleMessage,
                server_messages=Pong,
            )
        )
        assert document["operations"]["sendControl"]["channel"] == {
            "$ref": "#/channels/control"
        }
        assert document["operations"]["sendControl"]["messages"] == [
            {"$ref": "#/channels/control/messages/client.configure"},
            {"$ref": "#/channels/control/messages/client.pong"},
        ]
        assert_internal_refs_resolve(document)

    def test_same_wire_name_in_both_directions_does_not_collide(self):
        class ClientStatus(BaseModel):
            type: Literal["status"]
            requested: bool

        class ServerStatus(BaseModel):
            type: Literal["status"]
            ready: bool

        document = render_realtime(
            RealtimeContract(
                client_messages=ClientStatus,
                server_messages=ServerStatus,
            )
        )
        messages = document["channels"]["control"]["messages"]
        assert set(messages) == {"client.status", "server.status"}
        client_payload = document["components"]["messages"]["client.status"]["payload"][
            "$ref"
        ]
        server_payload = document["components"]["messages"]["server.status"]["payload"][
            "$ref"
        ]
        assert client_payload != server_payload
        assert_internal_refs_resolve(document)

    def test_session_params_are_not_reserved_without_wire_semantics(self):
        with pytest.raises(TypeError, match="session_params"):
            RealtimeContract(session_params=Configure)  # type: ignore[call-arg]

    def test_named_examples_are_validated_and_grouped_by_wire_type(self):
        document = render_realtime(
            RealtimeContract(
                client_messages=SampleMessage,
                client_message_examples=(
                    MessageExample(
                        name="Advanced",
                        summary="Begin at a higher level.",
                        payload={"type": "configure", "level": 4},
                    ),
                    MessageExample(name="Pong", payload={"type": "pong"}),
                ),
            )
        )
        messages = document["components"]["messages"]
        assert messages["client.configure"]["examples"] == [
            {
                "name": "Advanced",
                "summary": "Begin at a higher level.",
                "payload": {"type": "configure", "level": 4},
            }
        ]
        assert messages["client.pong"]["examples"] == [
            {"name": "Pong", "payload": {"type": "pong"}}
        ]

    def test_example_payloads_use_schema_defaults_and_omit_none(self):
        class OptionalMessage(BaseModel):
            type: Literal["optional"]
            enabled: bool = True
            note: Union[str, None] = None

        document = render_realtime(
            RealtimeContract(
                client_messages=OptionalMessage,
                client_message_examples=(
                    MessageExample(name="Default", payload={"type": "optional"}),
                ),
            )
        )
        example = document["components"]["messages"]["client.optional"]["examples"][0]
        assert example["payload"] == {"type": "optional", "enabled": True}

    def test_example_payloads_omit_nested_private_fields(self):
        class Settings(OrderedBaseModel):
            SCHEMA_IGNORES: ClassVar[set] = {"private_setting"}
            visible_setting: int = 1
            private_setting: int = 2

        class ChoiceA(OrderedBaseModel):
            SCHEMA_IGNORES: ClassVar[set] = {"private_a"}
            visible_a: int = 3
            private_a: int = 4

        class ChoiceB(OrderedBaseModel):
            SCHEMA_IGNORES: ClassVar[set] = {"private_b"}
            visible_b: int = 5
            private_b: int = 6

        class NestedMessage(BaseModel):
            model_config = {"extra": "forbid"}
            type: Literal["nested"]
            settings: Settings
            history: List[Settings]
            settings_by_name: Dict[str, Settings]
            choice: Union[ChoiceA, ChoiceB]

        document = render_realtime(
            RealtimeContract(
                client_messages=NestedMessage,
                client_message_examples=(
                    MessageExample(
                        name="Nested",
                        payload={
                            "type": "nested",
                            "settings": {},
                            "history": [{}],
                            "settings_by_name": {"primary": {}},
                            "choice": {},
                        },
                    ),
                ),
            )
        )
        example = document["components"]["messages"]["client.nested"]["examples"][0]
        assert example["payload"] == {
            "type": "nested",
            "settings": {"visible_setting": 1},
            "history": [{"visible_setting": 1}],
            "settings_by_name": {"primary": {"visible_setting": 1}},
            "choice": {"visible_a": 3},
        }

    def test_invalid_example_payload_is_refused(self):
        with pytest.raises(ValidationError):
            render_realtime(
                RealtimeContract(
                    client_messages=Configure,
                    client_message_examples=(
                        MessageExample(
                            name="Invalid",
                            payload={"type": "configure", "level": "high"},
                        ),
                    ),
                )
            )

    def test_examples_require_a_matching_message_schema(self):
        with pytest.raises(ValueError, match="require a client message schema"):
            RealtimeContract(
                client_message_examples=(
                    MessageExample(name="Orphan", payload={"type": "configure"}),
                )
            )


class TestSchemaPrefix:
    @pytest.mark.parametrize(
        ("class_name", "expected"),
        [
            ("WmaEchoApp", "WmaEcho"),
            ("ExampleWorldApp", "ExampleWorld"),
            ("Renderer", "Renderer"),
            ("App", "App"),
        ],
    )
    def test_the_app_suffix_is_dropped(self, class_name, expected):
        assert _schema_prefix(class_name) == expected


class TestPayloadSchemaGuards:
    def test_nested_optional_type_field_stays_optional(self):
        class NestedStyle(BaseModel):
            type: Optional[str] = None
            name: str

        class SetStyle(BaseModel):
            type: Literal["set_style"]
            style: NestedStyle

        document = render_realtime(
            RealtimeContract(media=MediaContract(), client_messages=SetStyle)
        )
        schemas = document["components"]["schemas"]
        assert "type" in schemas["SampleClientMessage"]["required"]
        assert "type" not in schemas["SampleClientNestedStyle"]["required"]

    def test_message_without_wire_type_is_refused(self):
        class Untyped(BaseModel):
            value: int

        with pytest.raises(ValueError, match="literal 'type'"):
            render_realtime(
                RealtimeContract(media=MediaContract(), client_messages=Untyped)
            )

    def test_nested_discriminated_union_publishes_a_string_discriminator(self):
        class Circle(BaseModel):
            type: Literal["circle"] = "circle"
            radius: float

        class Square(BaseModel):
            type: Literal["square"] = "square"
            side: float

        class Draw(BaseModel):
            type: Literal["draw"]
            shape: Annotated[Union[Circle, Square], Field(discriminator="type")]

        document = render_realtime(
            RealtimeContract(media=MediaContract(), client_messages=Draw)
        )
        shape = document["components"]["schemas"]["SampleClientMessage"]["properties"][
            "shape"
        ]
        # AsyncAPI Schema Objects define discriminator as a string, not the
        # OpenAPI {propertyName, mapping} object Pydantic emits.
        assert shape["discriminator"] == "type"
        # Runtime validation cannot select a union branch without the tag,
        # even though the member models default it — the published members
        # must require it.
        schemas = document["components"]["schemas"]
        assert "type" in schemas["SampleClientCircle"]["required"]
        assert "type" in schemas["SampleClientSquare"]["required"]
        assert_internal_refs_resolve(document)

    def test_non_string_wire_discriminator_is_refused(self):
        class Numeric(BaseModel):
            type: Literal[1]

        with pytest.raises(ValueError, match="must be strings"):
            render_realtime(
                RealtimeContract(media=MediaContract(), client_messages=Numeric)
            )

    def test_non_string_union_tags_are_refused(self):
        # The union path bypasses message_types(): Pydantic stringifies the
        # mapping keys while the member const values stay numeric, so the
        # guard must hold on the published payload schemas too.
        class One(BaseModel):
            type: Literal[1]

        class Two(BaseModel):
            type: Literal[2]

        numeric_union = Annotated[Union[One, Two], Field(discriminator="type")]
        with pytest.raises(ValueError, match="must be strings"):
            render_realtime(
                RealtimeContract(media=MediaContract(), client_messages=numeric_union)
            )

    def test_aliased_wire_discriminator_is_refused(self):
        # A single aliased model is caught by the no-literal-type guard; the
        # union path is the one that slips through message_types() via the
        # discriminator mapping while publishing payloads without ``type``.
        class Start(BaseModel):
            type: Literal["start"] = Field(alias="kind")

        class Stop(BaseModel):
            type: Literal["stop"] = Field(alias="kind")

        aliased_union = Annotated[Union[Start, Stop], Field(discriminator="type")]
        with pytest.raises(ValueError, match="aliases are not part"):
            render_realtime(
                RealtimeContract(media=MediaContract(), client_messages=aliased_union)
            )

    def test_nested_model_named_message_is_refused(self):
        class Message(BaseModel):
            text: str

        class Send(BaseModel):
            type: Literal["send"]
            message: Message

        with pytest.raises(ValueError, match="rename the nested model"):
            render_realtime(
                RealtimeContract(media=MediaContract(), client_messages=Send)
            )


class TestAppWithoutContract:
    def test_app_without_a_contract_publishes_no_extension(self):
        from fal.wma import App

        class Bare(App):  # type: ignore[misc]
            pass

        app = Bare(_allow_init=True)
        assert CONTRACT_EXTENSION not in app.openapi()["paths"][SESSION_PATH]
        with pytest.raises(ValueError, match="does not declare"):
            app.asyncapi()


class TestCommandsMessage:
    Commands = commands_message(
        "MovementCommands",
        commands=("forward", "backward", "left", "right"),
        conflict_groups=(("forward", "backward"),),
    )

    def test_schema_publishes_semantic_commands_and_conflicts(self):
        schema = self.Commands.model_json_schema()
        assert schema["properties"]["active"]["items"]["enum"] == [
            "forward",
            "backward",
            "left",
            "right",
        ]
        assert schema["x-fal-ui"] == {"conflictGroups": [["forward", "backward"]]}

    def test_generated_model_dispatches_as_commands(self):
        assert issubclass(self.Commands, CommandsMessage)

    def test_unknown_command_is_rejected(self):
        with pytest.raises(ValidationError):
            self.Commands(type="commands", active=["W"])


class TestLegacyKeysMessage:
    Arrows = keys_message(
        "Arrows",
        keys=("UP", "DOWN", "LEFT", "RIGHT"),
        conflict_groups=(("UP", "DOWN"),),
    )

    def test_legacy_schema_is_marked_deprecated(self):
        assert self.Arrows.model_json_schema()["deprecated"] is True

    @pytest.mark.parametrize(
        ("reported", "expected"),
        [
            (["up", "left"], ["UP", "LEFT"]),
            (["UP", "ESCAPE"], ["UP"]),
            (["UP", "up", "UP"], ["UP"]),
        ],
    )
    def test_legacy_reports_keep_their_lenient_wire_behavior(self, reported, expected):
        message = self.Arrows(type="keys", pressed=reported)
        assert message.pressed == expected

    def test_a_non_string_entry_is_still_an_error(self):
        with pytest.raises(ValidationError):
            self.Arrows(type="keys", pressed=["UP", 7])

    def test_legacy_key_input_remains_parseable_for_compatibility(self):
        message = parse_control_message(json.dumps({"type": "keys", "pressed": ["w"]}))
        assert isinstance(message, KeysMessage)
        assert message.pressed == ["w"]


def test_message_presentation_is_opt_in_and_validates_wire_types():
    from fal.wma.contract import MessagePresentation

    class Start(BaseModel):
        type: Literal["start"]

    base = RealtimeContract(client_messages=Start)
    before = render_realtime(base)
    declared = RealtimeContract(
        client_messages=Start,
        client_message_presentation=(
            MessagePresentation("start", "Start world", "Choose a scene."),
        ),
    )
    document = render_realtime(declared)
    assert document["components"]["messages"]["client.start"]["title"] == "Start world"
    assert (
        document["components"]["messages"]["client.start"]["summary"]
        == "Choose a scene."
    )
    assert render_realtime(base) == before
    assert (
        before["components"]["messages"]["client.start"]["title"]
        == "Client start message"
    )
    for items in (
        (MessagePresentation("missing", "Start world"),),
        (MessagePresentation("start", " "),),
        (MessagePresentation("start", "One"), MessagePresentation("start", "Two")),
    ):
        with pytest.raises(ValueError):
            render_realtime(
                RealtimeContract(
                    client_messages=Start, client_message_presentation=items
                )
            )
