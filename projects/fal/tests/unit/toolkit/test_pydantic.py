from typing import Any

import pytest
from pydantic import Field

from fal.toolkit.pydantic import (
    IS_PYDANTIC_V2,
    AudioField,
    FalBaseModel,
    FileField,
    Hidden,
    ImageField,
    VideoField,
    _is_hidden_field,
)


class MockUrlField:
    def __init__(self):
        self.bound_loc = None

    def _bind_context(self, loc):
        self.bound_loc = loc


def test_hidden_requires_field_info():
    """Hidden must receive a FieldInfo instance."""
    with pytest.raises(TypeError, match="requires a Field instance"):
        Hidden("not a field")


def test_hidden_requires_default():
    """Hidden fields must have a default value."""
    with pytest.raises(ValueError, match="must have a default"):
        Hidden(Field())


def test_hidden_with_default():
    """Hidden accepts fields with default values."""
    field = Hidden(Field(default=False))
    assert _is_hidden_field(field)


def test_hidden_with_default_factory():
    """Hidden accepts fields with default_factory."""
    field = Hidden(Field(default_factory=list))
    assert _is_hidden_field(field)


def test_hidden_with_none_default():
    """Hidden accepts fields with None as default."""
    field = Hidden(Field(default=None))
    assert _is_hidden_field(field)


def test_non_hidden_field_not_marked():
    """Regular fields should not be marked as hidden."""
    field = Field(default=False)
    assert not _is_hidden_field(field)


def test_field_orders_in_schema():
    """FIELD_ORDERS should reorder properties in schema."""

    class Model(FalBaseModel):
        FIELD_ORDERS = ["b", "a"]
        a: str = Field(default="")
        b: str = Field(default="")
        c: str = Field(default="")

    schema = Model.model_json_schema() if IS_PYDANTIC_V2 else Model.schema()
    props = list(schema["properties"].keys())
    assert props[:2] == ["b", "a"]


def test_schema_ignores_sets_hidden_true():
    """SCHEMA_IGNORES should set ui.hidden = True."""

    # Warning is emitted when class is defined
    with pytest.warns(DeprecationWarning, match="SCHEMA_IGNORES is deprecated"):

        class Model(FalBaseModel):
            SCHEMA_IGNORES = {"hidden_field"}
            visible: str = Field(default="")
            hidden_field: str = Field(default="")

    schema = Model.model_json_schema() if IS_PYDANTIC_V2 else Model.schema()

    assert schema["properties"]["hidden_field"]["ui"]["hidden"] is True
    # visible field should not have ui.hidden = True
    assert (
        "ui" not in schema["properties"]["visible"]
        or schema["properties"]["visible"].get("ui", {}).get("hidden") is not True
    )


def test_hidden_wrapper_in_model():
    """Hidden() wrapper should work in model definition."""

    class Model(FalBaseModel):
        visible: str = Field(default="")
        hidden: bool = Hidden(Field(default=False))

    schema = Model.model_json_schema() if IS_PYDANTIC_V2 else Model.schema()
    assert schema["properties"]["hidden"]["ui"]["hidden"] is True


def test_combined_field_orders_and_schema_ignores():
    """FIELD_ORDERS and SCHEMA_IGNORES should work together."""

    # Warning is emitted when class is defined
    with pytest.warns(DeprecationWarning, match="SCHEMA_IGNORES is deprecated"):

        class Model(FalBaseModel):
            FIELD_ORDERS = ["prompt", "image"]
            SCHEMA_IGNORES = {"internal"}
            a_first_alphabetically: str = Field(default="")
            prompt: str = Field(default="")
            image: str = Field(default="")
            internal: str = Field(default="hidden")

    schema = Model.model_json_schema() if IS_PYDANTIC_V2 else Model.schema()

    props = list(schema["properties"].keys())
    # prompt and image should come first
    assert props[0] == "prompt"
    assert props[1] == "image"
    # internal should have hidden = True
    assert schema["properties"]["internal"]["ui"]["hidden"] is True


def test_empty_field_orders():
    """Empty FIELD_ORDERS should not modify order."""

    class Model(FalBaseModel):
        FIELD_ORDERS = []
        a: str = Field(default="")
        b: str = Field(default="")

    schema = Model.model_json_schema() if IS_PYDANTIC_V2 else Model.schema()
    # Should have both fields
    assert "a" in schema["properties"]
    assert "b" in schema["properties"]


def test_schema_ignores_deprecation_warning():
    """SCHEMA_IGNORES should emit a deprecation warning."""

    # Warning is emitted when class is defined
    with pytest.warns(DeprecationWarning, match="SCHEMA_IGNORES is deprecated"):

        class Model(FalBaseModel):
            SCHEMA_IGNORES = {"field"}
            field: str = Field(default="")

    schema = Model.model_json_schema() if IS_PYDANTIC_V2 else Model.schema()

    assert "field" in schema["properties"]
    assert schema["properties"]["field"]["ui"]["hidden"] is True


def test_model_instantiation():
    """FalBaseModel should be instantiable like a normal Pydantic model."""

    class Model(FalBaseModel):
        name: str = Field(default="test")
        value: int = Field(default=42)

    instance = Model()
    assert instance.name == "test"
    assert instance.value == 42

    instance2 = Model(name="custom", value=100)
    assert instance2.name == "custom"
    assert instance2.value == 100


def test_inheritance():
    """FalBaseModel subclasses should properly inherit."""

    class BaseInput(FalBaseModel):
        FIELD_ORDERS = ["prompt"]
        prompt: str = Field(default="")

    class ExtendedInput(BaseInput):
        FIELD_ORDERS = ["prompt", "image"]
        image: str = Field(default="")

    schema = (
        ExtendedInput.model_json_schema() if IS_PYDANTIC_V2 else ExtendedInput.schema()
    )
    props = list(schema["properties"].keys())
    assert props[0] == "prompt"
    assert props[1] == "image"


def test_context_binding_calls_bind_context():
    """Model should call _bind_context on fields that support it."""

    class Model(FalBaseModel):
        if IS_PYDANTIC_V2:
            model_config = {"arbitrary_types_allowed": True}
        else:

            class Config:
                arbitrary_types_allowed = True

        url_field: Any = Field(default_factory=MockUrlField)

    # Create instance - this should trigger context binding
    instance = Model()
    # The mock should have been bound with the field name
    assert instance.url_field.bound_loc == ("url_field",)


def test_context_binding_nested_model():
    """Context binding should work recursively for nested models."""

    class InnerModel(FalBaseModel):
        if IS_PYDANTIC_V2:
            model_config = {"arbitrary_types_allowed": True}
        else:

            class Config:
                arbitrary_types_allowed = True

        inner_url: Any = Field(default_factory=MockUrlField)

    class OuterModel(InnerModel):
        pass

    instance = OuterModel()
    assert instance.inner_url.bound_loc == ("inner_url",)


def test_context_binding_list_fields():
    """Context binding should work for list fields."""

    class Model(FalBaseModel):
        urls: list = Field(default_factory=list)

    # Create with list containing mockable items
    mock1 = MockUrlField()
    mock2 = MockUrlField()

    urls = [mock1, mock2]
    instance = Model(urls=urls)

    assert instance.urls[0].bound_loc == ("urls", 0)
    assert instance.urls[1].bound_loc == ("urls", 1)


class TestFieldHelpers:
    """Tests for domain-specific Field helpers (FileField, ImageField, etc.)."""

    def test_file_field_init(self):
        """FileField should initialize and set ui.field = 'file' in schema."""

        class Model(FalBaseModel):
            file_input: str = FileField(default="", description="A file input")

        schema = Model.model_json_schema() if IS_PYDANTIC_V2 else Model.schema()
        assert schema["properties"]["file_input"]["ui"]["field"] == "file"

    def test_image_field_init(self):
        """ImageField should initialize and set ui.field = 'image' in schema."""

        class Model(FalBaseModel):
            image_input: str = ImageField(default="", description="An image input")

        schema = Model.model_json_schema() if IS_PYDANTIC_V2 else Model.schema()
        assert schema["properties"]["image_input"]["ui"]["field"] == "image"

    def test_video_field_init(self):
        """VideoField should initialize and set ui.field = 'video' in schema."""

        class Model(FalBaseModel):
            video_input: str = VideoField(default="", description="A video input")

        schema = Model.model_json_schema() if IS_PYDANTIC_V2 else Model.schema()
        assert schema["properties"]["video_input"]["ui"]["field"] == "video"

    def test_audio_field_init(self):
        """AudioField should initialize and set ui.field = 'audio' in schema."""

        class Model(FalBaseModel):
            audio_input: str = AudioField(default="", description="An audio input")

        schema = Model.model_json_schema() if IS_PYDANTIC_V2 else Model.schema()
        assert schema["properties"]["audio_input"]["ui"]["field"] == "audio"
