from __future__ import annotations

import warnings
from typing import Any, ClassVar, List, Set, TypeVar

import pydantic
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo

from fal.toolkit.audio import AudioField
from fal.toolkit.file import FileField
from fal.toolkit.image import ImageField
from fal.toolkit.video import VideoField

if not hasattr(pydantic, "__version__") or pydantic.__version__.startswith("1."):
    IS_PYDANTIC_V2 = False
else:
    from pydantic import model_validator

    IS_PYDANTIC_V2 = True


_T = TypeVar("_T")

# Marker for hidden fields in metadata (Pydantic v2) and extra dict (Pydantic v1)
_FAL_HIDDEN = "_fal_hidden"


def Hidden(field: _T) -> _T:
    """
    Wrapper that marks a Field as hidden in the UI.

    The field MUST have a default or default_factory set since hidden
    fields cannot be required inputs in the UI.

    Usage:
        class Input(FalBaseModel):
            prompt: str = Field(...)
            hidden_flag: bool = Hidden(Field(default=False))

    Args:
        field: A pydantic FieldInfo instance (result of Field(...))

    Returns:
        The same FieldInfo with hidden marker in metadata

    Raises:
        ValueError: If field has no default or default_factory
    """
    if not isinstance(field, FieldInfo):
        raise TypeError(
            f"Hidden() requires a Field instance, got {type(field).__name__}. "
            f"Usage: Hidden(Field(default=...))"
        )

    if IS_PYDANTIC_V2:
        from pydantic_core import PydanticUndefined

        has_default = (
            field.default is not PydanticUndefined or field.default_factory is not None
        )
    else:
        from pydantic.fields import Undefined

        has_default = (
            field.default is not Undefined or field.default_factory is not None
        )

    if not has_default:
        raise ValueError("Hidden fields must have a default or default_factory.")

    if IS_PYDANTIC_V2:
        if field.metadata is None:
            field.metadata = []
        field.metadata = list(field.metadata) + [_FAL_HIDDEN]
    else:
        if not hasattr(field, "extra") or field.extra is None:
            field.extra = {}
        field.extra[_FAL_HIDDEN] = True

    return field


def _is_hidden_field(field_info: FieldInfo) -> bool:
    """Check if a field has been marked as hidden."""
    if IS_PYDANTIC_V2:
        return _FAL_HIDDEN in (field_info.metadata or [])
    else:
        return getattr(field_info, "extra", {}).get(_FAL_HIDDEN, False)


def _bind_context_recursively(
    data_node: Any, path_prefix: tuple[str | int, ...]
) -> None:
    """
    Recursively bind location context for better error reporting.

    This enables proper error location tracking for nested structures like:
    {"items": [{"photo": "https://..."}]} -> loc = ("items", 0, "photo")
    """
    # Check for _bind_context method (used by HttpsOrDataUrl and similar types)
    if hasattr(data_node, "_bind_context") and callable(data_node._bind_context):
        data_node._bind_context(loc=path_prefix)
    elif isinstance(data_node, BaseModel):
        # Use model_fields for Pydantic v2, __fields__ for v1
        model_cls = type(data_node)
        fields = model_cls.model_fields if IS_PYDANTIC_V2 else model_cls.__fields__
        for field_name in fields:
            field_value = getattr(data_node, field_name, None)
            if field_value is not None:
                _bind_context_recursively(field_value, path_prefix + (field_name,))
    elif isinstance(data_node, list):
        for i, item in enumerate(data_node):
            _bind_context_recursively(item, path_prefix + (i,))


def _apply_schema_modifications(schema: dict[str, Any], model: type) -> None:
    """
    Apply SCHEMA_IGNORES, FIELD_ORDERS, and Hidden() modifications to a JSON schema.

    - SCHEMA_IGNORES: Sets ui.hidden = True on specified fields
    - FIELD_ORDERS: Reorders properties so listed fields appear first
    - Hidden() fields: Sets ui.hidden = True based on field metadata
    """
    properties = schema.get("properties", {})

    # Apply SCHEMA_IGNORES: set ui.hidden = True
    schema_ignores: Set[str] = getattr(model, "SCHEMA_IGNORES", set())

    # Set ui.hidden = True for hidden fields and fields in SCHEMA_IGNORES
    if IS_PYDANTIC_V2:
        fields = getattr(model, "model_fields", {})
    else:
        fields = getattr(model, "__fields__", {})

    for field_name, field in fields.items():
        if field_name not in properties:
            continue
        # In Pydantic v1, fields are ModelField objects; get field_info from them
        field_info = field if IS_PYDANTIC_V2 else field.field_info
        if _is_hidden_field(field_info) or field_name in schema_ignores:
            properties[field_name].setdefault("ui", {})["hidden"] = True

    # Reorder properties based on FIELD_ORDERS
    field_orders = getattr(model, "FIELD_ORDERS", [])
    if field_orders:
        ordered_props = {}
        for field_name in field_orders:
            if field_name in properties:
                ordered_props[field_name] = properties.pop(field_name)

        schema["properties"] = {**ordered_props, **properties}


class FalBaseModel(BaseModel):
    """
    Base model for fal applications with field ordering, visibility control,
    and context binding for error reporting.

    Features:
    - FIELD_ORDERS: Control field order in JSON schema, useful for nested
      models.
    - Hidden(Field(...)): Mark fields as hidden from OpenAPI schema, useful
      for hidden params.

    Example:
        from fal.toolkit.pydantic import FalBaseModel, Field, Hidden

        class Input(FalBaseModel):
            FIELD_ORDERS = ["prompt", "image_url"]

            prompt: str = Field(description="Text prompt")
            image_url: str = Field(description="Image URL")
            debug_mode: bool = Hidden(Field(default=False))
    """

    SCHEMA_IGNORES: ClassVar[Set[str]] = set()
    FIELD_ORDERS: ClassVar[List[str]] = []

    if IS_PYDANTIC_V2:
        # Pydantic v2: Use model_config and model_validator

        @classmethod
        def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
            super().__pydantic_init_subclass__(**kwargs)

            if "SCHEMA_IGNORES" in cls.__dict__ and cls.__dict__["SCHEMA_IGNORES"]:
                warnings.warn(
                    "FalBaseModel.SCHEMA_IGNORES is deprecated. "
                    "Use Hidden(Field(...)) instead.",
                    DeprecationWarning,
                    stacklevel=3,
                )

            def schema_extra(schema: dict[str, Any], model_cls: type = cls) -> None:
                _apply_schema_modifications(schema, model_cls)

            if hasattr(cls, "model_config") and cls.model_config:
                existing = dict(cls.model_config)
                existing["json_schema_extra"] = schema_extra
                cls.model_config = existing
            else:
                cls.model_config = {"json_schema_extra": schema_extra}

        @model_validator(mode="after")
        def _inject_context_paths(self) -> FalBaseModel:
            for field_name in self.__class__.model_fields:
                field_value = getattr(self, field_name, None)
                if field_value is not None:
                    _bind_context_recursively(field_value, path_prefix=(field_name,))
            return self

    else:
        # Pydantic v1: Use Config class and root_validator
        class Config:
            @staticmethod
            def schema_extra(schema: dict[str, Any], model: type) -> None:
                _apply_schema_modifications(schema, model)

        def __init_subclass__(cls, **kwargs: Any) -> None:
            super().__init_subclass__(**kwargs)

            if "SCHEMA_IGNORES" in cls.__dict__ and cls.__dict__["SCHEMA_IGNORES"]:
                warnings.warn(
                    "FalBaseModel.SCHEMA_IGNORES is deprecated. "
                    "Use Hidden(Field(...)) instead.",
                    DeprecationWarning,
                    stacklevel=3,
                )

        @pydantic.root_validator(pre=False)
        def _inject_context_paths_v1(cls, values: dict) -> dict:
            for field_name, field_value in values.items():
                if field_value is not None:
                    _bind_context_recursively(field_value, path_prefix=(field_name,))
            return values


__all__ = [
    # Base model
    "FalBaseModel",
    # Field utilities
    "Field",
    "Hidden",
    # Domain-specific fields
    "AudioField",
    "FileField",
    "ImageField",
    "VideoField",
    # Version detection
    "IS_PYDANTIC_V2",
]
