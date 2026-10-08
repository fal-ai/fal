"""Optional presentation hints for WMA schemas and message examples.

Types, defaults, constraints and descriptions belong to JSON Schema. These
hints only suggest how a client presents that schema; they never change what
an application accepts. Use
``Field(..., json_schema_extra=FieldUI(...).schema_extra())``.
"""

from typing import Any, Dict, Literal, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

UI_EXTENSION = "x-fal-ui"


class FieldUI(BaseModel):
    """Presentation of a WMA message property.

    ``field`` selects a specialized widget only when JSON Schema alone cannot
    express the intended input. A missing/unsupported widget falls back to the
    schema's type. Hidden values still participate in payload validation.
    Clients must reveal hidden required inputs they cannot supply. Lower
    ``order`` values appear first; ties retain schema property order.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    field: Union[Literal["text", "textarea", "image", "audio", "video"], None] = None
    hidden: Union[bool, None] = None
    advanced: Union[bool, None] = None
    order: Union[int, None] = Field(default=None, ge=0)

    def schema_extra(self) -> Dict[str, Any]:
        return {UI_EXTENSION: self.model_dump(mode="json", exclude_none=True)}


class ExampleUI(BaseModel):
    """Optional gallery presentation for a native named WMA message example.

    The native example name and summary remain its title and description.
    A thumbnail is a preview only, never an input to the model.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    category: Union[str, None] = Field(default=None, min_length=1)
    tags: Tuple[str, ...] = ()
    thumbnail_url: Union[HttpUrl, None] = None
    thumbnail_alt: Union[str, None] = None
    order: Union[int, None] = Field(default=None, ge=0)

    def schema_extra(self) -> Dict[str, Any]:
        return {
            UI_EXTENSION: self.model_dump(
                mode="json", exclude_none=True, exclude_defaults=True
            )
        }
