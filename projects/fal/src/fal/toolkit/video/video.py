from functools import wraps
from typing import Optional

from pydantic import Field

from fal.toolkit.constraints import VideoValidationConfig, to_xfal
from fal.toolkit.file.file import IS_PYDANTIC_V2, File


def _merge_ui(schema: dict, ui: Optional[dict]) -> None:
    """Merge caller ``ui`` into a schema dict, defaulting ``ui.field`` to video."""
    schema_ui = schema.setdefault("ui", {})
    if ui:
        schema_ui.update(ui)
    schema_ui.setdefault("field", "video")


@wraps(Field)
def VideoField(
    *args,
    constraints: Optional[VideoValidationConfig] = None,
    ui: Optional[dict] = None,
    **kwargs,
):
    """A ``Field`` for a video input that documents the videos it accepts.

    Pass ``constraints`` to emit the accepted-video limits under the ``x-fal``
    schema extension, and ``ui`` for UI metadata (e.g. ``{"important": True}``);
    all other arguments are forwarded to ``Field``. ``ui`` is taken as an explicit
    argument (rather than a passthrough kwarg) so it is not dropped when an
    explicit ``json_schema_extra`` is also emitted on Pydantic v2.
    """
    fal_extra: dict = {}
    if constraints is not None:
        data = to_xfal(constraints)
        if data:
            fal_extra["x-fal"] = data

    if IS_PYDANTIC_V2:
        # Pydantic v2: use json_schema_extra
        json_schema_extra = kwargs.pop("json_schema_extra", None) or {}
        if callable(json_schema_extra):
            # If it's a callable, wrap it to also add ui.field
            original_func = json_schema_extra

            def merged_schema_extra(schema):
                original_func(schema)
                _merge_ui(schema, ui)
                schema.update(fal_extra)

            kwargs["json_schema_extra"] = merged_schema_extra
        else:
            _merge_ui(json_schema_extra, ui)
            json_schema_extra.update(fal_extra)
            kwargs["json_schema_extra"] = json_schema_extra
    else:
        # Pydantic v1: extra kwargs are stored on the field and emitted as-is.
        merged_ui = dict(ui or {})
        merged_ui.setdefault("field", "video")
        kwargs["ui"] = merged_ui
        kwargs.update(fal_extra)
    return Field(*args, **kwargs)


class Video(File):
    if IS_PYDANTIC_V2:
        model_config = {"json_schema_extra": {"ui": {"field": "video"}}}
    else:

        class Config:
            @staticmethod
            def schema_extra(schema, model_type):
                schema.setdefault("ui", {})["field"] = "video"
