from functools import wraps

from pydantic import Field

from fal.toolkit.file.file import IS_PYDANTIC_V2, File


@wraps(Field)
def AudioField(*args, **kwargs):
    if IS_PYDANTIC_V2:
        # Pydantic v2: use json_schema_extra
        json_schema_extra = kwargs.pop("json_schema_extra", None) or {}
        if callable(json_schema_extra):
            # If it's a callable, wrap it to also add ui.field
            original_func = json_schema_extra

            def merged_schema_extra(schema):
                original_func(schema)
                schema.setdefault("ui", {}).setdefault("field", "audio")

            kwargs["json_schema_extra"] = merged_schema_extra
        else:
            json_schema_extra.setdefault("ui", {}).setdefault("field", "audio")
            kwargs["json_schema_extra"] = json_schema_extra
    else:
        # Pydantic v1: use ui kwarg (stored in extra)
        ui = kwargs.get("ui", {})
        ui.setdefault("field", "audio")
        kwargs["ui"] = ui
    return Field(*args, **kwargs)


class Audio(File):
    if IS_PYDANTIC_V2:
        model_config = {"json_schema_extra": {"ui": {"field": "audio"}}}
    else:

        class Config:
            @staticmethod
            def schema_extra(schema, model_type):
                schema.setdefault("ui", {})["field"] = "audio"
