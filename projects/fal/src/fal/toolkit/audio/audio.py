from functools import wraps

from pydantic import Field

from fal.toolkit.file.file import IS_PYDANTIC_V2, File


@wraps(Field)
def AudioField(*args, **kwargs):
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
