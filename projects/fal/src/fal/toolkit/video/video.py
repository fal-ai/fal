from fal.toolkit.file.file import IS_PYDANTIC_V2, File


class Video(File):
    if IS_PYDANTIC_V2:
        model_config = {"json_schema_extra": {"ui": {"field": "video"}}}
    else:

        class Config:
            @staticmethod
            def schema_extra(schema, model_type):
                schema.setdefault("ui", {})["field"] = "video"
