import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Union

import pydantic
from pydantic.utils import update_not_none

from fal.toolkit.image import read_image_from_url
from fal.toolkit.utils.download_utils import download_file

# https://github.com/pydantic/pydantic/pull/2573
if not hasattr(pydantic, "__version__") or pydantic.__version__.startswith("1."):
    IS_PYDANTIC_V2 = False
else:
    IS_PYDANTIC_V2 = True

MAX_DATA_URI_LENGTH = 10 * 1024 * 1024
MAX_HTTPS_URL_LENGTH = 2048

HTTP_URL_REGEX = (
    r"^https:\/\/(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?::\d{1,5})?(?:\/[^\s]*)?$"
)


class DownloadFileMixin:
    @contextmanager
    def as_temp_file(self) -> Generator[Path, None, None]:
        with tempfile.TemporaryDirectory() as temp_dir:
            yield download_file(str(self), temp_dir)


class DownloadImageMixin:
    def to_pil(self):
        return read_image_from_url(str(self))


class DataUri(DownloadFileMixin, str):
    if IS_PYDANTIC_V2:

        @classmethod
        def __get_pydantic_core_schema__(cls, source_type: Any, handler) -> Any:
            return {
                "type": "str",
                "pattern": "^data:",
                "max_length": MAX_DATA_URI_LENGTH,
                "strip_whitespace": True,
            }

        def __get_pydantic_json_schema__(cls, core_schema, handler) -> Dict[str, Any]:
            json_schema = handler(core_schema)
            json_schema.update(format="data-uri")
            return json_schema
    else:

        @classmethod
        def __get_validators__(cls):
            yield cls.validate

        @classmethod
        def validate(cls, value: Any) -> "DataUri":
            from pydantic.validators import str_validator

            value = str_validator(value)
            value = value.strip()

            if not value.startswith("data:"):
                raise ValueError("Data URI must start with 'data:'")

            if len(value) > MAX_DATA_URI_LENGTH:
                raise ValueError(
                    f"Data URI is too long. Max length is {MAX_DATA_URI_LENGTH} bytes."
                )

            return cls(value)

        @classmethod
        def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
            update_not_none(field_schema, format="data-uri")


class HttpsUrl(DownloadFileMixin, str):
    if IS_PYDANTIC_V2:

        @classmethod
        def __get_pydantic_core_schema__(cls, source_type: Any, handler) -> Any:
            return {
                "type": "str",
                "pattern": HTTP_URL_REGEX,
                "max_length": MAX_HTTPS_URL_LENGTH,
                "strip_whitespace": True,
            }

        def __get_pydantic_json_schema__(cls, core_schema, handler) -> Dict[str, Any]:
            json_schema = handler(core_schema)
            json_schema.update(format="https-url")
            return json_schema

    else:

        @classmethod
        def __get_validators__(cls):
            yield cls.validate

        @classmethod
        def validate(cls, value: Any) -> "HttpsUrl":
            from pydantic.validators import str_validator

            value = str_validator(value)
            value = value.strip()

            if not re.match(HTTP_URL_REGEX, value):
                raise ValueError(
                    "URL must start with 'https://' and follow the correct format."
                )

            if len(value) > MAX_HTTPS_URL_LENGTH:
                raise ValueError(
                    f"HTTPS URL is too long. Max length is "
                    f"{MAX_HTTPS_URL_LENGTH} characters."
                )

            return cls(value)

        @classmethod
        def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
            update_not_none(field_schema, format="https-url")


class ImageHttpsUrl(DownloadImageMixin, HttpsUrl):
    pass


class ImageDataUri(DownloadImageMixin, DataUri):
    pass


FileInput = Union[HttpsUrl, DataUri]
ImageInput = Union[ImageHttpsUrl, ImageDataUri]
