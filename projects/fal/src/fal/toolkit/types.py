import re
from typing import Any, Dict, Union

from pydantic.utils import update_not_none
from pydantic.validators import str_validator

MAX_DATA_URI_LENGTH = 10 * 1024 * 1024
MAX_HTTPS_URL_LENGTH = 2048


class DataUri(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value: Any) -> "DataUri":
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


class HttpsUrl(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value: Any) -> "HttpsUrl":
        value = str_validator(value)
        value = value.strip()

        # Regular expression for validating HTTPS URL format
        https_url_regex = (
            r"^https:\/\/(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?::\d{1,5})?(?:\/[^\s]*)?$"
        )

        if not re.match(https_url_regex, value):
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


FileInput = Union[HttpsUrl, DataUri]
ImageInput = Union[HttpsUrl, DataUri]
