from __future__ import annotations

from dataclasses import dataclass


class FalServerlessException(Exception):
    """Base exception type for fal Serverless related flows and APIs."""

    pass


@dataclass
class AppException(FalServerlessException):
    """
    Base exception class for application-specific errors.

    Attributes:
        message: A descriptive message explaining the error.
        status_code: The HTTP status code associated with the error.
    """

    message: str
    status_code: int


@dataclass
class FieldException(FalServerlessException):
    """Exception raised for errors related to specific fields.

    Attributes:
        field: The field that caused the error.
        message: A descriptive message explaining the error.
        status_code: The HTTP status code associated with the error. Defaults to 422
        type: The type of error. Defaults to "value_error"
    """

    field: str
    message: str
    status_code: int = 422
    type: str = "value_error"

    def to_pydantic_format(self) -> dict[str, list[dict]]:
        return dict(
            detail=[
                {
                    "loc": ["body", self.field],
                    "msg": self.message,
                    "type": self.type,
                }
            ]
        )


@dataclass
class RequestCancelledException(FalServerlessException):
    """Exception raised when the request is cancelled by the client."""

    message: str = "Request cancelled by the client."
