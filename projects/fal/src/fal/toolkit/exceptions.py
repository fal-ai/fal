from collections.abc import Sequence
from typing import Any, Union

from fastapi import HTTPException
from pydantic import BaseModel

NumericValue = int | float
ScalarValue = Union[NumericValue, str]
SequenceValue = Sequence
ConstraintValue = Union[ScalarValue, SequenceValue]

# we should not really have None values in the input
# and we should fix those cases where possible
InputType = Union[ScalarValue, SequenceValue, dict, None, BaseModel]

ERROR_URL = "https://docs.fal.ai/model-apis/errors"


class FalTookitException(Exception):
    """Base exception for all toolkit exceptions"""

    pass


class FileUploadException(FalTookitException):
    """Raised when file upload fails"""

    pass


class KVStoreException(FalTookitException):
    """Raised when KV store operation fails"""

    pass


class ToolkitFileSizeExceededException(FalTookitException):
    """Error raised when an image file size exceeds the maximum allowed size."""

    pass


class ToolkitFileDownloadException(FalTookitException):
    """Error raised when a file download fails."""

    pass


# picked ToolkitDataFormatError naming because
# - "data" covers any type of content (not just files)
# - "format" is broad enough to cover encoding, structure, corruption
# - its clear this is about data format issues, not transport/network issues
class ToolkitDataFormatException(FalTookitException):
    """Error raised when data format parsing fails (e.g.,
    invalid base64, malformed data URLs)."""

    pass


class ToolkitImageLoadException(FalTookitException):
    """Error raised when an image load fails."""

    pass


def _get_input_from_model(
    model: BaseModel | None, location: tuple[str | int, ...]
) -> Any:
    if model is None:
        return None

    value: dict[str | int, Any] = {"body": model.dict()}
    for key in location:
        value = value[key]
    return value


class ErrorDetail:
    msg: str
    type: str
    input: Any | None
    model: BaseModel | None
    loc: tuple[str | int, ...]
    ctx: dict[str, Any] | None

    def __init__(
        self,
        msg: str,
        type: str,
        input: str | None = None,
        model: BaseModel | None = None,
        location: tuple[str | int, ...] = (),
        ctx: dict[str, Any] | None = None,
    ):
        self.loc = ("body", *location)

        # if input is not provided, we use the model to get the input
        self.input = input or _get_input_from_model(model, self.loc)

        self.msg = msg
        self.type = type
        self.ctx = ctx

    def as_dict(self) -> dict[str, Any]:
        return {
            "msg": self.msg,
            "type": self.type,
            "input": self.input,
            "loc": self.loc,
            "ctx": self.ctx,
        }


class ToolkitHTTPException(HTTPException, FalTookitException):
    def __init__(
        self,
        status_code: int,
        errors: tuple[ErrorDetail, ...] = (),
        retryable: bool = False,
        billing_units: int | None = None,
    ):
        headers = {"x-fal-retryable": "true" if retryable else "false"}

        if billing_units is not None:
            headers["x-fal-billable-units"] = str(billing_units)

        HTTPException.__init__(
            self,
            status_code=status_code,
            detail=[error.as_dict() for error in errors],
            headers=headers,
        )


class InternalServerException(ToolkitHTTPException):
    def __init__(
        self,
        *,
        input: str | None = None,
        model: BaseModel | None = None,
        retryable: bool = False,
    ):
        error = ErrorDetail(
            input=input,
            model=model,
            msg="Internal server error",
            type="internal_server_error",
        )
        errors = (error,)

        super().__init__(
            status_code=504, errors=errors, retryable=retryable, billing_units=0
        )


class GenerationTimeoutException(ToolkitHTTPException):
    def __init__(
        self,
        *,
        input: str | None = None,
        model: BaseModel | None = None,
        retryable: bool = False,
    ):
        error = ErrorDetail(
            input=input,
            model=model,
            msg="Generation timeout",
            type="generation_timeout",
        )

        errors = (error,)

        # no need for billing since it is already 5XX
        super().__init__(
            status_code=504, errors=errors, retryable=retryable, billing_units=0
        )


class DownstreamServiceException(ToolkitHTTPException):
    # by default the billing units are 0
    # because we do not want to charge for downstream service errors
    def __init__(
        self,
        *,
        input: str | None = None,
        model: BaseModel | None = None,
        msg: str | None = None,
        exception: Exception,
        retryable: bool = False,
        billing_units: int = 0,
    ):
        error = ErrorDetail(
            input=input,
            model=model,
            msg="Downstream service error",
            type="downstream_service_error",
        )

        super().__init__(
            status_code=504,
            errors=(error,),
            retryable=retryable,
            billing_units=billing_units,
        )


class DownstreamServiceUnavailableException(ToolkitHTTPException):
    def __init__(
        self,
        *,
        exception: Exception,
        input: str | None = None,
        model: BaseModel | None = None,
        retryable: bool = False,
    ):
        error = ErrorDetail(
            input=input,
            model=model,
            msg="Downstream service unavailable",
            type="downstream_service_unavailable",
        )

        super().__init__(
            status_code=504, errors=(error,), retryable=retryable, billing_units=0
        )


class ImageTooSmallException(ToolkitHTTPException):
    def __init__(
        self,
        *,
        input: str | None = None,
        model: BaseModel | None = None,
        location: tuple[str | int, ...] = (),
        min_height: int,
        min_width: int,
        msg: str | None = None,
        billing_units: int | None = None,
    ):
        error = ErrorDetail(
            input=input,
            model=model,
            location=location,
            msg=(
                f"Image dimensions are too small. Minimum dimensions required: "
                f"{min_width}x{min_height} pixels."
            ),
            type="image_too_small",
            ctx={"min_height": min_height, "min_width": min_width},
        )
        super().__init__(status_code=422, errors=(error,), billing_units=billing_units)


class ImageTooLargeException(ToolkitHTTPException):
    def __init__(
        self,
        *,
        input: str | None = None,
        model: BaseModel | None = None,
        location: tuple[str | int, ...] = (),
        max_height: int | None = None,
        max_width: int | None = None,
        billing_units: int | None = None,
        max_resolution: int | None = None,
    ):
        if max_resolution is None and (max_height is None or max_width is None):
            raise ValueError(
                "Either 'max_resolution' must be provided, "
                "or both 'max_height' and 'max_width' must be provided"
            )

        if max_resolution is not None:
            error = ErrorDetail(
                input=input,
                model=model,
                location=location,
                msg="Image too large",
                type="image_too_large",
                ctx={"max_resolution": max_resolution},
            )
        else:
            error = ErrorDetail(
                input=input,
                model=model,
                location=location,
                msg="Image too large",
                type="image_too_large",
                ctx={"max_height": max_height, "max_width": max_width},
            )
        super().__init__(status_code=422, errors=(error,), billing_units=billing_units)


class ImageAspectRatioException(ToolkitHTTPException):
    def __init__(
        self,
        *,
        input: str | None = None,
        model: BaseModel | None = None,
        location: tuple[str | int, ...] = (),
        min_aspect_ratio: float | None = None,
        max_aspect_ratio: float | None = None,
        billing_units: int | None = None,
    ):
        if min_aspect_ratio is None and max_aspect_ratio is None:
            raise ValueError(
                "At least one of 'min_aspect_ratio' or 'max_aspect_ratio' must be "
                "provided."
            )

        if min_aspect_ratio is not None and max_aspect_ratio is not None:
            msg = (
                f"The aspect ratio of the image should be between {min_aspect_ratio} "
                f"and {max_aspect_ratio}."
            )
        elif min_aspect_ratio is not None:
            msg = (
                "The aspect ratio of the image should be greater "
                f"than {min_aspect_ratio}."
            )
        elif max_aspect_ratio is not None:
            msg = (
                f"The aspect ratio of the image should be less than {max_aspect_ratio}."
            )

        error = ErrorDetail(
            input=input,
            model=model,
            location=location,
            msg=msg,
            type="image_aspect_ratio_error",
            ctx={
                "min_aspect_ratio": min_aspect_ratio,
                "max_aspect_ratio": max_aspect_ratio,
            },
        )
        super().__init__(status_code=422, errors=(error,), billing_units=billing_units)


class ImageLoadException(ToolkitHTTPException):
    def __init__(
        self,
        *,
        input: str | None = None,
        model: BaseModel | None = None,
        location: tuple[str | int, ...] = (),
        billing_units: int | None = None,
    ):
        error = ErrorDetail(
            input=input,
            model=model,
            location=location,
            msg=(
                "Failed to load the image. Please ensure the image "
                "file is not corrupted and is in a supported format."
            ),
            type="image_load_error",
        )
        super().__init__(status_code=422, errors=(error,), billing_units=billing_units)


class FileDownloadException(ToolkitHTTPException):
    def __init__(
        self,
        *,
        input: str | None = None,
        model: BaseModel | None = None,
        location: tuple[str | int, ...] = (),
        billing_units: int | None = None,
    ):
        error = ErrorDetail(
            input=input,
            model=model,
            location=location,
            msg=(
                "Failed to download the file. Please check if the URL "
                "is accessible and try again."
            ),
            type="file_download_error",
        )
        super().__init__(status_code=422, errors=(error,), billing_units=billing_units)


class FileTooLargeException(ToolkitHTTPException):
    def __init__(
        self,
        *,
        input: str | None = None,
        model: BaseModel | None = None,
        location: tuple[str | int, ...] = (),
        max_size: int,
        msg: str | None = None,
        billing_units: int | None = None,
    ):
        error = ErrorDetail(
            input=input,
            model=model,
            location=location,
            msg=(
                f"File size exceeds the maximum allowed size of {max_size} bytes. "
                "Please upload a smaller file."
            ),
            type="file_too_large",
            ctx={"max_size": max_size},
        )
        super().__init__(status_code=422, errors=(error,), billing_units=billing_units)
