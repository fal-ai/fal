import asyncio
import base64
import dataclasses
import random
import re
import time
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypedDict, cast, overload

import pydantic
from pydantic import BaseModel
from typing_extensions import Unpack

if not hasattr(pydantic, "__version__") or pydantic.__version__.startswith("1."):
    _IS_PYDANTIC_V2 = False
else:
    from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler, model_validator
    from pydantic.json_schema import JsonSchemaValue
    from pydantic_core import CoreSchema, core_schema as pydantic_core_schema

    _IS_PYDANTIC_V2 = True

from fal.toolkit.exceptions import (
    FileDownloadException,
    FileTooLargeException,
    ImageAspectRatioException,
    ImageLoadException,
    ImageTooLargeException,
    ImageTooSmallException,
    ToolkitDataFormatException,
    ToolkitFileDownloadException,
    ToolkitFileSizeExceededException,
    ToolkitImageLoadException,
)

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


SUPPORTED_ARCHIVE_EXTENSIONS = [".zip"]
SUPPORTED_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]
SUPPORTED_VIDEO_EXTENSIONS = [".mp4", ".mov"]
SUPPORTED_AUDIO_EXTENSIONS = [".wav", ".mp3"]

# defined in https://github.com/fal-ai/fal/blob/eaaded2c850186405cb547f081926a4fb44fa224/projects/fal/src/fal/toolkit/image/image.py#L28-L34
MAX_SUPPORTED_IMAGE_RESOLUTION = 14142

MAX_IMAGE_FILE_SIZE = 20 * 1024 * 1024  # 20MB

MAX_SEED = 2**31 - 1


HTTPS_URL_REGEX = re.compile(
    r"^https?://"  # Must start with http:// or https://
    r"(www\.)?"  # Optional www.
    r"[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?"  # First domain label
    r"(\.[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?)*"  # Additional domain labels
    r"\.[a-zA-Z]{2,24}"  # TLD
    r"(:([1-9]|[1-9][0-9]|[1-9][0-9]{2}|[1-9][0-9]{3}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5]))?"  # Optional valid port number (1-65535)  # noqa: E501
    r"\b"  # Word boundary after TLD/port
    r"([\/?#][^\s]*)?"  # Optional path, query, or fragment
    r"$"
)

DATA_URI_REGEX = re.compile(
    r"^data:(?:[a-zA-Z0-9-+.]+/[a-zA-Z0-9-+.]+)?(?:;base64)?,.*$"
)


def get_retry_delay(
    num_retry: int,
    base_delay: float,
    max_delay: float,
    backoff_type: Literal["exponential", "fixed"] = "exponential",
    jitter: bool = False,
) -> float:
    if backoff_type == "exponential":
        delay = min(base_delay * (2 ** (num_retry - 1)), max_delay)
    else:
        delay = min(base_delay, max_delay)

    if jitter:
        delay *= random.uniform(0.5, 1.5)

    return min(delay, max_delay)


@lru_cache(maxsize=1)
def _pillow_has_avif_support() -> bool:
    import PIL

    major, minor, *_ = PIL.__version__.split(".")
    major = int(major)
    minor = int(minor)
    return (major, minor) >= (11, 2)


@lru_cache(maxsize=1)
def _register_heif_opener():
    from PIL.Image import MIME

    try:
        from pillow_heif import register_heif_opener
    except ModuleNotFoundError:
        print("HEIF: pillow-heif not installed.")
    except ImportError:
        print("HEIF: pillow-heif import failed.")
    else:
        if getattr(MIME, "HEIF", None) is None:
            register_heif_opener()


@lru_cache(maxsize=1)
def _register_avif_opener():
    from PIL.Image import MIME

    try:
        from pillow_heif import register_avif_opener
    except ModuleNotFoundError:
        print("AVIF: pillow-heif not installed.")
    except ImportError:
        print("AVIF: pillow-heif import failed.")
    else:
        if getattr(MIME, "AVIF", None) is None:
            register_avif_opener()


@lru_cache(maxsize=1)
def _register_custom_pil_openers():
    _register_heif_opener()
    if not _pillow_has_avif_support():
        _register_avif_opener()


@overload
async def download_file_async(
    url: str,
    output_path: None = None,
    timeout: int | float = 60,
    max_retries: int = 3,
    max_size: int | None = None,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    headers: dict | None = None,
    follow_redirects: bool = True,
) -> bytes: ...


@overload
async def download_file_async(
    url: str,
    output_path: Path | str,
    timeout: int | float = 60,
    max_retries: int = 3,
    max_size: int | None = None,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    headers: dict | None = None,
    follow_redirects: bool = True,
) -> Path: ...


async def download_file_async(
    url: str,
    output_path: Path | str | None = None,
    timeout: int | float = 60,
    max_retries: int = 3,
    max_size: int | None = None,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    headers: dict | None = None,
    follow_redirects: bool = True,
) -> bytes | Path | str:
    """
    Download a file asynchronously from a URL.

    Args:
        url: The URL to download the file from
        output_path: Optional path to save the file to. If None,
            returns the file content as bytes
        timeout: Timeout for the request in seconds
        max_retries: Maximum number of retries for failed requests
        max_size: Maximum allowed file size in bytes. None means no limit
        base_delay: Base delay for retry attempts
        max_delay: Maximum delay for retry attempts
        headers: Optional additional headers for the request
        follow_redirects: Whether to follow redirects

    Returns:
        Path object pointing to the downloaded file if output_path is provided,
        otherwise returns the file content as bytes
    """
    import httpx

    if url.startswith("data:"):
        content_str = url.split(",")[1]
        try:
            content = base64.b64decode(content_str)
        except Exception as e:
            raise ToolkitDataFormatException(
                f"Error decoding base64 data: {e.__class__.__name__} {str(e)}"
            )

        if max_size and len(content) > max_size:
            raise ToolkitFileSizeExceededException(
                f"File size ({len(content)} bytes) exceeds maximum allowed size "
                f"({max_size} bytes)"
            )

        if output_path:
            output_path = Path(output_path)
            # Create parent directories if they don't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save content to file
            with open(output_path, "wb") as f:
                f.write(content)
            return output_path

        return content

    if headers is None:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:21.0) "
                "Gecko/20100101 Firefox/21.0"
            ),
        }

    for num_retry in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(
                    url, headers=headers, follow_redirects=follow_redirects
                )
                response.raise_for_status()
                break

        except httpx.HTTPError as e:
            if num_retry < max_retries:
                delay = get_retry_delay(
                    num_retry,
                    base_delay,
                    max_delay,
                    backoff_type="exponential",
                    jitter=True,
                )

                if isinstance(e, httpx.HTTPStatusError):
                    status_code = e.response.status_code
                    # Only retry on 429 (too many requests) or 5xx (server errors)
                    if status_code != 429 and not (500 <= status_code < 600):
                        raise ToolkitFileDownloadException(
                            f"Error downloading file from {url}"
                        ) from e

                await asyncio.sleep(delay)
                continue

            raise ToolkitFileDownloadException(
                f"Error downloading file from {url}"
            ) from e

        except httpx.InvalidURL as e:
            raise ToolkitFileDownloadException(
                f"Error downloading file from {url}, URL too long"
            ) from e

    # Check file size before processing
    content = response.content
    if max_size and len(content) > max_size:
        raise ToolkitFileSizeExceededException(
            f"File size ({len(content)} bytes) exceeds maximum allowed size "
            f"({max_size} bytes)"
        )

    if output_path:
        output_path = Path(output_path)
        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save content to file
        with open(output_path, "wb") as f:
            f.write(content)
        return output_path

    # Return content as bytes
    return content


@overload
def download_file(
    url: str,
    output_path: None = None,
    timeout: int | float = 60,
    max_retries: int = 3,
    max_size: int | None = None,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    headers: dict | None = None,
    follow_redirects: bool = True,
) -> bytes: ...


@overload
def download_file(
    url: str,
    output_path: Path | str,
    timeout: int | float = 60,
    max_retries: int = 3,
    max_size: int | None = None,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    headers: dict | None = None,
    follow_redirects: bool = True,
) -> Path: ...


def download_file(
    url: str,
    output_path: Path | str | None = None,
    timeout: int | float = 60,
    max_retries: int = 3,
    max_size: int | None = None,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    headers: dict | None = None,
    follow_redirects: bool = True,
) -> bytes | Path | str:
    """
    Download a file synchronously from a URL.

    Args:
        url: The URL to download the file from
        output_path: Optional path to save the file to. If None,
            returns the file content as bytes
        timeout: Timeout for the request in seconds
        max_retries: Maximum number of retries for failed requests
        max_size: Maximum allowed file size in bytes. None means no limit
        base_delay: Base delay for retry attempts
        max_delay: Maximum delay for retry attempts
        headers: Optional additional headers for the request
        follow_redirects: Whether to follow redirects

    Returns:
        Path object pointing to the downloaded file if output_path is provided,
        otherwise returns the file content as bytes
    """
    import httpx

    if url.startswith("data:"):
        content_str = url.split(",")[1]
        try:
            content = base64.b64decode(content_str)
        except Exception as e:
            raise ToolkitDataFormatException(
                f"Error decoding base64 data: {e.__class__.__name__} {str(e)}"
            )

        if max_size and len(content) > max_size:
            raise ToolkitFileSizeExceededException(
                f"File size ({len(content)} bytes) exceeds maximum allowed size "
                f"({max_size} bytes)"
            )

        if output_path:
            output_path = Path(output_path)
            # Create parent directories if they don't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save content to file
            with open(output_path, "wb") as f:
                f.write(content)
            return output_path

        return content

    if headers is None:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:21.0) "
                "Gecko/20100101 Firefox/21.0"
            ),
        }

    for num_retry in range(max_retries + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.get(
                    url, headers=headers, follow_redirects=follow_redirects
                )
                response.raise_for_status()
                break

        except httpx.HTTPError as e:
            if num_retry < max_retries:
                delay = get_retry_delay(
                    num_retry,
                    base_delay,
                    max_delay,
                    backoff_type="exponential",
                    jitter=True,
                )

                if isinstance(e, httpx.HTTPStatusError):
                    status_code = e.response.status_code
                    # Only retry on 429 (too many requests) or 5xx (server errors)
                    if status_code != 429 and not (500 <= status_code < 600):
                        raise ToolkitFileDownloadException(
                            f"Error downloading file from {url}: "
                            f"{e.__class__.__name__} {str(e)}"
                        )

                time.sleep(delay)

            raise ToolkitFileDownloadException(
                f"Error downloading file from {url}: "
                f"{e.__class__.__name__} {str(e)}"
            )

        except httpx.InvalidURL as e:
            raise ToolkitFileDownloadException(
                f"Error downloading file from {url}, URL too "
                f"long: {e.__class__.__name__} {str(e)}"
            )

    # Check file size before processing
    content = response.content
    if max_size and len(content) > max_size:
        raise ToolkitFileSizeExceededException(
            f"File size ({len(content)} bytes) exceeds maximum allowed size "
            f"({max_size} bytes)"
        )

    if output_path:
        output_path = Path(output_path)
        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save content to file
        with open(output_path, "wb") as f:
            f.write(content)

        return output_path

    # Return content as bytes
    return content


def preprocess_image(
    image_pil, convert_to_rgb=True, fix_orientation=True
) -> "PILImage":
    from PIL import ImageOps, ImageSequence

    # For MPO (multi picture object) format images, we only need the first image
    images: list["PILImage"] = []
    for image in ImageSequence.Iterator(image_pil):
        img = image

        if convert_to_rgb:
            img = img.convert("RGB")

        if fix_orientation:
            try:
                transposed = ImageOps.exif_transpose(img, in_place=False)
                if transposed is not None:
                    img = transposed
            except Exception:
                pass

        images.append(img)

        break

    return images[0]


async def read_image_from_url_async(
    url: str,
    convert_to_rgb: bool = True,
    fix_orientation: bool = True,
    timeout: int | float = 60,
    max_retries: int = 3,
    max_size: int | None = MAX_IMAGE_FILE_SIZE,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
):
    from PIL import Image as PILImage

    _register_custom_pil_openers()

    loop = asyncio.get_running_loop()

    response = await download_file_async(
        url,
        timeout=timeout,
        max_retries=max_retries,
        max_size=max_size,
        base_delay=base_delay,
        max_delay=max_delay,
    )

    response_io = BytesIO(response)

    try:
        image_pil = PILImage.open(response_io)
        image_pil = await loop.run_in_executor(
            None, preprocess_image, image_pil, convert_to_rgb, fix_orientation
        )
    except Exception as e:
        raise ToolkitImageLoadException(
            f"Error loading image from {url}: {e.__class__.__name__} {str(e)}"
        )

    return image_pil


def read_image_from_url(
    url: str,
    convert_to_rgb: bool = True,
    fix_orientation: bool = True,
    timeout: int | float = 60,
    max_retries: int = 3,
    max_size: int | None = MAX_IMAGE_FILE_SIZE,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
):
    """
    Read an image from a URL synchronously.

    Args:
        url: The URL to download the image from
        convert_to_rgb: Whether to convert the image to RGB mode
        fix_orientation: Whether to fix the image orientation based on EXIF data
        timeout: Timeout for the request in seconds
        max_retries: Maximum number of retries for failed requests
        max_size: Maximum allowed file size in bytes. None means no limit
        base_delay: Base delay for retry attempts
        max_delay: Maximum delay for retry attempts

    Returns:
        PIL Image object
    """
    from PIL import Image as PILImage

    _register_custom_pil_openers()

    image_bytes = download_file(
        url,
        timeout=timeout,
        max_retries=max_retries,
        max_size=max_size,
        base_delay=base_delay,
        max_delay=max_delay,
    )
    # response is now bytes
    response_io = BytesIO(image_bytes)
    try:
        image_pil = PILImage.open(response_io)
        image_pil = preprocess_image(image_pil, convert_to_rgb, fix_orientation)
    except Exception as e:
        raise ToolkitImageLoadException(
            f"Error loading image from {url}: {e.__class__.__name__} {str(e)}"
        )
    return image_pil


@overload
def encode_image(
    image: "PILImage",
    format: Literal["png", "jpeg"] = ...,
    quality: int | None = ...,
    *,
    return_type: Literal["base64"] = "base64",
) -> str: ...


@overload
def encode_image(
    image: "PILImage",
    format: Literal["png", "jpeg"] = ...,
    quality: int | None = ...,
    *,
    return_type: Literal["bytes"],
) -> bytes: ...


@overload
def encode_image(
    image: "PILImage",
    format: Literal["png", "jpeg"] = ...,
    quality: int | None = ...,
    *,
    return_type: Literal["data_uri"],
) -> str: ...


def encode_image(
    image: "PILImage",
    format: Literal["png", "jpeg"] = "jpeg",
    quality: int | None = None,
    return_type: Literal["bytes", "base64", "data_uri"] = "base64",
) -> str | bytes:
    buffer = BytesIO()

    save_kwargs: dict[str, Any] = {"format": format}
    if format.lower() == "jpeg" and quality is not None:
        save_kwargs["quality"] = quality

    image.save(buffer, **save_kwargs)
    b64_image = base64.b64encode(buffer.getvalue()).decode()

    if return_type == "bytes":
        return buffer.getvalue()
    elif return_type == "base64":
        return b64_image
    return f"data:image/{format.lower()};base64,{b64_image}"


def _bind_context_recursively(data_node: Any, path_prefix: tuple[str | int, ...]):
    if isinstance(data_node, HttpsOrDataUrl):
        data_node._bind_context(loc=path_prefix)
    elif isinstance(data_node, BaseModel):
        model_cls = type(data_node)
        fields = model_cls.model_fields if _IS_PYDANTIC_V2 else model_cls.__fields__
        for field_name in fields:
            field_value = getattr(data_node, field_name, None)
            if field_value is not None:
                _bind_context_recursively(field_value, path_prefix + (field_name,))
    elif isinstance(data_node, list):
        for i, item in enumerate(data_node):
            _bind_context_recursively(item, path_prefix + (i,))


class FileValidationOptions(TypedDict, total=False):
    """Validation options for file processing."""

    max_file_size: int | None
    timeout: float | None


@dataclasses.dataclass(frozen=True)
class FileValidationConfig:
    """Validation options for file processing."""

    max_file_size: int | None = None
    timeout: float | None = None


class ImageValidationOptions(TypedDict, total=False):
    """Validation options for image processing."""

    max_file_size: int | None
    min_width: int | None
    min_height: int | None
    max_width: int | None
    max_height: int | None
    min_aspect_ratio: float | None
    max_aspect_ratio: float | None
    timeout: float | None


@dataclasses.dataclass(frozen=True)
class ImageValidationConfig:
    """A structured, type-safe configuration for image validation parameters."""

    max_file_size: int | None = None
    min_width: int | None = None
    min_height: int | None = None
    max_width: int | None = None
    max_height: int | None = None
    min_aspect_ratio: float | None = None
    max_aspect_ratio: float | None = None
    timeout: float = 20.0

    def __post_init__(self):
        min_provided = self.min_aspect_ratio is not None
        max_provided = self.max_aspect_ratio is not None

        # Both min_aspect_ratio and max_aspect_ratio must be provided together
        # or neither should be provided. Having only one doesn't make sense.
        #
        # This is because aspect ratio can be calculated as either width/height
        # or height/width, and with a single bound, it's impossible to create
        # a meaningful constraint that works for both orientations.
        #
        # For example, if we only have max_aspect_ratio=0.5:
        # - A portrait image (w=200, h=400) has ratio h/w = 0.5 - PASS
        # - A landscape image (w=400, h=200) has ratio w/h = 2.0 - FAIL
        #
        # The issue is that with a single bound, it's impossible to satisfy both
        # orientations.
        # Only with both bounds can we create a proper constraint that works
        # correctly for both landscape and portrait orientations.

        if min_provided != max_provided:
            raise ValueError(
                "Both min_aspect_ratio and max_aspect_ratio must be provided together, "
                "or neither should be provided. Having only one doesn't make sense."
            )


class HttpsOrDataUrl(str):
    """
    A custom Pydantic type for a string that is either a valid HTTPS URL
    or a Data URI. Inherits from `str` and uses `__get_validators__`.
    """

    _loc: tuple[str | int, ...] = ()
    _fal_ui_field_name = "url"

    _file_validation_config: ClassVar[FileValidationConfig] = FileValidationConfig()

    def _bind_context(self, loc: tuple[str | int, ...]) -> "HttpsOrDataUrl":
        """Toolkit method to store the name of the field this instance belongs to."""
        self._loc = loc
        return self

    @classmethod
    def validate(cls, v: Any) -> "HttpsOrDataUrl":
        if not isinstance(v, str):
            raise TypeError("string required")

        value = v.strip()

        if HTTPS_URL_REGEX.match(value) or DATA_URI_REGEX.match(value):
            return cast("HttpsOrDataUrl", cls(value))

        raise ValueError("Input must be a valid HTTPS URL or a Data URI")

    @classmethod
    def _modify_fal_ui_schema(cls, field_schema: dict, field, *args, **kwargs):
        field_info = getattr(field, "field_info", None)
        field_extras = getattr(field_info, "extra", {}) or {}

        # pydantic is doing multiple passes over the schema,
        # and we dont want to make any modifications on field schema
        # that has the type key
        if "type" in field_schema:
            return

        # we just want to update the ui bucket, not the whole field schema
        ui_bucket: dict[str, Any] = field_schema.setdefault("ui", {})
        ui_bucket.setdefault("field", cls._fal_ui_field_name)

        field_extras.update({"ui": ui_bucket})
        field_schema.update(field_extras)

    @classmethod
    def _inject_validation_config(cls, field_schema: dict):
        """Toolkit method to add validation configuration."""
        pass

    if _IS_PYDANTIC_V2:

        @classmethod
        def __get_pydantic_core_schema__(
            cls, source_type: Any, handler: GetCoreSchemaHandler
        ) -> "CoreSchema":
            return pydantic_core_schema.no_info_after_validator_function(
                cls.validate,
                handler(str),
            )

        @classmethod
        def __get_pydantic_json_schema__(
            cls, _core_schema: "CoreSchema", handler: GetJsonSchemaHandler
        ) -> "JsonSchemaValue":
            json_schema = handler(_core_schema)
            json_schema.update({"type": "string"})
            ui_bucket: dict[str, Any] = json_schema.setdefault("ui", {})
            ui_bucket.setdefault("field", cls._fal_ui_field_name)
            cls._inject_validation_config(json_schema)
            return json_schema

    else:

        @classmethod
        def __get_validators__(cls):
            yield cls.validate

        @classmethod
        def __modify_schema__(cls, field_schema: dict, field):
            cls._modify_fal_ui_schema(field_schema, field)
            cls._inject_validation_config(field_schema)

    def _handle_generic_download_Exception(
        self, exc: Exception, max_size: int | None
    ) -> None:
        if isinstance(exc, ToolkitFileDownloadException):
            raise FileDownloadException(
                input=self,
                location=self._loc,
                billable_units=0,
            )
        elif isinstance(exc, ToolkitFileSizeExceededException):
            raise FileTooLargeException(
                input=self,
                location=self._loc,
                max_size=max_size or 0,
                billable_units=0,
            )
        elif isinstance(exc, ToolkitDataFormatException):
            raise FileDownloadException(
                input=self,
                location=self._loc,
                billable_units=0,
            )

        raise exc

    def to_bytes(
        self, *, timeout: float | int = 20, max_file_size: int | None = None
    ) -> bytes:
        try:
            content = download_file(
                str(self),
                timeout=timeout,
                max_size=max_file_size,
            )
        except Exception as exc:
            self._handle_generic_download_Exception(exc, max_file_size)
        return content

    async def to_bytes_async(
        self, *, timeout: float | int = 20, max_file_size: int | None = None
    ) -> bytes:
        try:
            content = await download_file_async(
                str(self),
                timeout=timeout,
                max_size=max_file_size,
            )
        except Exception as exc:
            self._handle_generic_download_Exception(exc, max_file_size)
        return content

    def to_base64(
        self, *, timeout: float | int = 20, max_file_size: int | None = None
    ) -> str:
        # Fast-path for data URIs
        value = str(self)
        if value.startswith("data:"):
            try:
                return value.split(",", 1)[1]
            except Exception:
                # Fall back to decoding path to surface proper errors
                pass

        content = self.to_bytes(timeout=timeout, max_file_size=max_file_size)
        return base64.b64encode(content).decode("utf-8")

    async def to_base64_async(
        self, *, timeout: float | int = 20, max_file_size: int | None = None
    ) -> str:
        value = str(self)
        if value.startswith("data:"):
            try:
                return value.split(",", 1)[1]
            except Exception:
                pass

        content = await self.to_bytes_async(
            timeout=timeout, max_file_size=max_file_size
        )
        return base64.b64encode(content).decode("utf-8")

    def to_data_uri(
        self,
        *,
        timeout: float | int = 20,
        max_file_size: int | None = None,
        content_type: str = "application/octet-stream",
    ) -> str:
        value = str(self)
        if value.startswith("data:"):
            return value

        b64 = self.to_base64(timeout=timeout, max_file_size=max_file_size)
        return f"data:{content_type};base64,{b64}"

    async def to_data_uri_async(
        self,
        *,
        timeout: float | int = 20,
        max_file_size: int | None = None,
        content_type: str = "application/octet-stream",
    ) -> str:
        value = str(self)
        if value.startswith("data:"):
            return value

        b64 = await self.to_base64_async(timeout=timeout, max_file_size=max_file_size)
        return f"data:{content_type};base64,{b64}"


class ImageUrl(HttpsOrDataUrl):
    """General purpose image URL"""

    _image_validation_config: ClassVar[ImageValidationConfig] = ImageValidationConfig()
    _fal_ui_field_name = "image"

    def __new__(cls, value: str, loc: tuple[str | int, ...] = ()):
        instance = super().__new__(cls, value)
        url = cast("ImageUrl", instance)
        url._loc = loc
        return url

    @classmethod
    def _inject_validation_config(cls, field_schema: dict):
        config_params = {}
        validation_keys: tuple[str, ...] = tuple(
            ImageValidationConfig.__dataclass_fields__.keys()
        )

        for key in validation_keys:
            if key in field_schema:
                config_params[key] = field_schema.pop(key)

        # Process and add config if:
        # 1. We found field parameters, OR
        # 2. The class has a custom default config
        base_default = ImageUrl._image_validation_config
        has_custom_default = cls._image_validation_config is not base_default

        if config_params or has_custom_default:
            if config_params:
                # Merge field parameters with class default config
                default_dict = dataclasses.asdict(cls._image_validation_config)
                default_dict.update(config_params)
                field_config = ImageValidationConfig(**default_dict)
            else:
                # Use the custom default config as-is
                field_config = cls._image_validation_config

            # Add config limits to schema description
            cls._add_config_to_schema(field_schema, field_config)

    @classmethod
    def _add_config_to_schema(cls, field_schema: dict, config: ImageValidationConfig):
        """Add configuration limits to field schema description."""
        limits = []
        fal_config: dict[str, int | float] = {}

        if config.max_file_size is not None:
            mb_size = config.max_file_size / (1024 * 1024)
            limits.append(f"Max file size: {mb_size:.1f}MB")
            fal_config["max_file_size"] = config.max_file_size

        if config.min_width is not None:
            limits.append(f"Min width: {config.min_width}px")
            fal_config["min_width"] = config.min_width

        if config.min_height is not None:
            limits.append(f"Min height: {config.min_height}px")
            fal_config["min_height"] = config.min_height

        if config.max_width is not None:
            limits.append(f"Max width: {config.max_width}px")
            fal_config["max_width"] = config.max_width

        if config.max_height is not None:
            limits.append(f"Max height: {config.max_height}px")
            fal_config["max_height"] = config.max_height

        if config.min_aspect_ratio is not None:
            limits.append(f"Min aspect ratio: {config.min_aspect_ratio:.2f}")
            fal_config["min_aspect_ratio"] = config.min_aspect_ratio

        if config.max_aspect_ratio is not None:
            limits.append(f"Max aspect ratio: {config.max_aspect_ratio:.2f}")
            fal_config["max_aspect_ratio"] = config.max_aspect_ratio

        if config.timeout is not None:
            limits.append(f"Timeout: {config.timeout}s")
            fal_config["timeout"] = config.timeout

        if limits:
            limit_desc = ", ".join(limits)
            field_schema["limit_description"] = f"{limit_desc}"

        if fal_config:
            field_schema["x-fal"] = fal_config

    def _get_effective_config(self, **overrides) -> ImageValidationConfig:
        """
        Get the effective configuration by merging instance config,
        class default, and overrides.
        """
        return dataclasses.replace(self._image_validation_config, **overrides)

    def _handle_download_Exception(self, exc: Exception, config: ImageValidationConfig):
        if isinstance(exc, ToolkitImageLoadException):
            raise ImageLoadException(
                input=self,
                location=self._loc,
                billable_units=0,
            )
        elif isinstance(exc, ToolkitFileDownloadException):
            raise FileDownloadException(
                input=self,
                location=self._loc,
                billable_units=0,
            )
        elif isinstance(exc, ToolkitFileSizeExceededException):
            raise FileTooLargeException(
                input=self,
                location=self._loc,
                max_size=config.max_file_size,  # type: ignore
                billable_units=0,
            )
        elif isinstance(exc, ToolkitDataFormatException):
            raise ImageLoadException(
                input=self,
                location=self._loc,
                billable_units=0,
            )

        raise exc

    def _check_image(self, image: "PILImage", config: ImageValidationConfig):
        width, height = image.size

        if width == 0 or height == 0:
            raise ImageLoadException(
                input=self,
                location=self._loc,
                billable_units=0,
            )

        width_too_small = config.min_width is not None and width < config.min_width
        height_too_small = config.min_height is not None and height < config.min_height
        if width_too_small or height_too_small:
            raise ImageTooSmallException(
                input=self,
                location=self._loc,
                min_width=config.min_width,  # type: ignore
                min_height=config.min_height,  # type: ignore
                billable_units=0,
            )

        width_exceeds = config.max_width is not None and width > config.max_width
        height_exceeds = config.max_height is not None and height > config.max_height
        if width_exceeds or height_exceeds:
            raise ImageTooLargeException(
                input=self,
                location=self._loc,
                max_width=config.max_width,  # type: ignore[call-arg]
                max_height=config.max_height,  # type: ignore[call-arg]
                billable_units=0,
            )

        max_aspect_ratio = max(width, height) / min(width, height)
        min_aspect_ratio = min(width, height) / max(width, height)

        if (
            config.min_aspect_ratio is not None
            and min_aspect_ratio < config.min_aspect_ratio
        ):
            raise ImageAspectRatioException(
                input=self,
                location=self._loc,
                min_aspect_ratio=config.min_aspect_ratio,
                max_aspect_ratio=config.max_aspect_ratio,
                billable_units=0,
            )

        if (
            config.max_aspect_ratio is not None
            and max_aspect_ratio > config.max_aspect_ratio
        ):
            raise ImageAspectRatioException(
                input=self,
                location=self._loc,
                min_aspect_ratio=config.min_aspect_ratio,
                max_aspect_ratio=config.max_aspect_ratio,
                billable_units=0,
            )

    async def _load_and_validate_pil_async(self, **overrides):
        config = self._get_effective_config(**overrides)

        try:
            image = await read_image_from_url_async(
                self,
                timeout=config.timeout,
                max_size=config.max_file_size,
            )
        except Exception as exc:
            self._handle_download_Exception(exc, config)

        self._check_image(image, config)
        return image

    def _load_and_validate_pil(self, **overrides) -> "PILImage":
        config = self._get_effective_config(**overrides)
        try:
            image = read_image_from_url(
                self,
                timeout=config.timeout,
                max_size=config.max_file_size,
            )
        except Exception as exc:
            self._handle_download_Exception(exc, config)

        self._check_image(image, config)
        return image

    def to_pil(self, **options: Unpack[ImageValidationOptions]) -> "PILImage":
        """
        Downloads, validates, and returns a Pillow Image.
        """
        return self._load_and_validate_pil(**options)

    async def to_pil_async(self, **options: Unpack[ImageValidationOptions]):
        return await self._load_and_validate_pil_async(**options)

    def to_bytes(
        self,
        format: Literal["png", "jpeg"] = "jpeg",
        **options: Unpack[ImageValidationOptions],
    ) -> bytes:
        """Downloads, validates, and returns the image as bytes."""
        pil_image = self.to_pil(**options)
        return encode_image(pil_image, format=format, return_type="bytes")

    async def to_bytes_async(
        self,
        format: Literal["png", "jpeg"] = "jpeg",
        **options: Unpack[ImageValidationOptions],
    ) -> bytes:
        """Downloads, validates, and returns the image as bytes."""
        pil_image = await self.to_pil_async(**options)
        return encode_image(pil_image, format=format, return_type="bytes")

    def to_base64(
        self,
        format: Literal["png", "jpeg"] = "jpeg",
        **options: Unpack[ImageValidationOptions],
    ) -> str:
        """Downloads, validates, and returns the image as a Base64 encoded string."""
        pil_image = self.to_pil(**options)
        return encode_image(pil_image, format=format, return_type="base64")

    async def to_base64_async(
        self,
        format: Literal["png", "jpeg"] = "jpeg",
        **options: Unpack[ImageValidationOptions],
    ) -> str:
        """Downloads, validates, and returns the image as a Base64 encoded string."""
        pil_image = await self.to_pil_async(**options)
        return encode_image(pil_image, format=format, return_type="base64")

    def to_data_uri(  # type: ignore
        self,
        format: Literal["png", "jpeg"] = "jpeg",
        **options: Unpack[ImageValidationOptions],
    ) -> str:
        """Downloads, validates, and returns the image as a Base64 data URI."""
        pil_image = self.to_pil(**options)
        return encode_image(pil_image, format=format, return_type="data_uri")

    async def to_data_uri_async(  # type: ignore
        self,
        format: Literal["png", "jpeg"] = "jpeg",
        **options: Unpack[ImageValidationOptions],
    ) -> str:
        """Downloads, validates, and returns the image as a Base64 data URI."""
        pil_image = await self.to_pil_async(**options)
        return encode_image(pil_image, format=format, return_type="data_uri")


class FieldContextProviderModel(BaseModel):
    class Config:  # type: ignore[bad-override]
        @staticmethod
        def schema_extra(
            schema: dict[str, Any], model: "FieldContextProviderModel"
        ) -> None:
            for field_name, field_schema in schema["properties"].items():
                if not isinstance(field_schema, dict):
                    continue

                description = field_schema.get("description", "")
                limit_description = field_schema.pop("limit_description", "")
                if limit_description:
                    description += "\n\n" + limit_description
                if limit_description or description:
                    field_schema["description"] = description

    if _IS_PYDANTIC_V2:

        @model_validator(mode="after")
        def _inject_full_context_path(self):
            for field_name in self.__class__.model_fields:
                field_value = getattr(self, field_name, None)
                if field_value is not None:
                    _bind_context_recursively(
                        field_value, path_prefix=(field_name,)
                    )
            return self

    else:

        @pydantic.root_validator(pre=False)
        def _inject_full_context_path(cls, values: dict) -> dict:
            for field_name, field_value in values.items():
                _bind_context_recursively(
                    field_value, path_prefix=(field_name,)
                )
            return values


class OrderedBaseModel(BaseModel):
    SCHEMA_IGNORES: ClassVar[set[str]] = set()
    FIELD_ORDERS: ClassVar[list[str]] = []

    class Config:  # type: ignore[bad-override]
        @staticmethod
        def schema_extra(schema: dict[str, Any], model: "OrderedBaseModel") -> None:
            # Remove the model_name and scheduler fields from the schema
            for key in model.SCHEMA_IGNORES:
                schema["properties"].pop(key, None)

            # Reorder the fields to make sure FIELD_ORDERS are accurate,
            # any missing fields will be appearing at the end of the schema.
            properties = {}
            for field in model.FIELD_ORDERS:
                if props := schema["properties"].pop(field, None):
                    properties[field] = props

            schema["properties"] = {**properties, **schema["properties"]}

        @staticmethod
        def json_schema_extra(
            schema: dict[str, Any], model: "OrderedBaseModel"
        ) -> None:
            OrderedBaseModel.Config.schema_extra(schema, model)


class FalBaseModel(OrderedBaseModel, FieldContextProviderModel):
    # Inheriting Config from both OrderedBaseModel and FieldContextProviderModel
    # is not allowed.
    class Config:  # type: ignore[bad-override]
        @staticmethod
        def schema_extra(schema: dict[str, Any], model: "FalBaseModel") -> None:
            FieldContextProviderModel.Config.schema_extra(schema, model)
            OrderedBaseModel.Config.schema_extra(schema, model)


class ZipUrl(HttpsOrDataUrl):
    _fal_ui_field_name = "archive"


class ImageMaskUrl(ImageUrl):
    _fal_ui_field_name = "image_mask"


class VideoUrl(HttpsOrDataUrl):
    _fal_ui_field_name = "video"


class AudioUrl(HttpsOrDataUrl):
    _fal_ui_field_name = "audio"
