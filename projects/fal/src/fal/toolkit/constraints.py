from __future__ import annotations

import dataclasses
from typing import Any, TypedDict

# Runtime settings (download/processing), not client-checkable limits, so they
# are never emitted in the ``x-fal`` schema extension.
_NON_SCHEMA_FIELDS = {"timeout"}


def to_xfal(
    config: ImageSizeConstraints
    | ImageValidationConfig
    | VideoValidationConfig
    | VideoNormalizationConfig,
) -> dict[str, Any]:
    """Return a config's set (non-None) limits as the ``x-fal`` schema payload."""
    return {
        key: value
        for key, value in dataclasses.asdict(config).items()
        if value is not None and key not in _NON_SCHEMA_FIELDS
    }


_BOUND_PAIRS = (
    ("min_width", "max_width"),
    ("min_height", "max_height"),
    ("min_area", "max_area"),
    ("min_frames", "max_frames"),
    ("min_duration", "max_duration"),
    ("min_fps", "max_fps"),
    ("min_aspect_ratio", "max_aspect_ratio"),
)


def _validate_bounds(config: Any) -> None:
    """Reject a config no input could satisfy. Equal bounds pin an exact value."""
    for low, high in _BOUND_PAIRS:
        lower, upper = getattr(config, low, None), getattr(config, high, None)
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(f"{low} ({lower}) must not exceed {high} ({upper}).")


def _validate_aspect_ratio_pair(
    min_aspect_ratio: float | None, max_aspect_ratio: float | None
) -> None:
    # A single bound is ambiguous since aspect ratio can be read either way.
    if (min_aspect_ratio is None) != (max_aspect_ratio is None):
        raise ValueError(
            "min_aspect_ratio and max_aspect_ratio must be provided together."
        )


@dataclasses.dataclass(frozen=True)
class ImageSizeConstraints:
    """Advisory limits on the image size a model can generate.

    Attach to an ``image_size`` field via :func:`fal.toolkit.ImageSizeField` to
    surface the model's size envelope in the OpenAPI schema (under the ``x-fal``
    extension), so clients and UIs can validate sizes before a request. These are
    documentation hints and are not enforced by the SDK.
    """

    min_width: int | None = None
    min_height: int | None = None
    max_width: int | None = None
    max_height: int | None = None
    min_area: int | None = None
    max_area: int | None = None
    multiple_of: int | None = None
    min_aspect_ratio: float | None = None
    max_aspect_ratio: float | None = None

    def __post_init__(self) -> None:
        _validate_bounds(self)
        _validate_aspect_ratio_pair(self.min_aspect_ratio, self.max_aspect_ratio)


class ImageValidationOptions(TypedDict, total=False):
    """Validation options accepted by input-image helpers."""

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
    """Limits applied to an input image. Surfaced in the schema (``x-fal``); the
    SDK does not enforce them."""

    max_file_size: int | None = None
    min_width: int | None = None
    min_height: int | None = None
    max_width: int | None = None
    max_height: int | None = None
    min_aspect_ratio: float | None = None
    max_aspect_ratio: float | None = None
    timeout: float = 20.0

    def __post_init__(self) -> None:
        _validate_bounds(self)
        _validate_aspect_ratio_pair(self.min_aspect_ratio, self.max_aspect_ratio)


class VideoValidationOptions(TypedDict, total=False):
    """Validation options accepted by input-video helpers."""

    max_file_size: int | None
    min_width: int | None
    min_height: int | None
    max_width: int | None
    max_height: int | None
    min_area: int | None
    max_area: int | None
    min_aspect_ratio: float | None
    max_aspect_ratio: float | None
    min_frames: int | None
    max_frames: int | None
    min_duration: float | None
    max_duration: float | None
    min_fps: float | None
    max_fps: float | None


@dataclasses.dataclass(frozen=True)
class VideoValidationConfig:
    """Limits applied to an input video.

    Attach to a video input via :func:`fal.toolkit.VideoField` to surface the
    limits in the OpenAPI schema (under the ``x-fal`` extension), so clients and
    UIs can reject a video before uploading it. Areas are per frame, in pixels.
    These are documentation hints and are not enforced by the SDK. Fetch settings
    such as the download timeout belong to the downloader, not here.
    """

    max_file_size: int | None = None
    min_width: int | None = None
    min_height: int | None = None
    max_width: int | None = None
    max_height: int | None = None
    min_area: int | None = None
    max_area: int | None = None
    min_aspect_ratio: float | None = None
    max_aspect_ratio: float | None = None
    min_frames: int | None = None
    max_frames: int | None = None
    min_duration: float | None = None
    max_duration: float | None = None
    min_fps: float | None = None
    max_fps: float | None = None

    def __post_init__(self) -> None:
        _validate_bounds(self)
        _validate_aspect_ratio_pair(self.min_aspect_ratio, self.max_aspect_ratio)


@dataclasses.dataclass(frozen=True)
class VideoNormalizationConfig:
    """What a model does to an input video it accepts but has to reshape.

    The counterpart to :class:`VideoValidationConfig`: those limits reject a
    request, these describe a video that is accepted and then rescaled, trimmed
    or resampled. Declaring both lets a UI block what will fail and merely warn
    about what will change ("trimmed to 15s"). Areas are per frame, in pixels.
    """

    min_width: int | None = None
    min_height: int | None = None
    max_width: int | None = None
    max_height: int | None = None
    min_area: int | None = None
    max_area: int | None = None
    max_duration: float | None = None
    fps: float | None = None

    def __post_init__(self) -> None:
        _validate_bounds(self)
