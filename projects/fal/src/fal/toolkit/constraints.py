from __future__ import annotations

import dataclasses
from typing import Any, Optional, TypedDict

# Runtime settings (download/processing), not client-checkable limits, so they
# are never emitted in the ``x-fal`` schema extension.
_NON_SCHEMA_FIELDS = {"timeout"}


def to_xfal(
    config: ImageSizeConstraints | ImageValidationConfig,
) -> dict[str, Any]:
    """Return a config's set (non-None) limits as the ``x-fal`` schema payload."""
    return {
        key: value
        for key, value in dataclasses.asdict(config).items()
        if value is not None and key not in _NON_SCHEMA_FIELDS
    }


def _validate_aspect_ratio_pair(
    min_aspect_ratio: Optional[float], max_aspect_ratio: Optional[float]
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

    min_width: Optional[int] = None
    min_height: Optional[int] = None
    max_width: Optional[int] = None
    max_height: Optional[int] = None
    min_area: Optional[int] = None
    max_area: Optional[int] = None
    multiple_of: Optional[int] = None
    min_aspect_ratio: Optional[float] = None
    max_aspect_ratio: Optional[float] = None

    def __post_init__(self) -> None:
        _validate_aspect_ratio_pair(self.min_aspect_ratio, self.max_aspect_ratio)


class ImageValidationOptions(TypedDict, total=False):
    """Validation options accepted by input-image helpers."""

    max_file_size: Optional[int]
    min_width: Optional[int]
    min_height: Optional[int]
    max_width: Optional[int]
    max_height: Optional[int]
    min_aspect_ratio: Optional[float]
    max_aspect_ratio: Optional[float]
    timeout: Optional[float]


@dataclasses.dataclass(frozen=True)
class ImageValidationConfig:
    """Limits applied to an input image. Surfaced in the schema (``x-fal``); the
    SDK does not enforce them."""

    max_file_size: Optional[int] = None
    min_width: Optional[int] = None
    min_height: Optional[int] = None
    max_width: Optional[int] = None
    max_height: Optional[int] = None
    min_aspect_ratio: Optional[float] = None
    max_aspect_ratio: Optional[float] = None
    timeout: float = 20.0

    def __post_init__(self) -> None:
        _validate_aspect_ratio_pair(self.min_aspect_ratio, self.max_aspect_ratio)
