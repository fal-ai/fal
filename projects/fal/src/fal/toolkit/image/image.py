from __future__ import annotations

import io
from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal, Optional, Union
from urllib.parse import urlparse

from fastapi import Request
from pydantic import BaseModel, Field

from fal.toolkit.file.file import (
    DEFAULT_REPOSITORY,
    FALLBACK_REPOSITORY,
    IS_PYDANTIC_V2,
    File,
)
from fal.toolkit.file.types import FileRepository, RepositoryId
from fal.toolkit.utils.download_utils import TEMP_HEADERS, _download_file_python
from fal.toolkit.utils.ssrf import ssrf_safe_get_to_file

if TYPE_CHECKING:
    from PIL import Image as PILImage


MAX_IMAGE_DOWNLOAD_SIZE = 50 * 1024 * 1024

ImageSizePreset = Literal[
    "square",
    "portrait_4_3",
    "portrait_16_9",
    "landscape_4_3",
    "landscape_16_9",
]
ImageSizePresetHD = Literal[
    "square_hd",
    "portrait_4_3_hd",
    "portrait_16_9_hd",
    "landscape_4_3_hd",
    "landscape_16_9_hd",
]
ImageSizePresetFullHD = Literal[
    "square_fhd",
    "portrait_4_3_fhd",
    "portrait_16_9_fhd",
    "landscape_16_9_fhd",
    "landscape_4_3_fhd",
]
ImageSizePresetQuadHD = Literal[
    "square_qhd",
    "portrait_4_3_qhd",
    "portrait_16_9_qhd",
    "landscape_4_3_qhd",
    "landscape_16_9_qhd",
]
ImageSizePresetUltraHD = Literal[
    "square_uhd",
    "portrait_4_3_uhd",
    "portrait_16_9_uhd",
    "landscape_4_3_uhd",
    "landscape_16_9_uhd",
]

# This syntax properly combines literals at runtime for introspection
ImageSizePresetUpToHD = Literal[ImageSizePreset, ImageSizePresetHD]
ImageSizePresetUpToFullHD = Literal[ImageSizePresetUpToHD, ImageSizePresetFullHD]
ImageSizePresetUpToQuadHD = Literal[ImageSizePresetUpToFullHD, ImageSizePresetQuadHD]
ImageSizePresetUpToUltraHD = Literal[ImageSizePresetUpToQuadHD, ImageSizePresetUltraHD]


class ImageSize(BaseModel):
    width: int = Field(
        default=512, description="The width of the generated image.", gt=0, le=14142
    )
    height: int = Field(
        default=512, description="The height of the generated image.", gt=0, le=14142
    )


IMAGE_SIZE_PRESETS: dict[ImageSizePresetUpToUltraHD, ImageSize] = {
    # legacy presets
    "square": ImageSize(width=512, height=512),
    "portrait_4_3": ImageSize(width=768, height=1024),
    "portrait_16_9": ImageSize(width=576, height=1024),
    "landscape_4_3": ImageSize(width=1024, height=768),
    "landscape_16_9": ImageSize(width=1024, height=576),
    # hd presets
    "square_hd": ImageSize(width=1024, height=1024),  # only hd legacy preset
    "portrait_4_3_hd": ImageSize(width=960, height=1280),
    "portrait_16_9_hd": ImageSize(width=720, height=1280),
    "landscape_4_3_hd": ImageSize(width=1280, height=960),
    "landscape_16_9_hd": ImageSize(width=1280, height=720),
    # full HD presets
    "square_fhd": ImageSize(width=1440, height=1440),
    "portrait_4_3_fhd": ImageSize(width=1440, height=1920),
    "portrait_16_9_fhd": ImageSize(width=1080, height=1920),
    "landscape_16_9_fhd": ImageSize(width=1920, height=1080),
    "landscape_4_3_fhd": ImageSize(width=1920, height=1440),
    # quad HD presets
    "square_qhd": ImageSize(width=1920, height=1920),
    "portrait_4_3_qhd": ImageSize(width=1920, height=2560),
    "portrait_16_9_qhd": ImageSize(width=1440, height=2560),
    "landscape_16_9_qhd": ImageSize(width=2560, height=1440),
    "landscape_4_3_qhd": ImageSize(width=2560, height=1920),
    # ultra HD presets
    "square_uhd": ImageSize(width=2560, height=2560),
    "portrait_4_3_uhd": ImageSize(width=2880, height=3840),
    "portrait_16_9_uhd": ImageSize(width=2160, height=3840),
    "landscape_16_9_uhd": ImageSize(width=3840, height=2160),
    "landscape_4_3_uhd": ImageSize(width=3840, height=2880),
}

ImageSizeInput = Union[ImageSize, ImageSizePreset]
ImageSizeInputHD = Union[ImageSize, ImageSizePresetHD]
ImageSizeInputFullHD = Union[ImageSize, ImageSizePresetFullHD]
ImageSizeInputQuadHD = Union[ImageSize, ImageSizePresetQuadHD]
ImageSizeInputUltraHD = Union[ImageSize, ImageSizePresetUltraHD]

ImageSizeInputUpToHD = Union[ImageSize, ImageSizePresetUpToHD]
ImageSizeInputUpToFullHD = Union[ImageSize, ImageSizePresetUpToFullHD]
ImageSizeInputUpToQuadHD = Union[ImageSize, ImageSizePresetUpToQuadHD]
ImageSizeInputUpToUltraHD = Union[ImageSize, ImageSizePresetUpToUltraHD]


def get_image_size(source: ImageSizeInputUpToUltraHD) -> ImageSize:
    if isinstance(source, ImageSize):
        return source
    if isinstance(source, str) and source in IMAGE_SIZE_PRESETS:
        size = IMAGE_SIZE_PRESETS[source]
        return size
    raise TypeError(f"Invalid value for ImageSize: {source}")


ImageFormat = Literal["png", "jpeg", "jpg", "webp", "gif"]


@wraps(Field)
def ImageField(*args, **kwargs):
    if IS_PYDANTIC_V2:
        # Pydantic v2: use json_schema_extra
        json_schema_extra = kwargs.pop("json_schema_extra", None) or {}
        if callable(json_schema_extra):
            # If it's a callable, wrap it to also add ui.field
            original_func = json_schema_extra

            def merged_schema_extra(schema):
                original_func(schema)
                schema.setdefault("ui", {}).setdefault("field", "image")

            kwargs["json_schema_extra"] = merged_schema_extra
        else:
            json_schema_extra.setdefault("ui", {}).setdefault("field", "image")
            kwargs["json_schema_extra"] = json_schema_extra
    else:
        # Pydantic v1: use ui kwarg (stored in extra)
        ui = kwargs.get("ui", {})
        ui.setdefault("field", "image")
        kwargs["ui"] = ui
    return Field(*args, **kwargs)


class Image(File):
    """
    Represents an image file.
    """

    width: Optional[int] = Field(
        None,
        description="The width of the image in pixels.",
        examples=[1024],
    )
    height: Optional[int] = Field(
        None, description="The height of the image in pixels.", examples=[1024]
    )

    if IS_PYDANTIC_V2:
        model_config = {"json_schema_extra": {"ui": {"field": "image"}}}
    else:

        class Config:
            @staticmethod
            def schema_extra(schema, model_type):
                schema.setdefault("ui", {})["field"] = "image"

    @classmethod
    def from_bytes(  # type: ignore[override]
        cls,
        data: bytes,
        format: ImageFormat,
        size: ImageSize | None = None,
        file_name: str | None = None,
        repository: FileRepository | RepositoryId = DEFAULT_REPOSITORY,
        fallback_repository: Optional[
            FileRepository | RepositoryId | list[FileRepository | RepositoryId]
        ] = FALLBACK_REPOSITORY,
        request: Optional[Request] = None,
        save_kwargs: Optional[dict] = None,
        fallback_save_kwargs: Optional[dict] = None,
    ) -> Image:
        obj = super().from_bytes(
            data,
            content_type=f"image/{format}",
            file_name=file_name,
            repository=repository,
            fallback_repository=fallback_repository,
            request=request,
            save_kwargs=save_kwargs,
            fallback_save_kwargs=fallback_save_kwargs,
        )
        obj.width = size.width if size else None
        obj.height = size.height if size else None
        return obj

    @classmethod
    def from_pil(
        cls,
        pil_image: PILImage.Image,
        format: ImageFormat | None = None,
        file_name: str | None = None,
        repository: FileRepository | RepositoryId = DEFAULT_REPOSITORY,
        fallback_repository: Optional[
            FileRepository | RepositoryId | list[FileRepository | RepositoryId]
        ] = FALLBACK_REPOSITORY,
        request: Optional[Request] = None,
        save_kwargs: Optional[dict] = None,
        fallback_save_kwargs: Optional[dict] = None,
    ) -> Image:
        size = ImageSize(width=pil_image.width, height=pil_image.height)
        if format is None:
            format = pil_image.format or "png"  # type: ignore[assignment]
            assert format  # for type checker

        saving_options = {}
        if format == "png":
            # PNG compression is an extremely slow process, and for the
            # purposes of our client applications we want to get a good
            # enough result quickly to utilize the underlying resources
            # efficiently.
            saving_options["compress_level"] = 1
        elif format == "jpeg":
            # JPEG quality is set to 95 by default, which is a good balance
            # between file size and image quality.
            saving_options["quality"] = 95

        with io.BytesIO() as f:
            pil_image.save(f, format=format, **saving_options)
            raw_image = f.getvalue()

        return cls.from_bytes(
            raw_image,
            format,
            size,
            file_name,
            repository,
            fallback_repository=fallback_repository,
            request=request,
            save_kwargs=save_kwargs,
            fallback_save_kwargs=fallback_save_kwargs,
        )

    def to_pil(self, mode: str = "RGB") -> PILImage.Image:
        try:
            from PIL import Image as PILImage
            from PIL import ImageOps
        except ImportError:
            raise ImportError("The PIL package is required to use Image.to_pil().")

        # Stream the image data from url to a temp file and convert it to a PIL image
        with TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir) / "image"

            if urlparse(self.url).scheme == "data":
                _download_file_python(self.url, temp_file_path)
            else:
                ssrf_safe_get_to_file(
                    self.url,
                    temp_file_path,
                    headers=TEMP_HEADERS,
                    max_size=MAX_IMAGE_DOWNLOAD_SIZE,
                )

            img = PILImage.open(temp_file_path).convert(mode)
            img = ImageOps.exif_transpose(img)

            return img
