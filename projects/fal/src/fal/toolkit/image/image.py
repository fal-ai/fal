from __future__ import annotations

import io
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Literal, Optional, Union

from fastapi import Request
from pydantic import BaseModel, Field

from fal.toolkit.file.file import DEFAULT_REPOSITORY, FALLBACK_REPOSITORY, File
from fal.toolkit.file.types import FileRepository, RepositoryId
from fal.toolkit.utils.download_utils import _download_file_python

if TYPE_CHECKING:
    from PIL import Image as PILImage


ImageSizePreset = Literal[
    "square_hd",
    "square",
    "portrait_4_3",
    "portrait_16_9",
    "landscape_4_3",
    "landscape_16_9",
]


class ImageSize(BaseModel):
    width: int = Field(
        default=512, description="The width of the generated image.", gt=0, le=14142
    )
    height: int = Field(
        default=512, description="The height of the generated image.", gt=0, le=14142
    )


IMAGE_SIZE_PRESETS: dict[ImageSizePreset, ImageSize] = {
    "square_hd": ImageSize(width=1024, height=1024),
    "square": ImageSize(width=512, height=512),
    "portrait_4_3": ImageSize(width=768, height=1024),
    "portrait_16_9": ImageSize(width=576, height=1024),
    "landscape_4_3": ImageSize(width=1024, height=768),
    "landscape_16_9": ImageSize(width=1024, height=576),
}

ImageSizeInput = Union[ImageSize, ImageSizePreset]


def get_image_size(source: ImageSizeInput) -> ImageSize:
    if isinstance(source, ImageSize):
        return source
    if isinstance(source, str) and source in IMAGE_SIZE_PRESETS:
        size = IMAGE_SIZE_PRESETS[source]
        return size
    raise TypeError(f"Invalid value for ImageSize: {source}")


ImageFormat = Literal["png", "jpeg", "jpg", "webp", "gif"]


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

    @classmethod
    def from_bytes(  # type: ignore[override]
        cls,
        data: bytes,
        format: ImageFormat,
        size: ImageSize | None = None,
        file_name: str | None = None,
        repository: FileRepository | RepositoryId = DEFAULT_REPOSITORY,
        fallback_repository: Optional[
            FileRepository | RepositoryId
        ] = FALLBACK_REPOSITORY,
        request: Optional[Request] = None,
    ) -> Image:
        obj = super().from_bytes(
            data,
            content_type=f"image/{format}",
            file_name=file_name,
            repository=repository,
            fallback_repository=fallback_repository,
            request=request,
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
            FileRepository | RepositoryId
        ] = FALLBACK_REPOSITORY,
        request: Optional[Request] = None,
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
        )

    def to_pil(self, mode: str = "RGB") -> PILImage.Image:
        try:
            from PIL import Image as PILImage
            from PIL import ImageOps
        except ImportError:
            raise ImportError("The PIL package is required to use Image.to_pil().")

        # Stream the image data from url to a temp file and convert it to a PIL image
        with NamedTemporaryFile() as temp_file:
            temp_file_path = temp_file.name

            _download_file_python(self.url, temp_file_path)

            img = PILImage.open(temp_file_path).convert(mode)
            img = ImageOps.exif_transpose(img)

            return img
