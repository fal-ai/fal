from __future__ import annotations

import io
from typing import TYPE_CHECKING, Literal, Optional, Union

from fal_serverless.toolkit import mainify
from fal_serverless.toolkit.file import DEFAULT_REPOSITORY, File
from fal_serverless.toolkit.file.types import FileData, FileRepository, RepositoryId
from pydantic import Field
from pydantic.dataclasses import dataclass

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


@dataclass
@mainify
class ImageSize:
    width: int = Field(
        default=512, description="The width of the generated image.", gt=0, le=4096
    )
    height: int = Field(
        default=512, description="The height of the generated image.", gt=0, le=4096
    )

    # NOTE: This is a workaround for incompatibility between pydantic and mypy
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height


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


get_image_size.__module__ = "__main__"

ImageFormat = Literal["png", "jpeg", "jpg", "webp", "gif"]


@mainify
@dataclass
class Image(File):
    """
    Represents an image file.
    """

    width: int = Field(description="The width of the image in pixels.")
    height: int = Field(description="The height of the image in pixels.")

    def __init__(
        self,
        data: FileData,
        image_size: ImageSize | None,
        repository: FileRepository | RepositoryId,
    ):
        if image_size is not None:
            self.width = image_size.width
            self.height = image_size.height
        super().__init__(data, repository)

    @classmethod
    def from_bytes(  # type: ignore[override]
        cls,
        data: bytes,
        format: ImageFormat,
        size: ImageSize | None = None,
        file_name: str | None = None,
        repository: FileRepository | RepositoryId = DEFAULT_REPOSITORY,
    ) -> Image:
        return cls(
            FileData(data=data, content_type=f"image/{format}", file_name=file_name),
            size,
            repository,
        )

    @classmethod
    def from_pil(
        cls,
        pil_image: PILImage.Image,
        format: ImageFormat | None = None,
        size: ImageSize | None = None,
        file_name: str | None = None,
        repository: FileRepository | RepositoryId = DEFAULT_REPOSITORY,
    ) -> Image:
        if size is None:
            size = ImageSize(pil_image.size[0], pil_image.size[1])
        if format is None:
            format = pil_image.format

        with io.BytesIO() as f:
            pil_image.save(f, format=format or "png")
            raw_image = f.getvalue()

        return cls.from_bytes(raw_image, format, size, file_name, repository)
