from __future__ import annotations

import io
from typing import TYPE_CHECKING, Literal, Optional, Union

from fal_serverless.toolkit import mainify
from fal_serverless.toolkit.file.file import (
    DEFAULT_REPOSITORY,
    File,
    get_builtin_repository,
)
from fal_serverless.toolkit.file.types import FileData, FileRepository, RepositoryId
from pydantic import BaseModel, Field

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


@mainify
class ImageSize(BaseModel):
    width: int = Field(
        default=512, description="The width of the generated image.", gt=0, le=4096
    )
    height: int = Field(
        default=512, description="The height of the generated image.", gt=0, le=4096
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


get_image_size.__module__ = "__main__"

ImageFormat = Literal["png", "jpeg", "jpg", "webp", "gif"]


@mainify
class Image(File):
    """
    Represents an image file.
    """

    width: Optional[int] = Field(description="The width of the image in pixels.")
    height: Optional[int] = Field(description="The height of the image in pixels.")

    @classmethod
    def from_bytes(  # type: ignore[override]
        cls,
        data: bytes,
        format: ImageFormat,
        size: ImageSize | None = None,
        file_name: str | None = None,
        repository: FileRepository | RepositoryId = DEFAULT_REPOSITORY,
    ) -> Image:
        repo = (
            repository
            if isinstance(repository, FileRepository)
            else get_builtin_repository(repository)
        )
        filedata = FileData(
            data=data, content_type=f"image/{format}", file_name=file_name
        )

        return cls(
            width=size.width if size else None,
            height=size.height if size else None,
            url=repo.save(filedata),
            content_type=filedata.content_type,
            file_name=filedata.file_name,
            file_size=len(filedata.data),
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
            size = ImageSize(width=pil_image.size[0], height=pil_image.size[1])
        if format is None:
            format = pil_image.format or "png"

        with io.BytesIO() as f:
            pil_image.save(f, format=format)
            raw_image = f.getvalue()

        return cls.from_bytes(raw_image, format or "png", size, file_name, repository)
