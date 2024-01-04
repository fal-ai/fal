from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING, Literal, Optional, Union

from fal.toolkit.file.file import DEFAULT_REPOSITORY, File
from fal.toolkit.file.types import FileRepository, RepositoryId
from fal.toolkit.mainify import mainify
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

@mainify
def get_image_size(source: ImageSizeInput) -> ImageSize:
    if isinstance(source, ImageSize):
        return source
    if isinstance(source, str) and source in IMAGE_SIZE_PRESETS:
        size = IMAGE_SIZE_PRESETS[source]
        return size
    raise TypeError(f"Invalid value for ImageSize: {source}")


ImageFormat = Literal["png", "jpeg", "jpg", "webp", "gif"]


@mainify
class Image(File):
    """
    Represents an image file.
    """

    width: Optional[int] = Field(
        description="The width of the image in pixels.",
        examples=[1024],
    )
    height: Optional[int] = Field(
        description="The height of the image in pixels.", examples=[1024]
    )

    @classmethod
    def __bytes_to_pil(cls, data: bytes) -> PILImage.Image:
        from PIL import Image as PILImage

        image_buffer = BytesIO(data)
        return PILImage.open(image_buffer)

    @classmethod
    def from_bytes(  # type: ignore[override]
        cls,
        data: bytes,
        format: ImageFormat | None = None,
        file_name: str | None = None,
        repository: FileRepository | RepositoryId = DEFAULT_REPOSITORY,
    ) -> Image:
        pil_image = cls.__bytes_to_pil(data)

        return cls.from_pil(
            pil_image=pil_image,
            format=format,
            file_name=file_name,
            repository=repository,
        )

    @classmethod
    def _from_url(cls, url: str):
        return super()._from_url(url)

    @classmethod
    def from_pil(
        cls,
        pil_image: PILImage.Image,
        format: ImageFormat | None = None,
        file_name: str | None = None,
        repository: FileRepository | RepositoryId = DEFAULT_REPOSITORY,
    ) -> Image:
        if format is None:
            format = pil_image.format or "png"  # type: ignore[assignment]
            assert format  # for type checker
        
        content_type = f"image/{format}"

        saving_options = {}
        if format == "png":
            # PNG compression is an extremely slow process, and for the
            # purposes of our client applications we want to get a good
            # enough result quickly to utilize the underlying resources
            # efficiently.
            saving_options["compress_level"] = 1

        with BytesIO() as f:
            pil_image.save(f, format=format, **saving_options)
            raw_image = f.getvalue()

        fal_image = super().from_bytes(
            data=raw_image,
            repository=repository,
            content_type=content_type,
            file_name=file_name,
        )

        fal_image.width = pil_image.width
        fal_image.height = pil_image.height


        return fal_image

    def to_pil(self) -> PILImage.Image:
        return self.__bytes_to_pil(self.as_bytes())
