from __future__ import annotations

import io
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Literal, Optional, Union

from pydantic import BaseModel, Field

from fal.toolkit.file.file import DEFAULT_REPOSITORY, File
from fal.toolkit.file.types import FileData, FileRepository, RepositoryId
from fal.toolkit.mainify import mainify
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
    def from_bytes(  # type: ignore[override]
        cls,
        data: bytes,
        format: ImageFormat,
        size: ImageSize | None = None,
        file_name: str | None = None,
        repository: FileRepository | RepositoryId = DEFAULT_REPOSITORY,
    ) -> Image:
        file_data = FileData(
            data=data, content_type=f"image/{format}", file_name=file_name
        )
        return cls(
            file_data=file_data,
            repository=repository,
            width=size.width if size else None,
            height=size.height if size else None,
        )

    @classmethod
    def from_pil(
        cls,
        pil_image: PILImage.Image,
        format: ImageFormat | None = None,
        file_name: str | None = None,
        repository: FileRepository | RepositoryId = DEFAULT_REPOSITORY,
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

        with io.BytesIO() as f:
            pil_image.save(f, format=format, **saving_options)
            raw_image = f.getvalue()

        return cls.from_bytes(raw_image, format, size, file_name, repository)

    def to_pil(self, mode: str = "RGB") -> PILImage.Image:
        try:
            from PIL import Image as PILImage
            from PIL import ImageOps
        except ImportError:
            raise ImportError(
                "The PIL package is required to use Image.to_pil()."
            )

        # Stream the image data from url to a temp file and convert it to a PIL image
        with NamedTemporaryFile() as temp_file:
            temp_file_path = temp_file.name

            _download_file_python(self.url, temp_file_path)

            img = PILImage.open(temp_file_path).convert(mode)
            img = ImageOps.exif_transpose(img)

            return img

