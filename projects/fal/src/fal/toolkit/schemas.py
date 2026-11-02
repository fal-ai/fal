"""Typed generation results built from existing File fields and explicit metadata.

Upload through File.from_bytes() or File.from_path() first. Inherited upload
constructors cannot supply required media fields. Revisit those constructors if
media upload helpers become part of this API.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class ImageMedia(BaseModel):
    type: Literal["image"] = "image"
    width: int
    height: int


class VideoMedia(BaseModel):
    type: Literal["video"] = "video"
    width: int
    height: int
    duration: float
    fps: float


class AudioMedia(BaseModel):
    type: Literal["audio"] = "audio"
    duration: float
    sample_rate: int


Media = Annotated[ImageMedia | VideoMedia | AudioMedia, Field(discriminator="type")]


class GenerationOutput(BaseModel):
    media: list[Media]
