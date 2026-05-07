from enum import Enum


class DocumentQualifyRequestMediaType(str, Enum):
    AUDIO = "audio"
    IMAGE = "image"
    VALUE_3 = "3d"
    VIDEO = "video"

    def __str__(self) -> str:
        return str(self.value)
