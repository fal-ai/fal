from enum import Enum


class DocumentIngestRequestMediaType(str, Enum):
    AUDIO = "audio"
    IMAGE = "image"
    VALUE_3 = "3d"
    VIDEO = "video"

    def __str__(self) -> str:
        return str(self.value)
